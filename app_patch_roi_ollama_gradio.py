#!/usr/bin/env python
"""Local Patch Proposed ROI classification and Ollama explanation demo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_DEFAULT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DEFAULT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gradio as gr
import numpy as np
import PIL
import torch
import torchvision

from ollama_report_service import OllamaReportService, OllamaServiceError
from patch_roi_inference_service import (
    EXPECTED_MODEL_SHA256,
    PatchROIInferenceService,
    get_inference_service,
)
from patch_roi_report_prompt import CLASS_NAME_ZH, DISCLAIMER, prompt_sha256


CLASS_MAPPING = {
    0: "Aortic enlargement",
    1: "Cardiomegaly",
    2: "Pleural thickening",
    3: "Pulmonary fibrosis",
    4: "Pleural effusion",
}
DEFAULT_MODEL = Path(
    "outputs/raddino_convnext_tiny_patch_experiment_seed42/"
    "phase2_proposed_patch_distilled/checkpoints/patch_proposed_convnext_tiny_5class.pt"
)
DEFAULT_OUTPUT = Path("outputs/patch_roi_ollama_gradio_demo")
VALIDATION_SELECTION = Path(
    "outputs/patch_proposed_class2_class4_validation_demo/selected_validation_samples.csv"
)
CLASS0_EXAMPLE = Path(
    "data/processed/bbox_crops_224/0_aortic_enlargement/"
    "01c2b9fcb0384c84648ed76c736552a8_class0_radR10_bbox0006.png"
)
INFERENCE_FIELDS = [
    "timestamp",
    "request_id",
    "image_filename",
    "image_sha256",
    "original_size",
    "original_mode",
    "predicted_class_id",
    "predicted_class_name",
    "confidence",
    "probability_class_0",
    "probability_class_1",
    "probability_class_2",
    "probability_class_3",
    "probability_class_4",
    "ground_truth",
    "is_correct",
    "classification_seconds",
    "device",
    "model_sha256",
]
OLLAMA_FIELDS = [
    "timestamp",
    "request_id",
    "ollama_base_url",
    "ollama_model",
    "prompt_sha256",
    "predicted_class_id",
    "confidence",
    "status",
    "retry_count",
    "generation_seconds",
    "response_sha256",
    "error_type",
]
_log_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_csv_locked(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        is_new = not path.exists() or path.stat().st_size == 0
        encoding = "utf-8-sig" if is_new else "utf-8"
        with path.open("a", encoding=encoding, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in fields})


def append_error(path: Path, request_id: str, exc: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = (
        f"[{utc_now()}] request_id={request_id} error_type={type(exc).__name__}\n"
        f"{traceback.format_exc()}\n"
    )
    with _log_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def dependency_versions() -> dict[str, str]:
    packages = ["gradio", "numpy", "Pillow", "torch", "torchvision"]
    return {name: importlib.metadata.version(name) for name in packages}


def read_validation_examples(project_root: Path) -> list[tuple[str, int]]:
    examples: list[tuple[str, int]] = []
    class0 = (project_root / CLASS0_EXAMPLE).resolve()
    if not class0.is_file():
        raise FileNotFoundError(f"Fixed class 0 validation example is missing: {class0}")
    examples.append((str(class0), 0))
    selection_path = (project_root / VALIDATION_SELECTION).resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(f"Validation selection CSV is missing: {selection_path}")
    selected = {2: [], 4: []}
    with selection_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            class_id = int(row["class_id"])
            if class_id in selected and len(selected[class_id]) < 2:
                image_path = Path(row["image_path"]).expanduser().resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(f"Validation example is missing: {image_path}")
                selected[class_id].append(str(image_path))
    for class_id in (2, 4):
        if len(selected[class_id]) != 2:
            raise ValueError(f"Expected two validation examples for class {class_id}")
        examples.extend((path, class_id) for path in selected[class_id])
    if len({path.casefold() for path, _ in examples}) != 5:
        raise ValueError("Validation examples are not five unique ROI paths")
    return examples


def parse_ground_truth(value: Any) -> int | None:
    if value in {None, "", "未提供"}:
        return None
    class_id = int(str(value).split(maxsplit=1)[0])
    if class_id not in CLASS_MAPPING:
        raise ValueError("Ground Truth must be one of class 0 through class 4")
    return class_id


def classification_log_row(request_id: str, result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "timestamp": utc_now(),
        "request_id": request_id,
        "image_filename": result["image_filename"],
        "image_sha256": result["image_sha256"],
        "original_size": f'{result["original_width"]}x{result["original_height"]}',
        "original_mode": result["original_mode"],
        "predicted_class_id": result["predicted_class_id"],
        "predicted_class_name": result["predicted_class_name"],
        "confidence": f'{result["confidence"]:.10f}',
        "ground_truth": result["ground_truth_class_id"],
        "is_correct": result["is_correct"],
        "classification_seconds": f'{result["inference_seconds"]:.6f}',
        "device": result["device"],
        "model_sha256": result["model_sha256"],
    }
    for index, probability in enumerate(result["probabilities"]):
        row[f"probability_class_{index}"] = f"{probability:.10f}"
    return row


def format_outputs(result: dict[str, Any]) -> tuple[str, float, list[list[Any]], str, str, dict[str, Any], str]:
    class_id = result["predicted_class_id"]
    predicted = (
        f"## {class_id} | {result['predicted_class_name']} | {CLASS_NAME_ZH[class_id]}"
    )
    rows = [
        [index, CLASS_MAPPING[index], CLASS_NAME_ZH[index], probability]
        for index, probability in enumerate(result["probabilities"])
    ]
    ranking = sorted(rows, key=lambda row: row[3], reverse=True)
    ranking_text = "\n".join(
        f"{rank}. **{row[1]} / {row[2]}**: {row[3]:.2%}"
        for rank, row in enumerate(ranking, start=1)
    )
    if result["ground_truth_class_id"] is None:
        ground_truth_text = "未提供 Ground Truth，因此不計算 Correct/Incorrect。"
    else:
        status = "Correct" if result["is_correct"] else "Incorrect"
        ground_truth_text = (
            f"Ground Truth: {result['ground_truth_class_id']} | "
            f"{result['ground_truth_class_name']} | "
            f"{CLASS_NAME_ZH[result['ground_truth_class_id']]}  \n"
            f"True-class probability: {result['true_class_probability']:.2%}  \n"
            f"Result: **{status}**"
        )
    audit = {
        "request_number": result["request_number"],
        "input_tensor_shape": result["input_tensor_shape"],
        "logits_shape": result["logits_shape"],
        "probability_sum": result["probability_sum"],
        "nan_count": result["nan_count"],
        "inf_count": result["inf_count"],
        "strict_load": result["strict_load"],
        "missing_keys": result["missing_keys"],
        "unexpected_keys": result["unexpected_keys"],
        "model_load_count": result["model_load_count"],
        "device": result["device"],
        "preprocessing_seconds": result["preprocessing_seconds"],
        "inference_seconds": result["inference_seconds"],
        "optimizer_created": result["optimizer_created"],
        "backward_called": result["backward_called"],
        "test_images_read_count": result["test_images_read_count"],
    }
    warning_text = "\n".join(f"- {warning}" for warning in result["warnings"])
    if not warning_text:
        warning_text = "輸入格式檢查通過，沒有前處理警告。"
    return predicted, result["confidence"], rows, ranking_text, ground_truth_text, audit, warning_text


def run_demo_request(
    image: Any,
    ground_truth_value: Any,
    inference_service: PatchROIInferenceService,
    ollama_service: OllamaReportService,
    output_dir: Path,
) -> tuple[Any, ...]:
    request_id = uuid.uuid4().hex
    logs = output_dir / "logs"
    try:
        ground_truth = parse_ground_truth(ground_truth_value)
        result = inference_service.predict(image, ground_truth)
        append_csv_locked(
            logs / "inference_history.csv",
            INFERENCE_FIELDS,
            classification_log_row(request_id, result),
        )
        formatted = format_outputs(result)
    except Exception as exc:
        append_error(logs / "error.log", request_id, exc)
        raise gr.Error("ROI 分類失敗。請確認圖片格式與內容，詳細資訊已寫入本機 error.log。") from None

    report = ""
    ollama_status: dict[str, Any]
    generation_seconds = 0.0
    try:
        ollama_result = ollama_service.generate(result)
        report = ollama_result["response"]
        generation_seconds = ollama_result["generation_seconds"]
        ollama_status = {
            "status": "PASS",
            "model": ollama_result["model"],
            "retry_count": ollama_result["retry_count"],
            "prompt_sha256": ollama_result["prompt_sha256"],
            "response_sha256": ollama_result["response_sha256"],
        }
        ollama_row = {
            "timestamp": utc_now(),
            "request_id": request_id,
            "ollama_base_url": ollama_service.base_url,
            "ollama_model": ollama_result["model"],
            "prompt_sha256": ollama_result["prompt_sha256"],
            "predicted_class_id": result["predicted_class_id"],
            "confidence": f'{result["confidence"]:.10f}',
            "status": "PASS",
            "retry_count": ollama_result["retry_count"],
            "generation_seconds": f"{generation_seconds:.6f}",
            "response_sha256": ollama_result["response_sha256"],
            "error_type": "",
        }
    except Exception as exc:
        append_error(logs / "error.log", request_id, exc)
        report = (
            "分類結果已完成並保留。Local Ollama 中文輔助說明目前無法產生；"
            "請確認 Ollama 服務、模型與 timeout 設定。系統沒有使用任何雲端或替代模型。"
        )
        ollama_status = {
            "status": "ERROR",
            "model": ollama_service.selected_model,
            "error_type": type(exc).__name__,
        }
        ollama_row = {
            "timestamp": utc_now(),
            "request_id": request_id,
            "ollama_base_url": ollama_service.base_url,
            "ollama_model": ollama_service.selected_model or "",
            "prompt_sha256": prompt_sha256(),
            "predicted_class_id": result["predicted_class_id"],
            "confidence": f'{result["confidence"]:.10f}',
            "status": "ERROR",
            "retry_count": 0,
            "generation_seconds": "0.000000",
            "response_sha256": "",
            "error_type": type(exc).__name__,
        }
    append_csv_locked(logs / "ollama_history.csv", OLLAMA_FIELDS, ollama_row)
    return (*formatted, report, ollama_status, generation_seconds, DISCLAIMER)


def find_port_pid(port: int) -> list[int]:
    try:
        import psutil

        return sorted(
            {
                connection.pid
                for connection in psutil.net_connections(kind="tcp")
                if connection.pid
                and connection.laddr
                and connection.laddr.port == port
                and connection.status == psutil.CONN_LISTEN
            }
        )
    except Exception:
        return []


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
            return True
        except OSError:
            return False


def select_server_port(host: str, requested_port: int) -> tuple[int, list[int]]:
    if port_available(host, requested_port):
        return requested_port, []
    pids = find_port_pid(requested_port)
    if requested_port != 7860:
        raise RuntimeError(f"Server port {requested_port} is already in use; PID(s): {pids}")
    if not port_available(host, 7861):
        raise RuntimeError(f"Ports 7860 and 7861 are both in use; 7860 PID(s): {pids}")
    return 7861, pids


def create_startup_audit(
    args: argparse.Namespace,
    inference: PatchROIInferenceService,
    ollama: OllamaReportService,
    examples: list[tuple[str, int]],
    selected_port: int,
    occupied_port_pids: list[int],
) -> dict[str, Any]:
    health = inference.health()
    ollama_health = ollama.get_tags()
    selected_model = ollama.resolve_model()
    return {
        "status": "PASS",
        "timestamp": utc_now(),
        "dry_run": bool(args.dry_run),
        "project_root": str(args.project_root),
        "model": health,
        "expected_model_sha256": EXPECTED_MODEL_SHA256,
        "ollama": {**ollama_health, "selected_model": selected_model},
        "examples": [
            {"image_path": path, "ground_truth_class_id": class_id}
            for path, class_id in examples
        ],
        "example_count": len(examples),
        "test_images_read_count": 0,
        "server": {
            "server_name": args.server_name,
            "requested_port": args.server_port,
            "selected_port": selected_port,
            "occupied_requested_port_pids": occupied_port_pids,
            "share": False,
            "show_error": False,
            "public_api": False,
        },
        "logs": {
            "startup_audit": str(args.output_dir / "logs" / "app_startup_audit.json"),
            "inference_history": str(args.output_dir / "logs" / "inference_history.csv"),
            "ollama_history": str(args.output_dir / "logs" / "ollama_history.csv"),
            "error_log": str(args.output_dir / "logs" / "error.log"),
        },
        "dependencies": dependency_versions(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(inference.device) if inference.device.type == "cuda" else None
            ),
        },
        "prompt_sha256": prompt_sha256(),
        "optimizer_created": False,
        "backward_called": False,
    }


def build_demo(
    inference: PatchROIInferenceService,
    ollama: OllamaReportService,
    output_dir: Path,
    examples: list[tuple[str, int]],
) -> gr.Blocks:
    ground_truth_choices = [("未提供", "")] + [
        (f"{index} {name} / {CLASS_NAME_ZH[index]}", str(index))
        for index, name in CLASS_MAPPING.items()
    ]
    css = """
    .demo-shell { max-width: 1180px; margin: 0 auto; }
    .result-title { min-height: 72px; }
    .audit-panel { min-height: 180px; }
    """
    with gr.Blocks(title="胸腔 X 光 ROI 五分類與 AI 輔助說明 Demo", css=css) as demo:
        with gr.Column(elem_classes=["demo-shell"]):
            gr.Markdown("# 胸腔 X 光 ROI 五分類 + AI 輔助說明 Demo")
            gr.Markdown("**RAD-DINO 7x7 Patch Distillation + ConvNeXt-Tiny + Local Ollama**")
            gr.Markdown(
                "請輸入已裁切的胸腔 X 光病灶 ROI。此工具不接受完整胸腔影像，"
                "不執行偵測、BBox 預測或 No finding 判讀。"
            )
            with gr.Row():
                with gr.Column(scale=5):
                    image_input = gr.Image(
                        type="pil",
                        image_mode=None,
                        sources=["upload"],
                        label="病灶 ROI (PNG/JPG/JPEG)",
                    )
                    ground_truth = gr.Dropdown(
                        choices=ground_truth_choices,
                        value="",
                        label="Ground Truth (選填)",
                    )
                    run_button = gr.Button("執行分類與本機說明", variant="primary")
                    gr.Examples(
                        examples=[[path, str(class_id)] for path, class_id in examples],
                        example_labels=[
                            "Class 0 validation",
                            "Class 2 validation A",
                            "Class 2 validation B",
                            "Class 4 validation A",
                            "Class 4 validation B",
                        ],
                        inputs=[image_input, ground_truth],
                        cache_examples=False,
                        api_visibility="private",
                    )
                with gr.Column(scale=7):
                    predicted = gr.Markdown(label="Predicted class", elem_classes=["result-title"])
                    confidence = gr.Number(label="Softmax confidence", precision=6)
                    probability_table = gr.Dataframe(
                        headers=["class_id", "class_name_en", "class_name_zh", "probability"],
                        datatype=["number", "str", "str", "number"],
                        interactive=False,
                        label="五類機率",
                    )
                    ranking = gr.Markdown(label="機率排序")
                    ground_truth_output = gr.Markdown(label="Ground Truth 比對")
            with gr.Tabs():
                with gr.Tab("本機 Ollama 中文說明"):
                    report = gr.Markdown()
                    ollama_status = gr.JSON(label="Ollama 狀態")
                    generation_seconds = gr.Number(label="LLM generation seconds", precision=4)
                with gr.Tab("模型稽核"):
                    audit = gr.JSON(label="本次推論稽核", elem_classes=["audit-panel"])
                    warnings = gr.Markdown(label="輸入警告")
            disclaimer = gr.Markdown(f"**{DISCLAIMER}**")
            outputs = [
                predicted,
                confidence,
                probability_table,
                ranking,
                ground_truth_output,
                audit,
                warnings,
                report,
                ollama_status,
                generation_seconds,
                disclaimer,
            ]
            run_button.click(
                fn=lambda image, gt: run_demo_request(
                    image, gt, inference, ollama, output_dir
                ),
                inputs=[image_input, ground_truth],
                outputs=outputs,
                api_name=None,
                api_visibility="private",
                show_progress="full",
                concurrency_limit=1,
                concurrency_id="patch_roi_inference",
            )
    demo.queue(default_concurrency_limit=1, max_size=16)
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="auto")
    parser.add_argument("--ollama-timeout", type=float, default=120.0)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--inbrowser", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.project_root = args.project_root.expanduser().resolve()
    if not args.project_root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {args.project_root}")
    for name in ("model", "output_dir"):
        value = getattr(args, name).expanduser()
        if not value.is_absolute():
            value = args.project_root / value
        setattr(args, name, value.resolve())
    if args.server_name != "127.0.0.1":
        raise ValueError("server-name must be 127.0.0.1 for this local-only demo")
    return args


def main() -> int:
    args = resolve_paths(parse_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = read_validation_examples(args.project_root)
    inference = get_inference_service(args.model, args.device)
    ollama = OllamaReportService(
        args.ollama_base_url,
        args.ollama_model,
        args.ollama_timeout,
        args.project_root,
    )
    selected_port, occupied_pids = select_server_port(args.server_name, args.server_port)
    audit = create_startup_audit(
        args, inference, ollama, examples, selected_port, occupied_pids
    )
    write_json_atomic(args.output_dir / "logs" / "app_startup_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    demo = build_demo(inference, ollama, args.output_dir, examples)
    allowed_paths = [path for path, _ in examples]
    demo.launch(
        server_name=args.server_name,
        server_port=selected_port,
        share=False,
        show_error=False,
        inbrowser=args.inbrowser,
        footer_links=[],
        allowed_paths=allowed_paths,
        max_file_size="20mb",
        enable_monitoring=False,
        strict_cors=True,
        mcp_server=False,
        quiet=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
