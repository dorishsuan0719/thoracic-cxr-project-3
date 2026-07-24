from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps

from train_full_image_224_multilabel_patch_transfer import (
    CLASS_MAPPING,
    LABEL_FIELDS,
    NUM_CLASSES,
    FullImageMultilabelConvNeXt,
    FullImageTransform,
)


CLASS_NAMES_ZH = {
    0: "主動脈擴大",
    1: "心臟擴大",
    2: "胸膜增厚",
    3: "肺纖維化",
    4: "胸腔積液",
}
CLASS_SLUGS = {
    0: "aortic_enlargement",
    1: "cardiomegaly",
    2: "pleural_thickening",
    3: "pulmonary_fibrosis",
    4: "pleural_effusion",
}
EXPECTED_MAPPING = {
    0: "Aortic enlargement",
    1: "Cardiomegaly",
    2: "Pleural thickening",
    3: "Pulmonary fibrosis",
    4: "Pleural effusion",
}
DISCLAIMER = "本結果僅供研究與教學展示，不可取代醫師判讀或臨床診斷。"
NO_THRESHOLD_MESSAGE = "五個目標類別中沒有任何類別達到其 Validation 判定門檻。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one full-image five-class multilabel chest X-ray inference."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ground-truth-labels", type=int, nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def normalize_mapping(mapping: Any) -> dict[int, str]:
    if not isinstance(mapping, dict):
        raise RuntimeError("Checkpoint class_mapping is not a dictionary")
    try:
        return {int(key): str(value) for key, value in mapping.items()}
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Checkpoint class_mapping is invalid") from exc


def resolve_model_path(project_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        path = requested.resolve()
        if not path.is_file():
            raise RuntimeError(f"Requested model does not exist: {path}")
        return path

    checkpoint_dir = (
        project_root
        / "outputs"
        / "full_image_224_multilabel_seed42"
        / "phase2_patch_transfer"
        / "checkpoints"
    )
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"Formal checkpoint directory does not exist: {checkpoint_dir}")
    matches: list[Path] = []
    for candidate in sorted(checkpoint_dir.glob("*.pt")):
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("checkpoint_kind") == "full_image_multilabel_export"
            and payload.get("task") == "full-image multilabel five-class classification"
        ):
            matches.append(candidate.resolve())
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one formal full-image export checkpoint, found {len(matches)}: {matches}"
        )
    return matches[0]


def resolve_threshold_path(project_root: Path, requested: Path | None) -> Path:
    path = (
        requested.resolve()
        if requested is not None
        else (
            project_root
            / "outputs"
            / "full_image_224_multilabel_seed42"
            / "phase2_patch_transfer"
            / "validation_selected_thresholds.json"
        ).resolve()
    )
    if not path.is_file():
        raise RuntimeError(f"Validation threshold JSON does not exist: {path}")
    return path


def load_thresholds(path: Path) -> tuple[list[float], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_split") != "validation" or payload.get("test_used") is not False:
        raise RuntimeError("Threshold JSON is not a Validation-only threshold export")
    raw = payload.get("thresholds")
    if not isinstance(raw, dict):
        raise RuntimeError("Threshold JSON does not contain a thresholds dictionary")
    if set(raw) != {str(index) for index in range(NUM_CLASSES)}:
        raise RuntimeError(f"Threshold keys must be exactly 0..4, got {sorted(raw)}")
    thresholds = [float(raw[str(index)]) for index in range(NUM_CLASSES)]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in thresholds):
        raise RuntimeError(f"Thresholds must all be finite and within [0,1]: {thresholds}")
    per_class = payload.get("per_class_audit")
    if not isinstance(per_class, list) or len(per_class) != NUM_CLASSES:
        raise RuntimeError("Threshold per_class_audit must contain five classes")
    for index, item in enumerate(per_class):
        if (
            int(item.get("class_id", -1)) != index
            or item.get("class_name") != EXPECTED_MAPPING[index]
            or float(item.get("threshold", -1)) != thresholds[index]
        ):
            raise RuntimeError(f"Threshold class audit mismatch at class {index}")
    return thresholds, payload


def validate_preprocessing(config: Any) -> None:
    expected = {
        "source": "complete raw/full chest X-ray",
        "convert_mode": "RGB",
        "resize": [224, 224],
        "interpolation": "BILINEAR",
        "antialias": True,
        "center_crop": False,
        "random_resized_crop": False,
        "bbox": False,
        "roi_crop": False,
        "augmentation": False,
        "to_tensor": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "output_shape": [3, 224, 224],
    }
    if not isinstance(config, dict):
        raise RuntimeError("Checkpoint preprocessing metadata is missing")
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint preprocessing mismatch: {mismatches}")


def validate_checkpoint(payload: Any, thresholds: list[float]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("Formal checkpoint payload must be a dictionary")
    checks = {
        "checkpoint_kind": (payload.get("checkpoint_kind"), "full_image_multilabel_export"),
        "architecture": (payload.get("architecture"), "convnext_tiny"),
        "task": (payload.get("task"), "full-image multilabel five-class classification"),
        "activation_for_inference": (payload.get("activation_for_inference"), "sigmoid"),
        "test_evaluation_count": (payload.get("test_evaluation_count"), 1),
        "uses_bbox": (payload.get("uses_bbox"), False),
        "uses_roi_crop": (payload.get("uses_roi_crop"), False),
    }
    failures = [f"{key}: expected {expected!r}, got {actual!r}" for key, (actual, expected) in checks.items() if actual != expected]
    if failures:
        raise RuntimeError("Checkpoint metadata validation failed: " + "; ".join(failures))
    if normalize_mapping(payload.get("class_mapping")) != EXPECTED_MAPPING or CLASS_MAPPING != EXPECTED_MAPPING:
        raise RuntimeError("Checkpoint or training-script class mapping does not match the fixed five classes")
    if payload.get("label_fields") != LABEL_FIELDS or NUM_CLASSES != 5:
        raise RuntimeError("Checkpoint label fields or num_classes do not match five-class training")
    checkpoint_thresholds = [float(value) for value in payload.get("validation_selected_thresholds", [])]
    if checkpoint_thresholds != thresholds:
        raise RuntimeError(
            f"Checkpoint thresholds differ from Validation JSON: {checkpoint_thresholds} != {thresholds}"
        )
    training_config = payload.get("training_config", {})
    validate_preprocessing(training_config.get("preprocessing"))
    initialization = payload.get("initialization_audit")
    if not isinstance(initialization, dict) or initialization.get("status") != "PASS":
        raise RuntimeError("Checkpoint initialization audit is missing or not PASS")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint model_state_dict is missing")
    return state


def resolve_device(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def validate_ground_truth(values: list[int] | None) -> tuple[list[int] | None, list[int] | None]:
    if values is None:
        return None, None
    if len(values) != len(set(values)):
        raise RuntimeError(f"Ground Truth class IDs contain duplicates: {values}")
    if any(value not in EXPECTED_MAPPING for value in values):
        raise RuntimeError(f"Ground Truth class IDs must be within 0..4: {values}")
    class_ids = sorted(values)
    vector = [int(class_id in class_ids) for class_id in range(NUM_CLASSES)]
    return class_ids, vector


def validate_image(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise RuntimeError(f"Input image must be PNG, JPG, or JPEG: {path}")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Input image is missing or empty: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
    except Exception as exc:
        raise RuntimeError(f"Input image is unreadable: {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Input image has invalid dimensions: {width}x{height}")
    return {"original_width": width, "original_height": height, "original_mode": mode}


def preprocess_image(path: Path) -> tuple[torch.Tensor, Image.Image]:
    transform = FullImageTransform(image_size=224)
    with Image.open(path) as source:
        source.load()
        display_image = ImageOps.exif_transpose(source).convert("RGB")
        tensor = transform(source)
    batch = tensor.unsqueeze(0)
    if list(batch.shape) != [1, 3, 224, 224]:
        raise RuntimeError(f"Unexpected input tensor shape: {list(batch.shape)}")
    if not torch.isfinite(batch).all():
        raise RuntimeError("Preprocessed input contains NaN or Inf")
    return batch, display_image


def sample_metrics(predicted: list[int], truth: list[int] | None) -> dict[str, Any]:
    if truth is None:
        return {
            "correctly_detected_ids": None,
            "missed_ids": None,
            "extra_ids": None,
            "tp": None,
            "fp": None,
            "fn": None,
            "tn": None,
            "exact_match": None,
            "sample_precision": None,
            "sample_recall": None,
            "sample_f1": None,
        }
    tp = sum(pred == 1 and gt == 1 for pred, gt in zip(predicted, truth))
    fp = sum(pred == 1 and gt == 0 for pred, gt in zip(predicted, truth))
    fn = sum(pred == 0 and gt == 1 for pred, gt in zip(predicted, truth))
    tn = sum(pred == 0 and gt == 0 for pred, gt in zip(predicted, truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "correctly_detected_ids": [index for index in range(NUM_CLASSES) if predicted[index] and truth[index]],
        "missed_ids": [index for index in range(NUM_CLASSES) if not predicted[index] and truth[index]],
        "extra_ids": [index for index in range(NUM_CLASSES) if predicted[index] and not truth[index]],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "exact_match": predicted == truth,
        "sample_precision": precision,
        "sample_recall": recall,
        "sample_f1": f1,
    }


def names(ids: list[int] | None, mapping: dict[int, str]) -> list[str] | None:
    return None if ids is None else [mapping[class_id] for class_id in ids]


def atomic_save_visualization(
    path: Path,
    image: Image.Image,
    probabilities: list[float],
    thresholds: list[float],
    predicted_ids: list[int],
    ground_truth_ids: list[int] | None,
    metrics: dict[str, Any],
) -> None:
    figure = plt.figure(figsize=(16, 9.8), facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.22)
    image_axis = figure.add_subplot(grid[0, 0])
    image_axis.imshow(image)
    image_axis.set_title("Complete full chest X-ray\nNo BBox or ROI crop", fontsize=14)
    image_axis.axis("off")

    right_grid = grid[0, 1].subgridspec(2, 1, height_ratios=[3.15, 1.25], hspace=0.32)
    chart_axis = figure.add_subplot(right_grid[0, 0])
    x = np.arange(NUM_CLASSES)
    colors = ["#16866f" if index in predicted_ids else "#5a7894" for index in range(NUM_CLASSES)]
    bars = chart_axis.bar(x, probabilities, color=colors, width=0.66, label="Sigmoid probability")
    chart_axis.scatter(x, thresholds, marker="_", s=900, linewidths=3, color="#d1495b", label="Validation threshold", zorder=4)
    for index, bar in enumerate(bars):
        marker = "PASS" if index in predicted_ids else "below"
        near_top = probabilities[index] > 0.9
        chart_axis.text(
            bar.get_x() + bar.get_width() / 2,
            probabilities[index] - 0.025 if near_top else probabilities[index] + 0.025,
            f"{probabilities[index]:.4f}\n{marker}",
            ha="center",
            va="top" if near_top else "bottom",
            fontsize=10,
            fontweight="bold" if index in predicted_ids else "normal",
            color="white" if near_top else "black",
        )
    chart_axis.set_ylim(0, 1)
    chart_axis.set_ylabel("Probability")
    chart_axis.set_xticks(
        x,
        ["0 Aortic\nenlargement", "1 Cardio-\nmegaly", "2 Pleural\nthickening", "3 Pulmonary\nfibrosis", "4 Pleural\neffusion"],
    )
    chart_axis.set_title("Five independent Sigmoid outputs", fontsize=14)
    chart_axis.grid(axis="y", alpha=0.22)
    chart_axis.legend(loc="upper right")

    predicted_text = ", ".join(names(predicted_ids, EXPECTED_MAPPING) or []) or NO_THRESHOLD_MESSAGE
    info_axis = figure.add_subplot(right_grid[1, 0])
    info_axis.axis("off")
    lines = [textwrap.fill(f"Predicted: {predicted_text}", width=78)]
    if ground_truth_ids is None:
        lines.extend(["Ground Truth: not provided", "Accuracy comparison: not computed"])
    else:
        gt_text = ", ".join(names(ground_truth_ids, EXPECTED_MAPPING) or [])
        correct_text = ", ".join(names(metrics["correctly_detected_ids"], EXPECTED_MAPPING) or []) or "None"
        missed_text = ", ".join(names(metrics["missed_ids"], EXPECTED_MAPPING) or []) or "None"
        extra_text = ", ".join(names(metrics["extra_ids"], EXPECTED_MAPPING) or []) or "None"
        lines.extend([
            textwrap.fill(f"Ground Truth: {gt_text}", width=78),
            textwrap.fill(f"Correctly detected: {correct_text}", width=78),
            textwrap.fill(f"Missed: {missed_text}", width=78),
            textwrap.fill(f"Extra: {extra_text}", width=78),
            (
                f"TP/FP/FN/TN: {metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']} | "
                f"Exact Match: {'Yes' if metrics['exact_match'] else 'No'} | Sample F1: {metrics['sample_f1']:.4f}"
            ),
        ])
    lines.append("Research use only. Not for clinical diagnosis.")
    info_axis.text(0.0, 0.98, "\n".join(lines), fontsize=10.5, va="top", ha="left", linespacing=1.35)
    temporary = path.with_name(path.name + ".writing")
    figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)


def environment_payload(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
    return {
        "created_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": Image.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": gpu,
    }


def protected_paths(project_root: Path, model_path: Path, threshold_path: Path) -> dict[str, Path]:
    dataset_dir = project_root / "outputs" / "full_image_224_multilabel_seed42" / "phase0_dataset"
    return {
        "model": model_path,
        "thresholds": threshold_path,
        "validation_manifest": dataset_dir / "val_manifest.csv",
        "test_manifest": dataset_dir / "test_manifest.csv",
    }


def hash_protected(paths: dict[str, Path]) -> dict[str, str]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Protected files are missing: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def prediction_payload(
    timestamp: str,
    model_path: Path,
    model_sha: str,
    threshold_path: Path,
    threshold_sha: str,
    checkpoint: dict[str, Any],
    image_path: Path,
    image_sha: str,
    image_info: dict[str, Any],
    probabilities: list[float],
    thresholds: list[float],
    predicted: list[int],
    ground_truth_ids: list[int] | None,
    ground_truth: list[int] | None,
    metrics: dict[str, Any],
    classification_seconds: float,
) -> dict[str, Any]:
    predicted_ids = [index for index, value in enumerate(predicted) if value]
    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "threshold_path": str(threshold_path),
        "threshold_sha256": threshold_sha,
        "architecture": checkpoint["architecture"],
        "initialization": checkpoint["experiment"],
        "image_path": str(image_path),
        "image_sha256": image_sha,
        **image_info,
        "input_tensor_shape": [1, 3, 224, 224],
    }
    payload.update({f"probability_class_{index}": probabilities[index] for index in range(NUM_CLASSES)})
    payload.update({f"threshold_class_{index}": thresholds[index] for index in range(NUM_CLASSES)})
    payload.update({
        "predicted_label_vector": predicted,
        "predicted_class_ids": predicted_ids,
        "predicted_class_names_en": names(predicted_ids, EXPECTED_MAPPING),
        "predicted_class_names_zh": names(predicted_ids, CLASS_NAMES_ZH),
        "no_class_reached_threshold_message": NO_THRESHOLD_MESSAGE if not predicted_ids else None,
        "ground_truth_label_vector": ground_truth,
        "ground_truth_class_ids": ground_truth_ids,
        "ground_truth_class_names_en": names(ground_truth_ids, EXPECTED_MAPPING),
        "ground_truth_class_names_zh": names(ground_truth_ids, CLASS_NAMES_ZH),
        "correctly_detected_labels": names(metrics["correctly_detected_ids"], EXPECTED_MAPPING),
        "missed_labels": names(metrics["missed_ids"], EXPECTED_MAPPING),
        "extra_predicted_labels": names(metrics["extra_ids"], EXPECTED_MAPPING),
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "exact_match": metrics["exact_match"],
        "sample_precision": metrics["sample_precision"],
        "sample_recall": metrics["sample_recall"],
        "sample_f1": metrics["sample_f1"],
        "classification_seconds": classification_seconds,
        "disclaimer": DISCLAIMER,
    })
    return payload


def csv_row(payload: dict[str, Any]) -> dict[str, Any]:
    row = {
        "image_path": payload["image_path"],
        "image_filename": Path(payload["image_path"]).name,
        "image_sha256": payload["image_sha256"],
    }
    row.update({
        f"probability_{CLASS_SLUGS[index]}": f"{payload[f'probability_class_{index}']:.8f}"
        for index in range(NUM_CLASSES)
    })
    row.update({
        f"threshold_{CLASS_SLUGS[index]}": f"{payload[f'threshold_class_{index}']:.8f}"
        for index in range(NUM_CLASSES)
    })
    row.update({
        "predicted_label_vector": json.dumps(payload["predicted_label_vector"], separators=(",", ":")),
        "predicted_class_ids": "|".join(map(str, payload["predicted_class_ids"])),
        "predicted_class_names": "|".join(payload["predicted_class_names_en"]),
        "ground_truth_label_vector": "" if payload["ground_truth_label_vector"] is None else json.dumps(payload["ground_truth_label_vector"], separators=(",", ":")),
        "ground_truth_class_ids": "" if payload["ground_truth_class_ids"] is None else "|".join(map(str, payload["ground_truth_class_ids"])),
        "tp": payload["tp"],
        "fp": payload["fp"],
        "fn": payload["fn"],
        "tn": payload["tn"],
        "exact_match": payload["exact_match"],
        "sample_precision": payload["sample_precision"],
        "sample_recall": payload["sample_recall"],
        "sample_f1": payload["sample_f1"],
        "model_sha256": payload["model_sha256"],
        "threshold_sha256": payload["threshold_sha256"],
        "timestamp": payload["timestamp"],
    })
    return row


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        raise RuntimeError(f"Project root does not exist: {project_root}")
    model_path = resolve_model_path(project_root, args.model)
    threshold_path = resolve_threshold_path(project_root, args.thresholds)
    image_path = args.image.resolve()
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (project_root / "outputs" / "full_image_multilabel_single_inference").resolve()
    )
    device = resolve_device(args.device)
    ground_truth_ids, ground_truth = validate_ground_truth(args.ground_truth_labels)
    image_info = validate_image(image_path)
    image_sha_before = sha256_file(image_path)

    thresholds, threshold_payload = load_thresholds(threshold_path)
    protected = protected_paths(project_root, model_path, threshold_path)
    protected_before = hash_protected(protected)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state = validate_checkpoint(checkpoint, thresholds)

    model = FullImageMultilabelConvNeXt()
    incompatible = model.load_state_dict(state, strict=True)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(f"Strict load failed: missing={missing_keys}, unexpected={unexpected_keys}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("At least one model parameter remains trainable")
    batch, display_image = preprocess_image(image_path)

    dry_run_summary = {
        "status": "PASS",
        "dry_run": True,
        "model_path": str(model_path),
        "model_sha256": protected_before["model"],
        "threshold_path": str(threshold_path),
        "threshold_sha256": protected_before["thresholds"],
        "thresholds": thresholds,
        "strict_load_missing_keys": missing_keys,
        "strict_load_unexpected_keys": unexpected_keys,
        "input_tensor_shape": list(batch.shape),
        "ground_truth_label_vector": ground_truth,
        "output_created": False,
        "model_inference_count": 0,
        "test_images_read_count": 0,
        "uses_bbox": False,
        "uses_roi_crop": False,
        "optimizer_created": False,
        "backward_executed": False,
        "ollama_calls": 0,
    }
    if args.dry_run:
        protected_after = hash_protected(protected)
        if protected_before != protected_after or image_sha_before != sha256_file(image_path):
            raise RuntimeError("A protected input changed during dry-run")
        print(json.dumps(dry_run_summary, ensure_ascii=False, indent=2))
        return 0

    model.to(device)
    batch = batch.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(batch)
        probabilities_tensor = torch.sigmoid(logits)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    classification_seconds = time.perf_counter() - start

    logits_shape = list(logits.shape)
    probability_shape = list(probabilities_tensor.shape)
    if logits_shape != [1, 5] or probability_shape != [1, 5]:
        raise RuntimeError(f"Unexpected output shapes: logits={logits_shape}, probabilities={probability_shape}")
    nan_count = int(torch.isnan(probabilities_tensor).sum().item())
    inf_count = int(torch.isinf(probabilities_tensor).sum().item())
    if nan_count or inf_count:
        raise RuntimeError(f"Non-finite probabilities: NaN={nan_count}, Inf={inf_count}")
    probabilities = [float(value) for value in probabilities_tensor[0].detach().cpu().tolist()]
    if not all(0.0 <= value <= 1.0 for value in probabilities):
        raise RuntimeError(f"Probability outside [0,1]: {probabilities}")
    predicted = [int(probabilities[index] >= thresholds[index]) for index in range(NUM_CLASSES)]
    predicted_shape = [1, len(predicted)]
    predicted_ids = [index for index, value in enumerate(predicted) if value]
    metrics = sample_metrics(predicted, ground_truth)
    timestamp = utc_now()
    payload = prediction_payload(
        timestamp,
        model_path,
        protected_before["model"],
        threshold_path,
        protected_before["thresholds"],
        checkpoint,
        image_path,
        image_sha_before,
        image_info,
        probabilities,
        thresholds,
        predicted,
        ground_truth_ids,
        ground_truth,
        metrics,
        classification_seconds,
    )

    result_name = f"{image_path.stem}_{timestamp_slug()}"
    result_dir = output_root / result_name
    staging = output_root / f"{result_name}.writing"
    if result_dir.exists() or staging.exists():
        raise RuntimeError(f"Refusing to overwrite existing inference result: {result_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        atomic_write_json(staging / "prediction.json", payload)
        row = csv_row(payload)
        atomic_write_csv(staging / "prediction.csv", list(row), [row])
        atomic_save_visualization(
            staging / "prediction_visualization.png",
            display_image,
            probabilities,
            thresholds,
            predicted_ids,
            ground_truth_ids,
            metrics,
        )
        protected_after = hash_protected(protected)
        image_sha_after = sha256_file(image_path)
        protected_unchanged = protected_before == protected_after
        source_image_unchanged = image_sha_before == image_sha_after
        if not protected_unchanged or not source_image_unchanged:
            raise RuntimeError("A protected model, manifest, threshold, or source image changed")
        audit = {
            "status": "PASS",
            "timestamp": timestamp,
            "formal_full_image_export": True,
            "architecture": checkpoint["architecture"],
            "initialization": checkpoint["experiment"],
            "num_classes": NUM_CLASSES,
            "class_mapping": EXPECTED_MAPPING,
            "checkpoint_kind": checkpoint["checkpoint_kind"],
            "checkpoint_test_evaluation_count": checkpoint["test_evaluation_count"],
            "strict_load": True,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "model_eval": not model.training,
            "all_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
            "torch_inference_mode_used": True,
            "optimizer_created": False,
            "backward_executed": False,
            "softmax_used": False,
            "sigmoid_used": True,
            "individual_validation_thresholds_used": True,
            "threshold_source": threshold_payload["source_split"],
            "preprocessing": FullImageTransform(224).config(),
            "uses_bbox": False,
            "uses_roi_crop": False,
            "test_images_read_count": 0,
            "model_inference_count": 1,
            "ollama_calls": 0,
            "input_tensor_shape": [1, 3, 224, 224],
            "logits_shape": logits_shape,
            "probabilities_shape": probability_shape,
            "predicted_vector_shape": predicted_shape,
            "probability_nan_count": nan_count,
            "probability_inf_count": inf_count,
            "probabilities_within_0_1": all(0.0 <= value <= 1.0 for value in probabilities),
            "thresholds_within_0_1": all(0.0 <= value <= 1.0 for value in thresholds),
            "ground_truth_provided": ground_truth is not None,
            "classification_seconds": classification_seconds,
            "device": str(device),
            "protected_sha256_before": protected_before,
            "protected_sha256_after": protected_after,
            "protected_files_unchanged": protected_unchanged,
            "source_image_sha256_before": image_sha_before,
            "source_image_sha256_after": image_sha_after,
            "source_image_unchanged": source_image_unchanged,
            "research_disclaimer_present": payload["disclaimer"] == DISCLAIMER,
        }
        atomic_write_json(staging / "inference_audit.json", audit)
        atomic_write_json(staging / "environment.json", environment_payload(device))
        staging.rename(result_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    temporary_remnants = [
        str(path)
        for path in result_dir.rglob("*")
        if path.name.endswith((".tmp", ".writing"))
    ]
    if temporary_remnants:
        raise RuntimeError(f"Temporary output remnants found: {temporary_remnants}")
    print(json.dumps({
        "status": "PASS",
        "result_dir": str(result_dir),
        "probabilities": probabilities,
        "thresholds": thresholds,
        "predicted_label_vector": predicted,
        "predicted_class_ids": predicted_ids,
        "ground_truth_label_vector": ground_truth,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "exact_match": metrics["exact_match"],
        "sample_precision": metrics["sample_precision"],
        "sample_recall": metrics["sample_recall"],
        "sample_f1": metrics["sample_f1"],
        "classification_seconds": classification_seconds,
        "device": str(device),
        "test_images_read_count": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
