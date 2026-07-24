#!/usr/bin/env python
"""Cache frozen RAD-DINO teacher features for the balanced ROI manifest.

Phase 0-B only: this script reads the Step 1 manifest and source ROI images,
runs microsoft/rad-dino in inference mode, and adds teacher-feature artifacts
without modifying the Step 1 files or source images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
import PIL
import torch
import transformers
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "microsoft/rad-dino"
EXPECTED_ROWS = 4725
EXPECTED_DIM = 768
EXPECTED_MANIFEST_SHA256 = (
    "796f067d00bb5740a51b51292eed4acfefe9b2e84fd2eeb9b5dfd2df926d5233"
)
FEATURE_TYPE = "pooler_output_cls_token"

STEP1_FILES = (
    "roi_manifest.csv",
    "audit_summary.json",
    "audit_report.txt",
    "duplicate_sha256_groups.csv",
    "unresolved_records.csv",
    "manifest_metadata.json",
)

PHASE0B_FILES = (
    "teacher_features.pt",
    "teacher_features.pt.tmp",
    "teacher_feature_metadata.json",
    "teacher_feature_audit.txt",
    "teacher_feature_progress.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def validate_preconditions(output_dir: Path) -> tuple[Path, list[dict[str, str]], str]:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Step 1 output directory not found: {output_dir}")

    missing_step1 = [name for name in STEP1_FILES if not (output_dir / name).is_file()]
    if missing_step1:
        raise FileNotFoundError(f"Missing Step 1 files: {missing_step1}")

    existing_targets = [name for name in PHASE0B_FILES if (output_dir / name).exists()]
    if existing_targets:
        raise FileExistsError(
            "Phase 0-B target file(s) already exist; refusing to overwrite: "
            + ", ".join(existing_targets)
        )

    manifest_path = output_dir / "roi_manifest.csv"
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256.lower() != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"Manifest SHA256 mismatch: {manifest_sha256} != {EXPECTED_MANIFEST_SHA256}"
        )

    rows = read_manifest(manifest_path)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Manifest row count mismatch: {len(rows)} != {EXPECTED_ROWS}")

    try:
        indices = [int(row["feature_index"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Manifest contains an invalid feature_index") from exc
    expected_indices = list(range(EXPECTED_ROWS))
    if indices != expected_indices:
        raise ValueError("Manifest feature_index is not exactly 0..4724 in row order")

    missing_paths = []
    for row in rows:
        raw_path = row.get("image_path", "").strip()
        if not raw_path or not Path(raw_path).is_file():
            missing_paths.append(raw_path)
    if missing_paths:
        raise FileNotFoundError(
            f"Manifest references {len(missing_paths)} missing image(s); first: {missing_paths[0]}"
        )

    return manifest_path, rows, manifest_sha256


def environment_info(device: torch.device) -> dict[str, Any]:
    gpu: dict[str, Any] | None = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "cuda_runtime": torch.version.cuda,
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "pillow": PIL.__version__,
        "device": str(device),
        "gpu": gpu,
    }


def load_teacher(local_files_only: bool) -> tuple[Any, torch.nn.Module, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this Phase 0-B run")
    device = torch.device("cuda:0")
    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        local_files_only=local_files_only,
    )
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    if model.training:
        raise RuntimeError("Teacher model did not enter eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Teacher model is not fully frozen")
    return processor, model, device


def open_images(rows: list[dict[str, str]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for row in rows:
        path = Path(row["image_path"])
        with Image.open(path) as image:
            image.load()
            images.append(image.copy())
    return images


def forward_batch(
    rows: list[dict[str, str]],
    processor: Any,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    images = open_images(rows)
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    if getattr(outputs, "pooler_output", None) is not None:
        features = outputs.pooler_output
    elif getattr(outputs, "last_hidden_state", None) is not None:
        features = outputs.last_hidden_state[:, 0, :]
    else:
        raise RuntimeError("RAD-DINO output has neither pooler_output nor last_hidden_state")
    features = features.detach().to(device="cpu", dtype=torch.float32)
    if features.ndim != 2 or features.shape != (len(rows), EXPECTED_DIM):
        raise RuntimeError(
            f"Unexpected teacher feature shape: {tuple(features.shape)}; "
            f"expected {(len(rows), EXPECTED_DIM)}"
        )
    return features


def smoke_test(
    rows: list[dict[str, str]],
    processor: Any,
    model: torch.nn.Module,
    device: torch.device,
    requested_batch_size: int,
) -> tuple[int, dict[str, Any]]:
    batch_size = min(requested_batch_size, len(rows))
    attempts: list[dict[str, Any]] = []
    while batch_size >= 1:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            features = forward_batch(rows[:batch_size], processor, model, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            nan_count = int(torch.isnan(features).sum().item())
            inf_count = int(torch.isinf(features).sum().item())
            zero_norm_count = int((torch.linalg.vector_norm(features, dim=1) == 0).sum().item())
            if nan_count or inf_count or zero_norm_count:
                raise RuntimeError(
                    "Smoke-test features failed numeric validation: "
                    f"NaN={nan_count}, Inf={inf_count}, zero_norm={zero_norm_count}"
                )
            peak_memory = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            )
            return batch_size, {
                "status": "PASS",
                "batch_size": batch_size,
                "feature_shape": list(features.shape),
                "feature_dtype": str(features.dtype),
                "nan_count": nan_count,
                "inf_count": inf_count,
                "zero_norm_count": zero_norm_count,
                "elapsed_seconds": elapsed,
                "peak_gpu_memory_bytes": peak_memory,
                "oom_attempts": attempts,
            }
        except torch.OutOfMemoryError as exc:
            attempts.append({"batch_size": batch_size, "error": str(exc)})
            if device.type == "cuda":
                torch.cuda.empty_cache()
            batch_size //= 2
    raise RuntimeError(f"Smoke test OOM at every batch size: {attempts}")


def verify_saved_features(path: Path, manifest_sha256: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = payload["features"]
    indices = payload["feature_indices"]
    if not isinstance(features, torch.Tensor) or not isinstance(indices, torch.Tensor):
        raise TypeError("Saved features or feature_indices is not a tensor")
    if tuple(features.shape) != (EXPECTED_ROWS, EXPECTED_DIM):
        raise ValueError(f"Saved feature shape mismatch: {tuple(features.shape)}")
    if features.dtype != torch.float32 or features.device.type != "cpu":
        raise ValueError(f"Saved feature tensor must be CPU float32, got {features.device} {features.dtype}")
    if indices.dtype != torch.int64 or indices.device.type != "cpu":
        raise ValueError("Saved feature_indices must be a CPU int64 tensor")

    expected = torch.arange(EXPECTED_ROWS, dtype=torch.int64)
    missing_index_count = int((~torch.isin(expected, indices)).sum().item())
    unique_indices, counts = torch.unique(indices, return_counts=True)
    duplicate_index_count = int((counts - 1).clamp_min(0).sum().item())
    out_of_range_index_count = int(((indices < 0) | (indices >= EXPECTED_ROWS)).sum().item())
    nan_count = int(torch.isnan(features).sum().item())
    inf_count = int(torch.isinf(features).sum().item())
    zero_norm_count = int((torch.linalg.vector_norm(features, dim=1) == 0).sum().item())
    sha_match = payload.get("manifest_sha256") == manifest_sha256
    model_match = payload.get("model_name") == MODEL_NAME

    status = "PASS"
    if any(
        (
            missing_index_count,
            duplicate_index_count,
            out_of_range_index_count,
            nan_count,
            inf_count,
            zero_norm_count,
        )
    ) or not sha_match or not model_match or not torch.equal(indices, expected):
        status = "FAIL"

    return {
        "status": status,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "feature_device": str(features.device),
        "feature_index_shape": list(indices.shape),
        "feature_indices_exactly_0_to_4724": bool(torch.equal(indices, expected)),
        "unique_index_count": int(unique_indices.numel()),
        "missing_index_count": missing_index_count,
        "duplicate_index_count": duplicate_index_count,
        "out_of_range_index_count": out_of_range_index_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_norm_count": zero_norm_count,
        "manifest_sha256": payload.get("manifest_sha256"),
        "manifest_sha256_match": sha_match,
        "model_name": payload.get("model_name"),
        "model_name_match": model_match,
    }


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else project_root / "outputs" / "raddino_feature_cache" / "balanced_945_seed42"
    )
    manifest_path, rows, manifest_sha256 = validate_preconditions(output_dir)

    processor, model, device = load_teacher(args.local_files_only)
    batch_size, smoke = smoke_test(
        rows,
        processor,
        model,
        device,
        requested_batch_size=args.batch_size,
    )
    if args.smoke_only:
        print(json.dumps({"manifest_sha256": manifest_sha256, "smoke_test": smoke}, indent=2))
        return 0

    progress_path = output_dir / "teacher_feature_progress.json"
    started_utc = utc_now()
    started = time.perf_counter()
    environment = environment_info(device)
    processor_config = processor.to_dict()
    features = torch.empty((EXPECTED_ROWS, EXPECTED_DIM), dtype=torch.float32, device="cpu")

    progress: dict[str, Any] = {
        "status": "running",
        "started_at_utc": started_utc,
        "updated_at_utc": started_utc,
        "model_name": MODEL_NAME,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "total_records": EXPECTED_ROWS,
        "completed_records": 0,
        "batch_size": batch_size,
        "device": str(device),
        "smoke_test": smoke,
    }
    atomic_write_json(progress_path, progress)

    try:
        for start in range(0, EXPECTED_ROWS, batch_size):
            end = min(start + batch_size, EXPECTED_ROWS)
            batch_features = forward_batch(rows[start:end], processor, model, device)
            features[start:end].copy_(batch_features)
            if end == EXPECTED_ROWS or end % max(batch_size * args.progress_every_batches, 1) == 0:
                elapsed = time.perf_counter() - started
                progress.update(
                    {
                        "updated_at_utc": utc_now(),
                        "completed_records": end,
                        "elapsed_seconds": elapsed,
                        "images_per_second": end / elapsed if elapsed else None,
                    }
                )
                atomic_write_json(progress_path, progress)
                print(
                    f"Processed {end}/{EXPECTED_ROWS} "
                    f"({100.0 * end / EXPECTED_ROWS:.1f}%)",
                    flush=True,
                )

        feature_indices = torch.arange(EXPECTED_ROWS, dtype=torch.int64)
        payload = {
            "features": features,
            "feature_indices": feature_indices,
            "manifest_sha256": manifest_sha256,
            "model_name": MODEL_NAME,
            "feature_type": FEATURE_TYPE,
            "processor_config": processor_config,
            "environment": environment,
        }
        temporary_path = output_dir / "teacher_features.pt.tmp"
        final_path = output_dir / "teacher_features.pt"
        torch.save(payload, temporary_path)
        os.replace(temporary_path, final_path)

        verification = verify_saved_features(final_path, manifest_sha256)
        if verification["status"] != "PASS":
            raise RuntimeError(f"Saved teacher feature verification failed: {verification}")

        completed_utc = utc_now()
        elapsed = time.perf_counter() - started
        feature_file_sha256 = sha256_file(final_path)
        metadata = {
            "status": "PASS",
            "phase": "Phase 0-B",
            "created_at_utc": completed_utc,
            "model_name": MODEL_NAME,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "feature_type": FEATURE_TYPE,
            "feature_shape": [EXPECTED_ROWS, EXPECTED_DIM],
            "feature_dtype": "torch.float32",
            "feature_device": "cpu",
            "batch_size": batch_size,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "teacher_features_path": str(final_path),
            "teacher_features_sha256": feature_file_sha256,
            "processor_config": processor_config,
            "environment": environment,
            "smoke_test": smoke,
            "verification": verification,
            "elapsed_seconds": elapsed,
            "images_per_second": EXPECTED_ROWS / elapsed if elapsed else None,
            "convnext_loaded": False,
            "phase1_started": False,
            "train_val_test_split_created": False,
            "model_training_executed": False,
        }
        atomic_write_json(output_dir / "teacher_feature_metadata.json", metadata)

        audit_lines = [
            "RAD-DINO Teacher Feature Audit",
            "================================",
            "status: PASS",
            f"model_name: {MODEL_NAME}",
            f"feature_type: {FEATURE_TYPE}",
            f"manifest_rows: {len(rows)}",
            f"manifest_sha256: {manifest_sha256}",
            f"feature_shape: {verification['feature_shape']}",
            f"feature_dtype: {verification['feature_dtype']}",
            f"feature_device: {verification['feature_device']}",
            f"batch_size: {batch_size}",
            f"gpu: {environment['gpu']['name'] if environment['gpu'] else 'none'}",
            f"nan_count: {verification['nan_count']}",
            f"inf_count: {verification['inf_count']}",
            f"zero_norm_count: {verification['zero_norm_count']}",
            f"missing_index_count: {verification['missing_index_count']}",
            f"duplicate_index_count: {verification['duplicate_index_count']}",
            f"manifest_sha256_match: {verification['manifest_sha256_match']}",
            f"elapsed_seconds: {elapsed:.3f}",
            f"images_per_second: {EXPECTED_ROWS / elapsed:.3f}",
        ]
        atomic_write_text(
            output_dir / "teacher_feature_audit.txt",
            "\n".join(audit_lines) + "\n",
        )

        progress.update(
            {
                "status": "completed",
                "updated_at_utc": completed_utc,
                "completed_records": EXPECTED_ROWS,
                "elapsed_seconds": elapsed,
                "images_per_second": EXPECTED_ROWS / elapsed if elapsed else None,
                "teacher_features_sha256": feature_file_sha256,
                "verification_status": verification["status"],
            }
        )
        atomic_write_json(progress_path, progress)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        progress.update(
            {
                "status": "failed",
                "updated_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error_reason": str(exc),
            }
        )
        atomic_write_json(progress_path, progress)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache frozen microsoft/rad-dino teacher features for Phase 0-B."
    )
    parser.add_argument(
        "--project-root",
        default=r"C:\Users\09688\thoracic-cxr-project-3",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--progress-every-batches", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.progress_every_batches < 1:
        raise ValueError("--progress-every-batches must be at least 1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
