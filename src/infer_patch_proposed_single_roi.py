#!/usr/bin/env python
"""Run one research-only ROI classification with the exported Patch Proposed model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision
from PIL import Image, ImageOps

from train_phase2_convnext_tiny_finetune import (
    ARCHITECTURE,
    CLASS_MAPPING,
    FEATURE_DIM,
    NUM_CLASSES,
    ConvNeXtTinyClassifier,
    Phase2Transform,
)


EXPECTED_INITIALIZATION = "patch_distilled"
EXPECTED_INITIALIZATION_DESCRIPTION = "RAD-DINO 7x7 patch distilled"
EXPECTED_DISTILLATION_TYPE = "RAD-DINO 7x7 patch feature"
EXPECTED_HEAD = {
    "dropout": 0.2,
    "linear": [768, 5],
    "hidden_layer": False,
    "projector": False,
}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DISCLAIMER = (
    "Research use only. This AI output is not a clinical diagnosis and must not "
    "replace review by qualified medical professionals."
)
CSV_FIELDS = [
    "image_path",
    "image_filename",
    "image_sha256",
    "predicted_class_id",
    "predicted_class_name",
    "confidence",
    "probability_aortic_enlargement",
    "probability_cardiomegaly",
    "probability_pleural_thickening",
    "probability_pulmonary_fibrosis",
    "probability_pleural_effusion",
    "ground_truth_class_id",
    "ground_truth_class_name",
    "true_class_probability",
    "is_correct",
    "model_sha256",
    "timestamp",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_class_mapping(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        raise ValueError("Checkpoint class_mapping must be a dictionary")
    try:
        return {int(key): str(name) for key, name in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint class_mapping contains invalid keys") from exc


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is unavailable; detected {torch.cuda.device_count()} device(s)"
            )
        torch.cuda.set_device(index)
        torch.cuda.current_device()
    return device


def inspect_image(path: Path) -> tuple[Image.Image, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension {path.suffix!r}; expected PNG, JPG, or JPEG"
        )
    if path.stat().st_size <= 0:
        raise ValueError(f"Input image is empty: {path}")

    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).copy()
    except Exception as exc:
        raise ValueError(f"Input image cannot be decoded by Pillow: {path}") from exc

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Input image has invalid dimensions: {width}x{height}")
    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    if grayscale.size == 0 or int(grayscale.max()) == int(grayscale.min()):
        raise ValueError("Input image is blank or has no pixel variation")

    warning = None
    if (width, height) != (224, 224):
        warning = (
            f"Input ROI is {width}x{height}, not 224x224. The locked deterministic "
            "Phase 2 preprocessing will resize to 236 then center-crop to 224."
        )
    return image, {
        "width": width,
        "height": height,
        "mode": image.mode,
        "file_size_bytes": path.stat().st_size,
        "pixel_min": int(grayscale.min()),
        "pixel_max": int(grayscale.max()),
        "warning": warning,
    }


def validate_checkpoint_metadata(
    checkpoint: dict[str, Any], expected_preprocessing: dict[str, Any]
) -> dict[str, bool]:
    class_mapping = normalize_class_mapping(checkpoint.get("class_mapping"))
    required_metadata = {
        "patch_phase1_backbone_sha256": checkpoint.get("patch_phase1_backbone_sha256"),
        "shared_config_sha256": checkpoint.get("shared_config_sha256"),
        "best_epoch": checkpoint.get("best_epoch"),
        "validation_metrics": checkpoint.get("validation_metrics"),
        "test_metrics": checkpoint.get("test_metrics"),
    }
    checks = {
        "architecture": checkpoint.get("architecture") == ARCHITECTURE,
        "initialization": checkpoint.get("initialization") == EXPECTED_INITIALIZATION,
        "initialization_description": checkpoint.get("initialization_description")
        == EXPECTED_INITIALIZATION_DESCRIPTION,
        "distillation_type": checkpoint.get("distillation_type") == EXPECTED_DISTILLATION_TYPE,
        "num_classes": checkpoint.get("num_classes") == NUM_CLASSES,
        "feature_dim": checkpoint.get("feature_dim") == FEATURE_DIM,
        "class_mapping": class_mapping == CLASS_MAPPING,
        "no_no_finding_class": all(
            name.lower() not in {"no finding", "normal", "background"}
            for name in class_mapping.values()
        ),
        "preprocessing": checkpoint.get("preprocessing_config") == expected_preprocessing,
        "complete_model_state_dict": isinstance(checkpoint.get("model_state_dict"), dict)
        and bool(checkpoint["model_state_dict"]),
        "head_metadata": checkpoint.get("head_metadata") == EXPECTED_HEAD,
        "patch_phase1_backbone_sha256": isinstance(
            required_metadata["patch_phase1_backbone_sha256"], str
        )
        and len(required_metadata["patch_phase1_backbone_sha256"]) == 64,
        "shared_config_sha256": isinstance(required_metadata["shared_config_sha256"], str)
        and len(required_metadata["shared_config_sha256"]) == 64,
        "best_epoch": isinstance(required_metadata["best_epoch"], int)
        and required_metadata["best_epoch"] > 0,
        "validation_metrics_metadata": isinstance(required_metadata["validation_metrics"], dict)
        and bool(required_metadata["validation_metrics"]),
        "test_metrics_metadata": isinstance(required_metadata["test_metrics"], dict)
        and bool(required_metadata["test_metrics"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Patch Proposed export metadata validation failed: {failed}")
    return checks


def load_export_model(
    model_path: Path, device: torch.device
) -> tuple[ConvNeXtTinyClassifier, dict[str, Any], dict[str, Any], float]:
    started = time.perf_counter()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root object must be a dictionary")

    transform = Phase2Transform(training=False)
    preprocessing = transform.preprocessing_config()
    metadata_checks = validate_checkpoint_metadata(checkpoint, preprocessing)

    # "export" constructs only the architecture. The full Phase 2 export then replaces
    # every parameter through a strict state_dict load; no alternate weights are loaded.
    model = ConvNeXtTinyClassifier(initialization="export")
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Strict model load failed: missing={missing_keys}, unexpected={unexpected_keys}"
        )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("One or more model parameters still require gradients")

    load_audit = {
        "strict_load": True,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "state_dict_key_count": len(checkpoint["model_state_dict"]),
        "model_eval": not model.training,
        "trainable_parameter_count": sum(
            int(parameter.requires_grad) for parameter in model.parameters()
        ),
        "optimizer_created": False,
        "backward_called": False,
        "rad_dino_loaded": False,
        "teacher_cache_loaded": False,
        "cls_checkpoint_loaded": False,
        "baseline_checkpoint_loaded": False,
        "patch_phase1_checkpoint_loaded": False,
        "metadata_checks": metadata_checks,
    }
    return model, checkpoint, load_audit, time.perf_counter() - started


def environment_info(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "total_memory_gib": props.total_memory / (1024**3),
        }
    return {
        "timestamp": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": gpu,
    }


def serialize_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def serialize_csv(row: dict[str, Any]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=CSV_FIELDS, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return ("\ufeff" + text.getvalue()).encode("utf-8")


def visualization_bytes(
    image: Image.Image,
    probabilities: list[float],
    predicted_class_id: int,
    confidence: float,
    ground_truth_class_id: int | None,
    is_correct: bool | None,
) -> bytes:
    labels = [CLASS_MAPPING[index] for index in range(NUM_CLASSES)]
    colors = ["#176b87" if index == predicted_class_id else "#9aa9b2" for index in range(NUM_CLASSES)]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1, 1.35]})
    axes[0].imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Input ROI")
    axes[0].axis("off")

    y_positions = np.arange(NUM_CLASSES)
    axes[1].barh(y_positions, probabilities, color=colors)
    axes[1].set_yticks(y_positions, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Softmax probability")
    axes[1].grid(axis="x", alpha=0.22)
    for index, probability in enumerate(probabilities):
        axes[1].text(
            min(probability + 0.012, 0.94),
            index,
            f"{probability:.4f}",
            va="center",
            fontsize=9,
        )

    ground_truth = (
        "Not provided"
        if ground_truth_class_id is None
        else f"{ground_truth_class_id}: {CLASS_MAPPING[ground_truth_class_id]}"
    )
    correctness = "N/A" if is_correct is None else ("Correct" if is_correct else "Incorrect")
    figure.suptitle(
        f"Predicted: {predicted_class_id} - {CLASS_MAPPING[predicted_class_id]}  |  "
        f"Confidence: {confidence:.4f}\nGround truth: {ground_truth}  |  {correctness}",
        fontsize=13,
    )
    figure.text(0.5, 0.015, DISCLAIMER, ha="center", fontsize=8, color="#7a2f2a")
    figure.tight_layout(rect=[0, 0.05, 1, 0.90])
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160, facecolor="white")
    plt.close(figure)
    return buffer.getvalue()


def atomic_write_bundle(run_dir: Path, payloads: dict[str, bytes]) -> list[Path]:
    run_dir.mkdir(parents=True, exist_ok=False)
    temporary_paths: list[Path] = []
    final_paths: list[Path] = []
    try:
        for filename, content in payloads.items():
            destination = run_dir / filename
            temporary = run_dir / f".{filename}.tmp"
            if destination.exists() or temporary.exists():
                raise FileExistsError(f"Refusing to overwrite output: {destination}")
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_paths.append(temporary)
            final_paths.append(destination)
        for temporary, destination in zip(temporary_paths, final_paths, strict=True):
            os.replace(temporary, destination)
        return final_paths
    except Exception:
        for temporary in temporary_paths:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run one Patch Proposed ConvNeXt-Tiny five-class ROI inference."
    )
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda:0"])
    parser.add_argument("--ground-truth-class-id", type=int, choices=range(NUM_CLASSES))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")
    model_path = (
        args.model.expanduser().resolve()
        if args.model
        else project_root
        / "outputs"
        / "raddino_convnext_tiny_patch_experiment_seed42"
        / "phase2_proposed_patch_distilled"
        / "checkpoints"
        / "patch_proposed_convnext_tiny_5class.pt"
    )
    image_path = args.image.expanduser().resolve()
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else project_root / "outputs" / "patch_proposed_single_roi_inference"
    )
    if not model_path.is_file() or model_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Patch Proposed export checkpoint is missing or empty: {model_path}")

    pipeline_started = time.perf_counter()
    device = resolve_device(args.device)
    image, image_audit = inspect_image(image_path)
    model_sha256 = sha256_file(model_path)
    image_sha256 = sha256_file(image_path)
    timestamp = utc_now()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / f"{image_path.stem}_{run_stamp}"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device.index if device.index is not None else 0)
    model, checkpoint, load_audit, model_load_seconds = load_export_model(model_path, device)
    transform = Phase2Transform(training=False)
    preprocessing = transform.preprocessing_config()

    dry_run = {
        "status": "PASS",
        "dry_run": True,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "architecture": checkpoint["architecture"],
        "initialization": checkpoint["initialization_description"],
        "num_classes": checkpoint["num_classes"],
        "class_mapping": normalize_class_mapping(checkpoint["class_mapping"]),
        "state_dict_key_count": load_audit["state_dict_key_count"],
        "strict_load": load_audit["strict_load"],
        "missing_keys": load_audit["missing_keys"],
        "unexpected_keys": load_audit["unexpected_keys"],
        "image_path": str(image_path),
        "image_sha256": image_sha256,
        "image_metadata": image_audit,
        "preprocessing": preprocessing,
        "planned_output_directory": str(run_dir),
        "forward_executed": False,
        "files_written": 0,
        "test_images_read_count": 0,
    }
    if args.dry_run:
        print(json.dumps(dry_run, ensure_ascii=False, indent=2))
        return 0

    preprocessing_started = time.perf_counter()
    tensor = transform(image)
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    if list(tensor.shape) != [3, 224, 224]:
        raise RuntimeError(f"Unexpected preprocessed tensor shape: {list(tensor.shape)}")
    input_tensor = tensor.unsqueeze(0).to(device)
    tensor_nan_count = int(torch.isnan(input_tensor).sum().item())
    tensor_inf_count = int(torch.isinf(input_tensor).sum().item())
    if tensor_nan_count or tensor_inf_count:
        raise RuntimeError(
            f"Preprocessed input contains NaN={tensor_nan_count}, Inf={tensor_inf_count}"
        )

    inference_started = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities_tensor = torch.softmax(logits.float(), dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started

    logits_nan_count = int(torch.isnan(logits).sum().item())
    logits_inf_count = int(torch.isinf(logits).sum().item())
    probabilities_nan_count = int(torch.isnan(probabilities_tensor).sum().item())
    probabilities_inf_count = int(torch.isinf(probabilities_tensor).sum().item())
    if list(logits.shape) != [1, NUM_CLASSES] or list(probabilities_tensor.shape) != [1, NUM_CLASSES]:
        raise RuntimeError(
            f"Unexpected output shapes: logits={list(logits.shape)}, "
            f"probabilities={list(probabilities_tensor.shape)}"
        )
    if any(
        value > 0
        for value in (
            logits_nan_count,
            logits_inf_count,
            probabilities_nan_count,
            probabilities_inf_count,
        )
    ):
        raise RuntimeError("Model output contains NaN or Inf")

    probability_sum = float(probabilities_tensor.sum().item())
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"Softmax probabilities do not sum to 1: {probability_sum}")
    probabilities = [float(value) for value in probabilities_tensor[0].detach().cpu().tolist()]
    predicted_class_id = int(torch.argmax(probabilities_tensor, dim=1).item())
    confidence = probabilities[predicted_class_id]
    if predicted_class_id not in CLASS_MAPPING or not 0.0 <= confidence <= 1.0:
        raise RuntimeError("Predicted class or confidence is outside the valid range")

    ground_truth_class_id = args.ground_truth_class_id
    ground_truth_class_name = (
        CLASS_MAPPING[ground_truth_class_id] if ground_truth_class_id is not None else None
    )
    true_class_probability = (
        probabilities[ground_truth_class_id] if ground_truth_class_id is not None else None
    )
    is_correct = (
        predicted_class_id == ground_truth_class_id
        if ground_truth_class_id is not None
        else None
    )

    gpu_allocated_bytes = 0
    gpu_reserved_bytes = 0
    gpu_peak_allocated_bytes = 0
    gpu_peak_reserved_bytes = 0
    if device.type == "cuda":
        gpu_allocated_bytes = torch.cuda.memory_allocated(device)
        gpu_reserved_bytes = torch.cuda.memory_reserved(device)
        gpu_peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
        gpu_peak_reserved_bytes = torch.cuda.max_memory_reserved(device)

    prediction = {
        "timestamp": timestamp,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "initialization": checkpoint["initialization_description"],
        "architecture": checkpoint["architecture"],
        "device": str(device),
        "image_path": str(image_path),
        "image_sha256": image_sha256,
        "original_width": image_audit["width"],
        "original_height": image_audit["height"],
        "original_mode": image_audit["mode"],
        "preprocessing": preprocessing,
        "input_tensor_shape": list(input_tensor.shape),
        "predicted_class_id": predicted_class_id,
        "predicted_class_name": CLASS_MAPPING[predicted_class_id],
        "confidence": confidence,
        "probability_class_0": probabilities[0],
        "probability_class_1": probabilities[1],
        "probability_class_2": probabilities[2],
        "probability_class_3": probabilities[3],
        "probability_class_4": probabilities[4],
        "ground_truth_class_id": ground_truth_class_id,
        "ground_truth_class_name": ground_truth_class_name,
        "true_class_probability": true_class_probability,
        "is_correct": is_correct,
        "warning": image_audit["warning"],
        "disclaimer": DISCLAIMER,
    }
    csv_row = {
        "image_path": str(image_path),
        "image_filename": image_path.name,
        "image_sha256": image_sha256,
        "predicted_class_id": predicted_class_id,
        "predicted_class_name": CLASS_MAPPING[predicted_class_id],
        "confidence": confidence,
        "probability_aortic_enlargement": probabilities[0],
        "probability_cardiomegaly": probabilities[1],
        "probability_pleural_thickening": probabilities[2],
        "probability_pulmonary_fibrosis": probabilities[3],
        "probability_pleural_effusion": probabilities[4],
        "ground_truth_class_id": ground_truth_class_id,
        "ground_truth_class_name": ground_truth_class_name,
        "true_class_probability": true_class_probability,
        "is_correct": is_correct,
        "model_sha256": model_sha256,
        "timestamp": timestamp,
    }
    environment = environment_info(device)
    inference_audit = {
        "status": "PASS",
        "timestamp": timestamp,
        "model_load_seconds": model_load_seconds,
        "preprocessing_seconds": preprocessing_seconds,
        "inference_seconds": inference_seconds,
        "pipeline_seconds_before_output_serialization": time.perf_counter() - pipeline_started,
        "device": str(device),
        "device_name": environment["gpu"]["name"] if environment["gpu"] else "CPU",
        "cuda_used": device.type == "cuda",
        "gpu_allocated_memory_bytes": gpu_allocated_bytes,
        "gpu_reserved_memory_bytes": gpu_reserved_bytes,
        "gpu_peak_allocated_memory_bytes": gpu_peak_allocated_bytes,
        "gpu_peak_reserved_memory_bytes": gpu_peak_reserved_bytes,
        "checkpoint_metadata_validation": load_audit["metadata_checks"],
        "strict_state_dict_load": load_audit["strict_load"],
        "state_dict_key_count": load_audit["state_dict_key_count"],
        "missing_keys": load_audit["missing_keys"],
        "unexpected_keys": load_audit["unexpected_keys"],
        "model_eval": load_audit["model_eval"],
        "trainable_parameter_count": load_audit["trainable_parameter_count"],
        "optimizer_created": False,
        "backward_called": False,
        "threshold_used": False,
        "test_time_augmentation_used": False,
        "rad_dino_loaded": False,
        "teacher_cache_loaded": False,
        "cls_checkpoint_loaded": False,
        "baseline_checkpoint_loaded": False,
        "patch_phase1_checkpoint_loaded": False,
        "checkpoint_test_metrics_read_from_metadata_only": True,
        "validation_images_read_count": 0,
        "test_images_read_count": 0,
        "input_tensor_shape": list(input_tensor.shape),
        "input_tensor_dtype": str(input_tensor.dtype),
        "input_tensor_min": float(input_tensor.min().item()),
        "input_tensor_max": float(input_tensor.max().item()),
        "input_tensor_mean": float(input_tensor.mean().item()),
        "input_tensor_nan_count": tensor_nan_count,
        "input_tensor_inf_count": tensor_inf_count,
        "logits_shape": list(logits.shape),
        "probabilities_shape": list(probabilities_tensor.shape),
        "logits_nan_count": logits_nan_count,
        "logits_inf_count": logits_inf_count,
        "probabilities_nan_count": probabilities_nan_count,
        "probabilities_inf_count": probabilities_inf_count,
        "probability_sum": probability_sum,
        "predicted_class_in_range": predicted_class_id in CLASS_MAPPING,
        "confidence_in_range": 0.0 <= confidence <= 1.0,
        "ground_truth_supplied_by_cli": ground_truth_class_id is not None,
        "ground_truth_inferred_from_path": False,
        "source_image_modified": False,
        "clinical_use": False,
        "disclaimer": DISCLAIMER,
    }

    payloads = {
        "prediction.json": serialize_json(prediction),
        "prediction.csv": serialize_csv(csv_row),
        "prediction_visualization.png": visualization_bytes(
            image,
            probabilities,
            predicted_class_id,
            confidence,
            ground_truth_class_id,
            is_correct,
        ),
        "inference_audit.json": serialize_json(inference_audit),
        "environment.json": serialize_json(environment),
    }
    written = atomic_write_bundle(run_dir, payloads)
    residual_temporary_files = [str(path) for path in run_dir.rglob("*.tmp")]
    if residual_temporary_files:
        raise RuntimeError(f"Temporary files remain after atomic writes: {residual_temporary_files}")

    summary = {
        "status": "PASS",
        "output_directory": str(run_dir),
        "files": [str(path) for path in written],
        "predicted_class_id": predicted_class_id,
        "predicted_class_name": CLASS_MAPPING[predicted_class_id],
        "confidence": confidence,
        "probabilities": {str(index): probabilities[index] for index in range(NUM_CLASSES)},
        "ground_truth_class_id": ground_truth_class_id,
        "ground_truth_class_name": ground_truth_class_name,
        "is_correct": is_correct,
        "model_load_seconds": model_load_seconds,
        "preprocessing_seconds": preprocessing_seconds,
        "inference_seconds": inference_seconds,
        "device": str(device),
        "test_images_read_count": 0,
        "temporary_files_remaining": 0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
