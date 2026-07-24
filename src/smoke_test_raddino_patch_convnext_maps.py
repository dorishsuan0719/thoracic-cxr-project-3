#!/usr/bin/env python
"""Smoke-test RAD-DINO patch maps against ConvNeXt-Tiny 7x7 maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import PIL
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
import transformers
from transformers import AutoImageProcessor, AutoModel


EXPECTED_MANIFEST_SHA256 = "796f067d00bb5740a51b51292eed4acfefe9b2e84fd2eeb9b5dfd2df926d5233"
EXPECTED_ROWS = 4725
EXPECTED_FEATURE_DIM = 768
EXPECTED_PER_CLASS = 945
CLASS_MAPPING = {
    0: "Aortic enlargement",
    1: "Cardiomegaly",
    2: "Pleural thickening",
    3: "Pulmonary fibrosis",
    4: "Pleural effusion",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n",
    )


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Manifest missing or empty: {path}")
    digest = sha256_file(path)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Manifest SHA256 mismatch: {digest}")
    rows = read_csv(path)
    required = {
        "feature_index", "image_path", "class_id", "class_name", "source_image_id",
        "is_brightness_augmented", "image_width", "image_height", "image_mode",
    }
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Manifest rows={len(rows)}, expected {EXPECTED_ROWS}")
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest missing fields: {sorted(required - set(rows[0] if rows else {}))}")
    indices = [int(row["feature_index"]) for row in rows]
    if sorted(indices) != list(range(EXPECTED_ROWS)):
        raise ValueError("feature_index is missing, duplicated, or outside 0..4724")
    counts = Counter(int(row["class_id"]) for row in rows)
    if dict(sorted(counts.items())) != {index: EXPECTED_PER_CLASS for index in range(5)}:
        raise ValueError(f"Class counts mismatch: {dict(counts)}")
    missing = [row["image_path"] for row in rows if not Path(row["image_path"]).is_file()]
    invalid_class = [row["feature_index"] for row in rows if int(row["class_id"]) not in CLASS_MAPPING]
    bad_name = [row["feature_index"] for row in rows if row["class_name"] != CLASS_MAPPING[int(row["class_id"])]]
    bad_metadata = [row["feature_index"] for row in rows if int(row["image_width"]) != 224 or int(row["image_height"]) != 224 or row["image_mode"] != "L"]
    if missing or invalid_class or bad_name or bad_metadata:
        raise ValueError(
            f"Manifest audit failed: missing={len(missing)}, invalid_class={len(invalid_class)}, "
            f"bad_name={len(bad_name)}, bad_image_metadata={len(bad_metadata)}"
        )
    return {
        "rows": rows,
        "manifest_sha256": digest,
        "feature_index_min": min(indices),
        "feature_index_max": max(indices),
        "feature_index_unique": len(set(indices)),
        "class_counts": dict(sorted(counts.items())),
        "brightness_augmented_count": sum(parse_bool(row["is_brightness_augmented"]) for row in rows),
        "missing_images": 0,
    }


def sample_records(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    if count < 5:
        raise ValueError("--num-samples must be at least 5")
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    base_quota, extra = divmod(count, 5)
    for class_id in range(5):
        quota = base_quota + int(class_id < extra)
        class_rows = sorted(
            (row for row in rows if int(row["class_id"]) == class_id),
            key=lambda row: int(row["feature_index"]),
        )
        originals = [row for row in class_rows if not parse_bool(row["is_brightness_augmented"])]
        augmented = [row for row in class_rows if parse_bool(row["is_brightness_augmented"])]
        class_selected: list[dict[str, str]] = []
        if quota and originals:
            class_selected.append(rng.choice(originals))
        if len(class_selected) < quota and augmented:
            class_selected.append(rng.choice(augmented))
        selected_indices = {int(row["feature_index"]) for row in class_selected}
        remaining = [row for row in class_rows if int(row["feature_index"]) not in selected_indices]
        if len(class_selected) < quota:
            class_selected.extend(rng.sample(remaining, quota - len(class_selected)))
        selected.extend(class_selected)
    if len(selected) != count:
        raise RuntimeError(f"Sampling produced {len(selected)} records, expected {count}")
    if len({int(row["class_id"]) for row in selected}) != 5:
        raise RuntimeError("Sampling failed to include all five classes")
    if len({int(row["feature_index"]) for row in selected}) != len(selected):
        raise RuntimeError("Sampling produced duplicate feature_index")
    return sorted(selected, key=lambda row: int(row["feature_index"]))


def sampled_csv_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    fields = ("feature_index", "image_path", "class_id", "class_name", "source_image_id", "is_brightness_augmented")
    return [{field: row[field] for field in fields} for row in rows]


def open_rgb_images(rows: list[dict[str, str]]) -> list[Image.Image]:
    images = []
    for row in rows:
        path = Path(row["image_path"])
        with Image.open(path) as image:
            image.load()
            if image.size != (224, 224):
                raise ValueError(f"Expected 224x224 source image, got {image.size}: {path}")
            if image.mode != "L":
                raise ValueError(f"Expected source image mode L, got {image.mode}: {path}")
            images.append(image.convert("RGB").copy())
    return images


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "nan_count": int(torch.isnan(values).sum()),
        "inf_count": int(torch.isinf(values).sum()),
    }


def map_statistics(tensor: torch.Tensor, map_type: str, row: dict[str, str]) -> dict[str, Any]:
    values = tensor.float()
    norms = torch.linalg.vector_norm(values, ord=2, dim=0)
    return {
        "feature_index": int(row["feature_index"]),
        "class_id": int(row["class_id"]),
        "class_name": row["class_name"],
        "map_type": map_type,
        "shape": json.dumps(list(tensor.shape)),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "l2_norm_min": float(norms.min()),
        "l2_norm_mean": float(norms.mean()),
        "l2_norm_max": float(norms.max()),
        "nan_count": int(torch.isnan(values).sum()),
        "inf_count": int(torch.isinf(values).sum()),
        "zero_norm_count": int((norms == 0).sum()),
    }


def patch_dimensions(patch_size: Any) -> tuple[int, int]:
    if isinstance(patch_size, int):
        return patch_size, patch_size
    if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
        return int(patch_size[0]), int(patch_size[1])
    raise ValueError(f"Unsupported model.config.patch_size: {patch_size!r}")


def processor_audit(processor: Any) -> dict[str, Any]:
    resample = getattr(processor, "resample", None)
    return {
        "processor_class": processor.__class__.__name__,
        "do_resize": getattr(processor, "do_resize", None),
        "size": getattr(processor, "size", None),
        "do_center_crop": getattr(processor, "do_center_crop", None),
        "crop_size": getattr(processor, "crop_size", None),
        "do_rescale": getattr(processor, "do_rescale", None),
        "rescale_factor": getattr(processor, "rescale_factor", None),
        "do_normalize": getattr(processor, "do_normalize", None),
        "image_mean": getattr(processor, "image_mean", None),
        "image_std": getattr(processor, "image_std", None),
        "do_convert_rgb": getattr(processor, "do_convert_rgb", None),
        "resample": int(resample) if resample is not None else None,
        "interpolation": str(resample),
    }


def student_preprocessing_audit(transform: Any) -> dict[str, Any]:
    return {
        "weights_enum": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "resize_size": list(transform.resize_size),
        "crop_size": list(transform.crop_size),
        "interpolation": str(transform.interpolation),
        "antialias": transform.antialias,
        "mean": list(transform.mean),
        "std": list(transform.std),
        "random_augmentation": False,
    }


def environment_info(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu = {
            "name": props.name,
            "total_vram_bytes": props.total_memory,
            "total_vram_gb": props.total_memory / (1024**3),
        }
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "device": str(device),
        "gpu": gpu,
    }


def install_student_hooks(student: torch.nn.Module, shapes: dict[str, list[list[int]]]) -> list[Any]:
    names = {
        0: "stem_output",
        1: "stage1_output",
        2: "downsample1_output",
        3: "stage2_output",
        4: "downsample2_output",
        5: "stage3_output",
        6: "downsample3_output",
        7: "final_feature_output",
    }
    handles = []
    for index, module in enumerate(student.features):
        name = names.get(index, f"features_{index}_output")
        shapes.setdefault(name, [])
        handles.append(module.register_forward_hook(lambda _module, _inputs, output, key=name: shapes[key].append(list(output.shape))))
    return handles


def validate_stage_shapes(stage_shapes: dict[str, list[list[int]]], batches: int) -> dict[str, list[int]]:
    result = {}
    for name, shapes in stage_shapes.items():
        if len(shapes) != batches:
            raise RuntimeError(f"Hook {name} recorded {len(shapes)} batches, expected {batches}")
        channel_spatial = {tuple(shape[1:]) for shape in shapes}
        if len(channel_spatial) != 1:
            raise RuntimeError(f"Inconsistent {name} shapes: {shapes}")
        result[name] = shapes[0]
    return result


def smoke_forward(
    rows: list[dict[str, str]],
    processor: Any,
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    student_transform: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    patch_h, patch_w = patch_dimensions(teacher.config.patch_size)
    hidden_size = int(teacher.config.hidden_size)
    if hidden_size != EXPECTED_FEATURE_DIM:
        raise RuntimeError(f"Teacher hidden_size={hidden_size}, expected 768")
    stage_shapes_raw: dict[str, list[list[int]]] = {}
    handles = install_student_hooks(student, stage_shapes_raw)
    teacher_input_batches, student_input_batches, teacher_output_batches = [], [], []
    map_stats: list[dict[str, Any]] = []
    sample_cosines: list[dict[str, Any]] = []
    mse_weighted_sum = 0.0
    cosine_values: list[torch.Tensor] = []
    expected_grid: tuple[int, int] | None = None
    expected_patch_count: int | None = None
    special_token_count: int | None = None
    peak_allocated = 0
    peak_reserved = 0
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch_rows = rows[start:start + batch_size]
                images = open_rgb_images(batch_rows)
                teacher_inputs = processor(images=images, return_tensors="pt")
                if "pixel_values" not in teacher_inputs:
                    raise RuntimeError("RAD-DINO processor did not return pixel_values")
                teacher_pixels = teacher_inputs["pixel_values"].to(device)
                teacher_input_batches.append(tensor_summary(teacher_pixels))
                input_h, input_w = int(teacher_pixels.shape[-2]), int(teacher_pixels.shape[-1])
                if input_h % patch_h or input_w % patch_w:
                    raise RuntimeError(f"Teacher input {(input_h, input_w)} is not divisible by patch {(patch_h, patch_w)}")
                grid_h, grid_w = input_h // patch_h, input_w // patch_w
                patch_count = grid_h * grid_w
                if expected_grid is None:
                    expected_grid = (grid_h, grid_w)
                    expected_patch_count = patch_count
                elif expected_grid != (grid_h, grid_w) or expected_patch_count != patch_count:
                    raise RuntimeError("Teacher patch grid changed across batches")
                outputs = teacher(pixel_values=teacher_pixels)
                last_hidden = outputs.last_hidden_state
                if last_hidden.ndim != 3 or int(last_hidden.shape[-1]) != hidden_size:
                    raise RuntimeError(f"Unexpected last_hidden_state: {tuple(last_hidden.shape)}")
                inferred_special = int(last_hidden.shape[1]) - patch_count
                if inferred_special < 1:
                    raise RuntimeError(
                        f"Cannot infer special tokens: tokens={last_hidden.shape[1]}, patches={patch_count}"
                    )
                if special_token_count is None:
                    special_token_count = inferred_special
                elif special_token_count != inferred_special:
                    raise RuntimeError("Special token count changed across batches")
                patch_tokens = last_hidden[:, -patch_count:, :]
                if tuple(patch_tokens.shape[1:]) != (patch_count, hidden_size):
                    raise RuntimeError(f"Unexpected patch token shape: {tuple(patch_tokens.shape)}")
                teacher_native = patch_tokens.reshape(len(batch_rows), grid_h, grid_w, hidden_size).permute(0, 3, 1, 2).contiguous()
                teacher_pooled = F.adaptive_avg_pool2d(teacher_native, output_size=(7, 7))
                teacher_output_batches.append({
                    "last_hidden_state_shape": list(last_hidden.shape),
                    "patch_tokens_shape": list(patch_tokens.shape),
                    "teacher_native_patch_map_shape": list(teacher_native.shape),
                    "teacher_pooled_patch_map_shape": list(teacher_pooled.shape),
                    "total_token_count": int(last_hidden.shape[1]),
                    "expected_patch_count": patch_count,
                    "special_token_count": inferred_special,
                })

                student_tensors = torch.stack([student_transform(image) for image in images]).to(device)
                student_input_batches.append(tensor_summary(student_tensors))
                student_map = student.features(student_tensors)
                expected_shape = (len(batch_rows), EXPECTED_FEATURE_DIM, 7, 7)
                if tuple(student_map.shape) != expected_shape:
                    raise RuntimeError(f"Student final feature map is {tuple(student_map.shape)}, expected {expected_shape}")
                if tuple(teacher_pooled.shape) != expected_shape:
                    raise RuntimeError(f"Teacher pooled map is {tuple(teacher_pooled.shape)}, expected {expected_shape}")
                if not torch.isfinite(student_map).all() or not torch.isfinite(teacher_native).all() or not torch.isfinite(teacher_pooled).all():
                    raise FloatingPointError("Non-finite Teacher or Student feature map")

                teacher_norms = torch.linalg.vector_norm(teacher_pooled.float(), ord=2, dim=1)
                student_norms = torch.linalg.vector_norm(student_map.float(), ord=2, dim=1)
                if int((teacher_norms == 0).sum()) or int((student_norms == 0).sum()):
                    raise FloatingPointError("Zero-norm spatial feature vector")
                teacher_normalized = F.normalize(teacher_pooled.float(), p=2, dim=1)
                student_normalized = F.normalize(student_map.float(), p=2, dim=1)
                if not torch.isfinite(teacher_normalized).all() or not torch.isfinite(student_normalized).all():
                    raise FloatingPointError("Non-finite normalized patch map")
                patch_mse = F.mse_loss(student_normalized, teacher_normalized)
                teacher_positions = teacher_normalized.permute(0, 2, 3, 1).reshape(len(batch_rows), 49, hidden_size)
                student_positions = student_normalized.permute(0, 2, 3, 1).reshape(len(batch_rows), 49, hidden_size)
                cosine = F.cosine_similarity(student_positions, teacher_positions, dim=-1)
                if not torch.isfinite(patch_mse) or not torch.isfinite(cosine).all():
                    raise FloatingPointError("Non-finite patch MSE or cosine")
                mse_weighted_sum += float(patch_mse) * len(batch_rows)
                cosine_values.append(cosine.detach().cpu())
                for item, row in enumerate(batch_rows):
                    map_stats.append(map_statistics(teacher_native[item], "teacher_native_patch_map", row))
                    map_stats.append(map_statistics(teacher_pooled[item], "teacher_pooled_patch_map", row))
                    map_stats.append(map_statistics(student_map[item], "student_feature_map", row))
                    sample_cosines.append({
                        "feature_index": int(row["feature_index"]),
                        "mean_patch_cosine_similarity": float(cosine[item].mean()),
                    })
        batches = math.ceil(len(rows) / batch_size)
        stage_shapes = validate_stage_shapes(stage_shapes_raw, batches)
        all_cosines = torch.cat([values.reshape(-1) for values in cosine_values])
        peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        return {
            "hidden_size": hidden_size,
            "patch_size": [patch_h, patch_w],
            "processor_input_height": teacher_input_batches[0]["shape"][-2],
            "processor_input_width": teacher_input_batches[0]["shape"][-1],
            "grid_height": expected_grid[0],
            "grid_width": expected_grid[1],
            "expected_patch_count": expected_patch_count,
            "total_token_count": teacher_output_batches[0]["total_token_count"],
            "special_token_count": special_token_count,
            "extra_special_or_register_tokens": special_token_count > 1,
            "last_hidden_state_shape": teacher_output_batches[0]["last_hidden_state_shape"],
            "patch_tokens_shape": teacher_output_batches[0]["patch_tokens_shape"],
            "teacher_native_patch_map_shape": teacher_output_batches[0]["teacher_native_patch_map_shape"],
            "teacher_pooled_patch_map_shape": teacher_output_batches[0]["teacher_pooled_patch_map_shape"],
            "student_input_shape": student_input_batches[0]["shape"],
            "student_stage_shapes": stage_shapes,
            "student_final_feature_map_shape": stage_shapes["final_feature_output"],
            "teacher_student_shape_equal": teacher_output_batches[0]["teacher_pooled_patch_map_shape"] == stage_shapes["final_feature_output"],
            "teacher_input_batches": teacher_input_batches,
            "student_input_batches": student_input_batches,
            "teacher_output_batches": teacher_output_batches,
            "patch_mse_loss": mse_weighted_sum / len(rows),
            "patch_cosine_mean": float(all_cosines.mean()),
            "patch_cosine_min": float(all_cosines.min()),
            "patch_cosine_max": float(all_cosines.max()),
            "per_image_patch_cosine": sample_cosines,
            "feature_map_statistics": map_stats,
            "nan_count": sum(int(row["nan_count"]) for row in map_stats),
            "inf_count": sum(int(row["inf_count"]) for row in map_stats),
            "zero_norm_count": sum(int(row["zero_norm_count"]) for row in map_stats),
            "peak_allocated_vram_bytes": peak_allocated,
            "peak_reserved_vram_bytes": peak_reserved,
        }
    finally:
        for handle in handles:
            handle.remove()


def write_text_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "RAD-DINO Patch / ConvNeXt-Tiny Feature Map Shape Smoke Test",
        f"Status: {result['status']}",
        f"Manifest SHA256: {result['manifest']['manifest_sha256']}",
        f"Samples: {result['sampling']['sample_count']} ({result['sampling']['feature_indices']})",
        f"Teacher model: {result['teacher']['model_name']} @ {result['teacher']['model_revision']}",
        f"Teacher processor input: {result['shape_pipeline']['processor_input_height']}x{result['shape_pipeline']['processor_input_width']}",
        f"Teacher patch size: {result['shape_pipeline']['patch_size']}",
        f"Teacher patch grid: {result['shape_pipeline']['grid_height']}x{result['shape_pipeline']['grid_width']}",
        f"Expected patch count: {result['shape_pipeline']['expected_patch_count']}",
        f"Total tokens / special tokens: {result['shape_pipeline']['total_token_count']} / {result['shape_pipeline']['special_token_count']}",
        f"Last hidden state: {result['shape_pipeline']['last_hidden_state_shape']}",
        f"Patch tokens: {result['shape_pipeline']['patch_tokens_shape']}",
        f"Teacher native map: {result['shape_pipeline']['teacher_native_patch_map_shape']}",
        f"Teacher pooled map: {result['shape_pipeline']['teacher_pooled_patch_map_shape']}",
        f"Student input: {result['shape_pipeline']['student_input_shape']}",
        f"Student stages: {result['shape_pipeline']['student_stage_shapes']}",
        f"Student final map: {result['shape_pipeline']['student_final_feature_map_shape']}",
        f"Teacher/Student shape equal: {result['shape_pipeline']['teacher_student_shape_equal']}",
        f"Patch MSE: {result['loss_pipeline']['patch_mse_loss']}",
        f"Patch cosine mean/min/max: {result['loss_pipeline']['patch_cosine_mean']} / {result['loss_pipeline']['patch_cosine_min']} / {result['loss_pipeline']['patch_cosine_max']}",
        f"NaN / Inf / zero norm: {result['validation']['nan_count']} / {result['validation']['inf_count']} / {result['validation']['zero_norm_count']}",
        "Backward executed: False",
        "Optimizer created or stepped: False",
        "Complete patch cache created: False",
        "Existing experiments modified: False",
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> int:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        contents = [str(path) for path in sorted(args.output_dir.rglob("*"))[:100]]
        raise FileExistsError(f"Output directory is non-empty and will not be overwritten: {contents}")
    set_seed(args.seed)
    manifest = validate_manifest(args.manifest)
    selected = sample_records(manifest["rows"], args.num_samples, args.seed)
    preflight = {
        "status": "PASS",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_rows": len(manifest["rows"]),
        "class_counts": manifest["class_counts"],
        "sample_count": len(selected),
        "feature_indices": [int(row["feature_index"]) for row in selected],
        "sample_class_counts": dict(sorted(Counter(int(row["class_id"]) for row in selected).items())),
        "sample_brightness_augmented_count": sum(parse_bool(row["is_brightness_augmented"]) for row in selected),
        "model_name": args.model_name,
        "output_directory_exists": args.output_dir.exists(),
    }
    if args.dry_run:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA device is required for this smoke test")
    processor = AutoImageProcessor.from_pretrained(args.model_name, local_files_only=True)
    teacher = AutoModel.from_pretrained(args.model_name, local_files_only=True).to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    if teacher.training or any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("RAD-DINO teacher is not eval/frozen")
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    student_transform = weights.transforms()
    student = convnext_tiny(weights=weights).to(device)
    student.eval()
    student.requires_grad_(False)
    if student.training or any(parameter.requires_grad for parameter in student.parameters()):
        raise RuntimeError("ConvNeXt student is not eval/frozen")
    teacher_revision = getattr(teacher.config, "_commit_hash", None) or getattr(processor, "_commit_hash", None)

    try:
        shape = smoke_forward(selected, processor, teacher, student, student_transform, device, args.batch_size)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(f"CUDA OOM during shape smoke test: {exc}") from exc
    finally:
        del student, teacher
        torch.cuda.empty_cache()

    pass_checks = {
        "manifest_valid": manifest["manifest_sha256"] == EXPECTED_MANIFEST_SHA256,
        "teacher_hidden_size": shape["hidden_size"] == EXPECTED_FEATURE_DIM,
        "patch_grid_valid": shape["expected_patch_count"] == shape["grid_height"] * shape["grid_width"],
        "special_tokens_inferred": shape["special_token_count"] >= 1,
        "teacher_native_map_valid": shape["teacher_native_patch_map_shape"][1:] == [EXPECTED_FEATURE_DIM, shape["grid_height"], shape["grid_width"]],
        "teacher_pooled_map_valid": shape["teacher_pooled_patch_map_shape"][1:] == [EXPECTED_FEATURE_DIM, 7, 7],
        "student_final_map_valid": shape["student_final_feature_map_shape"][1:] == [EXPECTED_FEATURE_DIM, 7, 7],
        "teacher_student_shape_equal": shape["teacher_student_shape_equal"],
        "patch_mse_finite": math.isfinite(shape["patch_mse_loss"]),
        "patch_cosine_finite": all(math.isfinite(shape[key]) for key in ("patch_cosine_mean", "patch_cosine_min", "patch_cosine_max")),
        "nan_zero": shape["nan_count"] == 0,
        "inf_zero": shape["inf_count"] == 0,
        "zero_norm_zero": shape["zero_norm_count"] == 0,
        "teacher_frozen": True,
        "student_frozen": True,
        "backward_not_executed": True,
        "optimizer_not_created": True,
        "full_cache_not_created": True,
        "no_oom": True,
    }
    failed = [name for name, passed in pass_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Shape smoke validation failed: {failed}")

    environment = environment_info(device)
    result = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "manifest": {key: value for key, value in manifest.items() if key != "rows"},
        "sampling": {
            "seed": args.seed,
            "sample_count": len(selected),
            "batch_size": args.batch_size,
            "feature_indices": [int(row["feature_index"]) for row in selected],
            "class_counts": dict(sorted(Counter(int(row["class_id"]) for row in selected).items())),
            "brightness_augmented_count": sum(parse_bool(row["is_brightness_augmented"]) for row in selected),
        },
        "teacher": {
            "model_name": args.model_name,
            "model_revision": teacher_revision,
            "config_commit_hash": getattr(getattr(processor, "config", None), "_commit_hash", None),
            "processor": processor_audit(processor),
            "official_processor_used": True,
            "eval": True,
            "frozen": True,
        },
        "student": {
            "architecture": "convnext_tiny",
            "weights": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
            "preprocessing": student_preprocessing_audit(student_transform),
            "eval": True,
            "frozen": True,
            "classification_head_used": False,
        },
        "shape_pipeline": {key: value for key, value in shape.items() if key not in {"feature_map_statistics", "patch_mse_loss", "patch_cosine_mean", "patch_cosine_min", "patch_cosine_max", "per_image_patch_cosine", "nan_count", "inf_count", "zero_norm_count"}},
        "loss_pipeline": {
            "normalization": "L2 along channel dimension at every spatial location",
            "patch_mse_loss": shape["patch_mse_loss"],
            "patch_cosine_mean": shape["patch_cosine_mean"],
            "patch_cosine_min": shape["patch_cosine_min"],
            "patch_cosine_max": shape["patch_cosine_max"],
            "per_image_patch_cosine": shape["per_image_patch_cosine"],
            "backward_executed": False,
            "optimizer_created_or_stepped": False,
        },
        "validation": {
            "checks": pass_checks,
            "failed_checks": [],
            "nan_count": shape["nan_count"],
            "inf_count": shape["inf_count"],
            "zero_norm_count": shape["zero_norm_count"],
            "oom": False,
            "full_patch_cache_created": False,
            "existing_experiments_modified": False,
        },
        "environment": environment,
    }
    # AutoModel config is no longer resident after cleanup; the cached revision is deterministic.
    cache_ref = Path.home() / ".cache" / "huggingface" / "hub" / "models--microsoft--rad-dino" / "refs" / "main"
    if cache_ref.is_file():
        result["teacher"]["model_revision"] = cache_ref.read_text(encoding="utf-8").strip()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "patch_shape_smoke_test.json", result)
    write_text_report(args.output_dir / "patch_shape_smoke_test.txt", result)
    atomic_write_csv(args.output_dir / "sampled_records.csv", sampled_csv_rows(selected))
    atomic_write_json(args.output_dir / "environment.json", environment)
    atomic_write_csv(args.output_dir / "feature_map_statistics.csv", shape["feature_map_statistics"])
    required_outputs = [
        args.output_dir / "patch_shape_smoke_test.json",
        args.output_dir / "patch_shape_smoke_test.txt",
        args.output_dir / "sampled_records.csv",
        args.output_dir / "environment.json",
        args.output_dir / "feature_map_statistics.csv",
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required_outputs):
        raise RuntimeError("Atomic output verification failed")
    print(json.dumps({
        "status": "PASS",
        "output_dir": str(args.output_dir),
        "feature_indices": result["sampling"]["feature_indices"],
        "teacher_native_shape": shape["teacher_native_patch_map_shape"],
        "teacher_pooled_shape": shape["teacher_pooled_patch_map_shape"],
        "student_shape": shape["student_final_feature_map_shape"],
        "patch_mse": shape["patch_mse_loss"],
        "patch_cosine_mean": shape["patch_cosine_mean"],
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(r"C:\Users\09688\thoracic-cxr-project-3")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, default=root / "outputs" / "raddino_feature_cache" / "balanced_945_seed42" / "roi_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "raddino_convnext_tiny_patch_experiment_seed42" / "phase0_patch_shape_smoke_test")
    parser.add_argument("--model-name", default="microsoft/rad-dino")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.num_samples < 5 or args.num_samples > 20 or args.seed != 42:
        raise ValueError("Use batch_size >= 1, 5 <= num_samples <= 20, and locked seed 42")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
