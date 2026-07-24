#!/usr/bin/env python
"""Build the frozen RAD-DINO 7x7 patch-feature cache for all ROI records."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
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
from torch.utils.data import DataLoader, Dataset
import torchvision
import transformers
from transformers import AutoImageProcessor, AutoModel


EXPECTED_MANIFEST_SHA256 = "796f067d00bb5740a51b51292eed4acfefe9b2e84fd2eeb9b5dfd2df926d5233"
EXPECTED_MODEL_NAME = "microsoft/rad-dino"
EXPECTED_MODEL_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
EXPECTED_ROWS = 4725
EXPECTED_FEATURE_DIM = 768
EXPECTED_PER_CLASS = 945
EXPECTED_INPUT_SIZE = (518, 518)
EXPECTED_PATCH_SIZE = (14, 14)
EXPECTED_GRID = (37, 37)
EXPECTED_PATCH_COUNT = 1369
EXPECTED_TOTAL_TOKENS = 1370
EXPECTED_SPECIAL_TOKENS = 1
EXPECTED_POOL_SIZE = (7, 7)
EXPECTED_CACHE_SHAPE = (4725, 768, 7, 7)
EXPECTED_NUMEL = 177811200
MIN_FREE_DISK_BYTES = 3 * 1024**3
VERIFY_INDICES = [146, 796, 977, 1335, 2118, 2140, 2977, 3590, 3899, 3945]
CLASS_MAPPING = {
    0: "Aortic enlargement",
    1: "Cardiomegaly",
    2: "Pleural thickening",
    3: "Pulmonary fibrosis",
    4: "Pleural effusion",
}
FINAL_OUTPUT_NAMES = {
    "teacher_patch_features_7x7.pt",
    "teacher_patch_feature_metadata.json",
    "teacher_patch_feature_audit.txt",
    "teacher_patch_feature_sample_verification.csv",
    "environment.json",
}
INCOMPLETE_OUTPUT_NAMES = {
    "teacher_patch_feature_progress.json",
    "teacher_patch_feature_batch_metrics.csv",
    "teacher_patch_features_7x7.pt.tmp",
    "_resume_shards",
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


def parse_bool_strict(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid Boolean value: {value!r}")


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n",
    )


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    torch.save(value, temporary)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patch_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Unsupported patch_size: {value!r}")


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
        "full_config": processor.to_dict(),
    }


def environment_info(device: torch.device) -> dict[str, Any]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": {
            "index": index,
            "name": properties.name,
            "total_vram_bytes": int(properties.total_memory),
            "total_vram_gib": float(properties.total_memory / 1024**3),
        },
    }


def protected_artifact_paths(project_root: Path, manifest: Path) -> dict[str, Path]:
    experiment = project_root / "outputs" / "raddino_convnext_tiny_experiment_seed42"
    feature_cache = project_root / "outputs" / "raddino_feature_cache" / "balanced_945_seed42"
    return {
        "roi_manifest": manifest,
        "old_cls_teacher_cache": feature_cache / "teacher_features.pt",
        "distilled_convnext_tiny_backbone": experiment / "phase1_distillation" / "checkpoints" / "distilled_convnext_tiny_backbone.pt",
        "shared_phase2_finetune_config": experiment / "shared_phase2_finetune_config.json",
        "train_roi_manifest": experiment / "phase2_split" / "train_roi_manifest.csv",
        "val_roi_manifest": experiment / "phase2_split" / "val_roi_manifest.csv",
        "test_roi_manifest": experiment / "phase2_split" / "test_roi_manifest.csv",
    }


def hash_protected_artifacts(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected artifact(s) missing: {missing}")
    return {
        name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def compare_protected_artifacts(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    comparisons = {}
    for name in before:
        match = before[name]["sha256"] == after[name]["sha256"]
        comparisons[name] = {
            "path": before[name]["path"],
            "before_sha256": before[name]["sha256"],
            "after_sha256": after[name]["sha256"],
            "unchanged": match,
        }
    return {
        "all_unchanged": all(row["unchanged"] for row in comparisons.values()),
        "artifacts": comparisons,
    }


def guard_output_directory(output_dir: Path, resume: bool) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    entries = list(output_dir.iterdir())
    if not entries:
        return
    names = {entry.name for entry in entries}
    existing_formal = sorted(names & FINAL_OUTPUT_NAMES)
    if existing_formal:
        raise FileExistsError(f"Formal output already exists; refusing to overwrite: {existing_formal}")
    unknown = sorted(names - INCOMPLETE_OUTPUT_NAMES)
    if unknown:
        raise FileExistsError(f"Unknown output item(s); refusing to modify: {unknown}")
    if not resume:
        raise FileExistsError(
            "Recognized incomplete output exists. Re-run with --resume after inspection: "
            + ", ".join(sorted(names))
        )


def validate_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise FileNotFoundError(f"Manifest missing or empty: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256.lower() != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Manifest SHA256 mismatch: {manifest_sha256}")
    rows = read_csv(manifest_path)
    required = {
        "feature_index", "image_path", "class_id", "class_name", "source_image_id",
        "original_roi_id", "original_roi_path", "is_brightness_augmented",
        "image_width", "image_height", "image_mode",
    }
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Manifest rows={len(rows)}, expected {EXPECTED_ROWS}")
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(rows[0] if rows else {}))}")
    indices = [int(row["feature_index"]) for row in rows]
    if sorted(indices) != list(range(EXPECTED_ROWS)) or len(set(indices)) != EXPECTED_ROWS:
        raise ValueError("feature_index is not a complete unique 0..4724 sequence")
    rows.sort(key=lambda row: int(row["feature_index"]))
    class_counts = Counter(int(row["class_id"]) for row in rows)
    if dict(sorted(class_counts.items())) != {class_id: EXPECTED_PER_CLASS for class_id in range(5)}:
        raise ValueError(f"Class counts mismatch: {dict(class_counts)}")

    original_count = 0
    augmented_count = 0
    missing_paths: list[str] = []
    unreadable: list[str] = []
    metadata_errors: list[str] = []
    relation_errors: list[str] = []
    for row in rows:
        feature_index = int(row["feature_index"])
        class_id = int(row["class_id"])
        if class_id not in CLASS_MAPPING or row["class_name"] != CLASS_MAPPING[class_id]:
            raise ValueError(f"Class mapping error at feature_index={feature_index}")
        path = Path(row["image_path"])
        if not path.is_file():
            missing_paths.append(str(path))
            continue
        if int(row["image_width"]) != 224 or int(row["image_height"]) != 224 or row["image_mode"] != "L":
            metadata_errors.append(str(feature_index))
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != (224, 224) or image.mode != "L":
                    unreadable.append(f"{feature_index}:{image.size}:{image.mode}")
        except Exception as exc:
            unreadable.append(f"{feature_index}:{type(exc).__name__}:{exc}")

        is_augmented = parse_bool_strict(row["is_brightness_augmented"])
        original_id = row["original_roi_id"].strip()
        original_path = Path(row["original_roi_path"])
        if not original_id or not original_path.is_file() or original_path.stem != original_id:
            relation_errors.append(str(feature_index))
        elif is_augmented and not path.stem.startswith(original_id + "__aug_brightness_"):
            relation_errors.append(str(feature_index))
        elif not is_augmented and path.stem != original_id:
            relation_errors.append(str(feature_index))
        if is_augmented:
            augmented_count += 1
        else:
            original_count += 1

    if missing_paths:
        raise FileNotFoundError(f"Missing images={len(missing_paths)}, first={missing_paths[0]}")
    if unreadable:
        raise ValueError(f"Unreadable/wrong source images={len(unreadable)}, first={unreadable[0]}")
    if metadata_errors:
        raise ValueError(f"Manifest image metadata errors={len(metadata_errors)}, first={metadata_errors[0]}")
    if relation_errors:
        raise ValueError(f"ROI/brightness relation errors={len(relation_errors)}, first={relation_errors[0]}")
    return {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
        "rows": rows,
        "row_count": len(rows),
        "feature_index_min": min(indices),
        "feature_index_max": max(indices),
        "unique_feature_indices": len(set(indices)),
        "missing_index_count": 0,
        "duplicate_index_count": 0,
        "class_counts": dict(sorted(class_counts.items())),
        "original_roi_count": original_count,
        "brightness_augmented_count": augmented_count,
        "missing_images": 0,
        "unreadable_images": 0,
        "wrong_size": 0,
        "wrong_mode": 0,
        "relation_errors": 0,
    }


class ManifestImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, position: int) -> tuple[int, Image.Image]:
        row = self.rows[position]
        path = Path(row["image_path"])
        with Image.open(path) as image:
            image.load()
            if image.size != (224, 224) or image.mode != "L":
                raise ValueError(
                    f"Source image changed after preflight: {path}, size={image.size}, mode={image.mode}"
                )
            rgb = image.convert("RGB").copy()
        return int(row["feature_index"]), rgb


def collate_images(items: list[tuple[int, Image.Image]]) -> tuple[list[int], list[Image.Image]]:
    return [item[0] for item in items], [item[1] for item in items]


def load_teacher(args: argparse.Namespace) -> tuple[Any, torch.nn.Module, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is prohibited")
    device = torch.device(args.device)
    if device.type != "cuda" or str(device) != "cuda:0":
        raise ValueError("This run is locked to --device cuda:0")
    processor = AutoImageProcessor.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        local_files_only=True,
    )
    teacher = AutoModel.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        local_files_only=True,
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.to(device)
    if teacher.training or any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("RAD-DINO teacher is not eval/frozen")
    loaded_revision = getattr(teacher.config, "_commit_hash", None)
    if loaded_revision != args.model_revision:
        raise RuntimeError(f"Loaded model revision mismatch: {loaded_revision}")
    return processor, teacher, device


def tensor_input_stats(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "nan_count": int(torch.isnan(tensor).sum().item()),
        "inf_count": int(torch.isinf(tensor).sum().item()),
    }


def forward_patch_batch(
    images: list[Image.Image],
    processor: Any,
    teacher: torch.nn.Module,
    device: torch.device,
    pool_size: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    processed = processor(images=images, return_tensors="pt")
    if "pixel_values" not in processed:
        raise RuntimeError("Official RAD-DINO processor did not return pixel_values")
    pixel_values = processed["pixel_values"]
    if tuple(pixel_values.shape[1:]) != (3, *EXPECTED_INPUT_SIZE):
        raise RuntimeError(f"Unexpected processor output shape: {tuple(pixel_values.shape)}")
    if pixel_values.dtype != torch.float32:
        raise RuntimeError(f"Unexpected processor dtype: {pixel_values.dtype}")
    input_stats = tensor_input_stats(pixel_values)
    model_inputs = {}
    for key, value in processed.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.device.type == "cpu" and not value.is_pinned():
            value = value.pin_memory()
        model_inputs[key] = value.to(device, non_blocking=True)
    pixel_values_device = model_inputs["pixel_values"]
    input_height, input_width = int(pixel_values_device.shape[-2]), int(pixel_values_device.shape[-1])
    patch_height, patch_width = patch_pair(teacher.config.patch_size)
    if input_height % patch_height or input_width % patch_width:
        raise RuntimeError("Processor input is not divisible by model patch size")
    grid_height = input_height // patch_height
    grid_width = input_width // patch_width
    expected_patch_count = grid_height * grid_width

    with torch.inference_mode():
        outputs = teacher(**model_inputs)
        last_hidden_state = outputs.last_hidden_state
        if last_hidden_state.ndim != 3:
            raise RuntimeError(f"Unexpected last_hidden_state rank: {last_hidden_state.ndim}")
        batch_rows, total_tokens, hidden_size = map(int, last_hidden_state.shape)
        special_tokens = total_tokens - expected_patch_count
        if special_tokens < 1:
            raise RuntimeError(
                f"Token count cannot supply expected patches: total={total_tokens}, expected={expected_patch_count}"
            )
        patch_tokens = last_hidden_state[:, -expected_patch_count:, :]
        if tuple(patch_tokens.shape) != (batch_rows, expected_patch_count, hidden_size):
            raise RuntimeError(f"Unexpected patch token shape: {tuple(patch_tokens.shape)}")
        native_map = (
            patch_tokens.reshape(batch_rows, grid_height, grid_width, hidden_size)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pooled_map = F.adaptive_avg_pool2d(native_map, output_size=pool_size)
        pooled_cpu = pooled_map.to(device="cpu", dtype=torch.float32).contiguous()

    shape_audit = {
        "processor_input_shape": list(pixel_values_device.shape),
        "processor_input_stats": input_stats,
        "patch_size": [patch_height, patch_width],
        "grid_height": grid_height,
        "grid_width": grid_width,
        "expected_patch_count": expected_patch_count,
        "last_hidden_state_shape": list(last_hidden_state.shape),
        "total_token_count": total_tokens,
        "special_token_count": special_tokens,
        "hidden_size": hidden_size,
        "patch_tokens_shape": list(patch_tokens.shape),
        "native_patch_map_shape": list(native_map.shape),
        "pooled_patch_map_shape": list(pooled_cpu.shape),
    }
    expected = {
        "input": EXPECTED_INPUT_SIZE,
        "patch": EXPECTED_PATCH_SIZE,
        "grid": EXPECTED_GRID,
        "patch_count": EXPECTED_PATCH_COUNT,
        "total_tokens": EXPECTED_TOTAL_TOKENS,
        "special_tokens": EXPECTED_SPECIAL_TOKENS,
        "hidden": EXPECTED_FEATURE_DIM,
        "pooled": EXPECTED_POOL_SIZE,
    }
    actual = {
        "input": (input_height, input_width),
        "patch": (patch_height, patch_width),
        "grid": (grid_height, grid_width),
        "patch_count": expected_patch_count,
        "total_tokens": total_tokens,
        "special_tokens": special_tokens,
        "hidden": hidden_size,
        "pooled": tuple(pooled_cpu.shape[-2:]),
    }
    if actual != expected:
        raise RuntimeError(f"RAD-DINO patch shape contract mismatch: actual={actual}, expected={expected}")
    if tuple(pooled_cpu.shape) != (len(images), EXPECTED_FEATURE_DIM, *pool_size):
        raise RuntimeError(f"Unexpected pooled map shape: {tuple(pooled_cpu.shape)}")
    nan_count = int(torch.isnan(pooled_cpu).sum().item())
    inf_count = int(torch.isinf(pooled_cpu).sum().item())
    zero_norm_count = int((torch.linalg.vector_norm(pooled_cpu, dim=1) == 0).sum().item())
    if nan_count or inf_count or zero_norm_count:
        raise RuntimeError(
            f"Batch numeric validation failed: NaN={nan_count}, Inf={inf_count}, zero_norm={zero_norm_count}"
        )
    del processed, model_inputs, outputs, last_hidden_state, patch_tokens, native_map, pooled_map
    return pooled_cpu, shape_audit


def batch_metric(
    batch_number: int,
    feature_indices: list[int],
    features: torch.Tensor,
    batch_elapsed: float,
    cumulative_rows: int,
    extraction_elapsed: float,
    device: torch.device,
) -> dict[str, Any]:
    spatial_norms = torch.linalg.vector_norm(features, dim=1)
    return {
        "batch_number": batch_number,
        "first_feature_index": min(feature_indices),
        "last_feature_index": max(feature_indices),
        "batch_rows": len(feature_indices),
        "elapsed_seconds": batch_elapsed,
        "cumulative_rows": cumulative_rows,
        "images_per_second": cumulative_rows / extraction_elapsed if extraction_elapsed else None,
        "allocated_vram_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_vram_bytes": int(torch.cuda.memory_reserved(device)),
        "batch_min": float(features.min().item()),
        "batch_max": float(features.max().item()),
        "batch_mean": float(features.mean().item()),
        "batch_std": float(features.std(unbiased=False).item()),
        "nan_count": int(torch.isnan(features).sum().item()),
        "inf_count": int(torch.isinf(features).sum().item()),
        "zero_norm_count": int((spatial_norms == 0).sum().item()),
    }


BATCH_METRIC_FIELDS = [
    "batch_number", "first_feature_index", "last_feature_index", "batch_rows",
    "elapsed_seconds", "cumulative_rows", "images_per_second",
    "allocated_vram_bytes", "reserved_vram_bytes", "batch_min", "batch_max",
    "batch_mean", "batch_std", "nan_count", "inf_count", "zero_norm_count",
]


def shard_path(shard_dir: Path, feature_indices: list[int]) -> Path:
    return shard_dir / f"batch_{min(feature_indices):04d}_{max(feature_indices):04d}.pt"


def save_shard(
    path: Path,
    feature_indices: list[int],
    features: torch.Tensor,
    manifest_sha256: str,
    model_revision: str,
) -> None:
    payload = {
        "feature_indices": torch.tensor(feature_indices, dtype=torch.int64),
        "features": features.contiguous(),
        "manifest_sha256": manifest_sha256,
        "model_revision": model_revision,
    }
    atomic_torch_save(payload, path)


def load_resume_shards(
    shard_dir: Path,
    cache: torch.Tensor,
    filled_mask: torch.Tensor,
    manifest_sha256: str,
    model_revision: str,
) -> int:
    if not shard_dir.exists():
        return 0
    if not shard_dir.is_dir():
        raise NotADirectoryError(f"Resume shard path is not a directory: {shard_dir}")
    loaded_rows = 0
    for path in sorted(shard_dir.iterdir()):
        if not path.is_file() or not path.name.startswith("batch_") or path.suffix != ".pt":
            raise FileExistsError(f"Unknown resume shard item: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        indices = payload.get("feature_indices")
        features = payload.get("features")
        if payload.get("manifest_sha256") != manifest_sha256 or payload.get("model_revision") != model_revision:
            raise ValueError(f"Resume shard provenance mismatch: {path}")
        if not isinstance(indices, torch.Tensor) or not isinstance(features, torch.Tensor):
            raise TypeError(f"Malformed resume shard: {path}")
        indices = indices.to(dtype=torch.int64, device="cpu")
        features = features.to(dtype=torch.float32, device="cpu").contiguous()
        if features.shape != (indices.numel(), EXPECTED_FEATURE_DIM, *EXPECTED_POOL_SIZE):
            raise ValueError(f"Resume shard shape mismatch: {path}, {tuple(features.shape)}")
        if int(indices.min()) < 0 or int(indices.max()) >= EXPECTED_ROWS:
            raise ValueError(f"Resume shard index out of range: {path}")
        if indices.unique().numel() != indices.numel() or bool(filled_mask[indices].any()):
            raise ValueError(f"Resume shard contains duplicate writes: {path}")
        if int(torch.isnan(features).sum()) or int(torch.isinf(features).sum()):
            raise ValueError(f"Resume shard contains non-finite features: {path}")
        if int((torch.linalg.vector_norm(features, dim=1) == 0).sum()):
            raise ValueError(f"Resume shard contains zero-norm vectors: {path}")
        cache[indices] = features
        filled_mask[indices] = True
        loaded_rows += indices.numel()
    return int(loaded_rows)


def compute_cache_statistics(cache: torch.Tensor, class_ids: torch.Tensor) -> dict[str, Any]:
    totals = {class_id: {"sum": 0.0, "sum_sq": 0.0, "count": 0, "norm_sum": 0.0, "rows": 0} for class_id in range(5)}
    global_sum = 0.0
    global_sum_sq = 0.0
    global_count = 0
    global_min = math.inf
    global_max = -math.inf
    nan_count = 0
    inf_count = 0
    zero_spatial = 0
    zero_maps = 0
    feature_map_norm_min = math.inf
    feature_map_norm_max = -math.inf
    feature_map_norm_sum = 0.0
    chunk_rows = 64
    for start in range(0, cache.shape[0], chunk_rows):
        end = min(start + chunk_rows, cache.shape[0])
        chunk = cache[start:end]
        nan_count += int(torch.isnan(chunk).sum().item())
        inf_count += int(torch.isinf(chunk).sum().item())
        global_min = min(global_min, float(chunk.min().item()))
        global_max = max(global_max, float(chunk.max().item()))
        values = chunk.double()
        global_sum += float(values.sum().item())
        global_sum_sq += float((values * values).sum().item())
        global_count += chunk.numel()
        spatial_norms = torch.linalg.vector_norm(chunk, dim=1)
        zero_spatial += int((spatial_norms == 0).sum().item())
        map_norms = torch.linalg.vector_norm(chunk.flatten(1), dim=1)
        zero_maps += int((map_norms == 0).sum().item())
        feature_map_norm_min = min(feature_map_norm_min, float(map_norms.min().item()))
        feature_map_norm_max = max(feature_map_norm_max, float(map_norms.max().item()))
        feature_map_norm_sum += float(map_norms.double().sum().item())
        chunk_classes = class_ids[start:end]
        for class_id in range(5):
            mask = chunk_classes == class_id
            if not bool(mask.any()):
                continue
            selected = values[mask]
            selected_norms = map_norms[mask]
            totals[class_id]["sum"] += float(selected.sum().item())
            totals[class_id]["sum_sq"] += float((selected * selected).sum().item())
            totals[class_id]["count"] += selected.numel()
            totals[class_id]["norm_sum"] += float(selected_norms.double().sum().item())
            totals[class_id]["rows"] += int(mask.sum().item())

    mean = global_sum / global_count
    variance = max(global_sum_sq / global_count - mean * mean, 0.0)
    by_class = {}
    for class_id, total in totals.items():
        class_mean = total["sum"] / total["count"]
        class_variance = max(total["sum_sq"] / total["count"] - class_mean * class_mean, 0.0)
        by_class[str(class_id)] = {
            "class_name": CLASS_MAPPING[class_id],
            "feature_rows": total["rows"],
            "mean": class_mean,
            "std": math.sqrt(class_variance),
            "feature_map_norm_mean": total["norm_sum"] / total["rows"],
        }
    return {
        "shape": list(cache.shape),
        "dtype": str(cache.dtype),
        "device": str(cache.device),
        "contiguous": cache.is_contiguous(),
        "numel": cache.numel(),
        "min": global_min,
        "max": global_max,
        "mean": mean,
        "std": math.sqrt(variance),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_norm_spatial_vector_count": zero_spatial,
        "feature_map_norm_min": feature_map_norm_min,
        "feature_map_norm_mean": feature_map_norm_sum / cache.shape[0],
        "feature_map_norm_max": feature_map_norm_max,
        "zero_feature_map_count": zero_maps,
        "by_class": by_class,
    }


def verify_loaded_cache(cache: Any) -> dict[str, Any]:
    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"Saved cache is not a tensor: {type(cache).__name__}")
    checks = {
        "shape": tuple(cache.shape) == EXPECTED_CACHE_SHAPE,
        "dtype": cache.dtype == torch.float32,
        "device": cache.device.type == "cpu",
        "contiguous": cache.is_contiguous(),
        "numel": cache.numel() == EXPECTED_NUMEL,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Reloaded cache structural validation failed: {failed}")
    return checks


def sample_verification(
    rows: list[dict[str, str]],
    cached: torch.Tensor,
    processor: Any,
    teacher: torch.nn.Module,
    device: torch.device,
    pool_size: tuple[int, int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    batch_starts = sorted({(feature_index // batch_size) * batch_size for feature_index in VERIFY_INDICES})
    for batch_start in batch_starts:
        batch_end = min(batch_start + batch_size, len(rows))
        batch_rows = rows[batch_start:batch_end]
        images = []
        for row in batch_rows:
            with Image.open(row["image_path"]) as image:
                image.load()
                images.append(image.convert("RGB").copy())
        recomputed, _ = forward_patch_batch(images, processor, teacher, device, pool_size)
        batch_targets = [
            feature_index for feature_index in VERIFY_INDICES
            if batch_start <= feature_index < batch_end
        ]
        for feature_index in batch_targets:
            position = feature_index - batch_start
            difference = (cached[feature_index] - recomputed[position]).abs()
            is_close = bool(
                torch.allclose(
                    cached[feature_index],
                    recomputed[position],
                    rtol=1e-5,
                    atol=1e-6,
                )
            )
            records.append({
                "feature_index": feature_index,
                "recompute_batch_first_index": batch_start,
                "recompute_batch_last_index": batch_end - 1,
                "recompute_batch_rows": len(batch_rows),
                "max_abs_difference": float(difference.max().item()),
                "mean_abs_difference": float(difference.mean().item()),
                "rtol": 1e-5,
                "atol": 1e-6,
                "allclose": is_close,
            })
    records.sort(key=lambda row: row["feature_index"])
    summary = {
        "sample_count": len(records),
        "all_passed": all(row["allclose"] for row in records),
        "maximum_abs_difference": max(row["max_abs_difference"] for row in records),
        "maximum_mean_abs_difference": max(row["mean_abs_difference"] for row in records),
    }
    return records, summary


SAMPLE_FIELDS = [
    "feature_index", "recompute_batch_first_index", "recompute_batch_last_index",
    "recompute_batch_rows", "max_abs_difference", "mean_abs_difference", "rtol", "atol", "allclose",
]


def cleanup_resume_shards(shard_dir: Path) -> None:
    if not shard_dir.exists():
        return
    for path in list(shard_dir.iterdir()):
        if not path.is_file() or not path.name.startswith("batch_") or path.suffix != ".pt":
            raise FileExistsError(f"Refusing to clean unknown shard item: {path}")
        path.unlink()
    shard_dir.rmdir()


def write_audit(path: Path, metadata: dict[str, Any]) -> None:
    stats = metadata["cache_statistics"]
    sample = metadata["sample_verification"]
    shape = metadata["shape_contract"]
    lines = [
        "RAD-DINO 7x7 Patch Teacher Feature Cache Audit",
        "================================================",
        f"status: {metadata['status']}",
        f"manifest_rows: {metadata['manifest']['row_count']}",
        f"manifest_sha256: {metadata['manifest']['sha256']}",
        f"model: {metadata['model']['name']} @ {metadata['model']['revision']}",
        f"processor_input_shape: {shape['processor_input_shape']}",
        f"patch_size: {shape['patch_size']}",
        f"patch_grid: {shape['grid_height']}x{shape['grid_width']}",
        f"tokens_total_special_patch: {shape['total_token_count']}/{shape['special_token_count']}/{shape['expected_patch_count']}",
        f"native_patch_map_shape: {shape['native_patch_map_shape']}",
        f"pooled_patch_map_shape: {shape['pooled_patch_map_shape']}",
        f"cache_shape: {stats['shape']}",
        f"cache_dtype_device_contiguous: {stats['dtype']} / {stats['device']} / {stats['contiguous']}",
        f"cache_numel: {stats['numel']}",
        f"nan_inf_zero_spatial_zero_map: {stats['nan_count']} / {stats['inf_count']} / {stats['zero_norm_spatial_vector_count']} / {stats['zero_feature_map_count']}",
        f"missing_index_count: {metadata['index_integrity']['missing_index_count']}",
        f"duplicate_write_count: {metadata['index_integrity']['duplicate_write_count']}",
        f"sample_allclose: {sample['passed_count']}/{sample['sample_count']}",
        f"sample_max_abs_difference: {sample['maximum_abs_difference']}",
        f"cache_size_bytes: {metadata['cache_file']['size_bytes']}",
        f"cache_sha256: {metadata['cache_file']['sha256']}",
        f"batch_size_workers: {metadata['execution']['batch_size']} / {metadata['execution']['workers']}",
        f"wall_time_seconds: {metadata['execution']['wall_time_seconds']}",
        f"images_per_second: {metadata['execution']['images_per_second']}",
        f"peak_allocated_vram_bytes: {metadata['execution']['peak_allocated_vram_bytes']}",
        f"peak_reserved_vram_bytes: {metadata['execution']['peak_reserved_vram_bytes']}",
        f"protected_artifacts_unchanged: {metadata['protected_artifacts']['all_unchanged']}",
        "oom: False",
        "backward_executed: False",
        "optimizer_created_or_stepped: False",
        "student_created_or_trained: False",
        "patch_phase1_started: False",
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    guard_output_directory(output_dir, args.resume)
    disk_usage = shutil.disk_usage(output_dir.parent)
    if disk_usage.free < MIN_FREE_DISK_BYTES:
        raise RuntimeError(
            f"Insufficient disk space: {disk_usage.free} bytes available, {MIN_FREE_DISK_BYTES} required"
        )
    manifest = validate_manifest(manifest_path)
    protected_paths = protected_artifact_paths(project_root, manifest_path)
    protected_before = hash_protected_artifacts(protected_paths)
    preflight = {
        "status": "PASS",
        "manifest": {key: value for key, value in manifest.items() if key != "rows"},
        "disk_free_bytes": disk_usage.free,
        "protected_artifacts": protected_before,
        "output_directory_exists": output_dir.exists(),
        "resume_requested": args.resume,
    }
    if args.dry_run:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, default=json_default))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "teacher_patch_feature_progress.json"
    batch_metrics_path = output_dir / "teacher_patch_feature_batch_metrics.csv"
    shard_dir = output_dir / "_resume_shards"
    shard_dir.mkdir(exist_ok=True)
    previous_progress: dict[str, Any] = {}
    previous_batch_metrics: list[dict[str, Any]] = []
    previous_wall_time = 0.0
    previous_extraction_time = 0.0
    if args.resume and progress_path.is_file():
        previous_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if previous_progress.get("manifest_sha256") != manifest["sha256"]:
            raise ValueError("Resume progress Manifest SHA256 mismatch")
        if previous_progress.get("model_revision") != args.model_revision:
            raise ValueError("Resume progress model revision mismatch")
        previous_wall_time = float(previous_progress.get("elapsed_time_seconds") or 0.0)
    if args.resume and batch_metrics_path.is_file():
        previous_batch_metrics = read_csv(batch_metrics_path)
        if previous_batch_metrics:
            last_metric = max(previous_batch_metrics, key=lambda row: int(row["cumulative_rows"]))
            previous_rows = int(last_metric["cumulative_rows"])
            previous_rate = float(last_metric["images_per_second"])
            previous_extraction_time = previous_rows / previous_rate if previous_rate else 0.0
    resume_started_utc = utc_now()
    run_started_utc = previous_progress.get("started_at_utc", resume_started_utc)
    run_started = time.perf_counter()
    progress: dict[str, Any] = {
        "status": "RUNNING",
        "started_at_utc": run_started_utc,
        "updated_at_utc": run_started_utc,
        "completed_rows": 0,
        "total_rows": EXPECTED_ROWS,
        "last_completed_feature_index": None,
        "elapsed_time_seconds": 0.0,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "manifest_sha256": manifest["sha256"],
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "resume": args.resume,
        "resume_started_at_utc": resume_started_utc if args.resume else None,
        "previous_elapsed_time_seconds": previous_wall_time,
    }
    atomic_write_json(progress_path, progress)

    cache = torch.empty(EXPECTED_CACHE_SHAPE, dtype=torch.float32, device="cpu")
    filled_mask = torch.zeros(EXPECTED_ROWS, dtype=torch.bool)
    duplicate_write_count = 0
    batch_metrics: list[dict[str, Any]] = previous_batch_metrics
    shape_contract: dict[str, Any] | None = None
    teacher: torch.nn.Module | None = None
    try:
        resumed_rows = load_resume_shards(
            shard_dir,
            cache,
            filled_mask,
            manifest["sha256"],
            args.model_revision,
        ) if args.resume else 0
        progress["completed_rows"] = resumed_rows
        progress["last_completed_feature_index"] = int(torch.where(filled_mask)[0].max()) if resumed_rows else None
        atomic_write_json(progress_path, progress)

        processor, teacher, device = load_teacher(args)
        environment = environment_info(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        pending_rows = [row for row in manifest["rows"] if not filled_mask[int(row["feature_index"])] ]
        dataset = ManifestImageDataset(pending_rows)
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_images,
            generator=generator,
            persistent_workers=False,
            drop_last=False,
        )
        extraction_started = time.perf_counter()
        completed_rows = resumed_rows
        for loader_batch_number, (feature_indices, images) in enumerate(loader, start=1):
            batch_started = time.perf_counter()
            if len(feature_indices) != len(set(feature_indices)):
                raise RuntimeError(f"Duplicate feature_index inside batch: {feature_indices}")
            if min(feature_indices) < 0 or max(feature_indices) >= EXPECTED_ROWS:
                raise RuntimeError(f"Out-of-range feature_index in batch: {feature_indices}")
            index_tensor = torch.tensor(feature_indices, dtype=torch.int64)
            if bool(filled_mask[index_tensor].any()):
                duplicate_write_count += int(filled_mask[index_tensor].sum().item())
                raise RuntimeError(f"Attempted duplicate cache write: {feature_indices}")
            pooled, current_shape = forward_patch_batch(
                images, processor, teacher, device, (args.pool_height, args.pool_width)
            )
            if shape_contract is None:
                shape_contract = current_shape
            else:
                invariant_keys = [
                    "patch_size", "grid_height", "grid_width", "expected_patch_count",
                    "total_token_count", "special_token_count", "hidden_size",
                ]
                if any(current_shape[key] != shape_contract[key] for key in invariant_keys):
                    raise RuntimeError(f"Batch shape contract changed: {current_shape}")
            if pooled.dtype != torch.float32 or pooled.device.type != "cpu" or not pooled.is_contiguous():
                raise RuntimeError("Pooled batch must be contiguous CPU float32")
            save_shard(
                shard_path(shard_dir, feature_indices),
                feature_indices,
                pooled,
                manifest["sha256"],
                args.model_revision,
            )
            cache[index_tensor] = pooled
            filled_mask[index_tensor] = True
            completed_rows += len(feature_indices)
            torch.cuda.synchronize(device)
            batch_elapsed = time.perf_counter() - batch_started
            extraction_elapsed = previous_extraction_time + (time.perf_counter() - extraction_started)
            batch_number = (min(feature_indices) // args.batch_size) + 1
            metric = batch_metric(
                batch_number,
                feature_indices,
                pooled,
                batch_elapsed,
                completed_rows,
                extraction_elapsed,
                device,
            )
            batch_metrics.append(metric)
            atomic_write_csv(batch_metrics_path, batch_metrics, BATCH_METRIC_FIELDS)
            progress.update({
                "status": "RUNNING",
                "updated_at_utc": utc_now(),
                "completed_rows": completed_rows,
                "last_completed_feature_index": max(feature_indices),
                "elapsed_time_seconds": previous_wall_time + (time.perf_counter() - run_started),
                "images_per_second": completed_rows / extraction_elapsed if extraction_elapsed else None,
                "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
            })
            atomic_write_json(progress_path, progress)
            print(
                f"Processed {completed_rows}/{EXPECTED_ROWS} "
                f"({100.0 * completed_rows / EXPECTED_ROWS:.1f}%), "
                f"batch={len(feature_indices)}, {progress['images_per_second']:.2f} images/s",
                flush=True,
            )
            del pooled, index_tensor, images

        del loader, dataset
        if not bool(filled_mask.all()):
            missing_indices = torch.where(~filled_mask)[0].tolist()
            raise RuntimeError(f"Cache has missing feature indices: {missing_indices[:20]}")
        missing_index_count = int((~filled_mask).sum().item())
        if duplicate_write_count:
            raise RuntimeError(f"Duplicate cache writes={duplicate_write_count}")
        if shape_contract is None:
            shape_images = []
            for row in manifest["rows"][:args.batch_size]:
                with Image.open(row["image_path"]) as image:
                    image.load()
                    shape_images.append(image.convert("RGB").copy())
            _, shape_contract = forward_patch_batch(
                shape_images,
                processor,
                teacher,
                device,
                (args.pool_height, args.pool_width),
            )
        new_extraction_time = time.perf_counter() - extraction_started

        temporary_cache_path = output_dir / "teacher_patch_features_7x7.pt.tmp"
        final_cache_path = output_dir / "teacher_patch_features_7x7.pt"
        atomic_torch_save(cache, temporary_cache_path)
        del cache
        gc.collect()
        loaded_cache = torch.load(temporary_cache_path, map_location="cpu", weights_only=True)
        reload_checks = verify_loaded_cache(loaded_cache)
        class_ids = torch.tensor([int(row["class_id"]) for row in manifest["rows"]], dtype=torch.int64)
        cache_statistics = compute_cache_statistics(loaded_cache, class_ids)
        numeric_failures = {
            "nan_count": cache_statistics["nan_count"],
            "inf_count": cache_statistics["inf_count"],
            "zero_norm_spatial_vector_count": cache_statistics["zero_norm_spatial_vector_count"],
            "zero_feature_map_count": cache_statistics["zero_feature_map_count"],
        }
        if any(numeric_failures.values()):
            raise RuntimeError(f"Reloaded cache numeric validation failed: {numeric_failures}")

        verification_rows, verification_summary = sample_verification(
            manifest["rows"],
            loaded_cache,
            processor,
            teacher,
            device,
            (args.pool_height, args.pool_width),
            args.batch_size,
        )
        if not verification_summary["all_passed"]:
            raise RuntimeError(f"Sample recomputation failed allclose: {verification_rows}")

        protected_after = hash_protected_artifacts(protected_paths)
        protected_comparison = compare_protected_artifacts(protected_before, protected_after)
        if not protected_comparison["all_unchanged"]:
            raise RuntimeError(f"Protected artifacts changed: {protected_comparison}")
        os.replace(temporary_cache_path, final_cache_path)
        cache_sha256 = sha256_file(final_cache_path)
        cache_size_bytes = final_cache_path.stat().st_size
        extraction_elapsed = previous_extraction_time + new_extraction_time
        completed_utc = utc_now()
        wall_time = previous_wall_time + (time.perf_counter() - run_started)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        sample_summary = {
            **verification_summary,
            "passed_count": sum(row["allclose"] for row in verification_rows),
            "rtol": 1e-5,
            "atol": 1e-6,
            "feature_indices": VERIFY_INDICES,
        }
        metadata = {
            "status": "PASS",
            "phase": "Phase 0 Patch Teacher Cache",
            "started_at_utc": run_started_utc,
            "completed_at_utc": completed_utc,
            "manifest": {key: value for key, value in manifest.items() if key != "rows"},
            "model": {
                "name": args.model_name,
                "revision": args.model_revision,
                "model_class": teacher.__class__.__name__,
                "teacher_eval": True,
                "teacher_frozen": True,
                "official_processor_used": True,
                "processor": processor_audit(processor),
            },
            "shape_contract": shape_contract,
            "cache_file": {
                "path": str(final_cache_path),
                "sha256": cache_sha256,
                "size_bytes": cache_size_bytes,
                "raw_tensor_bytes": loaded_cache.numel() * loaded_cache.element_size(),
                "layout": "NCHW",
                "pooling_method": "torch.nn.functional.adaptive_avg_pool2d",
            },
            "cache_statistics": cache_statistics,
            "reload_validation": reload_checks,
            "index_integrity": {
                "feature_index_range": [0, EXPECTED_ROWS - 1],
                "filled_rows": int(filled_mask.sum().item()),
                "missing_index_count": missing_index_count,
                "duplicate_write_count": duplicate_write_count,
            },
            "sample_verification": sample_summary,
            "execution": {
                "batch_size": args.batch_size,
                "workers": args.workers,
                "pin_memory": True,
                "seed": args.seed,
                "device": args.device,
                "wall_time_seconds": wall_time,
                "extraction_time_seconds": extraction_elapsed,
                "images_per_second": EXPECTED_ROWS / extraction_elapsed,
                "peak_allocated_vram_bytes": peak_allocated,
                "peak_reserved_vram_bytes": peak_reserved,
                "oom": False,
                "autocast_used": False,
                "mixed_precision_used": False,
                "backward_executed": False,
                "optimizer_created_or_stepped": False,
                "student_created_or_trained": False,
                "patch_phase1_started": False,
            },
            "environment": environment,
            "protected_artifacts": protected_comparison,
        }
        atomic_write_csv(
            output_dir / "teacher_patch_feature_sample_verification.csv",
            verification_rows,
            SAMPLE_FIELDS,
        )
        atomic_write_json(output_dir / "environment.json", environment)
        atomic_write_json(output_dir / "teacher_patch_feature_metadata.json", metadata)
        write_audit(output_dir / "teacher_patch_feature_audit.txt", metadata)
        progress.update({
            "status": "PASS",
            "updated_at_utc": completed_utc,
            "completed_rows": EXPECTED_ROWS,
            "last_completed_feature_index": EXPECTED_ROWS - 1,
            "elapsed_time_seconds": wall_time,
            "images_per_second": EXPECTED_ROWS / extraction_elapsed,
            "peak_allocated_vram_bytes": peak_allocated,
            "peak_reserved_vram_bytes": peak_reserved,
            "cache_sha256": cache_sha256,
            "sample_allclose_passed": verification_summary["all_passed"],
        })
        atomic_write_json(progress_path, progress)
        cleanup_resume_shards(shard_dir)
        leftovers = [
            str(path) for path in output_dir.iterdir()
            if path.name.endswith(".tmp") or path.name.endswith(".writing") or path.name == "_resume_shards"
        ]
        if leftovers:
            raise RuntimeError(f"Temporary output remains after success: {leftovers}")
        print(json.dumps({
            "status": "PASS",
            "cache_path": str(final_cache_path),
            "cache_shape": list(loaded_cache.shape),
            "cache_sha256": cache_sha256,
            "cache_size_bytes": cache_size_bytes,
            "nan_count": cache_statistics["nan_count"],
            "inf_count": cache_statistics["inf_count"],
            "zero_norm_count": cache_statistics["zero_norm_spatial_vector_count"],
            "sample_allclose": verification_summary["all_passed"],
            "wall_time_seconds": wall_time,
            "images_per_second": EXPECTED_ROWS / extraction_elapsed,
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        progress.update({
            "status": "FAIL",
            "updated_at_utc": utc_now(),
            "elapsed_time_seconds": previous_wall_time + (time.perf_counter() - run_started),
            "error_type": type(exc).__name__,
            "error_reason": str(exc),
            "completed_rows": int(filled_mask.sum().item()),
            "last_completed_feature_index": (
                int(torch.where(filled_mask)[0].max()) if bool(filled_mask.any()) else None
            ),
            "duplicate_write_count": duplicate_write_count,
            "oom": isinstance(exc, torch.OutOfMemoryError),
        })
        if torch.cuda.is_available():
            progress["peak_allocated_vram_bytes"] = int(torch.cuda.max_memory_allocated())
            progress["peak_reserved_vram_bytes"] = int(torch.cuda.max_memory_reserved())
        atomic_write_json(progress_path, progress)
        raise
    finally:
        if teacher is not None:
            del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    root = Path(r"C:\Users\09688\thoracic-cxr-project-3")
    parser = argparse.ArgumentParser(
        description="Build the complete frozen RAD-DINO 7x7 patch teacher feature cache."
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "outputs" / "raddino_feature_cache" / "balanced_945_seed42" / "roi_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "raddino_convnext_tiny_patch_experiment_seed42" / "phase0_patch_teacher_cache",
    )
    parser.add_argument("--model-name", default=EXPECTED_MODEL_NAME)
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-height", type=int, default=7)
    parser.add_argument("--pool-width", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    locked = {
        "model_name": (args.model_name, EXPECTED_MODEL_NAME),
        "model_revision": (args.model_revision, EXPECTED_MODEL_REVISION),
        "batch_size": (args.batch_size, 32),
        "workers": (args.workers, 2),
        "device": (args.device, "cuda:0"),
        "seed": (args.seed, 42),
        "pool_height": (args.pool_height, 7),
        "pool_width": (args.pool_width, 7),
    }
    mismatches = {name: values for name, values in locked.items() if values[0] != values[1]}
    if mismatches:
        raise ValueError(f"This formal run has locked parameters: {mismatches}")
    set_seed(args.seed)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
