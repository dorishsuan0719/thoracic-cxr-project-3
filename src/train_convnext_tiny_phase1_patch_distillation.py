#!/usr/bin/env python
"""Phase 1: distill cached RAD-DINO 7x7 patch maps into ConvNeXt-Tiny."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.transforms import functional as TF


EXPECTED_MANIFEST_SHA256 = "796f067d00bb5740a51b51292eed4acfefe9b2e84fd2eeb9b5dfd2df926d5233"
EXPECTED_TEACHER_SHA256 = "082c626e9a5730023361f48566e68bb653ceceb2e0600c42f23e555336002828"
EXPECTED_PRETRAINED_SHA256 = "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"
EXPECTED_ROWS = 4725
EXPECTED_DIM = 768
EXPECTED_MAP_SHAPE = (768, 7, 7)
EXPECTED_CACHE_SHAPE = (4725, 768, 7, 7)
EXPECTED_ORIGINAL = 4256
EXPECTED_AUGMENTED = 469
EXPECTED_PER_CLASS = 945
PHASE = "phase1_patch_distillation"
ARCHITECTURE = "convnext_tiny"
CLASS_MAPPING = {
    0: "Aortic enlargement",
    1: "Cardiomegaly",
    2: "Pleural thickening",
    3: "Pulmonary fibrosis",
    4: "Pleural effusion",
}

EPOCH_FIELDS = [
    "epoch", "train_patch_mse", "train_patch_cosine_mean", "train_patch_cosine_min",
    "train_patch_cosine_max", "monitor_patch_mse", "monitor_patch_cosine_mean",
    "monitor_patch_cosine_min", "monitor_patch_cosine_max",
    "monitor_improvement_from_previous", "monitor_improvement_from_best",
    "monitor_recent_5_epoch_change", "monitor_recent_10_epoch_change",
    "monitor_recent_5_epoch_slope", "monitor_recent_10_epoch_slope", "learning_rate",
    "gradient_norm_mean", "gradient_norm_max", "student_feature_norm_mean",
    "teacher_feature_norm_mean", "epoch_train_seconds", "epoch_monitor_seconds",
    "epoch_total_seconds", "images_per_second", "gpu_allocated_peak_bytes",
    "gpu_reserved_peak_bytes", "nan_count", "inf_count", "non_finite_gradient_count",
    "patience_counter", "is_best", "stop_reason",
]
BATCH_FIELDS = [
    "epoch", "phase", "batch_number", "batch_rows", "first_feature_index",
    "last_feature_index", "patch_mse", "patch_cosine_mean", "patch_cosine_min",
    "patch_cosine_max", "student_feature_norm_mean", "teacher_feature_norm_mean",
    "gradient_norm", "learning_rate", "elapsed_seconds", "images_per_second",
    "allocated_vram_bytes", "reserved_vram_bytes", "nan_count", "inf_count",
    "non_finite_gradient_count",
]
IMAGE_COSINE_FIELDS = ["epoch", "phase", "feature_index", "mean_patch_cosine"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def replace_with_retry(source: Path, destination: Path, attempts: int = 20) -> None:
    """Replace atomically while tolerating short-lived Windows reader locks."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    replace_with_retry(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    replace_with_retry(temporary, path)


def atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    if loaded.get("phase") != PHASE or loaded.get("architecture") != ARCHITECTURE:
        raise RuntimeError(f"Checkpoint reload validation failed: {temporary}")
    replace_with_retry(temporary, path)


def atomic_save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, dpi=150, bbox_inches="tight")
    plt.close(figure)
    replace_with_retry(temporary, path)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def environment_info(device: torch.device) -> dict[str, Any]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "device": str(device),
        "gpu": {
            "index": index,
            "name": props.name,
            "total_vram_bytes": int(props.total_memory),
            "total_vram_gib": float(props.total_memory / 1024**3),
        },
    }


def configure_runtime(device: torch.device, seed: int) -> None:
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if device.type != "cuda":
        raise RuntimeError("Patch Phase 1 requires CUDA; CPU fallback is prohibited")


def validate_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Manifest missing or empty: {path}")
    sha = sha256_file(path)
    if sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Manifest SHA256 mismatch: {sha}")
    rows = read_csv(path)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Manifest rows={len(rows)}, expected {EXPECTED_ROWS}")
    indices = [int(row["feature_index"]) for row in rows]
    if sorted(indices) != list(range(EXPECTED_ROWS)) or len(set(indices)) != EXPECTED_ROWS:
        raise ValueError("feature_index is not a complete unique 0..4724 sequence")
    rows.sort(key=lambda row: int(row["feature_index"]))
    class_counts = Counter(int(row["class_id"]) for row in rows)
    if dict(sorted(class_counts.items())) != {index: EXPECTED_PER_CLASS for index in range(5)}:
        raise ValueError(f"Manifest class counts mismatch: {class_counts}")
    original = 0
    augmented = 0
    errors = []
    for row in rows:
        feature_index = int(row["feature_index"])
        path_value = Path(row["image_path"])
        try:
            with Image.open(path_value) as image:
                image.load()
                if image.size != (224, 224) or image.mode != "L":
                    errors.append(f"{feature_index}:{image.size}:{image.mode}")
        except Exception as exc:
            errors.append(f"{feature_index}:{type(exc).__name__}:{exc}")
        if parse_bool(row["is_brightness_augmented"]):
            augmented += 1
        else:
            original += 1
    if errors:
        raise ValueError(f"Manifest image errors={len(errors)}, first={errors[0]}")
    if original != EXPECTED_ORIGINAL or augmented != EXPECTED_AUGMENTED:
        raise ValueError(f"Original/augmented mismatch: {original}/{augmented}")
    return {
        "path": str(path),
        "sha256": sha,
        "rows": rows,
        "row_count": len(rows),
        "feature_index_range": [0, EXPECTED_ROWS - 1],
        "missing_index_count": 0,
        "duplicate_index_count": 0,
        "class_counts": dict(sorted(class_counts.items())),
        "original_roi_count": original,
        "brightness_augmented_roi_count": augmented,
        "missing_images": 0,
        "unreadable_images": 0,
        "wrong_size": 0,
        "wrong_mode": 0,
    }


def validate_teacher_cache(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Teacher cache missing or empty: {path}")
    sha = sha256_file(path)
    if sha != EXPECTED_TEACHER_SHA256:
        raise ValueError(f"Teacher cache SHA256 mismatch: {sha}")
    load_method = "torch.load(weights_only=True, mmap=True)"
    fallback_reason = None
    try:
        cache = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError, ValueError) as exc:
        fallback_reason = f"{type(exc).__name__}: {exc}"
        load_method = "torch.load(weights_only=True, mmap=False)"
        cache = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"Teacher cache must be a tensor, got {type(cache).__name__}")
    if tuple(cache.shape) != EXPECTED_CACHE_SHAPE:
        raise ValueError(f"Teacher cache shape mismatch: {tuple(cache.shape)}")
    if cache.dtype != torch.float32 or cache.device.type != "cpu" or not cache.is_contiguous():
        raise ValueError(f"Teacher cache contract mismatch: {cache.dtype}/{cache.device}/{cache.is_contiguous()}")
    nan_count = inf_count = zero_norm_count = 0
    for start in range(0, EXPECTED_ROWS, 64):
        chunk = cache[start:start + 64]
        nan_count += int(torch.isnan(chunk).sum().item())
        inf_count += int(torch.isinf(chunk).sum().item())
        zero_norm_count += int((torch.linalg.vector_norm(chunk, dim=1) == 0).sum().item())
    if nan_count or inf_count or zero_norm_count:
        raise ValueError(
            f"Teacher cache numeric failure: NaN={nan_count}, Inf={inf_count}, zero={zero_norm_count}"
        )
    return {
        "path": str(path),
        "sha256": sha,
        "tensor": cache,
        "type": type(cache).__name__,
        "shape": list(cache.shape),
        "dtype": str(cache.dtype),
        "device": str(cache.device),
        "contiguous": cache.is_contiguous(),
        "layout": "NCHW",
        "load_method": load_method,
        "mmap_fallback_reason": fallback_reason,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_norm_spatial_vector_count": zero_norm_count,
    }


class StudentTransform:
    def __init__(self, augment: bool) -> None:
        official = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
        self.resize_size = list(official.resize_size)
        self.crop_size = list(official.crop_size)
        self.interpolation = official.interpolation
        self.antialias = official.antialias
        self.mean = list(official.mean)
        self.std = list(official.std)
        self.augment = augment
        self.blur_probability = 0.20
        self.blur_kernel_size = 3
        self.blur_sigma = (0.1, 0.5)
        self.noise_probability = 0.30
        self.noise_std = (0.005, 0.015)

    def preprocessing_config(self) -> dict[str, Any]:
        return {
            "resize_size": self.resize_size,
            "crop_size": self.crop_size,
            "interpolation": str(self.interpolation),
            "antialias": self.antialias,
            "mean": self.mean,
            "std": self.std,
            "source_mode": "L",
            "student_mode": "RGB",
            "pipeline_order": [
                "in-memory L to RGB",
                "optional Gaussian blur",
                "official resize",
                "official center crop",
                "ToTensor [0,1]",
                "optional Gaussian noise and clamp [0,1]",
                "ImageNet normalization",
            ],
        }

    def augmentation_config(self) -> dict[str, Any]:
        return {
            "gaussian_blur": {
                "enabled": True,
                "probability": self.blur_probability,
                "kernel_size": self.blur_kernel_size,
                "sigma_range": list(self.blur_sigma),
            },
            "gaussian_noise": {
                "enabled": True,
                "probability": self.noise_probability,
                "mean": 0.0,
                "std_range": list(self.noise_std),
                "position": "after ToTensor and before Normalize",
                "clamp": [0.0, 1.0],
            },
            "brightness_transform": False,
            "contrast_transform": False,
            "other_transforms": [],
        }

    def apply(self, image: Image.Image, return_display: bool = False) -> Any:
        if image.mode != "RGB":
            image = image.convert("RGB")
        blur_applied = self.augment and random.random() < self.blur_probability
        blur_sigma = None
        if blur_applied:
            blur_sigma = random.uniform(*self.blur_sigma)
            image = TF.gaussian_blur(
                image,
                kernel_size=[self.blur_kernel_size, self.blur_kernel_size],
                sigma=[blur_sigma, blur_sigma],
            )
        image = TF.resize(
            image,
            self.resize_size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        image = TF.center_crop(image, self.crop_size)
        tensor = TF.pil_to_tensor(image).to(torch.float32).div_(255.0)
        noise_applied = self.augment and random.random() < self.noise_probability
        noise_std = None
        if noise_applied:
            noise_std = random.uniform(*self.noise_std)
            tensor = torch.clamp(tensor + torch.randn_like(tensor) * noise_std, 0.0, 1.0)
        display = tensor.clone() if return_display else None
        normalized = TF.normalize(tensor, self.mean, self.std)
        if return_display:
            return normalized, display, {
                "blur_applied": blur_applied,
                "blur_sigma": blur_sigma,
                "noise_applied": noise_applied,
                "noise_std": noise_std,
            }
        return normalized

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.apply(image, return_display=False)


class RoiDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform: StudentTransform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = Path(row["image_path"])
        with Image.open(path) as image:
            image.load()
            if image.size != (224, 224) or image.mode != "L":
                raise ValueError(f"ROI changed after validation: {path}, {image.size}, {image.mode}")
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "feature_index": int(row["feature_index"])}


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


class ConvNeXtTinyPatchStudent(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_tiny(weights=weights)
        self.features = base.features
        del base

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


def pretrained_weight_info() -> dict[str, Any]:
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not path.is_file():
        raise FileNotFoundError(f"Official ConvNeXt-Tiny weights are not cached: {path}")
    sha = sha256_file(path)
    if sha != EXPECTED_PRETRAINED_SHA256:
        raise ValueError(f"Pretrained weight SHA256 mismatch: {sha}")
    return {
        "weights_enum": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "url": weights.url,
        "cache_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha,
    }


def extract_teacher(cache: torch.Tensor, feature_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    selected = cache.index_select(0, feature_indices.to(device="cpu", dtype=torch.int64))
    if device.type == "cuda" and not selected.is_pinned():
        selected = selected.pin_memory()
    return selected.to(device, dtype=torch.float32, non_blocking=True).detach()


def patch_alignment(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    if student.shape != teacher.shape or tuple(student.shape[1:]) != EXPECTED_MAP_SHAPE:
        raise RuntimeError(f"Student/teacher patch shape mismatch: {student.shape}/{teacher.shape}")
    counts = {
        "nan_count": int(torch.isnan(student).sum().item() + torch.isnan(teacher).sum().item()),
        "inf_count": int(torch.isinf(student).sum().item() + torch.isinf(teacher).sum().item()),
        "zero_norm_count": int(
            (torch.linalg.vector_norm(student.float(), dim=1) == 0).sum().item()
            + (torch.linalg.vector_norm(teacher.float(), dim=1) == 0).sum().item()
        ),
    }
    if any(counts.values()):
        raise FloatingPointError(f"Raw patch feature numeric failure: {counts}")
    student_normalized = F.normalize(student.float(), p=2, dim=1, eps=1e-12)
    teacher_normalized = F.normalize(teacher.float(), p=2, dim=1, eps=1e-12)
    if not torch.isfinite(student_normalized).all() or not torch.isfinite(teacher_normalized).all():
        raise FloatingPointError("Normalized patch feature is non-finite")
    loss = F.mse_loss(student_normalized, teacher_normalized)
    batch = student.shape[0]
    student_positions = student_normalized.permute(0, 2, 3, 1).reshape(batch, 49, EXPECTED_DIM)
    teacher_positions = teacher_normalized.permute(0, 2, 3, 1).reshape(batch, 49, EXPECTED_DIM)
    cosine = F.cosine_similarity(student_positions, teacher_positions, dim=-1)
    if not torch.isfinite(loss) or not torch.isfinite(cosine).all():
        raise FloatingPointError("Patch loss/cosine is non-finite")
    return loss, cosine, student_normalized, teacher_normalized, counts


def gradient_status(model: nn.Module) -> tuple[bool, float, int]:
    sum_sq = 0.0
    found = False
    non_finite = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        found = True
        gradient = parameter.grad.detach()
        non_finite += int((~torch.isfinite(gradient)).sum().item())
        if non_finite == 0:
            sum_sq += float(torch.sum(gradient.float() ** 2).item())
    norm = math.sqrt(sum_sq) if non_finite == 0 else float("nan")
    return found and non_finite == 0 and math.isfinite(norm) and norm > 0, norm, non_finite


def make_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def make_scaler(device: torch.device) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")


def training_config(args: argparse.Namespace, transform: StudentTransform) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "architecture": ARCHITECTURE,
        "initialization": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "student_output": "student.features(images)",
        "student_output_shape": ["B", 768, 7, 7],
        "teacher_target_shape": ["B", 768, 7, 7],
        "projector_added": False,
        "classifier_head_retained": False,
        "global_pooling_used": False,
        "epochs": args.epochs,
        "minimum_epochs": args.minimum_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": args.epochs,
        "amp": True,
        "grad_scaler": True,
        "gradient_clip_max_norm": 1.0,
        "batch_size": args.batch_size,
        "accumulation_steps": 1,
        "effective_batch_size": args.batch_size,
        "workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "seed": args.seed,
        "device": args.device,
        "shuffle_train": True,
        "shuffle_monitor": False,
        "cudnn_benchmark": True,
        "tf32": True,
        "loss": "float32 MSE of L2-normalized teacher/student maps at every spatial position",
        "normalization_dimension": 1,
        "normalization_eps": 1e-12,
        "cosine_role": "monitoring only",
        "monitor": "full 4725-row deterministic no-augmentation feature-alignment pass every epoch",
        "preprocessing": transform.preprocessing_config(),
        "augmentation": transform.augmentation_config(),
        "disease_labels_used": False,
        "rad_dino_model_loaded": False,
        "split_created": False,
        "phase2_started": False,
    }


def protected_paths(project_root: Path, manifest: Path, teacher_cache: Path) -> dict[str, Path]:
    experiment = project_root / "outputs" / "raddino_convnext_tiny_experiment_seed42"
    old_cache = project_root / "outputs" / "raddino_feature_cache" / "balanced_945_seed42"
    return {
        "manifest": manifest,
        "patch_teacher_cache": teacher_cache,
        "old_cls_teacher_cache": old_cache / "teacher_features.pt",
        "old_cls_phase1_script": project_root / "src" / "train_convnext_tiny_phase1_distillation.py",
        "old_cls_phase1_best": experiment / "phase1_distillation" / "checkpoints" / "best.pt",
        "old_cls_phase1_backbone": experiment / "phase1_distillation" / "checkpoints" / "distilled_convnext_tiny_backbone.pt",
        "old_cls_phase1_metrics": experiment / "phase1_distillation" / "metrics" / "phase1_metrics.csv",
        "proposed_phase2_best": experiment / "phase2_proposed_distilled" / "checkpoints" / "best.pt",
        "baseline_phase2_best": experiment / "phase2_baseline_imagenet" / "checkpoints" / "best.pt",
        "train_manifest": experiment / "phase2_split" / "train_roi_manifest.csv",
        "val_manifest": experiment / "phase2_split" / "val_roi_manifest.csv",
        "test_manifest": experiment / "phase2_split" / "test_roi_manifest.csv",
        "shared_config": experiment / "shared_phase2_finetune_config.json",
        "final_comparison": experiment / "final_comparison" / "final_comparison_summary.json",
    }


def hash_paths(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected artifact missing: {missing}")
    return {
        name: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def compare_hashes(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records = {
        name: {
            "path": before[name]["path"],
            "before_sha256": before[name]["sha256"],
            "after_sha256": after[name]["sha256"],
            "unchanged": before[name]["sha256"] == after[name]["sha256"],
        }
        for name in before
    }
    return {"all_unchanged": all(item["unchanged"] for item in records.values()), "artifacts": records}


def create_output_tree(output_dir: Path) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "checkpoints": output_dir / "checkpoints",
        "metrics": output_dir / "metrics",
        "figures": output_dir / "figures",
        "diagnostics": output_dir / "diagnostics",
        "per_image": output_dir / "metrics" / "per_image_cosine",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def augmentation_preview(
    rows: list[dict[str, str]],
    transform: StudentTransform,
    path: Path,
    seed: int,
) -> dict[str, Any]:
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    torch.manual_seed(seed)
    indices = random.sample(range(len(rows)), 25)
    clean_images = []
    augmented_images = []
    records = []
    clean_transform = StudentTransform(augment=False)
    for index in indices:
        row = rows[index]
        with Image.open(row["image_path"]) as image:
            image.load()
            rgb = image.convert("RGB")
            _, clean, _ = clean_transform.apply(rgb, return_display=True)
            _, augmented, info = transform.apply(rgb, return_display=True)
        clean_images.append(clean)
        augmented_images.append(augmented)
        records.append({"feature_index": int(row["feature_index"]), **info})
    random.setstate(python_state)
    torch.set_rng_state(torch_state)
    figure, axes = plt.subplots(5, 10, figsize=(15, 8), constrained_layout=True)
    for item, (clean, augmented) in enumerate(zip(clean_images, augmented_images)):
        row_index, pair = divmod(item, 5)
        for column, tensor, title in (
            (pair * 2, clean, "clean"),
            (pair * 2 + 1, augmented, "aug"),
        ):
            axes[row_index, column].imshow(tensor.permute(1, 2, 0).clamp(0, 1).numpy())
            axes[row_index, column].set_title(f"{title} {item + 1}", fontsize=7)
            axes[row_index, column].axis("off")
    atomic_save_figure(figure, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "sample_count": len(records),
        "blur_applied_count": sum(item["blur_applied"] for item in records),
        "noise_applied_count": sum(item["noise_applied"] for item in records),
        "records": records,
    }


def initialization_audit(
    project_root: Path,
    config: dict[str, Any],
    transform: StudentTransform,
    weight_info: dict[str, Any],
    preview: dict[str, Any],
) -> dict[str, Any]:
    old_path = (
        project_root / "outputs" / "raddino_convnext_tiny_experiment_seed42"
        / "phase1_distillation" / "config" / "phase1_config.json"
    )
    old = json.loads(old_path.read_text(encoding="utf-8"))
    comparisons = {
        "optimizer": old["optimizer"] == config["optimizer"] == "AdamW",
        "learning_rate": float(old["learning_rate"]) == config["learning_rate"] == 1e-4,
        "weight_decay": float(old["weight_decay"]) == config["weight_decay"] == 1e-4,
        "scheduler": old["scheduler"] == config["scheduler"] == "CosineAnnealingLR",
        "gradient_clip": float(old["gradient_clip_max_norm"]) == config["gradient_clip_max_norm"] == 1.0,
        "amp": bool(old["amp"]) == config["amp"] is True,
        "seed": int(old["seed"]) == config["seed"] == 42,
        "workers": int(old["workers"]) == config["workers"] == 2,
        "preprocessing": old["preprocessing_config"] == transform.preprocessing_config(),
        "augmentation": old["augmentation_config"] == transform.augmentation_config(),
    }
    if not all(comparisons.values()):
        raise ValueError(f"CLS/Patch shared initialization settings differ unexpectedly: {comparisons}")
    probe = ConvNeXtTinyPatchStudent(pretrained=True)
    total_parameters = sum(parameter.numel() for parameter in probe.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in probe.parameters() if parameter.requires_grad)
    probe_shape = list(probe(torch.zeros(2, 3, 224, 224)).shape)
    del probe
    if probe_shape != [2, 768, 7, 7]:
        raise RuntimeError(f"ConvNeXt patch probe shape mismatch: {probe_shape}")
    return {
        "status": "PASS",
        "old_cls_config_path": str(old_path),
        "shared_setting_comparisons": comparisons,
        "intentional_differences": [
            "teacher target [B,768] -> [B,768,7,7]",
            "student global feature -> final 7x7 feature map",
            "normalized global MSE -> normalized spatial patch MSE",
            "epochs 30 -> maximum 100 with minimum 60",
            "full deterministic monitor each epoch",
            "new output directory",
        ],
        "pretrained_weights": weight_info,
        "pretrained_load_missing_keys": [],
        "pretrained_load_unexpected_keys": [],
        "random_initialization_fallback": False,
        "old_distilled_checkpoint_loaded": False,
        "classification_checkpoint_loaded": False,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "student_probe_shape": probe_shape,
        "classifier_head_retained": False,
        "augmentation_preview": preview,
    }


def pass_batch(
    model: ConvNeXtTinyPatchStudent,
    images: torch.Tensor,
    indices: torch.Tensor,
    teacher_cache: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    teacher = extract_teacher(teacher_cache, indices, device)
    with torch.amp.autocast(device_type=device.type, enabled=amp):
        student = model(images)
    loss, cosine, _, _, counts = patch_alignment(student, teacher)
    return loss, cosine, student.float(), teacher.float(), counts


def smoke_test(
    rows: list[dict[str, str]],
    teacher_cache: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    paths: dict[str, Path],
    manifest_sha: str,
    teacher_sha: str,
    pretrained_sha: str,
) -> dict[str, Any]:
    set_seed(config["seed"])
    train_dataset = RoiDataset(rows, StudentTransform(augment=True))
    monitor_dataset = RoiDataset(rows, StudentTransform(augment=False))
    train_loader = make_loader(train_dataset, config["batch_size"], config["workers"], config["seed"], True)
    monitor_loader = make_loader(monitor_dataset, config["batch_size"], config["workers"], config["seed"], False)
    model = ConvNeXtTinyPatchStudent(pretrained=True).to(device)
    optimizer = make_optimizer(model, config["learning_rate"], config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = make_scaler(device)
    train_records = []
    monitor_records = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        for batch_number, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, cosine, _, _, counts = pass_batch(
                model, images, batch["feature_index"], teacher_cache, device, amp=True
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite, checked_norm, non_finite = gradient_status(model)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not finite or non_finite or not math.isfinite(float(gradient_norm)):
                raise FloatingPointError("Smoke-test gradient failure")
            scaler.step(optimizer)
            scaler.update()
            train_records.append({
                "batch": batch_number,
                "student_shape": list((images.shape[0], *EXPECTED_MAP_SHAPE)),
                "teacher_shape": list((images.shape[0], *EXPECTED_MAP_SHAPE)),
                "loss": float(loss.detach().cpu()),
                "cosine": float(cosine.mean().detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "checked_gradient_norm": checked_norm,
                **counts,
            })
            if batch_number == 3:
                break
        model.eval()
        with torch.inference_mode():
            for batch_number, batch in enumerate(monitor_loader, start=1):
                images = batch["image"].to(device, non_blocking=True)
                loss, cosine, _, _, counts = pass_batch(
                    model, images, batch["feature_index"], teacher_cache, device, amp=True
                )
                monitor_records.append({
                    "batch": batch_number,
                    "loss": float(loss.cpu()),
                    "cosine": float(cosine.mean().cpu()),
                    **counts,
                })
                if batch_number == 3:
                    break
        torch.cuda.synchronize(device)
        smoke = {
            "status": "PASS",
            "created_at_utc": utc_now(),
            "train_batches": train_records,
            "monitor_batches": monitor_records,
            "batch_size": config["batch_size"],
            "workers": config["workers"],
            "optimizer_step_succeeded": True,
            "backward_succeeded": True,
            "student_shape": train_records[-1]["student_shape"],
            "teacher_shape": train_records[-1]["teacher_shape"],
            "loss_finite": True,
            "gradient_finite": True,
            "oom": False,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
            "class_labels_used": False,
            "rad_dino_loaded": False,
        }
        checkpoint = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=0,
            global_step=3,
            best_mse=float("inf"),
            best_cosine=float("-inf"),
            best_epoch=0,
            patience_counter=0,
            train_metrics={"smoke": True},
            monitor_metrics={"smoke": True},
            config=config,
            manifest_sha=manifest_sha,
            teacher_sha=teacher_sha,
            pretrained_sha=pretrained_sha,
            checkpoint_kind="smoke",
        )
        atomic_save_checkpoint(paths["checkpoints"] / "smoke_ready.pt", checkpoint)
        atomic_write_json(paths["diagnostics"] / "smoke_test_audit.json", smoke)
        atomic_write_json(output_dir / "phase1_patch_training_progress.json", {
            "status": "SMOKE_PASS",
            "updated_at_utc": utc_now(),
            "completed_epochs": 0,
            "smoke_test": smoke,
        })
        return smoke
    except torch.OutOfMemoryError as exc:
        failure = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error_reason": str(exc),
            "batch_size": config["batch_size"],
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
            "oom": True,
        }
        atomic_write_json(paths["diagnostics"] / "smoke_test_audit.json", failure)
        raise
    finally:
        del train_loader, monitor_loader, scaler, scheduler, optimizer, model
        torch.cuda.empty_cache()


def checkpoint_payload(
    model: ConvNeXtTinyPatchStudent,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_mse: float,
    best_cosine: float,
    best_epoch: int,
    patience_counter: int,
    train_metrics: dict[str, Any],
    monitor_metrics: dict[str, Any],
    config: dict[str, Any],
    manifest_sha: str,
    teacher_sha: str,
    pretrained_sha: str,
    checkpoint_kind: str,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "architecture": ARCHITECTURE,
        "checkpoint_kind": checkpoint_kind,
        "epoch": epoch,
        "global_step": global_step,
        "student_state_dict": model.features.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "best_monitor_patch_mse": best_mse,
        "best_monitor_patch_cosine": best_cosine,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
        "train_metrics": train_metrics,
        "monitor_metrics": monitor_metrics,
        "seed": config["seed"],
        "batch_size": config["batch_size"],
        "accumulation_steps": config["accumulation_steps"],
        "shared_phase1_config": config,
        "manifest_sha256": manifest_sha,
        "teacher_cache_sha256": teacher_sha,
        "pretrained_imagenet_weight_sha256": pretrained_sha,
        "rng_state": rng_state(),
        "student_output_shape": ["B", 768, 7, 7],
        "teacher_target_shape": ["B", 768, 7, 7],
        "class_labels_used": False,
        "classification_head_included": False,
        "rad_dino_loaded": False,
    }


def run_epoch_pass(
    epoch: int,
    phase: str,
    model: ConvNeXtTinyPatchStudent,
    dataset: Dataset,
    teacher_cache: torch.Tensor,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    progress_path: Path,
    completed_epochs: int,
    global_step: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int]:
    training = phase == "train"
    loader = make_loader(dataset, batch_size, workers, seed + epoch if training else seed, training)
    if training:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_rows = 0
    cosine_sum = 0.0
    cosine_count = 0
    cosine_min = math.inf
    cosine_max = -math.inf
    student_norm_sum = 0.0
    teacher_norm_sum = 0.0
    gradient_norms: list[float] = []
    nan_count = inf_count = non_finite_gradient_count = 0
    batch_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_number, batch in enumerate(loader, start=1):
            batch_started = time.perf_counter()
            images = batch["image"].to(device, non_blocking=True)
            indices = batch["feature_index"]
            if not torch.isfinite(images).all():
                raise FloatingPointError("Student input contains NaN/Inf")
            if training:
                assert optimizer is not None and scaler is not None
                optimizer.zero_grad(set_to_none=True)
            loss, cosine, student_features, teacher_features, counts = pass_batch(
                model, images, indices, teacher_cache, device, amp=True
            )
            nan_count += counts["nan_count"]
            inf_count += counts["inf_count"]
            gradient_norm = 0.0
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                finite, _, non_finite = gradient_status(model)
                non_finite_gradient_count += non_finite
                gradient_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                gradient_norm = float(gradient_tensor.detach().cpu())
                if not finite or non_finite or not math.isfinite(gradient_norm) or gradient_norm <= 0:
                    raise FloatingPointError(
                        f"Non-finite/zero gradient: finite={finite}, count={non_finite}, norm={gradient_norm}"
                    )
                gradient_norms.append(gradient_norm)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
            count = images.shape[0]
            patch_count = cosine.numel()
            total_loss += float(loss.detach().cpu()) * count
            total_rows += count
            cosine_sum += float(cosine.double().sum().detach().cpu())
            cosine_count += patch_count
            cosine_min = min(cosine_min, float(cosine.min().detach().cpu()))
            cosine_max = max(cosine_max, float(cosine.max().detach().cpu()))
            student_feature_norm = torch.linalg.vector_norm(student_features.flatten(1), dim=1)
            teacher_feature_norm = torch.linalg.vector_norm(teacher_features.flatten(1), dim=1)
            student_norm_sum += float(student_feature_norm.double().sum().detach().cpu())
            teacher_norm_sum += float(teacher_feature_norm.double().sum().detach().cpu())
            per_image_cosine = cosine.mean(dim=1).detach().cpu()
            image_rows.extend({
                "epoch": epoch,
                "phase": phase,
                "feature_index": int(feature_index),
                "mean_patch_cosine": float(value),
            } for feature_index, value in zip(indices.tolist(), per_image_cosine.tolist()))
            elapsed = time.perf_counter() - batch_started
            batch_rows.append({
                "epoch": epoch,
                "phase": phase,
                "batch_number": batch_number,
                "batch_rows": count,
                "first_feature_index": int(indices.min()),
                "last_feature_index": int(indices.max()),
                "patch_mse": float(loss.detach().cpu()),
                "patch_cosine_mean": float(cosine.mean().detach().cpu()),
                "patch_cosine_min": float(cosine.min().detach().cpu()),
                "patch_cosine_max": float(cosine.max().detach().cpu()),
                "student_feature_norm_mean": float(student_feature_norm.mean().detach().cpu()),
                "teacher_feature_norm_mean": float(teacher_feature_norm.mean().detach().cpu()),
                "gradient_norm": gradient_norm,
                "learning_rate": optimizer.param_groups[0]["lr"] if optimizer else 0.0,
                "elapsed_seconds": elapsed,
                "images_per_second": count / elapsed,
                "allocated_vram_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_vram_bytes": int(torch.cuda.memory_reserved(device)),
                "nan_count": counts["nan_count"],
                "inf_count": counts["inf_count"],
                "non_finite_gradient_count": 0,
            })
            atomic_write_json(progress_path, {
                "status": "RUNNING",
                "updated_at_utc": utc_now(),
                "completed_epochs": completed_epochs,
                "current_epoch": epoch,
                "current_phase": phase,
                "current_batch": batch_number,
                "total_batches": len(loader),
                "global_step": global_step,
            })
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "patch_mse": total_loss / total_rows,
        "patch_cosine_mean": cosine_sum / cosine_count,
        "patch_cosine_min": cosine_min,
        "patch_cosine_max": cosine_max,
        "student_feature_norm_mean": student_norm_sum / total_rows,
        "teacher_feature_norm_mean": teacher_norm_sum / total_rows,
        "gradient_norm_mean": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
        "gradient_norm_max": float(np.max(gradient_norms)) if gradient_norms else 0.0,
        "elapsed_seconds": elapsed,
        "images_per_second": total_rows / elapsed,
        "rows": total_rows,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "non_finite_gradient_count": non_finite_gradient_count,
    }
    del loader
    return result, batch_rows, image_rows, global_step


def recent_trends(metrics: list[dict[str, Any]], current_mse: float) -> dict[str, float]:
    values = [float(row["monitor_patch_mse"]) for row in metrics] + [current_mse]
    def trend(window: int) -> tuple[float, float]:
        selected = values[-window:]
        change = selected[0] - selected[-1] if len(selected) >= 2 else 0.0
        slope = float(np.polyfit(np.arange(len(selected)), selected, 1)[0]) if len(selected) >= 2 else 0.0
        return change, slope
    change5, slope5 = trend(5)
    change10, slope10 = trend(10)
    return {
        "monitor_recent_5_epoch_change": change5,
        "monitor_recent_10_epoch_change": change10,
        "monitor_recent_5_epoch_slope": slope5,
        "monitor_recent_10_epoch_slope": slope10,
    }


def save_plots(metrics: list[dict[str, Any]], figures: Path) -> None:
    epochs = [int(row["epoch"]) for row in metrics]
    specs = [
        ("patch_train_mse_curve.png", [("train_patch_mse", "Train patch MSE")], "Patch MSE"),
        ("patch_monitor_mse_curve.png", [("monitor_patch_mse", "Monitor patch MSE")], "Patch MSE"),
        ("patch_cosine_curve.png", [("train_patch_cosine_mean", "Train"), ("monitor_patch_cosine_mean", "Monitor")], "Patch cosine"),
        ("recent_loss_improvement_curve.png", [("monitor_recent_5_epoch_change", "5-epoch change"), ("monitor_recent_10_epoch_change", "10-epoch change")], "MSE improvement"),
        ("learning_rate_curve.png", [("learning_rate", "Learning rate")], "Learning rate"),
        ("gradient_norm_curve.png", [("gradient_norm_mean", "Mean"), ("gradient_norm_max", "Max")], "Gradient norm"),
        ("gpu_memory_curve.png", [("gpu_allocated_peak_bytes", "Allocated"), ("gpu_reserved_peak_bytes", "Reserved")], "VRAM bytes"),
    ]
    for filename, series, ylabel in specs:
        figure, axis = plt.subplots(figsize=(8, 5))
        for key, label in series:
            axis.plot(epochs, [float(row[key]) for row in metrics], label=label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if len(series) > 1:
            axis.legend()
        atomic_save_figure(figure, figures / filename)


def validate_resume(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    manifest_sha: str,
    teacher_sha: str,
    pretrained_sha: str,
    output_dir: Path,
) -> None:
    if checkpoint.get("checkpoint_kind") not in {"last", "best"}:
        raise ValueError("Resume is allowed only from a complete best/last epoch checkpoint")
    checks = {
        "phase": checkpoint.get("phase") == PHASE,
        "architecture": checkpoint.get("architecture") == ARCHITECTURE,
        "config": checkpoint.get("shared_phase1_config") == config,
        "manifest": checkpoint.get("manifest_sha256") == manifest_sha,
        "teacher": checkpoint.get("teacher_cache_sha256") == teacher_sha,
        "pretrained": checkpoint.get("pretrained_imagenet_weight_sha256") == pretrained_sha,
        "batch": checkpoint.get("batch_size") == config["batch_size"],
        "accumulation": checkpoint.get("accumulation_steps") == 1,
        "seed": checkpoint.get("seed") == 42,
        "output": Path(checkpoint.get("output_directory", str(output_dir))).resolve() == output_dir.resolve(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Resume checkpoint validation failed: {failed}")


def export_backbone(
    best_path: Path,
    output_path: Path,
    rows: list[dict[str, str]],
    transform: StudentTransform,
    device: torch.device,
    completed_epochs: int,
    stop_reason: str,
) -> dict[str, Any]:
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    state = best["student_state_dict"]
    state_sha = state_dict_sha256(state)
    payload = {
        "phase": PHASE,
        "architecture": ARCHITECTURE,
        "initialization": "ImageNet1K V1",
        "distillation_type": "RAD-DINO 7x7 patch feature",
        "student_state_dict": state,
        "input_preprocessing_metadata": transform.preprocessing_config(),
        "output_feature_shape": ["B", 768, 7, 7],
        "manifest_sha256": best["manifest_sha256"],
        "teacher_cache_sha256": best["teacher_cache_sha256"],
        "best_epoch": best["best_epoch"],
        "best_monitor_patch_mse": best["best_monitor_patch_mse"],
        "best_monitor_patch_cosine": best["best_monitor_patch_cosine"],
        "completed_epochs": completed_epochs,
        "stop_reason": stop_reason,
        "config": best["shared_phase1_config"],
        "export_sha256": state_sha,
        "export_sha256_scope": "serialized student_state_dict",
        "classifier_head_included": False,
    }
    atomic_save_checkpoint(output_path, {**payload, "checkpoint_kind": "export"})
    reloaded = torch.load(output_path, map_location="cpu", weights_only=False)
    model = ConvNeXtTinyPatchStudent(pretrained=False)
    incompatible = model.features.load_state_dict(reloaded["student_state_dict"], strict=True)
    images = []
    for row in rows[:2]:
        with Image.open(row["image_path"]) as image:
            image.load()
            images.append(transform(image.convert("RGB")))
    model.to(device).eval()
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=True):
        output = model(torch.stack(images).to(device))
    validation = {
        "status": "PASS",
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "output_shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all()),
        "state_dict_sha256": state_sha,
        "file_sha256": sha256_file(output_path),
        "file_size_bytes": output_path.stat().st_size,
    }
    if validation["missing_keys"] or validation["unexpected_keys"] or validation["output_shape"] != [2, 768, 7, 7] or not validation["finite"]:
        raise RuntimeError(f"Export reload validation failed: {validation}")
    return validation


def write_summary(
    path: Path,
    metrics: list[dict[str, Any]],
    final: dict[str, Any],
) -> None:
    by_epoch = {int(row["epoch"]): row for row in metrics}
    def metric_line(epoch: int) -> str:
        if epoch not in by_epoch:
            return f"- Epoch {epoch}: not reached"
        row = by_epoch[epoch]
        return (
            f"- Epoch {epoch}: train MSE={float(row['train_patch_mse']):.10f}, "
            f"monitor MSE={float(row['monitor_patch_mse']):.10f}, "
            f"train cosine={float(row['train_patch_cosine_mean']):.8f}, "
            f"monitor cosine={float(row['monitor_patch_cosine_mean']):.8f}"
        )
    lines = [
        "# RAD-DINO Patch Feature Distillation Summary",
        "",
        "## Purpose",
        "This Phase 1 transfers local spatial and texture information from cached RAD-DINO patch maps to ConvNeXt-Tiny.",
        "The prior experiment aligned a global RAD-DINO CLS vector with a globally pooled student vector; this experiment aligns 7x7 spatial maps directly.",
        "",
        "## Method",
        "- Teacher: frozen cached float32 RAD-DINO features shaped [4725,768,7,7].",
        "- Student: ImageNet1K V1 ConvNeXt-Tiny features output, with no classifier, projector, or global pooling.",
        "- Preprocessing: resize 236, center crop 224, bilinear antialiasing, RGB, ImageNet normalization.",
        "- Online augmentation: Gaussian blur p=0.20 and Gaussian noise p=0.30 only.",
        "- Loss: float32 MSE after L2 normalization along channel dimension 1 at each spatial position.",
        "- The 7x7 target preserves the native ConvNeXt final spatial resolution while matching pooled RAD-DINO patches.",
        "- At least 60 epochs are observed before convergence stopping; training is capped at 100 epochs.",
        "- Every epoch includes a complete deterministic no-augmentation monitor over all 4,725 ROI images.",
        "",
        "> This is a deterministic feature-alignment monitor over the complete unlabeled Phase 1 dataset; it does not represent independent-data generalization ability.",
        "",
        "This is transductive feature distillation over all 4,725 ROI images and does not use disease labels.",
        "",
        "## Results",
        f"- Status: {final['status']}",
        f"- Completed epochs: {final['completed_epochs']}",
        f"- Best epoch: {final['best_epoch']}",
        f"- Stop reason: {final['stop_reason']}",
        metric_line(1),
        metric_line(30),
        metric_line(60),
        metric_line(final["completed_epochs"]),
        f"- Best monitor patch MSE: {final['best_monitor_patch_mse']:.10f}",
        f"- Best monitor patch cosine: {final['best_monitor_patch_cosine']:.8f}",
        f"- Recent 5-epoch MSE change/slope: {final['recent_5_change']:.10f} / {final['recent_5_slope']:.10f}",
        f"- Recent 10-epoch MSE change/slope: {final['recent_10_change']:.10f} / {final['recent_10_slope']:.10f}",
        f"- Loss still decreasing after epoch 60: {final['loss_decreasing_after_epoch_60']}",
        f"- Plateau detected: {final['plateau_detected']}",
        f"- Average train+monitor images/s: {final['average_images_per_second']:.3f}",
        f"- Training wall time seconds: {final['training_wall_time_seconds']:.3f}",
        f"- Peak allocated/reserved VRAM bytes: {final['peak_allocated_vram_bytes']} / {final['peak_reserved_vram_bytes']}",
        f"- OOM/NaN/Inf/non-finite gradient: {final['oom_count']}/{final['nan_count']}/{final['inf_count']}/{final['non_finite_gradient_count']}",
        f"- Interrupted/resumed: {final['interrupted']} / {final['resumed']}",
        f"- Exported backbone: {final['export_path']}",
        f"- Exported backbone SHA256: {final['export_file_sha256']}",
        "",
        "The next permitted step is supervised five-class training with the existing Phase 2 split. This script did not start Phase 2.",
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    manifest_path = args.manifest.resolve()
    teacher_path = args.teacher_cache.resolve()
    output_dir = args.output_dir.resolve()
    device = torch.device(args.device)
    if str(device) != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("This run is locked to cuda:0")
    configure_runtime(device, args.seed)
    manifest = validate_manifest(manifest_path)
    teacher = validate_teacher_cache(teacher_path)
    weight_info = pretrained_weight_info()
    train_transform = StudentTransform(augment=True)
    monitor_transform = StudentTransform(augment=False)
    config = training_config(args, train_transform)
    protected = protected_paths(project_root, manifest_path, teacher_path)
    protected_before = hash_paths(protected)
    input_audit = {
        "status": "PASS",
        "manifest": {key: value for key, value in manifest.items() if key != "rows"},
        "teacher_cache": {key: value for key, value in teacher.items() if key != "tensor"},
        "feature_index_lookup_required": True,
        "dataloader_order_assumed": False,
        "disease_labels_used": False,
    }
    if args.dry_run:
        print(json.dumps(input_audit, ensure_ascii=False, indent=2))
        return 0

    if args.smoke_test:
        paths = create_output_tree(output_dir)
        preview = augmentation_preview(
            manifest["rows"], train_transform, paths["diagnostics"] / "augmentation_preview.png", args.seed
        )
        init_audit = initialization_audit(project_root, config, train_transform, weight_info, preview)
        environment = environment_info(device)
        atomic_write_json(paths["diagnostics"] / "phase1_patch_config.json", config)
        atomic_write_json(paths["diagnostics"] / "input_integrity_audit.json", input_audit)
        atomic_write_json(
            paths["diagnostics"] / "teacher_cache_audit.json",
            {key: value for key, value in teacher.items() if key != "tensor"},
        )
        atomic_write_json(paths["diagnostics"] / "initialization_audit.json", init_audit)
        atomic_write_json(paths["diagnostics"] / "resume_audit.json", {"status": "NOT_APPLICABLE_SMOKE"})
        atomic_write_json(paths["diagnostics"] / "protected_artifact_hashes_before.json", protected_before)
        atomic_write_json(output_dir / "environment.json", environment)
        smoke = smoke_test(
            manifest["rows"], teacher["tensor"], config, device, output_dir, paths,
            manifest["sha256"], teacher["sha256"], weight_info["sha256"],
        )
        atomic_write_json(paths["diagnostics"] / "numerical_stability_audit.json", {
            "status": "SMOKE_PASS", "nan_count": 0, "inf_count": 0,
            "non_finite_gradient_count": 0, "oom_count": 0,
        })
        print(json.dumps(smoke, ensure_ascii=False, indent=2))
        return 0

    required_smoke = [
        output_dir / "checkpoints" / "smoke_ready.pt",
        output_dir / "diagnostics" / "smoke_test_audit.json",
        output_dir / "diagnostics" / "phase1_patch_config.json",
    ]
    if any(not path.is_file() for path in required_smoke):
        raise FileNotFoundError("Formal training requires a completed smoke-test output")
    smoke_record = json.loads(required_smoke[1].read_text(encoding="utf-8"))
    if smoke_record.get("status") != "PASS" or smoke_record.get("batch_size") != 64:
        raise ValueError("Smoke test is not a batch-64 PASS")
    stored_config = json.loads(required_smoke[2].read_text(encoding="utf-8"))
    if stored_config != config:
        raise ValueError("Stored smoke config does not match formal config")
    paths = {
        "checkpoints": output_dir / "checkpoints",
        "metrics": output_dir / "metrics",
        "figures": output_dir / "figures",
        "diagnostics": output_dir / "diagnostics",
        "per_image": output_dir / "metrics" / "per_image_cosine",
    }
    environment = json.loads((output_dir / "environment.json").read_text(encoding="utf-8"))
    progress_path = output_dir / "phase1_patch_training_progress.json"
    metrics_path = paths["metrics"] / "phase1_patch_metrics.csv"
    batch_metrics_path = paths["metrics"] / "phase1_patch_batch_metrics.csv"
    epoch_metrics: list[dict[str, Any]] = []
    all_batch_metrics: list[dict[str, Any]] = []
    resumed = False

    if args.resume_checkpoint:
        resume_path = args.resume_checkpoint.resolve()
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume(
            checkpoint, config, manifest["sha256"], teacher["sha256"], weight_info["sha256"], output_dir
        )
        model = ConvNeXtTinyPatchStudent(pretrained=False).to(device)
        model.features.load_state_dict(checkpoint["student_state_dict"], strict=True)
        optimizer = make_optimizer(model, config["learning_rate"], config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
        scaler = make_scaler(device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_mse = float(checkpoint["best_monitor_patch_mse"])
        best_cosine = float(checkpoint["best_monitor_patch_cosine"])
        best_epoch = int(checkpoint["best_epoch"])
        patience_counter = int(checkpoint["patience_counter"])
        epoch_metrics = [row for row in read_csv(metrics_path) if int(row["epoch"]) < start_epoch]
        all_batch_metrics = [row for row in read_csv(batch_metrics_path) if int(row["epoch"]) < start_epoch]
        resumed = True
        resume_audit = {
            "status": "PASS", "resumed": True, "checkpoint": str(resume_path),
            "checkpoint_epoch": start_epoch - 1, "next_epoch": start_epoch,
        }
    else:
        if (paths["checkpoints"] / "best.pt").exists() or (paths["checkpoints"] / "last.pt").exists():
            raise FileExistsError("Formal checkpoint already exists; use --resume-checkpoint explicitly")
        set_seed(args.seed)
        model = ConvNeXtTinyPatchStudent(pretrained=True).to(device)
        optimizer = make_optimizer(model, config["learning_rate"], config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
        scaler = make_scaler(device)
        start_epoch = 1
        global_step = 0
        best_mse = float("inf")
        best_cosine = float("-inf")
        best_epoch = 0
        patience_counter = 0
        resume_audit = {
            "status": "PASS", "resumed": False, "formal_initialization": "fresh ImageNet1K V1",
            "smoke_checkpoint_loaded": False, "epoch_start": 1,
        }
    atomic_write_json(paths["diagnostics"] / "resume_audit.json", resume_audit)
    train_dataset = RoiDataset(manifest["rows"], train_transform)
    monitor_dataset = RoiDataset(manifest["rows"], monitor_transform)
    training_started = time.perf_counter()
    stop_reason = "none"
    current_epoch = start_epoch - 1
    current_phase = "initialization"
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            epoch_started = time.perf_counter()
            current_phase = "train"
            train_result, train_batches, train_images, global_step = run_epoch_pass(
                epoch, "train", model, train_dataset, teacher["tensor"], device,
                args.batch_size, args.workers, args.seed, optimizer, scaler,
                progress_path, epoch - 1, global_step,
            )
            current_phase = "monitor"
            monitor_result, monitor_batches, monitor_images, global_step = run_epoch_pass(
                epoch, "monitor", model, monitor_dataset, teacher["tensor"], device,
                args.batch_size, args.workers, args.seed, None, None,
                progress_path, epoch - 1, global_step,
            )
            previous_mse = float(epoch_metrics[-1]["monitor_patch_mse"]) if epoch_metrics else monitor_result["patch_mse"]
            previous_best = best_mse
            is_best = (
                monitor_result["patch_mse"] < best_mse
                or (
                    monitor_result["patch_mse"] == best_mse
                    and monitor_result["patch_cosine_mean"] > best_cosine
                )
            )
            if is_best:
                best_mse = monitor_result["patch_mse"]
                best_cosine = monitor_result["patch_cosine_mean"]
                best_epoch = epoch
            if epoch <= args.minimum_epochs:
                patience_counter = 0
            elif monitor_result["patch_mse"] < previous_best - args.min_delta:
                patience_counter = 0
            else:
                patience_counter += 1
            trends = recent_trends(epoch_metrics, monitor_result["patch_mse"])
            learning_rate = optimizer.param_groups[0]["lr"]
            scheduler.step()
            epoch_total = time.perf_counter() - epoch_started
            gpu_alloc = int(torch.cuda.max_memory_allocated(device))
            gpu_reserved = int(torch.cuda.max_memory_reserved(device))
            row = {
                "epoch": epoch,
                "train_patch_mse": train_result["patch_mse"],
                "train_patch_cosine_mean": train_result["patch_cosine_mean"],
                "train_patch_cosine_min": train_result["patch_cosine_min"],
                "train_patch_cosine_max": train_result["patch_cosine_max"],
                "monitor_patch_mse": monitor_result["patch_mse"],
                "monitor_patch_cosine_mean": monitor_result["patch_cosine_mean"],
                "monitor_patch_cosine_min": monitor_result["patch_cosine_min"],
                "monitor_patch_cosine_max": monitor_result["patch_cosine_max"],
                "monitor_improvement_from_previous": previous_mse - monitor_result["patch_mse"],
                "monitor_improvement_from_best": (
                    0.0 if math.isinf(previous_best) else previous_best - monitor_result["patch_mse"]
                ),
                **trends,
                "learning_rate": learning_rate,
                "gradient_norm_mean": train_result["gradient_norm_mean"],
                "gradient_norm_max": train_result["gradient_norm_max"],
                "student_feature_norm_mean": monitor_result["student_feature_norm_mean"],
                "teacher_feature_norm_mean": monitor_result["teacher_feature_norm_mean"],
                "epoch_train_seconds": train_result["elapsed_seconds"],
                "epoch_monitor_seconds": monitor_result["elapsed_seconds"],
                "epoch_total_seconds": epoch_total,
                "images_per_second": (EXPECTED_ROWS * 2) / (train_result["elapsed_seconds"] + monitor_result["elapsed_seconds"]),
                "gpu_allocated_peak_bytes": gpu_alloc,
                "gpu_reserved_peak_bytes": gpu_reserved,
                "nan_count": train_result["nan_count"] + monitor_result["nan_count"],
                "inf_count": train_result["inf_count"] + monitor_result["inf_count"],
                "non_finite_gradient_count": train_result["non_finite_gradient_count"],
                "patience_counter": patience_counter,
                "is_best": is_best,
                "stop_reason": "none",
            }
            if any(row[key] for key in ("nan_count", "inf_count", "non_finite_gradient_count")):
                raise FloatingPointError(f"Epoch numerical stability failure: {row}")
            epoch_metrics.append(row)
            all_batch_metrics.extend(train_batches + monitor_batches)
            atomic_write_csv(metrics_path, epoch_metrics, EPOCH_FIELDS)
            atomic_write_csv(batch_metrics_path, all_batch_metrics, BATCH_FIELDS)
            atomic_write_csv(
                paths["per_image"] / f"epoch_{epoch:03d}.csv",
                train_images + monitor_images,
                IMAGE_COSINE_FIELDS,
            )
            payload = checkpoint_payload(
                model, optimizer, scheduler, scaler, epoch, global_step, best_mse, best_cosine,
                best_epoch, patience_counter, train_result, monitor_result, config,
                manifest["sha256"], teacher["sha256"], weight_info["sha256"], "last",
            )
            payload["output_directory"] = str(output_dir)
            atomic_save_checkpoint(paths["checkpoints"] / "last.pt", payload)
            if is_best:
                best_payload = {**payload, "checkpoint_kind": "best"}
                atomic_save_checkpoint(paths["checkpoints"] / "best.pt", best_payload)
            save_plots(epoch_metrics, paths["figures"])
            atomic_write_json(progress_path, {
                "status": "RUNNING",
                "updated_at_utc": utc_now(),
                "completed_epochs": epoch,
                "current_epoch": epoch,
                "current_phase": "epoch_complete",
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_monitor_patch_mse": best_mse,
                "patience_counter": patience_counter,
                "last_epoch_metrics": row,
                "training_wall_time_seconds": time.perf_counter() - training_started,
            })
            print(
                f"epoch={epoch:03d} train_mse={train_result['patch_mse']:.10f} "
                f"monitor_mse={monitor_result['patch_mse']:.10f} "
                f"monitor_cos={monitor_result['patch_cosine_mean']:.8f} "
                f"lr={learning_rate:.8g} patience={patience_counter} best={is_best}",
                flush=True,
            )
            if epoch >= args.minimum_epochs and patience_counter >= args.patience:
                stop_reason = "convergence_patience_after_minimum_epochs"
                break
        if stop_reason == "none":
            recent = epoch_metrics[-1]
            still_improving = (
                float(recent["monitor_recent_10_epoch_change"]) > args.min_delta
                and float(recent["monitor_recent_10_epoch_slope"]) < 0
            )
            stop_reason = (
                "reached_maximum_epoch_while_still_improving"
                if still_improving else "converged_or_plateaued_near_maximum_epoch"
            )
        epoch_metrics[-1]["stop_reason"] = stop_reason
        atomic_write_csv(metrics_path, epoch_metrics, EPOCH_FIELDS)
    except (KeyboardInterrupt, Exception) as exc:
        failure = {
            "status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAIL",
            "created_at_utc": utc_now(),
            "epoch": current_epoch,
            "phase": current_phase,
            "error_type": type(exc).__name__,
            "error_reason": str(exc),
            "traceback": traceback.format_exc(),
            "oom": isinstance(exc, torch.OutOfMemoryError),
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        atomic_write_json(paths["diagnostics"] / "training_failure.json", failure)
        interrupted = checkpoint_payload(
            model, optimizer, scheduler, scaler, current_epoch, global_step, best_mse,
            best_cosine, best_epoch, patience_counter, {"incomplete": True},
            {"incomplete": True}, config, manifest["sha256"], teacher["sha256"],
            weight_info["sha256"], "interrupted",
        )
        interrupted["interruption"] = failure
        atomic_save_checkpoint(paths["checkpoints"] / "interrupted.pt", interrupted)
        atomic_write_json(progress_path, {**failure, "completed_epochs": current_epoch - 1})
        raise

    completed_epochs = int(epoch_metrics[-1]["epoch"])
    if completed_epochs < args.minimum_epochs:
        raise RuntimeError(f"Completed only {completed_epochs} epochs; minimum is {args.minimum_epochs}")
    best_path = paths["checkpoints"] / "best.pt"
    last_path = paths["checkpoints"] / "last.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError("best.pt/last.pt missing after training")
    export_path = paths["checkpoints"] / "patch_distilled_convnext_tiny_backbone.pt"
    export_validation = export_backbone(
        best_path, export_path, manifest["rows"], monitor_transform, device, completed_epochs, stop_reason
    )
    protected_after = hash_paths(protected)
    protected_comparison = compare_hashes(protected_before, protected_after)
    if not protected_comparison["all_unchanged"]:
        raise RuntimeError(f"Protected artifacts changed: {protected_comparison}")
    last = epoch_metrics[-1]
    epoch60 = next(row for row in epoch_metrics if int(row["epoch"]) == 60)
    post60 = [float(row["monitor_patch_mse"]) for row in epoch_metrics if int(row["epoch"]) >= 60]
    loss_decreasing_after60 = len(post60) >= 2 and post60[-1] < post60[0]
    training_wall = time.perf_counter() - training_started
    final = {
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_monitor_patch_mse": best_mse,
        "best_monitor_patch_cosine": best_cosine,
        "stop_reason": stop_reason,
        "recent_5_change": float(last["monitor_recent_5_epoch_change"]),
        "recent_5_slope": float(last["monitor_recent_5_epoch_slope"]),
        "recent_10_change": float(last["monitor_recent_10_epoch_change"]),
        "recent_10_slope": float(last["monitor_recent_10_epoch_slope"]),
        "loss_decreasing_after_epoch_60": loss_decreasing_after60,
        "epoch_60_monitor_mse": float(epoch60["monitor_patch_mse"]),
        "plateau_detected": "plateau" in stop_reason or "patience" in stop_reason,
        "average_images_per_second": float(np.mean([float(row["images_per_second"]) for row in epoch_metrics])),
        "training_wall_time_seconds": training_wall,
        "peak_allocated_vram_bytes": max(int(row["gpu_allocated_peak_bytes"]) for row in epoch_metrics),
        "peak_reserved_vram_bytes": max(int(row["gpu_reserved_peak_bytes"]) for row in epoch_metrics),
        "oom_count": 0,
        "nan_count": sum(int(row["nan_count"]) for row in epoch_metrics),
        "inf_count": sum(int(row["inf_count"]) for row in epoch_metrics),
        "non_finite_gradient_count": sum(int(row["non_finite_gradient_count"]) for row in epoch_metrics),
        "interrupted": False,
        "resumed": resumed,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "export_path": str(export_path),
        "export_file_sha256": export_validation["file_sha256"],
        "export_reload_validation": export_validation,
        "manifest_sha256": manifest["sha256"],
        "teacher_cache_sha256": teacher["sha256"],
        "pretrained_weight_sha256": weight_info["sha256"],
        "class_labels_used": False,
        "test_evaluation_executed": False,
        "phase2_started": False,
        "protected_artifacts": protected_comparison,
    }
    atomic_write_json(paths["diagnostics"] / "numerical_stability_audit.json", {
        "status": "PASS", "nan_count": final["nan_count"], "inf_count": final["inf_count"],
        "non_finite_gradient_count": final["non_finite_gradient_count"], "oom_count": 0,
    })
    atomic_write_json(paths["diagnostics"] / "phase1_patch_final_audit.json", final)
    write_summary(output_dir / "phase1_patch_training_summary.md", epoch_metrics, final)
    atomic_write_json(progress_path, {
        "status": "PASS",
        "updated_at_utc": utc_now(),
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_monitor_patch_mse": best_mse,
        "stop_reason": stop_reason,
        "training_wall_time_seconds": training_wall,
    })
    leftovers = [str(path) for path in output_dir.rglob("*") if path.name.endswith(".tmp")]
    if leftovers:
        raise RuntimeError(f"Temporary files remain: {leftovers}")
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(r"C:\Users\09688\thoracic-cxr-project-3")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--manifest", type=Path,
        default=root / "outputs" / "raddino_feature_cache" / "balanced_945_seed42" / "roi_manifest.csv",
    )
    parser.add_argument(
        "--teacher-cache", type=Path,
        default=root / "outputs" / "raddino_convnext_tiny_patch_experiment_seed42"
        / "phase0_patch_teacher_cache" / "teacher_patch_features_7x7.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "outputs" / "raddino_convnext_tiny_patch_experiment_seed42"
        / "phase1_patch_distillation",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--minimum-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    locked = {
        "epochs": (args.epochs, 100),
        "minimum_epochs": (args.minimum_epochs, 60),
        "patience": (args.patience, 15),
        "min_delta": (args.min_delta, 1e-6),
        "batch_size": (args.batch_size, 64),
        "workers": (args.workers, 2),
        "device": (args.device, "cuda:0"),
        "seed": (args.seed, 42),
    }
    mismatches = {name: values for name, values in locked.items() if values[0] != values[1]}
    if mismatches:
        raise ValueError(f"Formal Patch Phase 1 parameters are locked: {mismatches}")
    if args.smoke_test and args.resume_checkpoint:
        raise ValueError("--smoke-test and --resume-checkpoint are mutually exclusive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
