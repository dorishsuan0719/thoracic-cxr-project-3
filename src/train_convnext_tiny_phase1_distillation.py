#!/usr/bin/env python
"""Phase 1: unlabeled RAD-DINO to ConvNeXt-Tiny feature distillation.

The RAD-DINO teacher is never loaded here. Fixed teacher features are read
from the Phase 0-B cache and matched to ROI images by feature_index.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


EXPECTED_MANIFEST_SHA256 = (
    "796f067d00bb5740a51b51292eed4acfefe9b2e84fd2eeb9b5dfd2df926d5233"
)
EXPECTED_ROWS = 4725
EXPECTED_FEATURE_DIM = 768
EXPECTED_ORIGINAL = 4256
EXPECTED_BRIGHTNESS_AUGMENTED = 469
EXPECTED_PER_CLASS = 945
MODEL_ARCHITECTURE = "convnext_tiny"
PHASE_NAME = "phase1_feature_distillation"

CLASS_MAPPING = {
    0: "Aortic enlargement",
    1: "Cardiomegaly",
    2: "Pleural thickening",
    3: "Pulmonary fibrosis",
    4: "Pleural effusion",
}

METRIC_FIELDS = [
    "epoch",
    "average_distillation_loss",
    "average_cosine_similarity",
    "learning_rate",
    "batch_size",
    "accumulation_steps",
    "effective_batch_size",
    "images_per_second",
    "epoch_seconds",
    "gpu_peak_allocated_gb",
    "gpu_peak_reserved_gb",
    "gradient_norm_mean",
    "gradient_norm_max",
    "convergence_counter",
    "is_best",
    "oom_status",
]


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


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_random_states() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_states(states: dict[str, Any]) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if torch.cuda.is_available() and states.get("torch_cuda"):
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def get_environment(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
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
        "cudnn": torch.backends.cudnn.version(),
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "device": str(device),
        "gpu": gpu,
    }


def validate_inputs(manifest_path: Path, teacher_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file() or not teacher_path.is_file():
        raise FileNotFoundError("Manifest or teacher feature cache does not exist")
    manifest_sha = sha256_file(manifest_path)
    teacher_sha = sha256_file(teacher_path)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Manifest SHA256 mismatch: {manifest_sha}")

    rows = read_csv(manifest_path)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Manifest rows {len(rows)} != {EXPECTED_ROWS}")
    indices = [int(row["feature_index"]) for row in rows]
    expected_indices = list(range(EXPECTED_ROWS))
    if indices != expected_indices:
        raise ValueError("Manifest feature_index is not exactly 0..4724 in row order")
    if len(set(indices)) != EXPECTED_ROWS:
        raise ValueError("Manifest contains duplicate feature_index values")

    missing_paths = [row["image_path"] for row in rows if not Path(row["image_path"]).is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Manifest contains {len(missing_paths)} missing images")
    class_counts = Counter(int(row["class_id"]) for row in rows)
    if class_counts != Counter({class_id: EXPECTED_PER_CLASS for class_id in CLASS_MAPPING}):
        raise ValueError(f"Unexpected per-class counts: {dict(class_counts)}")
    augmented_count = sum(parse_bool(row["is_brightness_augmented"]) for row in rows)
    original_count = len(rows) - augmented_count
    if (original_count, augmented_count) != (EXPECTED_ORIGINAL, EXPECTED_BRIGHTNESS_AUGMENTED):
        raise ValueError(
            f"Original/augmented counts mismatch: {original_count}/{augmented_count}"
        )

    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    features = teacher.get("features")
    feature_indices = teacher.get("feature_indices")
    if not isinstance(features, torch.Tensor) or not isinstance(feature_indices, torch.Tensor):
        raise TypeError("Teacher cache tensors are missing")
    if tuple(features.shape) != (EXPECTED_ROWS, EXPECTED_FEATURE_DIM):
        raise ValueError(f"Teacher feature shape mismatch: {tuple(features.shape)}")
    if features.dtype != torch.float32 or features.device.type != "cpu":
        raise ValueError("Teacher features must be a CPU float32 tensor")
    expected_tensor_indices = torch.arange(EXPECTED_ROWS, dtype=torch.int64)
    if not torch.equal(feature_indices.cpu(), expected_tensor_indices):
        raise ValueError("Teacher feature_indices are not exactly 0..4724")
    if teacher.get("manifest_sha256") != manifest_sha:
        raise ValueError("Teacher cache manifest SHA256 does not match the manifest")
    if teacher.get("model_name") != "microsoft/rad-dino":
        raise ValueError("Unexpected teacher model name")
    nan_count = int(torch.isnan(features).sum().item())
    inf_count = int(torch.isinf(features).sum().item())
    zero_norm_count = int((torch.linalg.vector_norm(features, dim=1) == 0).sum().item())
    if nan_count or inf_count or zero_norm_count:
        raise ValueError(
            f"Invalid teacher features: NaN={nan_count}, Inf={inf_count}, zero={zero_norm_count}"
        )

    return {
        "rows": rows,
        "teacher_features": features.contiguous(),
        "manifest_sha256": manifest_sha,
        "teacher_cache_sha256": teacher_sha,
        "class_counts": dict(sorted(class_counts.items())),
        "original_count": original_count,
        "brightness_augmented_count": augmented_count,
        "teacher_model_metadata": {
            "model_name": teacher.get("model_name"),
            "feature_type": teacher.get("feature_type"),
            "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype),
            "processor_config": teacher.get("processor_config"),
            "environment": teacher.get("environment"),
        },
    }


class StudentTransform:
    def __init__(self) -> None:
        weights_transform = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
        self.resize_size = list(weights_transform.resize_size)
        self.crop_size = list(weights_transform.crop_size)
        self.interpolation = weights_transform.interpolation
        self.antialias = weights_transform.antialias
        self.mean = list(weights_transform.mean)
        self.std = list(weights_transform.std)
        self.blur_probability = 0.20
        self.blur_kernel_size = 3
        self.blur_sigma = (0.1, 0.5)
        self.noise_probability = 0.30
        self.noise_std = (0.005, 0.015)

    def config(self) -> dict[str, Any]:
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

    def apply(self, image: Image.Image, audit: bool = False) -> Any:
        if image.mode != "RGB":
            image = image.convert("RGB")
        blur_applied = random.random() < self.blur_probability
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
        tensor = TF.pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)
        pre_noise = tensor.clone() if audit else None
        noise_applied = random.random() < self.noise_probability
        noise_std = None
        if noise_applied:
            noise_std = random.uniform(*self.noise_std)
            tensor = torch.clamp(tensor + torch.randn_like(tensor) * noise_std, 0.0, 1.0)
        display_tensor = tensor.clone() if audit else None
        normalized = TF.normalize(tensor, mean=self.mean, std=self.std)
        if not audit:
            return normalized
        return normalized, {
            "blur_applied": blur_applied,
            "blur_sigma": blur_sigma,
            "noise_applied": noise_applied,
            "noise_std": noise_std,
            "pre_noise_tensor": pre_noise,
            "display_tensor": display_tensor,
        }

    def clean_display_tensor(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = TF.resize(
            image,
            self.resize_size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        image = TF.center_crop(image, self.crop_size)
        return TF.pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.apply(image, audit=False)


class RoiDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform: StudentTransform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = Path(row["image_path"])
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != (224, 224):
                    raise ValueError(f"Expected 224x224, got {image.size}")
                if image.mode != "L":
                    raise ValueError(f"Expected source mode L, got {image.mode}")
                rgb = image.convert("RGB")
                tensor = self.transform(rgb)
        except Exception as exc:
            raise RuntimeError(f"Failed to read ROI {path}: {exc}") from exc
        return {
            "image": tensor,
            "feature_index": int(row["feature_index"]),
            "source_image_id": row["source_image_id"],
            "class_id": int(row["class_id"]),
            "image_path": str(path),
            "is_brightness_augmented": parse_bool(row["is_brightness_augmented"]),
        }


class ConvNeXtTinyStudent(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_tiny(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        self.final_norm = base.classifier[0]
        self.flatten = base.classifier[1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.avgpool(x)
        x = self.final_norm(x)
        x = self.flatten(x)
        return x


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


def make_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def make_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device.type, enabled=enabled)


def extract_teacher_batch(
    teacher_features: torch.Tensor,
    feature_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    selected = teacher_features.index_select(0, feature_indices.cpu())
    if device.type == "cuda":
        selected = selected.pin_memory()
    return selected.to(device, non_blocking=True)


def check_gradients(model: nn.Module) -> tuple[bool, float]:
    squared_norm = 0.0
    found = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        found = True
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            return False, float("nan")
        squared_norm += float(torch.sum(gradient.float() ** 2).item())
    norm = math.sqrt(squared_norm)
    return found and math.isfinite(norm) and norm > 0.0, norm


def one_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    images: torch.Tensor,
    teachers: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
        student = model(images)
    if tuple(student.shape) != (images.shape[0], EXPECTED_FEATURE_DIM):
        raise RuntimeError(f"Student feature shape mismatch: {tuple(student.shape)}")
    if not torch.isfinite(student).all() or not torch.isfinite(teachers).all():
        raise FloatingPointError("Non-finite raw student or teacher feature")
    student_norm = F.normalize(student.float(), p=2, dim=1)
    teacher_norm = F.normalize(teachers.float(), p=2, dim=1)
    if not torch.isfinite(student_norm).all() or not torch.isfinite(teacher_norm).all():
        raise FloatingPointError("Non-finite normalized feature")
    loss = F.mse_loss(student_norm, teacher_norm)
    cosine = F.cosine_similarity(student_norm, teacher_norm, dim=1).mean()
    if not torch.isfinite(loss) or not torch.isfinite(cosine):
        raise FloatingPointError("Non-finite loss or cosine similarity")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    finite_gradients, checked_norm = check_gradients(model)
    if not finite_gradients or not math.isfinite(float(gradient_norm)) or float(gradient_norm) <= 0:
        raise FloatingPointError("Gradient is non-finite or zero")
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": float(loss.detach().cpu()),
        "cosine": float(cosine.detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "checked_gradient_norm": checked_norm,
        "student_shape": list(student.shape),
        "teacher_shape": list(teachers.shape),
    }


def auto_probe_batch_size(
    dataset: Dataset,
    teacher_features: torch.Tensor,
    device: torch.device,
    workers: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    requested: str,
) -> tuple[int, list[dict[str, Any]]]:
    candidates = [64, 32, 16, 8, 4] if requested == "auto" else [int(requested)]
    attempts: list[dict[str, Any]] = []
    for batch_size in candidates:
        model = optimizer = scaler = loader = batch = None
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            set_seed(seed)
            model = ConvNeXtTinyStudent(pretrained=True).to(device)
            model.train()
            optimizer = make_optimizer(model, learning_rate, weight_decay)
            scaler = make_scaler(device, enabled=device.type == "cuda")
            loader = make_loader(dataset, batch_size, workers=0, seed=seed, shuffle=True)
            batch = next(iter(loader))
            images = batch["image"].to(device, non_blocking=True)
            teachers = extract_teacher_batch(teacher_features, batch["feature_index"], device)
            result = one_training_step(
                model, optimizer, scaler, images, teachers, device, device.type == "cuda"
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            attempts.append(
                {
                    "batch_size": batch_size,
                    "status": "PASS",
                    "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
                    "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
                    **result,
                }
            )
            return batch_size, attempts
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
            attempts.append(
                {
                    "batch_size": batch_size,
                    "status": "OOM" if is_oom else "FAIL",
                    "error_type": type(exc).__name__,
                    "error_reason": str(exc),
                }
            )
            if not is_oom:
                raise
        finally:
            del batch, loader, scaler, optimizer, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError(f"No batch size passed the complete training-step probe: {attempts}")


def stage0_smoke_test(
    dataset: Dataset,
    teacher_features: torch.Tensor,
    device: torch.device,
    batch_size: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    set_seed(seed)
    model = ConvNeXtTinyStudent(pretrained=True).to(device)
    optimizer = make_optimizer(model, learning_rate, weight_decay)
    scaler = make_scaler(device, enabled=device.type == "cuda")
    loader = make_loader(dataset, batch_size, workers=0, seed=seed, shuffle=True)
    model.train()
    results = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for batch_number, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            teachers = extract_teacher_batch(teacher_features, batch["feature_index"], device)
            result = one_training_step(
                model, optimizer, scaler, images, teachers, device, device.type == "cuda"
            )
            result["batch_number"] = batch_number
            result["feature_indices"] = batch["feature_index"].tolist()
            results.append(result)
            if batch_number >= 3:
                break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            "status": "PASS",
            "tested_batches": len(results),
            "batch_size": batch_size,
            "student_feature_shape": results[-1]["student_shape"],
            "teacher_feature_shape": results[-1]["teacher_shape"],
            "losses": [item["loss"] for item in results],
            "cosine_similarities": [item["cosine"] for item in results],
            "gradient_norms": [item["gradient_norm"] for item in results],
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
            "rad_dino_loaded": False,
            "split_created": False,
            "class_id_used_in_loss": False,
            "brightness_transform_used": False,
            "contrast_transform_used": False,
        }
    finally:
        del loader, scaler, optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def create_augmentation_preview(
    rows: list[dict[str, str]],
    transform: StudentTransform,
    preview_path: Path,
    audit_path: Path,
    seed: int,
) -> dict[str, Any]:
    saved_python_state = random.getstate()
    saved_torch_state = torch.get_rng_state()
    random.seed(seed)
    torch.manual_seed(seed)
    sample_indices = random.sample(range(len(rows)), 25)
    originals: list[torch.Tensor] = []
    augmented: list[torch.Tensor] = []
    records = []
    for index in sample_indices:
        row = rows[index]
        with Image.open(row["image_path"]) as image:
            image.load()
            if image.size != (224, 224) or image.mode != "L":
                raise ValueError(f"Preview source is not 224x224 L: {row['image_path']}")
            rgb = image.convert("RGB")
            clean = transform.clean_display_tensor(rgb)
            normalized, info = transform.apply(rgb, audit=True)
        del normalized
        originals.append(clean)
        augmented.append(info["display_tensor"])
        records.append(
            {
                "feature_index": int(row["feature_index"]),
                "image_path": row["image_path"],
                "blur_applied": info["blur_applied"],
                "blur_sigma": info["blur_sigma"],
                "noise_applied": info["noise_applied"],
                "noise_std": info["noise_std"],
            }
        )
    random.setstate(saved_python_state)
    torch.set_rng_state(saved_torch_state)

    original_stack = torch.stack(originals)
    augmented_stack = torch.stack(augmented)
    finite = bool(torch.isfinite(original_stack).all() and torch.isfinite(augmented_stack).all())
    channels_ok = original_stack.shape[1] == augmented_stack.shape[1] == 3
    size_ok = tuple(original_stack.shape[-2:]) == tuple(augmented_stack.shape[-2:]) == (224, 224)
    if not finite or not channels_ok or not size_ok:
        raise ValueError("Augmentation preview failed finite/channel/size validation")

    figure, axes = plt.subplots(5, 10, figsize=(15, 8), constrained_layout=True)
    for item, (clean, aug) in enumerate(zip(originals, augmented)):
        row = item // 5
        pair = item % 5
        clean_ax = axes[row, pair * 2]
        aug_ax = axes[row, pair * 2 + 1]
        clean_ax.imshow(clean.permute(1, 2, 0).clamp(0, 1).numpy(), cmap="gray")
        aug_ax.imshow(aug.permute(1, 2, 0).clamp(0, 1).numpy(), cmap="gray")
        clean_ax.set_title(f"Original {item + 1}", fontsize=8)
        aug_ax.set_title(f"Student {item + 1}", fontsize=8)
        clean_ax.axis("off")
        aug_ax.axis("off")
    figure.savefig(preview_path, dpi=160)
    plt.close(figure)

    audit = {
        "status": "PASS",
        "seed": seed,
        "sample_count": 25,
        "gaussian_blur_applied_count": sum(item["blur_applied"] for item in records),
        "gaussian_noise_applied_count": sum(item["noise_applied"] for item in records),
        "brightness_transform_used": False,
        "contrast_transform_used": False,
        "original_tensor": {
            "min": float(original_stack.min()),
            "max": float(original_stack.max()),
            "mean": float(original_stack.mean()),
        },
        "augmented_tensor": {
            "min": float(augmented_stack.min()),
            "max": float(augmented_stack.max()),
            "mean": float(augmented_stack.mean()),
        },
        "nan_count": int(torch.isnan(augmented_stack).sum()),
        "inf_count": int(torch.isinf(augmented_stack).sum()),
        "three_channels": channels_ok,
        "input_size": list(augmented_stack.shape[1:]),
        "records": records,
    }
    atomic_write_json(audit_path, audit)
    return audit


def create_directories(output_dir: Path) -> dict[str, Path]:
    paths = {
        name: output_dir / name
        for name in ("checkpoints", "metrics", "figures", "logs", "config", "diagnostics")
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for path in paths.values():
        path.mkdir()
    return paths


def log_message(log_path: Path | None, message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def checkpoint_payload(
    model: ConvNeXtTinyStudent,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_loss: float,
    best_cosine: float,
    best_epoch: int,
    convergence_counter: int,
    batch_size: int,
    accumulation_steps: int,
    validation: dict[str, Any],
    transform: StudentTransform,
    environment: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": PHASE_NAME,
        "architecture": MODEL_ARCHITECTURE,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "backbone_state_dict": model.features.state_dict(),
        "final_norm_state_dict": model.final_norm.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "current_epoch": epoch,
        "best_distillation_loss": best_loss,
        "best_cosine_similarity": best_cosine,
        "best_epoch": best_epoch,
        "convergence_counter": convergence_counter,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "class_mapping_audit_only": CLASS_MAPPING,
        "manifest_sha256": validation["manifest_sha256"],
        "teacher_cache_sha256": validation["teacher_cache_sha256"],
        "teacher_model_metadata": validation["teacher_model_metadata"],
        "augmentation_config": transform.augmentation_config(),
        "preprocessing_config": transform.config(),
        "random_states": get_random_states(),
        "environment": environment,
        "training_config": training_config,
        "classifier_head_included": False,
        "rad_dino_loaded": False,
        "split_created": False,
        "class_label_used_in_loss": False,
    }


def atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    for key in ("phase", "architecture", "feature_dim", "manifest_sha256", "teacher_cache_sha256"):
        if loaded.get(key) != payload.get(key):
            raise RuntimeError(f"Checkpoint reload validation failed for {key}")
    if loaded.get("current_epoch") != payload.get("current_epoch"):
        raise RuntimeError("Checkpoint epoch changed during reload validation")
    os.replace(temporary, path)


def validate_resume(
    checkpoint: dict[str, Any],
    validation: dict[str, Any],
    transform: StudentTransform,
) -> None:
    checks = {
        "phase": checkpoint.get("phase") == PHASE_NAME,
        "architecture": checkpoint.get("architecture") == MODEL_ARCHITECTURE,
        "feature_dim": checkpoint.get("feature_dim") == EXPECTED_FEATURE_DIM,
        "manifest_sha256": checkpoint.get("manifest_sha256") == validation["manifest_sha256"],
        "teacher_cache_sha256": checkpoint.get("teacher_cache_sha256")
        == validation["teacher_cache_sha256"],
        "augmentation_config": checkpoint.get("augmentation_config")
        == transform.augmentation_config(),
        "preprocessing_config": checkpoint.get("preprocessing_config") == transform.config(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Resume checkpoint validation failed: {failed}")


def save_plots(metrics: list[dict[str, Any]], figures_dir: Path) -> None:
    if not metrics:
        return
    epochs = [int(row["epoch"]) for row in metrics]
    plots = [
        ("average_distillation_loss", "Distillation MSE Loss", "distillation_loss_curve.png"),
        ("average_cosine_similarity", "Cosine Similarity", "cosine_similarity_curve.png"),
        ("learning_rate", "Learning Rate", "learning_rate_curve.png"),
        ("gpu_peak_allocated_gb", "GPU Memory (GB)", "gpu_memory_curve.png"),
    ]
    for field, ylabel, filename in plots:
        values = [float(row[field]) for row in metrics]
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(epochs, values, marker="o", linewidth=1.5, markersize=3)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(figures_dir / filename, dpi=160)
        plt.close(figure)


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return read_csv(path)


def export_distilled_backbone(best_path: Path, output_path: Path) -> None:
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    export = {
        "backbone_state_dict": best["backbone_state_dict"],
        "final_norm_state_dict": best["final_norm_state_dict"],
        "architecture": MODEL_ARCHITECTURE,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "source_checkpoint": str(best_path),
        "best_distillation_loss": best["best_distillation_loss"],
        "best_cosine_similarity": best["best_cosine_similarity"],
        "best_epoch": best["best_epoch"],
        "manifest_sha256": best["manifest_sha256"],
        "teacher_cache_sha256": best["teacher_cache_sha256"],
        "classifier_head_included": False,
    }
    atomic_save_generic(output_path, export)


def atomic_save_generic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    if loaded.get("architecture") != MODEL_ARCHITECTURE or loaded.get("feature_dim") != 768:
        raise RuntimeError("Export reload validation failed")
    os.replace(temporary, path)


def train_one_epoch(
    epoch: int,
    model: ConvNeXtTinyStudent,
    dataset: Dataset,
    teacher_features: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    batch_size: int,
    accumulation_steps: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    loader = make_loader(dataset, batch_size, workers, seed + epoch, shuffle=True)
    num_batches = len(loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_cosine = 0.0
    total_images = 0
    gradient_norms: list[float] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch_number, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        indices = batch["feature_index"]
        teachers = extract_teacher_batch(teacher_features, indices, device)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            students = model(images)
        if tuple(students.shape) != (images.shape[0], EXPECTED_FEATURE_DIM):
            raise RuntimeError(f"Student feature shape mismatch: {tuple(students.shape)}")
        if not torch.isfinite(students).all() or not torch.isfinite(teachers).all():
            raise FloatingPointError("Non-finite raw feature")
        student_norm = F.normalize(students.float(), p=2, dim=1)
        teacher_norm = F.normalize(teachers.float(), p=2, dim=1)
        if not torch.isfinite(student_norm).all() or not torch.isfinite(teacher_norm).all():
            raise FloatingPointError("Non-finite normalized feature")
        loss = F.mse_loss(student_norm, teacher_norm)
        cosine = F.cosine_similarity(student_norm, teacher_norm, dim=1).mean()
        if not torch.isfinite(loss) or not torch.isfinite(cosine):
            raise FloatingPointError("Non-finite loss or cosine similarity")

        window_start = ((batch_number - 1) // accumulation_steps) * accumulation_steps
        window_size = min(accumulation_steps, num_batches - window_start)
        scaler.scale(loss / window_size).backward()
        finite_gradient, _ = check_gradients(model)
        if not finite_gradient:
            raise FloatingPointError("Non-finite or zero gradient after batch backward")

        should_step = batch_number % accumulation_steps == 0 or batch_number == num_batches
        if should_step:
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not math.isfinite(float(gradient_norm)) or float(gradient_norm) <= 0:
                raise FloatingPointError("Non-finite or zero optimizer-step gradient")
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        count = images.shape[0]
        total_loss += float(loss.detach().cpu()) * count
        total_cosine += float(cosine.detach().cpu()) * count
        total_images += count

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "average_distillation_loss": total_loss / total_images,
        "average_cosine_similarity": total_cosine / total_images,
        "images_per_second": total_images / elapsed,
        "epoch_seconds": elapsed,
        "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
        "gradient_norm_mean": float(np.mean(gradient_norms)),
        "gradient_norm_max": float(np.max(gradient_norms)),
    }


def write_summary(
    path: Path,
    metrics: list[dict[str, Any]],
    completed_epochs: int,
    stopped: bool,
    best_epoch: int,
    best_loss: float,
    best_cosine: float,
    batch_size: int,
    accumulation_steps: int,
    validation: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    last = metrics[-1]
    average_speed = float(np.mean([float(row["images_per_second"]) for row in metrics]))
    lines = [
        "# Phase 1 Training Summary",
        "",
        "- Status: PASS",
        f"- Completed epochs: {completed_epochs}",
        f"- Convergence stopping triggered: {stopped}",
        f"- Best epoch: {best_epoch}",
        f"- Best average distillation loss: {best_loss:.10f}",
        f"- Best average cosine similarity: {best_cosine:.10f}",
        f"- Last epoch loss: {float(last['average_distillation_loss']):.10f}",
        f"- Batch size: {batch_size}",
        f"- Accumulation steps: {accumulation_steps}",
        f"- Effective batch size: {batch_size * accumulation_steps}",
        f"- Average images/s: {average_speed:.3f}",
        f"- Peak allocated VRAM GB: {max(float(row['gpu_peak_allocated_gb']) for row in metrics):.3f}",
        f"- Peak reserved VRAM GB: {max(float(row['gpu_peak_reserved_gb']) for row in metrics):.3f}",
        f"- Manifest SHA256: {validation['manifest_sha256']}",
        f"- Teacher cache SHA256: {validation['teacher_cache_sha256']}",
        f"- GPU: {environment['gpu']['name'] if environment['gpu'] else 'none'}",
        "- RAD-DINO loaded: False",
        "- Class labels used in loss: False",
        "- Train/validation/test split created: False",
        "- Phase 2 or baseline started: False",
        "- NaN/Inf/non-finite gradient: False",
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")


def build_training_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "phase": PHASE_NAME,
        "architecture": MODEL_ARCHITECTURE,
        "initialization": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "feature_dim": EXPECTED_FEATURE_DIM,
        "epochs": args.epochs,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "gradient_clip_max_norm": 1.0,
        "amp": True,
        "convergence_patience": args.patience,
        "convergence_min_delta": args.min_delta,
        "seed": args.seed,
        "workers": args.workers,
        "requested_batch_size": args.batch_size,
        "target_effective_batch_size": 64,
        "shuffle": True,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "cache_images_in_ram": False,
        "cudnn_benchmark": True,
        "tf32": True,
        "loss": "MSE(L2-normalized student, L2-normalized teacher)",
        "cosine_similarity_role": "monitoring only",
        "class_labels_used_in_loss": False,
        "split_created": False,
    }


def configure_runtime(device: torch.device, seed: int) -> None:
    set_seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def resolve_device(value: str) -> torch.device:
    device = torch.device("cuda:0" if value == "auto" and torch.cuda.is_available() else value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 1 requires the requested CUDA GPU")
    return device


def prepare_smoke_artifacts(
    output_dir: Path,
    paths: dict[str, Path],
    validation: dict[str, Any],
    transform: StudentTransform,
    environment: dict[str, Any],
    training_config: dict[str, Any],
    batch_size: int,
    accumulation_steps: int,
    probe_attempts: list[dict[str, Any]],
    smoke: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    config = {
        **training_config,
        "selected_batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "manifest_path": str(args.manifest),
        "teacher_cache_path": str(args.teacher_cache),
        "manifest_sha256": validation["manifest_sha256"],
        "teacher_cache_sha256": validation["teacher_cache_sha256"],
        "dataset_rows": EXPECTED_ROWS,
        "original_roi_count": validation["original_count"],
        "brightness_augmented_roi_count": validation["brightness_augmented_count"],
        "class_counts_audit_only": validation["class_counts"],
        "preprocessing_config": transform.config(),
        "augmentation_config": transform.augmentation_config(),
        "student_feature_pipeline": [
            "ConvNeXt-Tiny features",
            "AdaptiveAvgPool2d(1)",
            "official final LayerNorm2d",
            "Flatten",
        ],
        "linear_projector_added": False,
        "rad_dino_loaded": False,
    }
    atomic_write_json(paths["config"] / "phase1_config.json", config)
    atomic_write_json(paths["config"] / "environment.json", environment)
    pip_result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    atomic_write_text(paths["config"] / "pip_freeze.txt", pip_result.stdout)
    preview_audit = create_augmentation_preview(
        validation["rows"],
        transform,
        paths["figures"] / "augmentation_preview.png",
        paths["diagnostics"] / "augmentation_audit.json",
        args.seed,
    )
    smoke_record = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "manifest_validation": "PASS",
        "teacher_cache_validation": "PASS",
        "batch_probe_attempts": probe_attempts,
        "selected_batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "stage0_smoke_test": smoke,
        "augmentation_audit": preview_audit,
        "rad_dino_loaded": False,
        "split_created": False,
        "class_label_used_in_loss": False,
    }
    atomic_write_json(paths["diagnostics"] / "stage0_smoke_test.json", smoke_record)

    model = ConvNeXtTinyStudent(pretrained=True).to(device)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = make_scaler(device, enabled=True)
    payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=0,
        best_loss=float("inf"),
        best_cosine=float("-inf"),
        best_epoch=0,
        convergence_counter=0,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        validation=validation,
        transform=transform,
        environment=environment,
        training_config=training_config,
    )
    smoke_ready = paths["checkpoints"] / "smoke_ready.pt"
    atomic_save_checkpoint(smoke_ready, payload)
    del scaler, scheduler, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log_message(paths["logs"] / "phase1.log", f"Stage 0 smoke test PASS; output={output_dir}")
    return smoke_ready


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    teacher_path = Path(args.teacher_cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    args.manifest = str(manifest_path)
    args.teacher_cache = str(teacher_path)
    device = resolve_device(args.device)
    configure_runtime(device, args.seed)
    validation = validate_inputs(manifest_path, teacher_path)
    transform = StudentTransform()
    dataset = RoiDataset(validation["rows"], transform)
    environment = get_environment(device)
    training_config = build_training_config(args)
    input_report = {
        "status": "PASS",
        "manifest_sha256": validation["manifest_sha256"],
        "teacher_cache_sha256": validation["teacher_cache_sha256"],
        "dataset_rows": len(dataset),
        "original_roi_count": validation["original_count"],
        "brightness_augmented_roi_count": validation["brightness_augmented_count"],
        "class_counts": validation["class_counts"],
        "teacher_feature_shape": list(validation["teacher_features"].shape),
        "teacher_feature_device": str(validation["teacher_features"].device),
        "teacher_feature_dtype": str(validation["teacher_features"].dtype),
        "environment": environment,
    }
    if args.dry_run:
        print(json.dumps(input_report, ensure_ascii=False, indent=2))
        return 0

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        if not output_dir.is_dir() or not any(output_dir.iterdir()):
            raise FileNotFoundError("Resume requires an existing non-empty output directory")
        paths = {name: output_dir / name for name in ("checkpoints", "metrics", "figures", "logs", "config", "diagnostics")}
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume(resume, validation, transform)
        if resume.get("training_config") != training_config:
            raise ValueError("Resume training configuration does not exactly match")
        batch_size = int(resume["batch_size"])
        accumulation_steps = int(resume["accumulation_steps"])
        smoke_record = json.loads(
            (paths["diagnostics"] / "stage0_smoke_test.json").read_text(encoding="utf-8")
        )
        if smoke_record.get("status") != "PASS":
            raise ValueError("Stage 0 smoke test record is not PASS")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            existing = [str(path) for path in sorted(output_dir.rglob("*"))]
            raise FileExistsError(
                "Phase 1 output is non-empty and --resume-checkpoint was not supplied:\n"
                + "\n".join(existing)
            )
        batch_size, probe_attempts = auto_probe_batch_size(
            dataset,
            validation["teacher_features"],
            device,
            args.workers,
            args.seed,
            args.learning_rate,
            args.weight_decay,
            args.batch_size,
        )
        accumulation_steps = max(1, 64 // batch_size)
        smoke = stage0_smoke_test(
            dataset,
            validation["teacher_features"],
            device,
            batch_size,
            args.seed,
            args.learning_rate,
            args.weight_decay,
        )
        if args.smoke_test_only:
            paths = create_directories(output_dir)
            smoke_ready = prepare_smoke_artifacts(
                output_dir,
                paths,
                validation,
                transform,
                environment,
                training_config,
                batch_size,
                accumulation_steps,
                probe_attempts,
                smoke,
                args,
                device,
            )
            print(
                json.dumps(
                    {
                        **input_report,
                        "stage0_smoke_test": smoke,
                        "batch_probe_attempts": probe_attempts,
                        "selected_batch_size": batch_size,
                        "accumulation_steps": accumulation_steps,
                        "effective_batch_size": batch_size * accumulation_steps,
                        "smoke_ready_checkpoint": str(smoke_ready),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        raise RuntimeError(
            "Run --smoke-test-only first, inspect augmentation_preview.png, then resume "
            "from checkpoints/smoke_ready.pt"
        )

    log_path = paths["logs"] / "phase1.log"
    log_message(log_path, f"Starting/resuming Phase 1 from {args.resume_checkpoint}")
    model = ConvNeXtTinyStudent(pretrained=False).to(device)
    model.features.load_state_dict(resume["backbone_state_dict"])
    model.final_norm.load_state_dict(resume["final_norm_state_dict"])
    probe = model(torch.zeros(2, 3, 224, 224, device=device))
    if tuple(probe.shape) != (2, EXPECTED_FEATURE_DIM):
        raise RuntimeError(f"Native ConvNeXt feature shape is {tuple(probe.shape)}")
    del probe
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = make_scaler(device, enabled=True)
    optimizer.load_state_dict(resume["optimizer_state_dict"])
    scheduler.load_state_dict(resume["scheduler_state_dict"])
    scaler.load_state_dict(resume["grad_scaler_state_dict"])
    restore_random_states(resume["random_states"])

    start_epoch = int(resume["current_epoch"]) + 1
    best_loss = float(resume["best_distillation_loss"])
    best_cosine = float(resume["best_cosine_similarity"])
    best_epoch = int(resume["best_epoch"])
    convergence_counter = int(resume["convergence_counter"])
    convergence_reference = best_loss
    metrics_path = paths["metrics"] / "phase1_metrics.csv"
    metrics = read_metrics(metrics_path)
    committed_epoch = start_epoch - 1
    committed_metrics = [row for row in metrics if int(row["epoch"]) <= committed_epoch]
    if len(committed_metrics) != len(metrics):
        log_message(
            log_path,
            f"Discarding {len(metrics) - len(committed_metrics)} metric row(s) newer than "
            f"resume checkpoint epoch {committed_epoch}",
        )
        metrics = committed_metrics
        atomic_write_csv(metrics_path, metrics)
    convergence_stopped = False
    current_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            epoch_result = train_one_epoch(
                epoch,
                model,
                dataset,
                validation["teacher_features"],
                optimizer,
                scaler,
                device,
                batch_size,
                accumulation_steps,
                args.workers,
                args.seed,
            )
            epoch_loss = epoch_result["average_distillation_loss"]
            epoch_cosine = epoch_result["average_cosine_similarity"]
            learning_rate = optimizer.param_groups[0]["lr"]

            is_best = (
                epoch_loss < best_loss
                or (epoch_loss == best_loss and epoch_cosine > best_cosine)
                or (epoch_loss == best_loss and epoch_cosine == best_cosine and epoch < best_epoch)
            )
            if is_best:
                best_loss = epoch_loss
                best_cosine = epoch_cosine
                best_epoch = epoch
            if epoch_loss < convergence_reference - args.min_delta:
                convergence_reference = epoch_loss
                convergence_counter = 0
            else:
                convergence_counter += 1

            scheduler.step()
            row = {
                "epoch": epoch,
                "average_distillation_loss": epoch_loss,
                "average_cosine_similarity": epoch_cosine,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "accumulation_steps": accumulation_steps,
                "effective_batch_size": batch_size * accumulation_steps,
                "images_per_second": epoch_result["images_per_second"],
                "epoch_seconds": epoch_result["epoch_seconds"],
                "gpu_peak_allocated_gb": epoch_result["gpu_peak_allocated_gb"],
                "gpu_peak_reserved_gb": epoch_result["gpu_peak_reserved_gb"],
                "gradient_norm_mean": epoch_result["gradient_norm_mean"],
                "gradient_norm_max": epoch_result["gradient_norm_max"],
                "convergence_counter": convergence_counter,
                "is_best": is_best,
                "oom_status": "none",
            }
            metrics.append(row)
            atomic_write_csv(metrics_path, metrics)

            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_loss,
                best_cosine,
                best_epoch,
                convergence_counter,
                batch_size,
                accumulation_steps,
                validation,
                transform,
                environment,
                training_config,
            )
            atomic_save_checkpoint(paths["checkpoints"] / "last.pt", payload)
            if is_best:
                atomic_save_checkpoint(paths["checkpoints"] / "best.pt", payload)
            if epoch % 5 == 0:
                atomic_save_checkpoint(paths["checkpoints"] / f"epoch_{epoch:03d}.pt", payload)
            save_plots(metrics, paths["figures"])
            log_message(
                log_path,
                f"epoch={epoch} loss={epoch_loss:.10f} cosine={epoch_cosine:.8f} "
                f"batch={batch_size} accum={accumulation_steps} "
                f"speed={epoch_result['images_per_second']:.2f}/s "
                f"peak_alloc={epoch_result['gpu_peak_allocated_gb']:.3f}GB "
                f"counter={convergence_counter} best={is_best}",
            )
            if convergence_counter >= args.patience:
                convergence_stopped = True
                log_message(log_path, f"Convergence stopping triggered at epoch {epoch}")
                break
    except (KeyboardInterrupt, Exception) as exc:
        diagnostic = {
            "status": "FAIL",
            "created_at_utc": utc_now(),
            "epoch": current_epoch,
            "error_type": type(exc).__name__,
            "error_reason": str(exc),
            "traceback": traceback.format_exc(),
            "rad_dino_loaded": False,
        }
        atomic_write_json(paths["diagnostics"] / "training_failure.json", diagnostic)
        try:
            interrupted = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                current_epoch,
                best_loss,
                best_cosine,
                best_epoch,
                convergence_counter,
                batch_size,
                accumulation_steps,
                validation,
                transform,
                environment,
                training_config,
            )
            interrupted["interruption"] = diagnostic
            atomic_save_checkpoint(paths["checkpoints"] / "interrupted.pt", interrupted)
        except Exception as checkpoint_exc:
            diagnostic["interrupted_checkpoint_error"] = str(checkpoint_exc)
            atomic_write_json(paths["diagnostics"] / "training_failure.json", diagnostic)
        raise

    best_path = paths["checkpoints"] / "best.pt"
    last_path = paths["checkpoints"] / "last.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError("best.pt or last.pt was not created")
    distilled_path = paths["checkpoints"] / "distilled_convnext_tiny_backbone.pt"
    export_distilled_backbone(best_path, distilled_path)
    write_summary(
        output_dir / "phase1_training_summary.md",
        metrics,
        current_epoch,
        convergence_stopped,
        best_epoch,
        best_loss,
        best_cosine,
        batch_size,
        accumulation_steps,
        validation,
        environment,
    )
    final_audit = {
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "completed_epochs": current_epoch,
        "convergence_stopping_triggered": convergence_stopped,
        "best_epoch": best_epoch,
        "best_average_distillation_loss": best_loss,
        "best_average_cosine_similarity": best_cosine,
        "last_epoch_loss": float(metrics[-1]["average_distillation_loss"]),
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "peak_allocated_gb": max(float(row["gpu_peak_allocated_gb"]) for row in metrics),
        "peak_reserved_gb": max(float(row["gpu_peak_reserved_gb"]) for row in metrics),
        "average_images_per_second": float(
            np.mean([float(row["images_per_second"]) for row in metrics])
        ),
        "nan_inf_nonfinite_gradient": False,
        "oom_during_training": False,
        "rad_dino_loaded": False,
        "class_label_used_in_loss": False,
        "split_created": False,
        "phase2_started": False,
        "baseline_started": False,
        "manifest_sha256": validation["manifest_sha256"],
        "teacher_cache_sha256": validation["teacher_cache_sha256"],
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "distilled_backbone": str(distilled_path),
    }
    atomic_write_json(paths["diagnostics"] / "phase1_final_audit.json", final_audit)
    log_message(log_path, json.dumps(final_audit, ensure_ascii=False))
    print(json.dumps(final_audit, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    default_root = Path(r"C:\Users\09688\thoracic-cxr-project-3")
    cache_root = default_root / "outputs" / "raddino_feature_cache" / "balanced_945_seed42"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(default_root))
    parser.add_argument("--manifest", default=str(cache_root / "roi_manifest.csv"))
    parser.add_argument("--teacher-cache", default=str(cache_root / "teacher_features.pt"))
    parser.add_argument(
        "--output-dir",
        default=str(
            default_root
            / "outputs"
            / "raddino_convnext_tiny_experiment_seed42"
            / "phase1_distillation"
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--smoke-test-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.workers < 0 or args.patience < 1:
        raise ValueError("epochs/workers/patience arguments are invalid")
    if args.batch_size != "auto":
        batch = int(args.batch_size)
        if batch not in {64, 32, 16, 8, 4}:
            raise ValueError("Explicit batch size must be one of 64, 32, 16, 8, 4")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
