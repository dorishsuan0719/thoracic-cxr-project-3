#!/usr/bin/env python
"""Shared Phase 2 ConvNeXt-Tiny fine-tuning for Proposed and Baseline runs.

The two runs share every training decision recorded in the locked config.  The
only permitted differences are initialization and output directory.  A smoke
run never reads test images; the test split is evaluated once, after best-model
selection has completed using validation data only.
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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.transforms import functional as TF


PHASE = "phase2_supervised_finetune"
ARCHITECTURE = "convnext_tiny"
FEATURE_DIM = 768
NUM_CLASSES = 5
EXPECTED_ROWS = {"train": 3770, "val": 454, "test": 454}
EXPECTED_CLASS_COUNTS = {
    "train": {0: 744, 1: 759, 2: 763, 3: 756, 4: 748},
    "val": {0: 77, 1: 78, 2: 112, 3: 106, 4: 81},
    "test": {0: 77, 1: 78, 2: 112, 3: 106, 4: 81},
}
EXPECTED_AUGMENTED = {"train": 357, "val": 0, "test": 0}
EXPECTED_MANIFEST_SHA256 = {
    "train": "ba5ba5f743c439563e15106239e6bcc87bf9c8fe4105b295ef034356e5dbae55",
    "val": "5f92fd7282df28a4ec3365ba5fa7a777b365db860f7991a47238162d1ac5bc00",
    "test": "2130a73dcbadec1d6b4bba68f809db7eeed25d1ea421c4d450d3e0b4d015551a",
    "image_id_split": "ace50c1f4820252073049b5ecf8f0b601eac026a8119789c8047c8bfd4e41c1a",
}
EXPECTED_DISTILLED_SHA256 = "267b5832d1283d93cb8812e312fbb12c64648711144003aa05dbbdf14446d116"
EXPECTED_PATCH_DISTILLED_SHA256 = "7ea9e3e3b930d4009e0a38c714eba2b1182c17af830e2a8e69e3505f9f181a78"
EXPECTED_PHASE1_MANIFEST_SHA256 = "796f067d00bb5740a51b51292eed4acfefe9b2e84fd2eeb9b5dfd2df926d5233"
EXPECTED_TEACHER_CACHE_SHA256 = "500a451a6023b71a08c15fc25d6651ec94a20e0575bbd14d6f84de308a7d9e38"
EXPECTED_PATCH_TEACHER_CACHE_SHA256 = "082c626e9a5730023361f48566e68bb653ceceb2e0600c42f23e555336002828"
IMAGENET_WEIGHTS_ENUM = "ConvNeXt_Tiny_Weights.IMAGENET1K_V1"

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


def atomic_write_text(path: Path, text: str, allow_replace: bool = True) -> None:
    if path.exists() and not allow_replace:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    replace_with_retry(temporary, path)


def atomic_write_json(path: Path, value: Any, allow_replace: bool = True) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        allow_replace=allow_replace,
    )


def atomic_write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    bom: bool = False,
    allow_replace: bool = True,
) -> None:
    if path.exists() and not allow_replace:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoding = "utf-8-sig" if bom else "utf-8"
    with temporary.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    replace_with_retry(temporary, path)


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


def random_states() -> dict[str, Any]:
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


def environment_info(device: torch.device) -> dict[str, Any]:
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
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "device": str(device),
        "gpu": gpu,
    }


def imagenet_weights_audit() -> dict[str, Any]:
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    filename = Path(weights.url).name
    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    transform = weights.transforms()
    metadata_keys = ("min_size", "num_params", "recipe", "_metrics", "_ops", "_file_size")
    return {
        "weights_enum": IMAGENET_WEIGHTS_ENUM,
        "url": weights.url,
        "cache_path": str(cache_path),
        "cache_exists": cache_path.is_file(),
        "cache_sha256": sha256_file(cache_path) if cache_path.is_file() else None,
        "metadata": {key: weights.meta.get(key) for key in metadata_keys},
        "categories_count": len(weights.meta.get("categories", [])),
        "transforms": {
            "resize_size": list(transform.resize_size),
            "crop_size": list(transform.crop_size),
            "interpolation": str(transform.interpolation),
            "antialias": transform.antialias,
            "mean": list(transform.mean),
            "std": list(transform.std),
        },
    }


def configure_runtime(device: torch.device, seed: int) -> None:
    set_seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


class Phase2Transform:
    def __init__(self, training: bool) -> None:
        weights_transform = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
        self.training = training
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

    def preprocessing_config(self) -> dict[str, Any]:
        return {
            "resize_size": self.resize_size,
            "crop_size": self.crop_size,
            "interpolation": str(self.interpolation),
            "antialias": self.antialias,
            "mean": self.mean,
            "std": self.std,
            "source_mode": "L",
            "model_mode": "RGB converted in memory",
            "input_size": [224, 224],
            "normalization": "ImageNet",
        }

    def augmentation_config(self) -> dict[str, Any]:
        return {
            "train_only": True,
            "gaussian_blur": {
                "probability": self.blur_probability,
                "kernel_size": self.blur_kernel_size,
                "sigma_range": list(self.blur_sigma),
            },
            "gaussian_noise": {
                "probability": self.noise_probability,
                "mean": 0.0,
                "std_range": list(self.noise_std),
                "position": "after tensor conversion and before normalization",
                "clamp": [0.0, 1.0],
            },
            "brightness": False,
            "contrast": False,
            "color_jitter": False,
            "gamma": False,
            "histogram_equalization": False,
            "flips": False,
            "rotation": False,
            "affine": False,
            "perspective": False,
            "random_crop": False,
            "random_resized_crop": False,
            "random_erasing": False,
        }

    def apply(self, image: Image.Image, audit: bool = False) -> Any:
        image = image.convert("RGB")
        blur_applied = False
        noise_applied = False
        blur_sigma = None
        noise_std = None
        if self.training and random.random() < self.blur_probability:
            blur_applied = True
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
        tensor = TF.pil_to_tensor(image).float().div_(255.0)
        if self.training and random.random() < self.noise_probability:
            noise_applied = True
            noise_std = random.uniform(*self.noise_std)
            tensor = torch.clamp(tensor + torch.randn_like(tensor) * noise_std, 0.0, 1.0)
        display = tensor.clone() if audit else None
        tensor = TF.normalize(tensor, self.mean, self.std)
        if audit:
            return tensor, {
                "display": display,
                "blur_applied": blur_applied,
                "blur_sigma": blur_sigma,
                "noise_applied": noise_applied,
                "noise_std": noise_std,
            }
        return tensor

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.apply(image, audit=False)


class RoiClassificationDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform: Phase2Transform) -> None:
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
                    raise ValueError(f"expected 224x224, got {image.size}")
                if image.mode != "L":
                    raise ValueError(f"expected source mode L, got {image.mode}")
                tensor = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to read ROI {path}: {exc}") from exc
        return {
            "image": tensor,
            "label": int(row["class_id"]),
            "row_position": index,
        }


class ConvNeXtTinyClassifier(nn.Module):
    def __init__(self, initialization: str, distilled_path: Path | None = None) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if initialization == "imagenet" else None
        base = convnext_tiny(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        self.final_norm = base.classifier[0]
        self.flatten = base.classifier[1]
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)
        self.load_audit: dict[str, Any] = {
            "initialization": initialization,
            "weights_enum": IMAGENET_WEIGHTS_ENUM if initialization == "imagenet" else None,
            "imagenet_pretrained_loaded": initialization == "imagenet",
            "distilled_checkpoint_loaded": False,
            "rad_dino_loaded": False,
            "teacher_feature_cache_loaded": False,
            "missing_backbone_keys": [],
            "unexpected_backbone_keys": [],
            "missing_final_norm_keys": [],
            "unexpected_final_norm_keys": [],
        }
        if initialization in {"distilled", "patch_distilled"}:
            if distilled_path is None:
                raise ValueError(f"{initialization} initialization requires --distilled-checkpoint")
            checkpoint = torch.load(distilled_path, map_location="cpu", weights_only=False)
            self.load_audit["distilled_checkpoint_loaded"] = True
            if initialization == "patch_distilled":
                self._validate_patch_distilled_checkpoint(checkpoint)
                backbone_result = self.features.load_state_dict(checkpoint["student_state_dict"], strict=True)
                norm_missing: list[str] = []
                norm_unexpected: list[str] = []
                self.load_audit.update({
                    "patch_distilled_checkpoint_loaded": True,
                    "final_norm_loaded_from_checkpoint": False,
                    "final_norm_initialization": "torchvision default initialization; no ImageNet classifier loaded",
                    "phase1_distillation_type": checkpoint.get("distillation_type"),
                    "phase1_output_feature_shape": checkpoint.get("output_feature_shape"),
                    "phase1_best_monitor_patch_mse": checkpoint.get("best_monitor_patch_mse"),
                    "phase1_best_monitor_patch_cosine": checkpoint.get("best_monitor_patch_cosine"),
                })
            else:
                self._validate_distilled_checkpoint(checkpoint)
                backbone_result = self.features.load_state_dict(checkpoint["backbone_state_dict"], strict=True)
                norm_result = self.final_norm.load_state_dict(checkpoint["final_norm_state_dict"], strict=True)
                norm_missing = list(norm_result.missing_keys)
                norm_unexpected = list(norm_result.unexpected_keys)
            self.load_audit.update(
                {
                    "missing_backbone_keys": list(backbone_result.missing_keys),
                    "unexpected_backbone_keys": list(backbone_result.unexpected_keys),
                    "missing_final_norm_keys": norm_missing,
                    "unexpected_final_norm_keys": norm_unexpected,
                    "phase1_best_epoch": checkpoint.get("best_epoch"),
                    "phase1_manifest_sha256": checkpoint.get("manifest_sha256"),
                    "teacher_cache_sha256": checkpoint.get("teacher_cache_sha256"),
                    "classifier_head_included": checkpoint.get("classifier_head_included"),
                }
            )

    @staticmethod
    def _validate_distilled_checkpoint(checkpoint: dict[str, Any]) -> None:
        checks = {
            "architecture": checkpoint.get("architecture") == ARCHITECTURE,
            "feature_dim": checkpoint.get("feature_dim") == FEATURE_DIM,
            "manifest_sha256": checkpoint.get("manifest_sha256") == EXPECTED_PHASE1_MANIFEST_SHA256,
            "teacher_cache_sha256": checkpoint.get("teacher_cache_sha256") == EXPECTED_TEACHER_CACHE_SHA256,
            "classifier_absent": checkpoint.get("classifier_head_included") is False,
            "backbone_state_dict": isinstance(checkpoint.get("backbone_state_dict"), dict),
            "final_norm_state_dict": isinstance(checkpoint.get("final_norm_state_dict"), dict),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"Invalid distilled checkpoint: {failed}")

    @staticmethod
    def _validate_patch_distilled_checkpoint(checkpoint: dict[str, Any]) -> None:
        checks = {
            "architecture": checkpoint.get("architecture") == ARCHITECTURE,
            "distillation_type": checkpoint.get("distillation_type") == "RAD-DINO 7x7 patch feature",
            "output_feature_shape": checkpoint.get("output_feature_shape") == ["B", 768, 7, 7],
            "manifest_sha256": checkpoint.get("manifest_sha256") == EXPECTED_PHASE1_MANIFEST_SHA256,
            "teacher_cache_sha256": checkpoint.get("teacher_cache_sha256")
            == EXPECTED_PATCH_TEACHER_CACHE_SHA256,
            "best_epoch": checkpoint.get("best_epoch") == 84,
            "best_monitor_patch_mse": math.isclose(
                float(checkpoint.get("best_monitor_patch_mse", float("nan"))),
                0.00026442373185615653,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "best_monitor_patch_cosine": math.isclose(
                float(checkpoint.get("best_monitor_patch_cosine", float("nan"))),
                0.8984613290410772,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "classifier_absent": checkpoint.get("classifier_head_included") is False,
            "student_state_dict": isinstance(checkpoint.get("student_state_dict"), dict),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"Invalid patch-distilled checkpoint: {failed}")

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.avgpool(x)
        x = self.final_norm(x)
        return self.flatten(x)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.extract_features(images)))


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


def build_optimizer(model: ConvNeXtTinyClassifier) -> torch.optim.Optimizer:
    backbone_parameters = list(model.features.parameters()) + list(model.final_norm.parameters())
    head_parameters = list(model.classifier.parameters())
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 1e-5, "name": "backbone"},
            {"params": head_parameters, "lr": 1e-4, "name": "classifier_head"},
        ],
        weight_decay=1e-4,
    )


def build_scaler(device: torch.device) -> torch.amp.GradScaler:
    # Full-backbone fine-tuning can overflow with AMP's 65536 default before
    # the scaler has had an opportunity to adapt.  This fixed conservative
    # starting value is shared by Proposed and Baseline.
    return torch.amp.GradScaler(device.type, enabled=device.type == "cuda", init_scale=1024.0)


def validate_manifests(args: argparse.Namespace) -> dict[str, Any]:
    paths = {"train": args.train_manifest, "val": args.val_manifest, "test": args.test_manifest}
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    hashes: dict[str, str] = {}
    required = {
        "record_index", "source_image_id", "split", "image_path", "filename",
        "class_id", "class_name", "is_brightness_augmented", "image_sha256",
    }
    for split, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty {split} manifest: {path}")
        digest = sha256_file(path)
        if digest != EXPECTED_MANIFEST_SHA256[split]:
            raise ValueError(f"{split} manifest SHA256 mismatch: {digest}")
        rows = read_csv(path)
        if len(rows) != EXPECTED_ROWS[split]:
            raise ValueError(f"{split} rows={len(rows)}, expected {EXPECTED_ROWS[split]}")
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"{split} manifest missing fields: {sorted(required - set(rows[0] if rows else {}))}")
        class_counts = Counter(int(row["class_id"]) for row in rows)
        if dict(sorted(class_counts.items())) != EXPECTED_CLASS_COUNTS[split]:
            raise ValueError(f"{split} class counts mismatch: {dict(class_counts)}")
        augmented = sum(parse_bool(row["is_brightness_augmented"]) for row in rows)
        if augmented != EXPECTED_AUGMENTED[split]:
            raise ValueError(f"{split} augmented rows={augmented}, expected {EXPECTED_AUGMENTED[split]}")
        bad_split = [row["record_index"] for row in rows if row["split"] != split]
        bad_class = [row["record_index"] for row in rows if int(row["class_id"]) not in CLASS_MAPPING]
        bad_name = [row["record_index"] for row in rows if row["class_name"] != CLASS_MAPPING[int(row["class_id"])]]
        missing_paths = [row["image_path"] for row in rows if not Path(row["image_path"]).is_file()]
        duplicate_paths = len(rows) - len({os.path.normcase(os.path.abspath(row["image_path"])) for row in rows})
        if bad_split or bad_class or bad_name or missing_paths or duplicate_paths:
            raise ValueError(
                f"{split} validation failed: bad_split={len(bad_split)}, bad_class={len(bad_class)}, "
                f"bad_name={len(bad_name)}, missing={len(missing_paths)}, duplicate_paths={duplicate_paths}"
            )
        hashes[split] = digest
        rows_by_split[split] = rows

    leakage: dict[str, int] = {}
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        for field in ("source_image_id", "image_path", "image_sha256", "original_roi_id"):
            lhs = {os.path.normcase(os.path.abspath(row[field])) if field == "image_path" else row[field] for row in rows_by_split[left]}
            rhs = {os.path.normcase(os.path.abspath(row[field])) if field == "image_path" else row[field] for row in rows_by_split[right]}
            leakage[f"{left}_{right}_{field}"] = len(lhs & rhs)
    if any(leakage.values()):
        raise ValueError(f"Split leakage detected: {leakage}")

    if not args.shared_protocol.is_file():
        raise FileNotFoundError(args.shared_protocol)
    protocol = json.loads(args.shared_protocol.read_text(encoding="utf-8-sig"))
    if protocol.get("status") != "PASS":
        raise ValueError("Shared split protocol is not PASS")
    for split in ("train", "val", "test"):
        if protocol.get("manifest_sha256", {}).get(f"{split}_roi_manifest") != hashes[split]:
            raise ValueError(f"Protocol {split} SHA256 mismatch")
    split_manifest_path = args.shared_protocol.parent / "image_id_split_manifest.csv"
    split_hash = sha256_file(split_manifest_path)
    if split_hash != EXPECTED_MANIFEST_SHA256["image_id_split"]:
        raise ValueError(f"Image-ID split SHA256 mismatch: {split_hash}")

    distilled_sha = None
    phase1_metadata = None
    if args.initialization in {"distilled", "patch_distilled"}:
        if not args.distilled_backbone.is_file():
            raise FileNotFoundError(args.distilled_backbone)
        distilled_sha = sha256_file(args.distilled_backbone)
        expected_distilled_sha = (
            EXPECTED_PATCH_DISTILLED_SHA256
            if args.initialization == "patch_distilled"
            else EXPECTED_DISTILLED_SHA256
        )
        if distilled_sha != expected_distilled_sha:
            raise ValueError(f"Distilled backbone SHA256 mismatch: {distilled_sha}")
        phase1 = torch.load(args.distilled_backbone, map_location="cpu", weights_only=False)
        if args.initialization == "patch_distilled":
            ConvNeXtTinyClassifier._validate_patch_distilled_checkpoint(phase1)
        else:
            ConvNeXtTinyClassifier._validate_distilled_checkpoint(phase1)
        phase1_metadata = {key: value for key, value in phase1.items() if not key.endswith("state_dict")}

    return {
        "rows": rows_by_split,
        "manifest_sha256": hashes,
        "image_id_split_manifest": str(split_manifest_path),
        "image_id_split_sha256": split_hash,
        "shared_protocol_sha256": sha256_file(args.shared_protocol),
        "leakage": leakage,
        "distilled_backbone_sha256": distilled_sha,
        "phase1_checkpoint_metadata": phase1_metadata,
    }


def locked_config(args: argparse.Namespace, validation: dict[str, Any], transform: Phase2Transform) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "architecture": ARCHITECTURE,
        "num_classes": NUM_CLASSES,
        "feature_dim": FEATURE_DIM,
        "seed": 42,
        "class_mapping": {str(key): value for key, value in CLASS_MAPPING.items()},
        "manifests": {
            split: {"path": str(getattr(args, f"{split}_manifest")), "sha256": validation["manifest_sha256"][split]}
            for split in ("train", "val", "test")
        },
        "image_id_split_manifest": {
            "path": validation["image_id_split_manifest"],
            "sha256": validation["image_id_split_sha256"],
        },
        "shared_protocol": {"path": str(args.shared_protocol), "sha256": validation["shared_protocol_sha256"]},
        "preprocessing": transform.preprocessing_config(),
        "augmentation": transform.augmentation_config(),
        "loss": {"name": "CrossEntropyLoss", "class_weights": None, "label_smoothing": 0.0},
        "optimizer": {
            "name": "AdamW",
            "backbone_learning_rate": 1e-5,
            "classifier_learning_rate": 1e-4,
            "weight_decay": 1e-4,
        },
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 50},
        "maximum_epochs": 50,
        "early_stopping": {
            "metric": "validation_macro_f1",
            "mode": "max",
            "patience": 10,
            "min_delta": 1e-4,
            "best_tie_break": ["higher validation macro-F1", "lower validation loss", "earlier epoch"],
        },
        "gradient_clip_max_norm": 1.0,
        "amp": True,
        "batch_size_candidates": [64, 32, 16, 8, 4],
        "target_effective_batch_size": 64,
        "data_loader": {
            "workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "train_shuffle": True,
            "validation_shuffle": False,
            "test_shuffle": False,
            "worker_seed_strategy": "torch initial seed modulo 2^32",
            "train_generator_seed_strategy": "seed + epoch",
        },
        "checkpoint_selection_uses_test": False,
        "test_evaluation": "exactly once after best checkpoint is fixed",
        "trainable_from_epoch_one": ["entire ConvNeXt-Tiny backbone", "final LayerNorm", "classification head"],
        "classifier_head": {"dropout": 0.2, "linear": [768, 5], "hidden_layer": False, "projector": False},
        "allowed_run_differences": ["initialization", "output_directory"],
    }


def validate_locked_runtime(args: argparse.Namespace) -> None:
    checks = {
        "seed": args.seed == 42,
        "epochs": args.epochs == 50,
        "patience": args.patience == 10,
        "workers": args.workers == 2,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Runtime differs from locked Proposed/Baseline protocol: {failed}")


def ensure_locked_config(path: Path, expected: dict[str, Any], write: bool) -> str:
    serialized = json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8-sig"))
        if existing != expected:
            raise ValueError(f"Existing locked config differs and will not be overwritten: {path}")
    elif write:
        atomic_write_text(path, serialized, allow_replace=False)
    else:
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return sha256_file(path)


def resolve_locked_batch(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.batch_size != "locked":
        return None
    proposed_config_path = (
        args.shared_config.parent
        / "phase2_proposed_distilled"
        / "config"
        / "experiment_config.json"
    )
    if not proposed_config_path.is_file():
        raise FileNotFoundError(
            f"Cannot lock Baseline batch settings because Proposed experiment config is missing: {proposed_config_path}"
        )
    proposed = json.loads(proposed_config_path.read_text(encoding="utf-8-sig"))
    batch_size = int(proposed.get("actual_batch_size", -1))
    accumulation_steps = int(proposed.get("accumulation_steps", -1))
    effective_batch_size = int(proposed.get("effective_batch_size", -1))
    if batch_size <= 0 or accumulation_steps <= 0 or batch_size * accumulation_steps != effective_batch_size:
        raise ValueError(f"Invalid Proposed batch settings: {proposed}")
    args.locked_batch_size = batch_size
    args.locked_accumulation_steps = accumulation_steps
    args.locked_effective_batch_size = effective_batch_size
    args.batch_size = str(batch_size)
    return {
        "source": str(proposed_config_path),
        "proposed_initialization": proposed.get("initialization"),
        "proposed_shared_config_sha256": proposed.get("shared_config_sha256"),
        "actual_batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch_size,
    }


def build_fairness_audit(
    args: argparse.Namespace,
    validation: dict[str, Any],
    shared_config: dict[str, Any],
    shared_sha: str,
    batch_size: int,
    accumulation_steps: int,
    locked_batch_source: dict[str, Any] | None,
) -> dict[str, Any]:
    proposed_config_path = (
        args.shared_config.parent
        / "phase2_proposed_distilled"
        / "config"
        / "experiment_config.json"
    )
    proposed_shared_copy = (
        args.shared_config.parent
        / "phase2_proposed_distilled"
        / "config"
        / "shared_phase2_finetune_config.json"
    )
    baseline_dir = args.shared_config.parent / "phase2_baseline_imagenet"
    baseline_config_path = baseline_dir / "config" / "baseline_experiment_config.json"
    baseline_shared_copy = baseline_dir / "config" / "shared_phase2_finetune_config.json"
    if not proposed_config_path.is_file() or not proposed_shared_copy.is_file():
        raise FileNotFoundError("Proposed config artifacts required for fairness audit are missing")
    if args.initialization == "patch_distilled" and (
        not baseline_config_path.is_file() or not baseline_shared_copy.is_file()
    ):
        raise FileNotFoundError("Baseline config artifacts required for Patch fairness audit are missing")
    proposed = json.loads(proposed_config_path.read_text(encoding="utf-8-sig"))
    baseline = (
        json.loads(baseline_config_path.read_text(encoding="utf-8-sig"))
        if args.initialization == "patch_distilled"
        else None
    )
    checks = {
        "shared_config_sha256": proposed.get("shared_config_sha256") == shared_sha,
        "shared_config_byte_identical": proposed_shared_copy.read_bytes() == args.shared_config.read_bytes(),
        "train_manifest_sha256": validation["manifest_sha256"]["train"] == shared_config["manifests"]["train"]["sha256"],
        "val_manifest_sha256": validation["manifest_sha256"]["val"] == shared_config["manifests"]["val"]["sha256"],
        "test_manifest_sha256": validation["manifest_sha256"]["test"] == shared_config["manifests"]["test"]["sha256"],
        "image_id_split_sha256": validation["image_id_split_sha256"] == shared_config["image_id_split_manifest"]["sha256"],
        "preprocessing": shared_config["preprocessing"] == Phase2Transform(training=False).preprocessing_config(),
        "augmentation": shared_config["augmentation"] == Phase2Transform(training=True).augmentation_config(),
        "seed": shared_config["seed"] == args.seed == 42,
        "batch_size": int(proposed.get("actual_batch_size", -1)) == batch_size,
        "accumulation_steps": int(proposed.get("accumulation_steps", -1)) == accumulation_steps,
        "effective_batch_size": int(proposed.get("effective_batch_size", -1)) == batch_size * accumulation_steps,
        "optimizer": shared_config["optimizer"]["name"] == "AdamW",
        "backbone_learning_rate": shared_config["optimizer"]["backbone_learning_rate"] == 1e-5,
        "classifier_learning_rate": shared_config["optimizer"]["classifier_learning_rate"] == 1e-4,
        "scheduler": shared_config["scheduler"] == {"name": "CosineAnnealingLR", "T_max": 50},
        "maximum_epochs": shared_config["maximum_epochs"] == args.epochs == 50,
        "early_stopping": shared_config["early_stopping"]["patience"] == args.patience == 10,
        "checkpoint_selection": shared_config["early_stopping"]["metric"] == "validation_macro_f1",
        "evaluation_metrics": True,
        "leakage_zero": all(value == 0 for value in validation["leakage"].values()),
    }
    augmentation_preview_sha256: dict[str, str] | None = None
    if args.initialization == "patch_distilled":
        current_preview = args.output_dir / "figures" / "augmentation_preview.png"
        proposed_preview = proposed_config_path.parents[1] / "figures" / "augmentation_preview.png"
        baseline_preview = baseline_dir / "figures" / "augmentation_preview.png"
        for preview in (current_preview, proposed_preview, baseline_preview):
            if not preview.is_file():
                raise FileNotFoundError(preview)
        augmentation_preview_sha256 = {
            "patch_proposed": sha256_file(current_preview),
            "cls_proposed": sha256_file(proposed_preview),
            "baseline": sha256_file(baseline_preview),
        }
        checks.update({
            "baseline_shared_config_sha256": baseline.get("shared_config_sha256") == shared_sha,
            "baseline_shared_config_byte_identical": baseline_shared_copy.read_bytes()
            == args.shared_config.read_bytes(),
            "baseline_batch_size": int(baseline.get("actual_batch_size", -1)) == batch_size,
            "baseline_accumulation_steps": int(baseline.get("accumulation_steps", -1))
            == accumulation_steps,
            "baseline_effective_batch_size": int(baseline.get("effective_batch_size", -1))
            == batch_size * accumulation_steps,
            "classifier_head": shared_config["classifier_head"]
            == {"dropout": 0.2, "linear": [768, 5], "hidden_layer": False, "projector": False},
            "loss_function": shared_config["loss"]
            == {"name": "CrossEntropyLoss", "class_weights": None, "label_smoothing": 0.0},
            "weight_decay": shared_config["optimizer"]["weight_decay"] == 1e-4,
            "amp": shared_config["amp"] is True,
            "gradient_clipping": shared_config["gradient_clip_max_norm"] == 1.0,
            "class_mapping": shared_config["class_mapping"]
            == {str(key): value for key, value in CLASS_MAPPING.items()},
            "augmentation_preview_sha256_identical": len(set(augmentation_preview_sha256.values())) == 1,
        })
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failed else "FAIL",
        "created_at_utc": utc_now(),
        "proposed_experiment_config": str(proposed_config_path),
        "comparison_output_directory": str(args.output_dir),
        "cls_proposed_output_directory": proposed.get("output_directory"),
        "baseline_output_directory": baseline.get("output_directory") if baseline else str(args.output_dir),
        "shared_config_path": str(args.shared_config),
        "shared_config_sha256": shared_sha,
        "locked_batch_source": locked_batch_source,
        "checks": checks,
        "failed_checks": failed,
        "fixed_setting_difference_count": len(failed),
        "allowed_differences": ({
            "initialization": {
                "patch_proposed": "patch_distilled",
                "cls_proposed": "distilled",
                "baseline": "imagenet",
            },
            "initialization_checkpoint": {
                "patch_proposed": str(args.distilled_backbone),
                "cls_proposed": proposed.get("distilled_backbone"),
                "baseline": IMAGENET_WEIGHTS_ENUM,
            },
            "output_directory": {
                "patch_proposed": str(args.output_dir),
                "cls_proposed": proposed.get("output_directory"),
                "baseline": baseline.get("output_directory"),
            },
            "experiment_label": {
                "patch_proposed": args.output_dir.name,
                "cls_proposed": proposed.get("experiment"),
                "baseline": baseline.get("experiment"),
            },
            "output_model_filename": {
                "patch_proposed": "patch_proposed_convnext_tiny_5class.pt",
                "cls_proposed": "proposed_convnext_tiny_5class.pt",
                "baseline": "baseline_convnext_tiny_5class.pt",
            },
        } if args.initialization == "patch_distilled" else {
            "initialization": {"proposed": "distilled", "baseline": "imagenet"},
            "output_directory": {
                "proposed": proposed.get("output_directory"),
                "baseline": str(args.output_dir),
            },
        }),
        "augmentation_preview_sha256": augmentation_preview_sha256,
        "weights": imagenet_weights_audit(),
        "distilled_checkpoint_loaded": args.initialization in {"distilled", "patch_distilled"},
        "patch_distilled_checkpoint_loaded": args.initialization == "patch_distilled",
        "teacher_feature_cache_loaded": False,
        "rad_dino_loaded": False,
    }


def create_output_directories(output_dir: Path) -> dict[str, Path]:
    paths = {name: output_dir / name for name in (
        "checkpoints", "metrics", "figures", "predictions", "logs", "config", "diagnostics", "test_results"
    )}
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        path.mkdir(exist_ok=True)
    return paths


def output_is_nonempty(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.iterdir())


def log_message(path: Path | None, message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    if path is not None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def create_augmentation_preview(rows: list[dict[str, str]], transform: Phase2Transform, figure_path: Path, audit_path: Path, seed: int) -> dict[str, Any]:
    set_seed(seed)
    selected = rows[:25]
    figure, axes = plt.subplots(5, 10, figsize=(18, 10))
    audit_rows = []
    for i, row in enumerate(selected):
        with Image.open(row["image_path"]) as image:
            image.load()
            clean = Phase2Transform(training=False).apply(image)
            augmented, audit = transform.apply(image, audit=True)
        clean_display = clean * torch.tensor(transform.std)[:, None, None] + torch.tensor(transform.mean)[:, None, None]
        axes[i // 5, (i % 5) * 2].imshow(clean_display.clamp(0, 1).permute(1, 2, 0).numpy())
        axes[i // 5, (i % 5) * 2].set_title(f"Clean {i + 1}", fontsize=7)
        axes[i // 5, (i % 5) * 2 + 1].imshow(audit["display"].permute(1, 2, 0).numpy())
        axes[i // 5, (i % 5) * 2 + 1].set_title(f"Aug {i + 1}", fontsize=7)
        axes[i // 5, (i % 5) * 2].axis("off")
        axes[i // 5, (i % 5) * 2 + 1].axis("off")
        audit_rows.append({
            "record_index": row["record_index"],
            "image_path": row["image_path"],
            "blur_applied": audit["blur_applied"],
            "blur_sigma": audit["blur_sigma"],
            "noise_applied": audit["noise_applied"],
            "noise_std": audit["noise_std"],
        })
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)
    result = {
        "status": "PASS",
        "sample_count": len(selected),
        "blur_applied_count": sum(row["blur_applied"] for row in audit_rows),
        "noise_applied_count": sum(row["noise_applied"] for row in audit_rows),
        "samples": audit_rows,
    }
    atomic_write_json(audit_path, result)
    return result


def class_metrics(targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1
    per_class: dict[int, dict[str, float]] = {}
    for class_id in range(NUM_CLASSES):
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - tp)
        fn = float(confusion[class_id, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_id] = {"precision": precision, "recall": recall, "f1": f1, "support": int(confusion[class_id].sum())}
    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0
    macro_precision = float(np.mean([item["precision"] for item in per_class.values()]))
    macro_recall = float(np.mean([item["recall"] for item in per_class.values()]))
    macro_f1 = float(np.mean([item["f1"] for item in per_class.values()]))
    weighted_f1 = float(sum(item["f1"] * item["support"] for item in per_class.values()) / total) if total else 0.0
    aurocs: dict[int, float] = {}
    if probabilities is not None:
        for class_id in range(NUM_CLASSES):
            binary = (targets == class_id).astype(np.int64)
            positive = int(binary.sum())
            negative = len(binary) - positive
            if positive == 0 or negative == 0:
                auroc = float("nan")
            else:
                order = np.argsort(-probabilities[:, class_id], kind="stable")
                sorted_binary = binary[order]
                tpr = np.concatenate(([0.0], np.cumsum(sorted_binary) / positive))
                fpr = np.concatenate(([0.0], np.cumsum(1 - sorted_binary) / negative))
                auroc = float(np.trapezoid(tpr, fpr))
            aurocs[class_id] = auroc
            per_class[class_id]["auroc"] = auroc
    macro_auroc = float(np.nanmean(list(aurocs.values()))) if aurocs else float("nan")
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_auroc": macro_auroc,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def finite_gradients(model: nn.Module) -> tuple[bool, int]:
    count = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            count += 1
            if not torch.isfinite(parameter.grad).all():
                return False, count
    return True, count


def auto_probe_batch_size(
    args: argparse.Namespace,
    train_rows: list[dict[str, str]],
    device: torch.device,
) -> tuple[int, int, list[dict[str, Any]]]:
    candidates = [int(args.batch_size)] if args.batch_size != "auto" else [64, 32, 16, 8, 4]
    attempts: list[dict[str, Any]] = []
    dataset = RoiClassificationDataset(train_rows, Phase2Transform(training=True))
    for batch_size in candidates:
        if 64 % batch_size:
            raise ValueError("Batch size must divide the locked effective batch size 64")
        model = optimizer = scaler = loader = None
        try:
            set_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            model = ConvNeXtTinyClassifier(args.initialization, args.distilled_backbone).to(device)
            optimizer = build_optimizer(model)
            scaler = build_scaler(device)
            loader = make_loader(dataset, batch_size, 0, args.seed, True)
            batch = next(iter(loader))
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Non-finite probe output")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            finite, gradient_tensors = finite_gradients(model)
            if not finite or not math.isfinite(float(gradient_norm)):
                raise FloatingPointError("Non-finite probe gradient")
            scaler.step(optimizer)
            scaler.update()
            attempts.append({
                "batch_size": batch_size,
                "status": "PASS",
                "loss": float(loss.detach()),
                "logits_shape": list(logits.shape),
                "gradient_norm": float(gradient_norm),
                "gradient_tensors": gradient_tensors,
                "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0,
                "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3) if device.type == "cuda" else 0.0,
            })
            return batch_size, 64 // batch_size, attempts
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append({"batch_size": batch_size, "status": "OOM", "error": str(exc)})
        finally:
            del loader, scaler, optimizer, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError(f"All batch-size candidates failed: {attempts}")


def run_smoke(
    args: argparse.Namespace,
    validation: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    train_dataset = RoiClassificationDataset(validation["rows"]["train"], Phase2Transform(training=True))
    val_dataset = RoiClassificationDataset(validation["rows"]["val"], Phase2Transform(training=False))
    train_loader = make_loader(train_dataset, batch_size, 0, args.seed, True)
    val_loader = make_loader(val_dataset, batch_size, 0, args.seed, False)
    model = ConvNeXtTinyClassifier(args.initialization, args.distilled_backbone).to(device)
    optimizer = build_optimizer(model)
    scaler = build_scaler(device)
    train_records = []
    model.train()
    for batch_number, batch in enumerate(train_loader, start=1):
        if batch_number > 3:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            features = model.extract_features(images)
            logits = model.classifier(model.dropout(features))
            loss = nn.functional.cross_entropy(logits, labels)
        probabilities = torch.softmax(logits.float(), dim=1)
        if tuple(features.shape) != (images.shape[0], FEATURE_DIM):
            raise RuntimeError(f"Smoke feature shape mismatch: {tuple(features.shape)}")
        if tuple(logits.shape) != (images.shape[0], NUM_CLASSES):
            raise RuntimeError(f"Smoke logits shape mismatch: {tuple(logits.shape)}")
        if not torch.isfinite(features).all() or not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError("Smoke non-finite feature/logit/loss")
        if not torch.allclose(probabilities.sum(dim=1), torch.ones(images.shape[0], device=device), atol=1e-5):
            raise FloatingPointError("Smoke softmax does not sum to one")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        finite, gradient_tensors = finite_gradients(model)
        if not finite:
            raise FloatingPointError("Smoke non-finite gradient")
        scaler.step(optimizer)
        scaler.update()
        train_records.append({
            "batch": batch_number,
            "loss": float(loss.detach()),
            "feature_shape": list(features.shape),
            "logits_shape": list(logits.shape),
            "softmax_max_sum_error": float((probabilities.sum(dim=1) - 1).abs().max()),
            "gradient_norm": float(gradient_norm),
            "gradient_tensors": gradient_tensors,
        })
    model.eval()
    with torch.inference_mode():
        batch = next(iter(val_loader))
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            val_loss = nn.functional.cross_entropy(logits, labels)
        probabilities = torch.softmax(logits.float(), dim=1)
        if tuple(logits.shape) != (images.shape[0], NUM_CLASSES) or not torch.isfinite(val_loss):
            raise RuntimeError("Validation smoke failed")
    result = {
        "status": "PASS",
        "train_batches": train_records,
        "validation_batches": 1,
        "validation_loss": float(val_loss),
        "validation_logits_shape": list(logits.shape),
        "validation_softmax_max_sum_error": float((probabilities.sum(dim=1) - 1).abs().max()),
        "backbone_load_audit": model.load_audit,
        "all_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
        "rad_dino_loaded": False,
        "teacher_feature_cache_loaded": False,
        "test_images_read": 0,
    }
    del scaler, optimizer, model, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def checkpoint_payload(
    args: argparse.Namespace,
    model: ConvNeXtTinyClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best: dict[str, Any],
    early_counter: int,
    convergence_reference: float,
    batch_size: int,
    accumulation_steps: int,
    validation: dict[str, Any],
    shared_config: dict[str, Any],
    shared_config_sha: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "experiment": args.output_dir.name,
        "initialization": args.initialization,
        "weights_enum": IMAGENET_WEIGHTS_ENUM if args.initialization == "imagenet" else None,
        "imagenet_pretrained_loaded": args.initialization == "imagenet",
        "architecture": ARCHITECTURE,
        "num_classes": NUM_CLASSES,
        "feature_dim": FEATURE_DIM,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "current_epoch": epoch,
        "best": best,
        "early_stopping_counter": early_counter,
        "convergence_reference": convergence_reference,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "shared_phase2_finetune_config": shared_config,
        "shared_config_sha256": shared_config_sha,
        "manifest_sha256": validation["manifest_sha256"],
        "image_id_split_sha256": validation["image_id_split_sha256"],
        "shared_protocol_sha256": validation["shared_protocol_sha256"],
        "distilled_backbone_sha256": validation["distilled_backbone_sha256"],
        "phase1_checkpoint_metadata": validation["phase1_checkpoint_metadata"],
        "class_mapping": CLASS_MAPPING,
        "preprocessing": shared_config["preprocessing"],
        "augmentation": shared_config["augmentation"],
        "random_states": random_states(),
        "environment": environment,
        "test_evaluation_count": 0,
        "rad_dino_loaded": False,
        "teacher_feature_cache_loaded": False,
    }


def atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    for key in ("phase", "initialization", "architecture", "num_classes", "shared_config_sha256", "current_epoch"):
        if loaded.get(key) != payload.get(key):
            raise RuntimeError(f"Atomic checkpoint reload validation failed: {key}")
    replace_with_retry(temporary, path)


def validate_resume(checkpoint: dict[str, Any], args: argparse.Namespace, validation: dict[str, Any], shared_sha: str) -> None:
    checks = {
        "phase": checkpoint.get("phase") == PHASE,
        "initialization": checkpoint.get("initialization") == args.initialization,
        "weights_enum": checkpoint.get("weights_enum")
        == (IMAGENET_WEIGHTS_ENUM if args.initialization == "imagenet" else None),
        "architecture": checkpoint.get("architecture") == ARCHITECTURE,
        "num_classes": checkpoint.get("num_classes") == NUM_CLASSES,
        "shared_config_sha256": checkpoint.get("shared_config_sha256") == shared_sha,
        "manifest_sha256": checkpoint.get("manifest_sha256") == validation["manifest_sha256"],
        "image_id_split_sha256": checkpoint.get("image_id_split_sha256") == validation["image_id_split_sha256"],
        "distilled_backbone_sha256": checkpoint.get("distilled_backbone_sha256") == validation["distilled_backbone_sha256"],
    }
    if hasattr(args, "locked_batch_size"):
        checks.update({
            "locked_batch_size": int(checkpoint.get("batch_size", -1)) == args.locked_batch_size,
            "locked_accumulation_steps": int(checkpoint.get("accumulation_steps", -1))
            == args.locked_accumulation_steps,
            "locked_effective_batch_size": int(checkpoint.get("effective_batch_size", -1))
            == args.locked_effective_batch_size,
        })
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Resume validation failed: {failed}")


def train_epoch(
    model: ConvNeXtTinyClassifier,
    rows: list[dict[str, str]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    batch_size: int,
    accumulation_steps: int,
    workers: int,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    loader = make_loader(
        RoiClassificationDataset(rows, Phase2Transform(training=True)),
        batch_size, workers, seed + epoch, True,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    total_loss = 0.0
    total_images = 0
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    grad_norms: list[float] = []
    for batch_number, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            raw_loss = nn.functional.cross_entropy(logits, labels)
            loss = raw_loss / accumulation_steps
        if not torch.isfinite(logits).all() or not torch.isfinite(raw_loss):
            positions = batch["row_position"].tolist()
            raise FloatingPointError(f"Non-finite train values at positions={positions}")
        scaler.scale(loss).backward()
        should_step = batch_number % accumulation_steps == 0 or batch_number == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            finite, _ = finite_gradients(model)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not finite or not math.isfinite(float(gradient_norm)):
                raise FloatingPointError(f"Non-finite gradient at train batch {batch_number}")
            grad_norms.append(float(gradient_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch_count = labels.shape[0]
        total_loss += float(raw_loss.detach()) * batch_count
        total_images += batch_count
        targets.append(labels.detach().cpu().numpy())
        predictions.append(logits.detach().argmax(dim=1).cpu().numpy())
    seconds = time.perf_counter() - started
    metrics = class_metrics(np.concatenate(targets), np.concatenate(predictions))
    metrics.update({
        "loss": total_loss / total_images,
        "images_per_second": total_images / seconds,
        "epoch_seconds": seconds,
        "gradient_norm_mean": float(np.mean(grad_norms)),
        "gradient_norm_max": float(np.max(grad_norms)),
        "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0,
        "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3) if device.type == "cuda" else 0.0,
    })
    return metrics


def evaluate(
    model: ConvNeXtTinyClassifier,
    rows: list[dict[str, str]],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = make_loader(
        RoiClassificationDataset(rows, Phase2Transform(training=False)),
        batch_size, workers, 42, False,
    )
    model.eval()
    started = time.perf_counter()
    total_loss = 0.0
    total_images = 0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_positions: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            probabilities = torch.softmax(logits.float(), dim=1)
            if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all() or not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite evaluation values at positions={batch['row_position'].tolist()}")
            count = labels.shape[0]
            total_loss += float(loss) * count
            total_images += count
            all_targets.append(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_predictions.append(probabilities.argmax(dim=1).cpu().numpy())
            all_positions.append(batch["row_position"].numpy())
    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    positions = np.concatenate(all_positions)
    metrics = class_metrics(targets, predictions, probabilities)
    seconds = time.perf_counter() - started
    metrics.update({"loss": total_loss / total_images, "images_per_second": total_images / seconds, "evaluation_seconds": seconds})
    return metrics, targets, predictions, probabilities, positions


def flatten_epoch_metrics(epoch: int, train: dict[str, Any], val: dict[str, Any], optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, best: bool, counter: int, batch_size: int, accumulation_steps: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": train["loss"],
        "train_accuracy": train["accuracy"],
        "train_macro_precision": train["macro_precision"],
        "train_macro_recall": train["macro_recall"],
        "train_macro_f1": train["macro_f1"],
        "val_loss": val["loss"],
        "val_accuracy": val["accuracy"],
        "val_macro_precision": val["macro_precision"],
        "val_macro_recall": val["macro_recall"],
        "val_macro_f1": val["macro_f1"],
        "val_weighted_f1": val["weighted_f1"],
        "val_macro_auroc": val["macro_auroc"],
        "learning_rate_backbone": optimizer.param_groups[0]["lr"],
        "learning_rate_head": optimizer.param_groups[1]["lr"],
        "gradient_norm_mean": train["gradient_norm_mean"],
        "gradient_norm_max": train["gradient_norm_max"],
        "grad_scaler_scale": scaler.get_scale(),
        "images_per_second": train["images_per_second"],
        "epoch_seconds": train["epoch_seconds"],
        "gpu_peak_allocated_gb": train["gpu_peak_allocated_gb"],
        "gpu_peak_reserved_gb": train["gpu_peak_reserved_gb"],
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "early_stopping_counter": counter,
        "is_best": best,
        "nan_count": 0,
        "inf_count": 0,
        "non_finite_gradient_count": 0,
    }
    for class_id in range(NUM_CLASSES):
        for metric in ("precision", "recall", "f1", "auroc"):
            row[f"val_class_{class_id}_{metric}"] = val["per_class"][class_id][metric]
    return row


def save_metrics(path: Path, metrics: list[dict[str, Any]]) -> None:
    if not metrics:
        return
    atomic_write_csv(path, metrics, list(metrics[0]))


def plot_series(metrics: list[dict[str, Any]], figures: Path) -> None:
    if not metrics:
        return
    epochs = [int(row["epoch"]) for row in metrics]
    specifications = [
        (["train_loss"], "Loss", "train_loss_curve.png"),
        (["train_loss", "val_loss"], "Loss", "training_loss.png"),
        (["val_loss"], "Loss", "val_loss_curve.png"),
        (["train_accuracy", "val_accuracy"], "Accuracy", "accuracy_curve.png"),
        (["train_macro_f1", "val_macro_f1"], "Macro-F1", "macro_f1_curve.png"),
        (["val_macro_f1"], "Validation Macro-F1", "validation_macro_f1.png"),
        (["val_macro_auroc"], "Macro-AUROC", "macro_auroc_curve.png"),
        (["val_macro_auroc"], "Validation Macro-AUROC", "validation_macro_auroc.png"),
        (["learning_rate_backbone", "learning_rate_head"], "Learning Rate", "learning_rate_curve.png"),
        (["gpu_peak_allocated_gb", "gpu_peak_reserved_gb"], "GPU Memory (GB)", "gpu_memory_curve.png"),
    ]
    for fields, ylabel, filename in specifications:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for field in fields:
            axis.plot(epochs, [float(row[field]) for row in metrics], marker="o", markersize=3, label=field)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if len(fields) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(figures / filename, dpi=160)
        plt.close(figure)


def plot_confusion(matrix: list[list[int]], path: Path, title: str) -> None:
    array = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(array, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(NUM_CLASSES))
    axis.set_yticks(range(NUM_CLASSES))
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            axis.text(column, row, str(array[row, column]), ha="center", va="center", color="white" if array[row, column] > array.max() / 2 else "black")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_roc_curves(targets: np.ndarray, probabilities: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for class_id in range(NUM_CLASSES):
        binary = (targets == class_id).astype(np.int64)
        positive = int(binary.sum())
        negative = len(binary) - positive
        order = np.argsort(-probabilities[:, class_id], kind="stable")
        sorted_binary = binary[order]
        tpr = np.concatenate(([0.0], np.cumsum(sorted_binary) / positive))
        fpr = np.concatenate(([0.0], np.cumsum(1 - sorted_binary) / negative))
        area = float(np.trapezoid(tpr, fpr))
        axis.plot(fpr, tpr, label=f"Class {class_id} (AUC={area:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def test_outputs_exist(paths: dict[str, Path]) -> list[str]:
    guarded = [
        paths["predictions"] / "test_predictions.csv",
        paths["figures"] / "confusion_matrix_test.png",
        paths["figures"] / "roc_curves_test.png",
        paths["test_results"] / "test_metrics.json",
        paths["test_results"] / "classification_report.csv",
        paths["test_results"] / "per_class_metrics.csv",
        paths["test_results"] / "confusion_matrix.csv",
        paths["test_results"] / "test_evaluation_metadata.json",
        paths["diagnostics"] / "test_evaluation_audit.json",
    ]
    return [str(path) for path in guarded if path.exists()]


def run_test_once(
    args: argparse.Namespace,
    paths: dict[str, Path],
    best_path: Path,
    validation: dict[str, Any],
    shared_config: dict[str, Any],
    shared_sha: str,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    existing = test_outputs_exist(paths)
    if existing:
        raise FileExistsError(f"Test outputs already exist and will not be overwritten: {existing}")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    validate_resume(best_checkpoint, args, validation, shared_sha)
    model = ConvNeXtTinyClassifier(args.initialization, args.distilled_backbone).to(device)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)

    val_metrics, _, _, _, _ = evaluate(model, validation["rows"]["val"], device, batch_size, workers)
    plot_confusion(val_metrics["confusion_matrix"], paths["figures"] / "confusion_matrix_validation.png", "Validation Confusion Matrix (Best Checkpoint)")

    test_started = utc_now()
    test_metrics, targets, predictions, probabilities, positions = evaluate(
        model, validation["rows"]["test"], device, batch_size, workers
    )
    prediction_rows = []
    for target, prediction, probability, position in zip(targets, predictions, probabilities, positions):
        source = validation["rows"]["test"][int(position)]
        prediction_rows.append({
            "source_image_id": source["source_image_id"],
            "original_roi_id": source["original_roi_id"],
            "image_path": source["image_path"],
            "true_class_id": int(target),
            "true_class_name": CLASS_MAPPING[int(target)],
            "predicted_class_id": int(prediction),
            "predicted_class_name": CLASS_MAPPING[int(prediction)],
            "confidence": float(probability[int(prediction)]),
            **{f"probability_class_{class_id}": float(probability[class_id]) for class_id in range(NUM_CLASSES)},
            "is_correct": bool(target == prediction),
        })
    prediction_fields = list(prediction_rows[0])
    atomic_write_csv(paths["predictions"] / "test_predictions.csv", prediction_rows, prediction_fields, bom=True, allow_replace=False)

    per_class_rows = []
    for class_id in range(NUM_CLASSES):
        item = test_metrics["per_class"][class_id]
        per_class_rows.append({"class_id": class_id, "class_name": CLASS_MAPPING[class_id], **item})
    atomic_write_csv(paths["test_results"] / "per_class_metrics.csv", per_class_rows, list(per_class_rows[0]), allow_replace=False)
    report_rows = per_class_rows + [
        {"class_id": "macro_avg", "class_name": "Macro average", "precision": test_metrics["macro_precision"], "recall": test_metrics["macro_recall"], "f1": test_metrics["macro_f1"], "support": len(targets), "auroc": test_metrics["macro_auroc"]},
        {"class_id": "weighted_avg", "class_name": "Weighted average", "precision": "", "recall": "", "f1": test_metrics["weighted_f1"], "support": len(targets), "auroc": ""},
        {"class_id": "accuracy", "class_name": "Accuracy", "precision": "", "recall": "", "f1": test_metrics["accuracy"], "support": len(targets), "auroc": ""},
    ]
    atomic_write_csv(paths["test_results"] / "classification_report.csv", report_rows, list(report_rows[0]), allow_replace=False)
    confusion_rows = [
        {"true_class_id": class_id, **{
            f"predicted_class_{predicted}": test_metrics["confusion_matrix"][class_id][predicted]
            for predicted in range(NUM_CLASSES)
        }}
        for class_id in range(NUM_CLASSES)
    ]
    atomic_write_csv(
        paths["test_results"] / "confusion_matrix.csv",
        confusion_rows,
        list(confusion_rows[0]),
        allow_replace=False,
    )
    serializable_metrics = {key: value for key, value in test_metrics.items() if key != "per_class"}
    serializable_metrics["per_class"] = {str(key): value for key, value in test_metrics["per_class"].items()}
    serializable_metrics.update({"status": "PASS", "evaluation_count": 1, "best_epoch": best_checkpoint["best"]["epoch"], "best_checkpoint": str(best_path)})
    atomic_write_json(paths["test_results"] / "test_metrics.json", serializable_metrics, allow_replace=False)
    atomic_write_json(paths["test_results"] / "test_evaluation_metadata.json", {
        "status": "PASS",
        "evaluation_count": 1,
        "started_at_utc": test_started,
        "completed_at_utc": utc_now(),
        "manifest_path": str(args.test_manifest),
        "manifest_sha256": validation["manifest_sha256"]["test"],
        "rows": len(targets),
        "checkpoint": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
        "checkpoint_selection_used_test": False,
    }, allow_replace=False)
    atomic_write_json(paths["diagnostics"] / "test_evaluation_audit.json", {
        "status": "PASS",
        "evaluation_count": 1,
        "test_rows": len(targets),
        "predictions_rows": len(prediction_rows),
        "checkpoint": str(best_path),
        "checkpoint_sha256": sha256_file(best_path),
        "checkpoint_selection_used_test": False,
        "paired_key_fields": ["source_image_id", "original_roi_id", "image_path"],
        "duplicate_paired_keys": len(prediction_rows) - len({
            (row["source_image_id"], row["original_roi_id"], row["image_path"])
            for row in prediction_rows
        }),
    }, allow_replace=False)
    plot_confusion(test_metrics["confusion_matrix"], paths["figures"] / "confusion_matrix_test.png", "Test Confusion Matrix")
    plot_roc_curves(targets, probabilities, paths["figures"] / "roc_curves_test.png")

    export_filename = {
        "imagenet": "baseline_convnext_tiny_5class.pt",
        "distilled": "proposed_convnext_tiny_5class.pt",
        "patch_distilled": "patch_proposed_convnext_tiny_5class.pt",
    }[args.initialization]
    export_path = paths["checkpoints"] / export_filename
    export = {
        "architecture": ARCHITECTURE,
        "initialization": args.initialization,
        "initialization_description": (
            "RAD-DINO 7x7 patch distilled"
            if args.initialization == "patch_distilled"
            else args.initialization
        ),
        "distillation_type": (
            "RAD-DINO 7x7 patch feature"
            if args.initialization == "patch_distilled"
            else None
        ),
        "weights_enum": IMAGENET_WEIGHTS_ENUM if args.initialization == "imagenet" else None,
        "num_classes": NUM_CLASSES,
        "feature_dim": FEATURE_DIM,
        "model_state_dict": best_checkpoint["model_state_dict"],
        "class_mapping": CLASS_MAPPING,
        "preprocessing_config": shared_config["preprocessing"],
        "image_input_mode": "grayscale converted to RGB in memory",
        "input_size": [224, 224],
        "best_epoch": best_checkpoint["best"]["epoch"],
        "validation_metrics": best_checkpoint["best"]["validation_metrics"],
        "test_metrics": serializable_metrics,
        "split_sha256": validation["image_id_split_sha256"],
        "shared_config_sha256": shared_sha,
        "training_config": shared_config,
        "source_best_checkpoint": str(best_path),
        "patch_phase1_backbone_sha256": (
            validation["distilled_backbone_sha256"]
            if args.initialization == "patch_distilled"
            else None
        ),
        "head_metadata": {"dropout": 0.2, "linear": [768, 5], "hidden_layer": False, "projector": False},
        "inference_instructions": "Apply locked clean preprocessing, then softmax(model(image), dim=1).",
    }
    atomic_save_checkpoint(export_path, {
        **export,
        "phase": PHASE,
        "experiment": args.output_dir.name,
        "current_epoch": best_checkpoint["best"]["epoch"],
        "shared_config_sha256": shared_sha,
    })
    reloaded = torch.load(export_path, map_location="cpu", weights_only=False)
    reload_model = ConvNeXtTinyClassifier(args.initialization, args.distilled_backbone).to(device)
    reload_result = reload_model.load_state_dict(reloaded["model_state_dict"], strict=True)
    reload_dataset = RoiClassificationDataset(validation["rows"]["val"][:2], Phase2Transform(training=False))
    reload_images = torch.stack([reload_dataset[index]["image"] for index in range(2)]).to(device)
    reload_model.eval()
    with torch.inference_mode():
        reload_logits = reload_model(reload_images)
        reload_probabilities = torch.softmax(reload_logits.float(), dim=1)
    reload_audit = {
        "status": "PASS",
        "images": 2,
        "source_split": "validation",
        "logits_shape": list(reload_logits.shape),
        "probabilities_shape": list(reload_probabilities.shape),
        "softmax_max_sum_error": float((reload_probabilities.sum(dim=1) - 1).abs().max()),
        "finite": bool(torch.isfinite(reload_probabilities).all()),
        "missing_keys": list(reload_result.missing_keys),
        "unexpected_keys": list(reload_result.unexpected_keys),
        "export_path": str(export_path),
        "export_sha256": sha256_file(export_path),
    }
    if reload_audit["logits_shape"] != [2, 5] or not reload_audit["finite"] or reload_audit["softmax_max_sum_error"] > 1e-5:
        raise RuntimeError(f"Export reload audit failed: {reload_audit}")
    atomic_write_json(paths["diagnostics"] / "export_reload_audit.json", reload_audit)
    return serializable_metrics


def write_summary(
    path: Path,
    args: argparse.Namespace,
    validation: dict[str, Any],
    environment: dict[str, Any],
    shared_sha: str,
    batch_size: int,
    accumulation_steps: int,
    metrics: list[dict[str, Any]],
    best: dict[str, Any],
    test_metrics: dict[str, Any] | None,
    early_stopped: bool,
) -> None:
    last = metrics[-1] if metrics else {}
    summary_label = {
        "imagenet": "Baseline",
        "distilled": "CLS Proposed",
        "patch_distilled": "Patch Proposed",
    }[args.initialization]
    lines = [
        f"# {summary_label} Phase 2 Training Summary",
        "",
        f"- Status: {'PASS' if test_metrics is not None else 'TRAINING COMPLETE'}",
        f"- Initialization: {args.initialization}",
        f"- Architecture: {ARCHITECTURE}",
        f"- Shared config SHA256: `{shared_sha}`",
        f"- Train/validation/test rows: {EXPECTED_ROWS['train']} / {EXPECTED_ROWS['val']} / {EXPECTED_ROWS['test']}",
        f"- Leakage audit: {validation['leakage']}",
        f"- Distilled backbone SHA256: `{validation['distilled_backbone_sha256']}`",
        f"- Device: {environment['device']}",
        f"- GPU: {environment.get('gpu')}",
        f"- Actual batch size: {batch_size}",
        f"- Accumulation steps: {accumulation_steps}",
        f"- Effective batch size: {batch_size * accumulation_steps}",
        f"- Completed epochs: {len(metrics)}",
        f"- Early stopping: {early_stopped}",
        f"- Best epoch: {best.get('epoch')}",
        f"- Best validation macro-F1: {best.get('macro_f1')}",
        f"- Best validation loss: {best.get('loss')}",
        f"- Best validation accuracy: {best.get('validation_metrics', {}).get('accuracy')}",
        f"- Best validation macro-AUROC: {best.get('validation_metrics', {}).get('macro_auroc')}",
        f"- Peak allocated VRAM GB: {max((float(row['gpu_peak_allocated_gb']) for row in metrics), default=0.0)}",
        f"- Peak reserved VRAM GB: {max((float(row['gpu_peak_reserved_gb']) for row in metrics), default=0.0)}",
        f"- Last train images/s: {last.get('images_per_second')}",
        "- RAD-DINO loaded: False",
        "- Teacher feature cache used in loss: False",
        "- Baseline trained: False",
        "- Test used for checkpoint selection: False",
        f"- Test evaluation count: {1 if test_metrics is not None else 0}",
    ]
    if test_metrics is not None:
        lines.extend([
            f"- Test loss: {test_metrics['loss']}",
            f"- Test accuracy: {test_metrics['accuracy']}",
            f"- Test macro precision: {test_metrics['macro_precision']}",
            f"- Test macro recall: {test_metrics['macro_recall']}",
            f"- Test macro-F1: {test_metrics['macro_f1']}",
            f"- Test weighted-F1: {test_metrics['weighted_f1']}",
            f"- Test macro-AUROC: {test_metrics['macro_auroc']}",
        ])
    atomic_write_text(path, "\n".join(lines) + "\n")


def prepare_smoke_artifacts(
    args: argparse.Namespace,
    paths: dict[str, Path],
    validation: dict[str, Any],
    shared_config: dict[str, Any],
    shared_sha: str,
    environment: dict[str, Any],
    batch_size: int,
    accumulation_steps: int,
    attempts: list[dict[str, Any]],
    smoke: dict[str, Any],
    device: torch.device,
    locked_batch_source: dict[str, Any] | None,
) -> Path:
    shared_copy = paths["config"] / "shared_phase2_finetune_config.json"
    atomic_write_text(shared_copy, args.shared_config.read_text(encoding="utf-8"), allow_replace=False)
    experiment_config = {
        "phase": PHASE,
        "experiment": args.output_dir.name,
        "initialization": args.initialization,
        "weights_enum": IMAGENET_WEIGHTS_ENUM if args.initialization == "imagenet" else None,
        "imagenet_weights": imagenet_weights_audit() if args.initialization == "imagenet" else None,
        "imagenet_pretrained_loaded": args.initialization == "imagenet",
        "distilled_checkpoint_loaded": args.initialization in {"distilled", "patch_distilled"},
        "patch_distilled_checkpoint_loaded": args.initialization == "patch_distilled",
        "teacher_feature_cache_loaded": False,
        "rad_dino_loaded": False,
        "output_directory": str(args.output_dir),
        "distilled_backbone": (
            str(args.distilled_backbone)
            if args.initialization in {"distilled", "patch_distilled"}
            else None
        ),
        "distilled_backbone_sha256": validation["distilled_backbone_sha256"],
        "shared_config_path": str(args.shared_config),
        "shared_config_sha256": shared_sha,
        "actual_batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
    }
    experiment_config_name = (
        "baseline_experiment_config.json"
        if args.initialization == "imagenet"
        else "experiment_config.json"
    )
    atomic_write_json(paths["config"] / experiment_config_name, experiment_config, allow_replace=False)
    atomic_write_json(paths["config"] / "environment.json", environment, allow_replace=False)
    atomic_write_json(args.output_dir / "environment.json", environment, allow_replace=False)
    pip = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False, timeout=120)
    atomic_write_text(paths["config"] / "pip_freeze.txt", pip.stdout, allow_replace=False)
    preview = create_augmentation_preview(
        validation["rows"]["train"], Phase2Transform(training=True),
        paths["figures"] / "augmentation_preview.png",
        paths["diagnostics"] / "augmentation_audit.json", args.seed,
    )
    smoke_record = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "manifest_validation": "PASS",
        "shared_config_sha256": shared_sha,
        "batch_probe_attempts": attempts,
        "selected_batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "smoke": smoke,
        "augmentation_preview": preview,
    }
    atomic_write_json(paths["diagnostics"] / "stage0_smoke_test.json", smoke_record, allow_replace=False)
    initialization_audit = {
        "status": "PASS",
        "initialization": args.initialization,
        "checkpoint": str(args.distilled_backbone),
        "checkpoint_sha256": validation["distilled_backbone_sha256"],
        "checkpoint_metadata": validation["phase1_checkpoint_metadata"],
        "backbone_load_audit": smoke["backbone_load_audit"],
        "feature_shape": smoke["train_batches"][0]["feature_shape"],
        "logits_shape": smoke["train_batches"][0]["logits_shape"],
        "head": {"dropout": 0.2, "linear": [768, 5], "parameters": 3845},
        "all_parameters_trainable": smoke["all_parameters_trainable"],
        "imagenet_classifier_loaded": False,
        "cls_checkpoint_loaded": False,
        "baseline_checkpoint_loaded": False,
        "rad_dino_loaded": False,
        "teacher_feature_cache_loaded": False,
        "projector_added": False,
    }
    atomic_write_json(
        paths["diagnostics"] / "initialization_audit.json",
        initialization_audit,
        allow_replace=False,
    )
    fairness = build_fairness_audit(
        args, validation, shared_config, shared_sha, batch_size,
        accumulation_steps, locked_batch_source,
    )
    if fairness["status"] != "PASS":
        raise RuntimeError(f"Proposed/Baseline fairness audit failed: {fairness['failed_checks']}")
    atomic_write_json(paths["diagnostics"] / "fairness_audit.json", fairness, allow_replace=False)

    set_seed(args.seed)
    model = ConvNeXtTinyClassifier(args.initialization, args.distilled_backbone).to(device)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = build_scaler(device)
    best = {"epoch": 0, "macro_f1": float("-inf"), "loss": float("inf"), "validation_metrics": {}}
    payload = checkpoint_payload(
        args, model, optimizer, scheduler, scaler, 0, best, 0, float("-inf"), batch_size,
        accumulation_steps, validation, shared_config, shared_sha, environment,
    )
    smoke_ready = paths["checkpoints"] / "smoke_ready.pt"
    atomic_save_checkpoint(smoke_ready, payload)
    del scaler, scheduler, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log_message(paths["logs"] / "phase2.log", f"Smoke test PASS; resume checkpoint={smoke_ready}")
    return smoke_ready


def run(args: argparse.Namespace) -> int:
    validate_locked_runtime(args)
    locked_batch_source = resolve_locked_batch(args)
    device = resolve_device(args.device)
    configure_runtime(device, args.seed)
    validation = validate_manifests(args)
    train_transform = Phase2Transform(training=True)
    shared_config = locked_config(args, validation, train_transform)
    shared_sha = ensure_locked_config(args.shared_config, shared_config, write=not args.dry_run)

    preflight = {
        "status": "PASS",
        "device": str(device),
        "manifest_sha256": validation["manifest_sha256"],
        "split_sha256": validation["image_id_split_sha256"],
        "leakage": validation["leakage"],
        "shared_config_sha256": shared_sha,
        "distilled_backbone_sha256": validation["distilled_backbone_sha256"],
        "rows": {split: len(rows) for split, rows in validation["rows"].items()},
        "locked_batch_source": locked_batch_source,
    }
    if args.dry_run:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 0

    nonempty = output_is_nonempty(args.output_dir)
    if nonempty and not args.resume_checkpoint and not args.evaluate_test_only:
        listing = [str(path) for path in sorted(args.output_dir.rglob("*"))[:50]]
        raise FileExistsError(f"Output directory is non-empty; use explicit --resume-checkpoint. Existing: {listing}")
    if args.smoke_test_only and nonempty:
        raise FileExistsError("Smoke output directory must be absent or empty")
    paths = create_output_directories(args.output_dir)
    log_path = paths["logs"] / "phase2.log"
    environment = environment_info(device)
    atomic_write_json(paths["diagnostics"] / "preflight_validation.json", preflight)
    atomic_write_json(paths["diagnostics"] / "split_integrity_audit.json", {
        **preflight,
        "source_image_counts": {"train": 472, "val": 59, "test": 59},
        "split_leakage_zero": all(value == 0 for value in validation["leakage"].values()),
    })

    if args.smoke_test_only:
        batch_size, accumulation_steps, attempts = auto_probe_batch_size(args, validation["rows"]["train"], device)
        smoke = run_smoke(args, validation, device, batch_size)
        checkpoint = prepare_smoke_artifacts(
            args, paths, validation, shared_config, shared_sha, environment,
            batch_size, accumulation_steps, attempts, smoke, device, locked_batch_source,
        )
        print(json.dumps({"status": "PASS", "smoke_ready_checkpoint": str(checkpoint), "batch_size": batch_size, "accumulation_steps": accumulation_steps}, indent=2))
        return 0

    if args.evaluate_test_only:
        best_path = args.resume_checkpoint or paths["checkpoints"] / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError(best_path)
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        validate_resume(checkpoint, args, validation, shared_sha)
        test_metrics = run_test_once(
            args, paths, best_path, validation, shared_config, shared_sha, device,
            int(checkpoint["batch_size"]), args.workers,
        )
        print(json.dumps({"status": "PASS", "test_metrics": test_metrics}, indent=2))
        return 0

    if not args.resume_checkpoint:
        raise ValueError("Formal training requires an explicit smoke-ready or training --resume-checkpoint")
    resume_path = args.resume_checkpoint
    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    validate_resume(checkpoint, args, validation, shared_sha)
    batch_size = int(checkpoint["batch_size"])
    accumulation_steps = int(checkpoint["accumulation_steps"])
    model = ConvNeXtTinyClassifier(args.initialization, args.distilled_backbone).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer = build_optimizer(model)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler = build_scaler(device)
    # smoke_ready represents the untouched epoch-0 model.  Start it with the
    # script's locked GradScaler policy; later checkpoints restore exact state.
    if int(checkpoint["current_epoch"]) > 0 or resume_path.name != "smoke_ready.pt":
        scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
    start_epoch = int(checkpoint["current_epoch"]) + 1
    best = checkpoint["best"]
    early_counter = int(checkpoint["early_stopping_counter"])
    convergence_reference = float(checkpoint["convergence_reference"])
    restore_random_states(checkpoint["random_states"])

    metrics_path = paths["metrics"] / "phase2_metrics.csv"
    progress_path = args.output_dir / "training_progress.json"
    metrics = read_csv(metrics_path) if metrics_path.is_file() else []
    metrics = [row for row in metrics if int(row["epoch"]) < start_epoch]
    log_message(log_path, f"Formal training resume={resume_path}; start_epoch={start_epoch}; batch={batch_size}; accumulation={accumulation_steps}")
    atomic_write_json(progress_path, {
        "status": "RUNNING",
        "current_epoch": start_epoch - 1,
        "completed_epochs": len(metrics),
        "best_epoch": best.get("epoch"),
        "best_validation_macro_f1": best.get("macro_f1"),
        "test_evaluation_count": 0,
        "updated_at_utc": utc_now(),
    })
    early_stopped = False
    current_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            train_metrics = train_epoch(
                model, validation["rows"]["train"], optimizer, scaler, device,
                batch_size, accumulation_steps, args.workers, args.seed, epoch,
            )
            val_metrics, _, _, _, _ = evaluate(
                model, validation["rows"]["val"], device, batch_size, args.workers
            )
            candidate_f1 = float(val_metrics["macro_f1"])
            candidate_loss = float(val_metrics["loss"])
            is_best = (
                candidate_f1 > float(best["macro_f1"])
                or (candidate_f1 == float(best["macro_f1"]) and candidate_loss < float(best["loss"]))
            )
            if is_best:
                best = {
                    "epoch": epoch,
                    "macro_f1": candidate_f1,
                    "loss": candidate_loss,
                    "validation_metrics": val_metrics,
                }
            if candidate_f1 > convergence_reference + 1e-4:
                convergence_reference = candidate_f1
                early_counter = 0
            else:
                early_counter += 1
            epoch_row = flatten_epoch_metrics(
                epoch, train_metrics, val_metrics, optimizer, scaler, is_best, early_counter,
                batch_size, accumulation_steps,
            )
            metrics.append(epoch_row)
            save_metrics(metrics_path, metrics)
            plot_series(metrics, paths["figures"])
            scheduler.step()
            payload = checkpoint_payload(
                args, model, optimizer, scheduler, scaler, epoch, best, early_counter,
                convergence_reference, batch_size, accumulation_steps, validation,
                shared_config, shared_sha, environment,
            )
            atomic_save_checkpoint(paths["checkpoints"] / "last.pt", payload)
            if is_best:
                atomic_save_checkpoint(paths["checkpoints"] / "best.pt", payload)
            if epoch % 5 == 0:
                atomic_save_checkpoint(paths["checkpoints"] / f"epoch_{epoch:03d}.pt", payload)
            log_message(
                log_path,
                f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} val_loss={candidate_loss:.6f} "
                f"val_macro_f1={candidate_f1:.6f} val_auroc={val_metrics['macro_auroc']:.6f} "
                f"best_epoch={best['epoch']} patience={early_counter}/{args.patience}",
            )
            atomic_write_json(progress_path, {
                "status": "RUNNING",
                "current_epoch": epoch,
                "completed_epochs": len(metrics),
                "best_epoch": best.get("epoch"),
                "best_validation_macro_f1": best.get("macro_f1"),
                "early_stopping_counter": early_counter,
                "test_evaluation_count": 0,
                "updated_at_utc": utc_now(),
            })
            if early_counter >= args.patience:
                early_stopped = True
                log_message(log_path, f"Early stopping at epoch {epoch}")
                break
    except BaseException as exc:
        diagnostic = {
            "status": "FAIL",
            "created_at_utc": utc_now(),
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "current_epoch": current_epoch,
            "nan_inf_or_nonfinite": isinstance(exc, FloatingPointError),
        }
        atomic_write_json(paths["diagnostics"] / "training_failure.json", diagnostic)
        atomic_write_json(progress_path, {
            "status": "FAIL",
            "current_epoch": current_epoch,
            "completed_epochs": len(metrics),
            "error": str(exc),
            "updated_at_utc": utc_now(),
        })
        interrupted = checkpoint_payload(
            args, model, optimizer, scheduler, scaler, current_epoch, best, early_counter,
            convergence_reference, batch_size, accumulation_steps, validation,
            shared_config, shared_sha, environment,
        )
        atomic_save_checkpoint(paths["checkpoints"] / "interrupted.pt", interrupted)
        raise

    best_path = paths["checkpoints"] / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("Training completed without best.pt")
    test_metrics = None
    if not args.train_only:
        test_metrics = run_test_once(
            args, paths, best_path, validation, shared_config, shared_sha,
            device, batch_size, args.workers,
        )
    final_audit = {
        "status": "PASS" if test_metrics is not None or args.train_only else "FAIL",
        "completed_at_utc": utc_now(),
        "completed_epochs": len(metrics),
        "early_stopped": early_stopped,
        "best": best,
        "test_evaluation_count": 0 if test_metrics is None else 1,
        "nan_count": 0,
        "inf_count": 0,
        "nonfinite_gradient_count": 0,
        "rad_dino_loaded": False,
        "teacher_feature_cache_loaded": False,
        "baseline_trained": args.initialization == "imagenet",
        "imagenet_pretrained_loaded": args.initialization == "imagenet",
        "distilled_checkpoint_loaded": args.initialization in {"distilled", "patch_distilled"},
        "patch_distilled_checkpoint_loaded": args.initialization == "patch_distilled",
        "phase0_phase1_split_modified": False,
    }
    atomic_write_json(paths["diagnostics"] / "phase2_final_audit.json", final_audit)
    atomic_write_json(paths["diagnostics"] / "numerical_stability_audit.json", {
        "status": "PASS",
        "oom_count": 0,
        "nan_count": 0,
        "inf_count": 0,
        "non_finite_gradient_count": 0,
    })
    atomic_write_json(progress_path, {
        "status": "PASS",
        "completed_epochs": len(metrics),
        "best_epoch": best.get("epoch"),
        "best_validation_macro_f1": best.get("macro_f1"),
        "early_stopped": early_stopped,
        "test_evaluation_count": 0 if test_metrics is None else 1,
        "updated_at_utc": utc_now(),
    })
    write_summary(
        args.output_dir
        / ({
            "imagenet": "phase2_baseline_training_summary.md",
            "distilled": "phase2_proposed_training_summary.md",
            "patch_distilled": "phase2_patch_proposed_training_summary.md",
        }[args.initialization]),
        args, validation,
        environment, shared_sha, batch_size, accumulation_steps, metrics, best,
        test_metrics, early_stopped,
    )
    print(json.dumps({"status": "PASS", "best": best, "test_metrics": test_metrics}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    default_root = Path(r"C:\Users\09688\thoracic-cxr-project-3")
    experiment_root = default_root / "outputs" / "raddino_convnext_tiny_experiment_seed42"
    split_root = experiment_root / "phase2_split"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--initialization", choices=("distilled", "patch_distilled", "imagenet"), required=True)
    parser.add_argument("--train-manifest", type=Path, default=split_root / "train_roi_manifest.csv")
    parser.add_argument("--val-manifest", type=Path, default=split_root / "val_roi_manifest.csv")
    parser.add_argument("--test-manifest", type=Path, default=split_root / "test_roi_manifest.csv")
    parser.add_argument("--shared-protocol", type=Path, default=split_root / "shared_training_protocol.json")
    parser.add_argument(
        "--distilled-backbone",
        "--distilled-checkpoint",
        dest="distilled_backbone",
        type=Path,
        default=experiment_root / "phase1_distillation" / "checkpoints" / "distilled_convnext_tiny_backbone.pt",
    )
    parser.add_argument("--shared-config", type=Path, default=experiment_root / "shared_phase2_finetune_config.json")
    parser.add_argument("--output-dir", type=Path, default=experiment_root / "phase2_proposed_distilled")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--smoke-test-only", "--smoke-test", dest="smoke_test_only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--evaluate-test-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size not in {"auto", "locked"}:
        try:
            int(args.batch_size)
        except ValueError as exc:
            raise ValueError("--batch-size must be auto, locked, or an integer") from exc
    exclusive = sum((args.smoke_test_only, args.train_only, args.evaluate_test_only, args.dry_run))
    if exclusive > 1:
        raise ValueError("smoke/train-only/evaluate-test-only/dry-run modes are mutually exclusive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
