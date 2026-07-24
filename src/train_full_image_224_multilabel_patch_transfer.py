#!/usr/bin/env python
"""Fine-tune full images from the ROI Patch Proposed transfer initialization."""

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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL
import sklearn
import torch
import torchvision
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_tiny
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from infer_patch_proposed_single_roi import load_export_model


CLASS_MAPPING = {
    0: "Aortic enlargement",
    1: "Cardiomegaly",
    2: "Pleural thickening",
    3: "Pulmonary fibrosis",
    4: "Pleural effusion",
}
LABEL_FIELDS = [
    "label_0_aortic_enlargement",
    "label_1_cardiomegaly",
    "label_2_pleural_thickening",
    "label_3_pulmonary_fibrosis",
    "label_4_pleural_effusion",
]
EXPECTED_CHECKPOINT_SHA256 = "8a68d68b901d721c63a38b5e75ee3291a8c06d13195572d20f29fd34a56485e5"
FEATURE_DIM = 768
NUM_CLASSES = 5
TRAIN_FIELDS = [
    "epoch",
    "train_loss",
    "backbone_lr",
    "head_lr",
    "grad_scale",
    "epoch_seconds",
]
VAL_FIELDS = [
    "epoch",
    "validation_loss",
    "macro_auroc",
    "micro_auroc",
    "macro_average_precision",
    "micro_average_precision",
    "is_best",
    "early_stopping_counter",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_atomic(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(text, encoding="utf-8")
    replace_atomic(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    replace_atomic(temporary, path)


def atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    check = torch.load(temporary, map_location="cpu", weights_only=False)
    for key in ("architecture", "task", "current_epoch", "test_evaluation_count"):
        if check.get(key) != payload.get(key):
            raise RuntimeError(f"Checkpoint atomic reload failed for {key}")
    replace_atomic(temporary, path)


def atomic_save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    replace_atomic(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class FullImageTransform:
    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = TF.pil_to_tensor(image).float().div_(255.0)
        return TF.normalize(tensor, self.mean, self.std)

    def config(self) -> dict[str, Any]:
        return {
            "source": "complete raw/full chest X-ray",
            "convert_mode": "RGB",
            "resize": [self.image_size, self.image_size],
            "interpolation": "BILINEAR",
            "antialias": True,
            "center_crop": False,
            "random_resized_crop": False,
            "bbox": False,
            "roi_crop": False,
            "augmentation": False,
            "to_tensor": True,
            "mean": self.mean,
            "std": self.std,
            "output_shape": [3, self.image_size, self.image_size],
        }


class FullImageMultilabelDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform: FullImageTransform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = Path(row["full_image_path"])
        try:
            with Image.open(path) as image:
                image.load()
                tensor = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to read full image {path}: {exc}") from exc
        label = torch.tensor([float(row[field]) for field in LABEL_FIELDS], dtype=torch.float32)
        return {"image": tensor, "label": label, "row_index": index}


class FullImageMultilabelConvNeXt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = convnext_tiny(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.final_norm = base.classifier[0]
        self.flatten = base.classifier[1]
        self.dropout = nn.Dropout(p=0.2)
        self.multilabel_head = nn.Linear(FEATURE_DIM, NUM_CLASSES)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = self.avgpool(features)
        features = self.final_norm(features)
        return self.flatten(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.multilabel_head(self.dropout(self.extract_features(images)))


def initialize_from_roi_export(
    checkpoint_path: Path, device: torch.device
) -> tuple[FullImageMultilabelConvNeXt, dict[str, Any]]:
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("ROI Patch Proposed checkpoint SHA256 mismatch")
    source_model, checkpoint, strict_audit, load_seconds = load_export_model(
        checkpoint_path, torch.device("cpu")
    )
    source_state = checkpoint["model_state_dict"]
    source_head = {
        "classifier.weight": source_state["classifier.weight"].clone(),
        "classifier.bias": source_state["classifier.bias"].clone(),
    }
    model = FullImageMultilabelConvNeXt()
    feature_result = model.features.load_state_dict(source_model.features.state_dict(), strict=True)
    norm_result = model.final_norm.load_state_dict(source_model.final_norm.state_dict(), strict=True)
    transferred_keys = sorted(
        [key for key in source_state if key.startswith("features.") or key.startswith("final_norm.")]
    )
    feature_equal = all(
        torch.equal(model.features.state_dict()[key], source_model.features.state_dict()[key])
        for key in model.features.state_dict()
    )
    norm_equal = all(
        torch.equal(model.final_norm.state_dict()[key], source_model.final_norm.state_dict()[key])
        for key in model.final_norm.state_dict()
    )
    head_reinitialized = not torch.equal(model.multilabel_head.weight, source_head["classifier.weight"])
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.to(device)
    audit = {
        "status": "PASS",
        "experiment_description": "ROI Patch Proposed transfer initialization -> Full-image multilabel fine-tuning",
        "not_full_image_patch_distillation": True,
        "not_full_image_rad_dino_teacher_training": True,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_state_dict_key_count": len(source_state),
        "source_strict_load": strict_audit["strict_load"],
        "source_missing_keys": strict_audit["missing_keys"],
        "source_unexpected_keys": strict_audit["unexpected_keys"],
        "source_load_seconds": load_seconds,
        "transferred_state_dict_key_count": len(transferred_keys),
        "transferred_state_dict_keys": transferred_keys,
        "feature_load_missing_keys": list(feature_result.missing_keys),
        "feature_load_unexpected_keys": list(feature_result.unexpected_keys),
        "final_norm_load_missing_keys": list(norm_result.missing_keys),
        "final_norm_load_unexpected_keys": list(norm_result.unexpected_keys),
        "feature_tensors_exactly_equal": feature_equal,
        "final_norm_tensors_exactly_equal": norm_equal,
        "discarded_old_head_keys": ["classifier.weight", "classifier.bias"],
        "old_roi_head_loaded_into_new_head": False,
        "new_head": "Dropout(0.2) -> Linear(768,5)",
        "new_head_parameter_count": sum(parameter.numel() for parameter in model.multilabel_head.parameters()),
        "new_head_reinitialized": head_reinitialized,
        "all_model_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
        "rad_dino_loaded": False,
        "teacher_cache_loaded": False,
    }
    required = [
        audit["source_strict_load"],
        not audit["source_missing_keys"],
        not audit["source_unexpected_keys"],
        len(transferred_keys) == 180,
        feature_equal,
        norm_equal,
        head_reinitialized,
        audit["new_head_parameter_count"] == 3845,
    ]
    if not all(required):
        audit["status"] = "FAIL"
        raise RuntimeError(f"Transfer initialization audit failed: {audit}")
    del source_model, checkpoint, source_head
    return model, audit


def read_manifest(path: Path, expected_split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"split", "image_id", "full_image_path", "image_sha256", *LABEL_FIELDS}
        if not required.issubset(fields):
            raise ValueError(f"Manifest {path} is missing fields: {sorted(required - fields)}")
        rows = list(reader)
    for row in rows:
        if row["split"] != expected_split:
            raise ValueError(f"Manifest split mismatch in {path}")
        labels = [int(row[field]) for field in LABEL_FIELDS]
        if any(value not in {0, 1} for value in labels) or sum(labels) <= 0:
            raise ValueError(f"Invalid multilabel vector for {row['image_id']}")
        image_path = Path(row["full_image_path"])
        lowered = str(image_path).casefold()
        if not image_path.is_file() or "data\\raw\\images" not in lowered:
            raise ValueError(f"Manifest does not reference an existing raw full image: {image_path}")
        if any(token in lowered for token in ("bbox_crops", "roi_balanced", "augmentation")):
            raise ValueError(f"ROI or augmented path found in full-image manifest: {image_path}")
    return rows


def dataset_integrity(
    train: list[dict[str, str]], val: list[dict[str, str]], test: list[dict[str, str]], paths: dict[str, Path]
) -> dict[str, Any]:
    splits = {"train": train, "val": val, "test": test}
    expected = {"train": 472, "val": 59, "test": 59}
    intersections = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for field in ("image_id", "image_sha256"):
            intersections[f"{left}_{right}_{field}"] = sorted(
                {row[field] for row in splits[left]} & {row[field] for row in splits[right]}
            )
    counts = {
        name: {
            str(index): sum(int(row[LABEL_FIELDS[index]]) for row in rows)
            for index in CLASS_MAPPING
        }
        for name, rows in splits.items()
    }
    failures = []
    if any(len(splits[name]) != expected[name] for name in expected):
        failures.append("split_rows")
    if any(intersections.values()):
        failures.append("leakage")
    for name, rows in splits.items():
        for index in CLASS_MAPPING:
            positive = counts[name][str(index)]
            if positive <= 0 or positive >= len(rows):
                failures.append(f"{name}_class_{index}_positive_negative")
    all_rows = train + val + test
    if len({row["image_id"] for row in all_rows}) != 590:
        failures.append("unique_image_ids")
    if len({row["full_image_path"] for row in all_rows}) != 590:
        failures.append("unique_paths")
    if len({row["image_sha256"] for row in all_rows}) != 590:
        failures.append("duplicate_content")
    return {
        "status": "PASS" if not failures else "FAIL",
        "split_rows": {name: len(rows) for name, rows in splits.items()},
        "class_positive_counts": counts,
        "intersections": intersections,
        "manifest_sha256": {name: sha256_file(paths[name]) for name in splits},
        "uses_raw_full_images": True,
        "uses_bbox": False,
        "uses_roi_crop": False,
        "augmentation": False,
        "test_images_read_during_preflight": 0,
        "failures": failures,
    }


def make_loader(
    rows: list[dict[str, str]], batch_size: int, workers: int, seed: int, shuffle: bool
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        FullImageMultilabelDataset(rows, FullImageTransform(224)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def shutdown_loader(loader: DataLoader) -> None:
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def build_optimizer(model: FullImageMultilabelConvNeXt) -> torch.optim.Optimizer:
    backbone_parameters = list(model.features.parameters()) + list(model.final_norm.parameters())
    head_parameters = list(model.multilabel_head.parameters())
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 1e-5, "name": "backbone_and_final_norm"},
            {"params": head_parameters, "lr": 1e-4, "name": "new_multilabel_head"},
        ],
        weight_decay=1e-4,
    )


def compute_threshold_free(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "macro_auroc": float(roc_auc_score(labels, probabilities, average="macro")),
        "micro_auroc": float(roc_auc_score(labels, probabilities, average="micro")),
        "macro_average_precision": float(average_precision_score(labels, probabilities, average="macro")),
        "micro_average_precision": float(average_precision_score(labels, probabilities, average="micro")),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, list[int]]:
    model.eval()
    losses = 0.0
    labels_parts = []
    probability_parts = []
    row_indices = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits.float())
            if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
                raise RuntimeError("NaN or Inf encountered during evaluation")
            losses += float(loss.item()) * images.shape[0]
            labels_parts.append(labels.cpu().numpy())
            probability_parts.append(probabilities.cpu().numpy())
            row_indices.extend(int(value) for value in batch["row_index"].tolist())
    label_array = np.concatenate(labels_parts)
    probability_array = np.concatenate(probability_parts)
    metrics = {"loss": losses / len(loader.dataset), **compute_threshold_free(label_array, probability_array)}
    return metrics, label_array, probability_array, row_indices


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[float, int, int]:
    model.train()
    total_loss = 0.0
    nan_count = 0
    inf_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        nan_count += int(torch.isnan(logits).sum().item()) + int(torch.isnan(loss).sum().item())
        inf_count += int(torch.isinf(logits).sum().item()) + int(torch.isinf(loss).sum().item())
        if nan_count or inf_count:
            raise RuntimeError(f"Training numerical failure NaN={nan_count}, Inf={inf_count}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * images.shape[0]
    return total_loss / len(loader.dataset), nan_count, inf_count


def select_thresholds(labels: np.ndarray, probabilities: np.ndarray) -> tuple[list[float], list[dict[str, Any]]]:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    thresholds = []
    audit = []
    for class_id in range(NUM_CLASSES):
        candidates = []
        for threshold in grid:
            predictions = (probabilities[:, class_id] >= threshold).astype(np.int64)
            score = float(f1_score(labels[:, class_id], predictions, zero_division=0))
            candidates.append((score, -abs(float(threshold) - 0.5), -float(threshold), float(threshold)))
        best = max(candidates)
        thresholds.append(best[3])
        audit.append(
            {
                "class_id": class_id,
                "class_name": CLASS_MAPPING[class_id],
                "threshold": best[3],
                "validation_f1": best[0],
                "tie_break": "closest_to_0.5_then_lower_threshold",
            }
        )
    return thresholds, audit


def threshold_metrics(
    labels: np.ndarray, probabilities: np.ndarray, thresholds: list[float]
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    predictions = (probabilities >= np.asarray(thresholds)[None, :]).astype(np.int64)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    samples_f1 = f1_score(labels, predictions, average="samples", zero_division=0)
    overall = {
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "hamming_loss": float(hamming_loss(labels, predictions)),
        "exact_match_subset_accuracy": float(accuracy_score(labels, predictions)),
        "samples_f1": float(samples_f1),
    }
    per_class = []
    for class_id in range(NUM_CLASSES):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels[:, class_id], predictions[:, class_id], average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(
            labels[:, class_id], predictions[:, class_id], labels=[0, 1]
        ).ravel()
        per_class.append(
            {
                "class_id": class_id,
                "class_name": CLASS_MAPPING[class_id],
                "threshold": thresholds[class_id],
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auroc": float(roc_auc_score(labels[:, class_id], probabilities[:, class_id])),
                "average_precision": float(average_precision_score(labels[:, class_id], probabilities[:, class_id])),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }
        )
    return overall, per_class, predictions


def environment_info(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(device)),
        }
    return {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": gpu,
    }


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_epoch: int,
    best_macro_auroc: float,
    initialization_audit: dict[str, Any],
    training_config: dict[str, Any],
    test_evaluation_count: int = 0,
) -> dict[str, Any]:
    return {
        "architecture": "convnext_tiny",
        "task": "full-image multilabel five-class classification",
        "experiment": "ROI Patch Proposed transfer initialization -> Full-image multilabel fine-tuning",
        "current_epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_macro_auroc": best_macro_auroc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "class_mapping": CLASS_MAPPING,
        "label_fields": LABEL_FIELDS,
        "initialization_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "initialization_audit": initialization_audit,
        "training_config": training_config,
        "activation_for_inference": "sigmoid",
        "loss": "BCEWithLogitsLoss",
        "test_evaluation_count": test_evaluation_count,
        "rad_dino_teacher_cache_rebuilt": False,
        "uses_bbox": False,
        "uses_roi_crop": False,
    }


def plot_training(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], figures: Path) -> None:
    epochs = [int(row["epoch"]) for row in train_rows]
    plots = [
        ("training_loss.png", "Training loss", [float(row["train_loss"]) for row in train_rows], "Loss"),
        ("validation_loss.png", "Validation loss", [float(row["validation_loss"]) for row in val_rows], "Loss"),
        ("validation_macro_auroc.png", "Validation Macro-AUROC", [float(row["macro_auroc"]) for row in val_rows], "Macro-AUROC"),
        ("validation_macro_ap.png", "Validation Macro Average Precision", [float(row["macro_average_precision"]) for row in val_rows], "Macro AP"),
    ]
    for filename, title, values, ylabel in plots:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(epochs, values, marker="o", linewidth=1.8)
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(True, alpha=0.3)
        atomic_save_figure(figure, figures / filename)


def plot_test_results(
    per_class: list[dict[str, Any]], thresholds: list[float], figures: Path
) -> None:
    labels = [f"{row['class_id']} {row['class_name']}" for row in per_class]
    for field, filename, title in (
        ("f1", "per_class_test_f1.png", "Per-class Test F1"),
        ("auroc", "per_class_test_auroc.png", "Per-class Test AUROC"),
        ("average_precision", "per_class_test_ap.png", "Per-class Test Average Precision"),
    ):
        figure, axis = plt.subplots(figsize=(9, 4.8))
        axis.bar(range(NUM_CLASSES), [float(row[field]) for row in per_class])
        axis.set_xticks(range(NUM_CLASSES), labels, rotation=25, ha="right")
        axis.set_ylim(0, 1)
        axis.set(title=title, ylabel=field)
        axis.grid(True, axis="y", alpha=0.3)
        atomic_save_figure(figure, figures / filename)

    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(range(NUM_CLASSES), thresholds)
    axis.set_xticks(range(NUM_CLASSES), labels, rotation=25, ha="right")
    axis.set_ylim(0, 1)
    axis.set(title="Validation-selected thresholds", ylabel="Threshold")
    axis.grid(True, axis="y", alpha=0.3)
    atomic_save_figure(figure, figures / "thresholds.png")

    figure, axes = plt.subplots(1, NUM_CLASSES, figsize=(16, 3.4))
    for axis, row in zip(axes, per_class):
        matrix = np.asarray([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"Class {row['class_id']}")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        for y in range(2):
            for x in range(2):
                axis.text(x, y, str(matrix[y, x]), ha="center", va="center")
    figure.suptitle("Per-class binary confusion matrices")
    atomic_save_figure(figure, figures / "per_class_binary_confusion_matrices.png")


def run_smoke_test(
    args: argparse.Namespace,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    device: torch.device,
    dataset_audit: dict[str, Any],
) -> dict[str, Any]:
    set_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, initialization = initialize_from_roi_export(args.initialization_checkpoint, device)
    optimizer = build_optimizer(model)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda", init_scale=1024.0)
    train_loader = make_loader(train_rows, args.batch_size, args.workers, args.seed, True)
    val_loader = make_loader(val_rows, args.batch_size, args.workers, args.seed, False)
    model.train()
    train_batches = 0
    last_shapes: dict[str, Any] = {}
    nan_count = 0
    inf_count = 0
    started = time.perf_counter()
    for batch in train_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        probabilities = torch.sigmoid(logits.float())
        nan_count += int(torch.isnan(logits).sum().item()) + int(torch.isnan(probabilities).sum().item())
        inf_count += int(torch.isinf(logits).sum().item()) + int(torch.isinf(probabilities).sum().item())
        if nan_count or inf_count:
            raise RuntimeError("Smoke test produced NaN or Inf")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        scaler.step(optimizer)
        scaler.update()
        train_batches += 1
        last_shapes = {
            "input": list(images.shape),
            "labels": list(labels.shape),
            "logits": list(logits.shape),
            "sigmoid_probabilities": list(probabilities.shape),
            "loss": float(loss.item()),
            "gradient_norm_before_clip": gradient_norm,
        }
        if train_batches == 3:
            break
    model.eval()
    with torch.inference_mode():
        validation_batch = next(iter(val_loader))
        val_images = validation_batch["image"].to(device, non_blocking=True)
        val_labels = validation_batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            val_logits = model(val_images)
            val_loss = criterion(val_logits, val_labels)
        val_probabilities = torch.sigmoid(val_logits.float())
    if not torch.isfinite(val_logits).all() or not torch.isfinite(val_probabilities).all():
        raise RuntimeError("Smoke validation produced NaN or Inf")
    audit = {
        "status": "PASS",
        "created_at": utc_now(),
        "initialization": initialization,
        "dataset_integrity": dataset_audit,
        "train_batches": train_batches,
        "validation_batches": 1,
        "test_images_read": 0,
        "backward_completed": True,
        "train_shapes": last_shapes,
        "validation_shapes": {
            "input": list(val_images.shape),
            "labels": list(val_labels.shape),
            "logits": list(val_logits.shape),
            "sigmoid_probabilities": list(val_probabilities.shape),
            "loss": float(val_loss.item()),
        },
        "nan_count": nan_count,
        "inf_count": inf_count,
        "oom": False,
        "seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "checkpoint_created": False,
        "formal_training_must_reinitialize": True,
    }
    del model, optimizer, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--initialization-checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "project_root",
        "initialization_checkpoint",
        "train_manifest",
        "val_manifest",
        "test_manifest",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.dry_run and args.smoke_test:
        raise ValueError("Choose either --dry-run or --smoke-test")
    locked = {
        "epochs": 50,
        "batch_size": 64,
        "workers": 2,
        "patience": 10,
        "device": "cuda:0",
        "seed": 42,
    }
    failed = [name for name, value in locked.items() if getattr(args, name) != value]
    if failed:
        raise ValueError(f"Locked experiment arguments differ: {failed}")
    required_files = [
        args.initialization_checkpoint,
        args.train_manifest,
        args.val_manifest,
        args.test_manifest,
    ]
    if not args.project_root.is_dir() or any(not path.is_file() for path in required_files):
        raise FileNotFoundError("Project, checkpoint, or manifest is missing")
    return args


def main() -> int:
    args = resolve_args(parse_args())
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Locked cuda:0 device is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    manifest_paths = {
        "train": args.train_manifest,
        "val": args.val_manifest,
        "test": args.test_manifest,
    }
    train_rows = read_manifest(args.train_manifest, "train")
    val_rows = read_manifest(args.val_manifest, "val")
    test_rows = read_manifest(args.test_manifest, "test")
    integrity = dataset_integrity(train_rows, val_rows, test_rows, manifest_paths)
    if integrity["status"] != "PASS":
        raise RuntimeError(f"Dataset integrity failed: {integrity}")
    if sha256_file(args.initialization_checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Initialization checkpoint SHA256 mismatch")
    transform = FullImageTransform(224)
    config = {
        "experiment": "ROI Patch Proposed transfer initialization -> Full-image multilabel fine-tuning",
        "seed": 42,
        "epochs_upper_limit": 50,
        "patience": 10,
        "batch_size": 64,
        "gradient_accumulation": 1,
        "effective_batch_size": 64,
        "workers": 2,
        "device": "cuda:0",
        "amp": True,
        "grad_scaler_initial_scale": 1024.0,
        "gradient_clipping": 1.0,
        "optimizer": "AdamW",
        "backbone_lr": 1e-5,
        "head_lr": 1e-4,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": 50,
        "loss": "BCEWithLogitsLoss",
        "pos_weight": None,
        "inference_activation": "Sigmoid",
        "best_checkpoint_metric": "Validation Macro-AUROC",
        "threshold_source": "Validation only",
        "threshold_grid": {"minimum": 0.05, "maximum": 0.95, "step": 0.01},
        "preprocessing": transform.config(),
        "augmentation": False,
        "uses_bbox": False,
        "uses_roi_crop": False,
        "rad_dino_teacher_cache_rebuilt": False,
    }
    dry_run = {
        "status": "PASS",
        "created_at": utc_now(),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "dataset_integrity": integrity,
        "training_config": config,
        "environment": environment_info(device),
        "test_images_read": 0,
        "model_inference_executed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        atomic_write_json(args.output_dir / "dry_run_audit.json", dry_run)
        print(json.dumps(dry_run, ensure_ascii=False, indent=2))
        return 0
    if args.smoke_test:
        if (args.output_dir / "smoke_test_audit.json").exists():
            raise FileExistsError("Smoke test audit already exists; refusing to overwrite")
        smoke = run_smoke_test(args, train_rows, val_rows, device, integrity)
        atomic_write_json(args.output_dir / "smoke_test_audit.json", smoke)
        print(json.dumps(smoke, ensure_ascii=False, indent=2))
        return 0

    formal_targets = [
        args.output_dir / "checkpoints" / "best.pt",
        args.output_dir / "checkpoints" / "last.pt",
        args.output_dir / "checkpoints" / "full_image_multilabel_patch_transfer.pt",
        args.output_dir / "train_metrics.csv",
        args.output_dir / "test_metrics.json",
    ]
    if any(path.exists() for path in formal_targets):
        raise FileExistsError("Formal output already exists; refusing to overwrite")
    smoke_path = args.output_dir / "smoke_test_audit.json"
    if not smoke_path.is_file() or json.loads(smoke_path.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("A passing smoke_test_audit.json is required")

    checkpoint_hash_before = sha256_file(args.initialization_checkpoint)
    torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    model, initialization = initialize_from_roi_export(args.initialization_checkpoint, device)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = torch.amp.GradScaler(device.type, enabled=True, init_scale=1024.0)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = make_loader(train_rows, args.batch_size, args.workers, args.seed, True)
    val_loader = make_loader(val_rows, args.batch_size, args.workers, args.seed, False)
    train_metrics: list[dict[str, Any]] = []
    validation_metrics: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro_auroc = -math.inf
    early_counter = 0
    completed_epochs = 0
    total_nan = 0
    total_inf = 0

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, nan_count, inf_count = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        total_nan += nan_count
        total_inf += inf_count
        validation, _, _, _ = evaluate(model, val_loader, criterion, device)
        improved = validation["macro_auroc"] > best_macro_auroc + 1e-12
        if improved:
            best_macro_auroc = validation["macro_auroc"]
            best_epoch = epoch
            early_counter = 0
        else:
            early_counter += 1
        train_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "grad_scale": scaler.get_scale(),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        val_row = {
            "epoch": epoch,
            "validation_loss": validation["loss"],
            "macro_auroc": validation["macro_auroc"],
            "micro_auroc": validation["micro_auroc"],
            "macro_average_precision": validation["macro_average_precision"],
            "micro_average_precision": validation["micro_average_precision"],
            "is_best": improved,
            "early_stopping_counter": early_counter,
        }
        train_metrics.append(train_row)
        validation_metrics.append(val_row)
        completed_epochs = epoch
        payload = checkpoint_payload(
            model, optimizer, scheduler, scaler, epoch, best_epoch, best_macro_auroc,
            initialization, config, 0
        )
        atomic_save_checkpoint(args.output_dir / "checkpoints" / "last.pt", payload)
        if improved:
            atomic_save_checkpoint(args.output_dir / "checkpoints" / "best.pt", payload)
        atomic_write_csv(args.output_dir / "train_metrics.csv", train_metrics, TRAIN_FIELDS)
        atomic_write_csv(args.output_dir / "validation_metrics.csv", validation_metrics, VAL_FIELDS)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} val_loss={validation['loss']:.6f} "
            f"val_macro_auroc={validation['macro_auroc']:.6f} best={best_macro_auroc:.6f} "
            f"early={early_counter}/{args.patience}",
            flush=True,
        )
        scheduler.step()
        if early_counter >= args.patience:
            break

    best_checkpoint = torch.load(
        args.output_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    model.to(device)
    validation_best, validation_labels, validation_probabilities, _ = evaluate(
        model, val_loader, criterion, device
    )
    thresholds, threshold_audit = select_thresholds(validation_labels, validation_probabilities)
    threshold_payload = {
        "status": "PASS",
        "source_split": "validation",
        "best_checkpoint_epoch": best_epoch,
        "search_minimum": 0.05,
        "search_maximum": 0.95,
        "search_step": 0.01,
        "selection_metric": "per-class Validation F1",
        "thresholds": {str(index): thresholds[index] for index in range(NUM_CLASSES)},
        "per_class_audit": threshold_audit,
        "test_used": False,
    }
    atomic_write_json(args.output_dir / "validation_selected_thresholds.json", threshold_payload)

    test_evaluation_count = 0
    shutdown_loader(train_loader)
    shutdown_loader(val_loader)
    del train_loader, val_loader
    test_loader = make_loader(test_rows, args.batch_size, args.workers, args.seed, False)
    test_metrics_free, test_labels, test_probabilities, test_row_indices = evaluate(
        model, test_loader, criterion, device
    )
    test_evaluation_count += 1
    shutdown_loader(test_loader)
    del test_loader
    test_threshold_metrics, per_class, test_predictions = threshold_metrics(
        test_labels, test_probabilities, thresholds
    )
    test_metrics = {
        "status": "PASS",
        "test_evaluation_count": test_evaluation_count,
        "best_epoch": best_epoch,
        "threshold_source": "validation",
        "test_loss": test_metrics_free["loss"],
        "macro_auroc": test_metrics_free["macro_auroc"],
        "micro_auroc": test_metrics_free["micro_auroc"],
        "macro_average_precision": test_metrics_free["macro_average_precision"],
        "micro_average_precision": test_metrics_free["micro_average_precision"],
        **test_threshold_metrics,
    }
    prediction_rows = []
    for position, row_index in enumerate(test_row_indices):
        source = test_rows[row_index]
        row = {
            "image_id": source["image_id"],
            "source_image_id": source["source_image_id"],
            "full_image_path": source["full_image_path"],
        }
        for class_id in range(NUM_CLASSES):
            row[f"label_{class_id}"] = int(test_labels[position, class_id])
            row[f"probability_{class_id}"] = float(test_probabilities[position, class_id])
            row[f"threshold_{class_id}"] = thresholds[class_id]
            row[f"prediction_{class_id}"] = int(test_predictions[position, class_id])
        prediction_rows.append(row)
    prediction_fields = ["image_id", "source_image_id", "full_image_path"]
    for class_id in range(NUM_CLASSES):
        prediction_fields.extend(
            [f"label_{class_id}", f"probability_{class_id}", f"threshold_{class_id}", f"prediction_{class_id}"]
        )
    atomic_write_json(args.output_dir / "test_metrics.json", test_metrics)
    atomic_write_csv(
        args.output_dir / "per_class_test_metrics.csv",
        per_class,
        list(per_class[0]),
    )
    atomic_write_csv(args.output_dir / "test_predictions.csv", prediction_rows, prediction_fields)
    atomic_write_json(args.output_dir / "initialization_audit.json", initialization)
    atomic_write_json(args.output_dir / "dataset_integrity_audit.json", integrity)
    atomic_write_json(args.output_dir / "training_config.json", config)
    atomic_write_json(args.output_dir / "environment.json", environment_info(device))
    test_audit = {
        "status": "PASS" if test_evaluation_count == 1 else "FAIL",
        "test_evaluation_count": test_evaluation_count,
        "best_checkpoint_fixed_before_test": True,
        "validation_thresholds_fixed_before_test": True,
        "test_used_for_checkpoint_selection": False,
        "test_used_for_threshold_selection": False,
        "test_rows": len(test_rows),
    }
    atomic_write_json(args.output_dir / "test_evaluation_audit.json", test_audit)
    plot_training(train_metrics, validation_metrics, args.output_dir / "figures")
    plot_test_results(per_class, thresholds, args.output_dir / "figures")
    training_seconds = time.perf_counter() - training_started
    peak_vram = torch.cuda.max_memory_allocated(device)
    export_payload = {
        **checkpoint_payload(
            model, optimizer, scheduler, scaler, best_epoch, best_epoch, best_macro_auroc,
            initialization, config, test_evaluation_count
        ),
        "checkpoint_kind": "full_image_multilabel_export",
        "validation_selected_thresholds": thresholds,
        "validation_metrics_at_best_reload": validation_best,
        "test_metrics": test_metrics,
        "per_class_test_metrics": per_class,
        "training_seconds": training_seconds,
        "peak_vram_bytes": peak_vram,
    }
    atomic_save_checkpoint(
        args.output_dir / "checkpoints" / "full_image_multilabel_patch_transfer.pt",
        export_payload,
    )
    summary_lines = [
        "# Full-image Multilabel Patch Transfer Summary",
        "",
        "- Status: **PASS**",
        "- Experiment: ROI Patch Proposed transfer initialization -> Full-image multilabel fine-tuning",
        "- Input: complete raw chest X-ray resized directly to 224x224",
        "- BBox / ROI crop / Center Crop: not used",
        "- RAD-DINO teacher cache rebuilt: no",
        f"- Completed epochs: {completed_epochs}",
        f"- Best epoch: {best_epoch}",
        f"- Best Validation Macro-AUROC: {best_macro_auroc:.8f}",
        f"- Test Macro/Micro AUROC: {test_metrics['macro_auroc']:.8f} / {test_metrics['micro_auroc']:.8f}",
        f"- Test Macro/Micro AP: {test_metrics['macro_average_precision']:.8f} / {test_metrics['micro_average_precision']:.8f}",
        f"- Test Macro/Micro F1: {test_metrics['macro_f1']:.8f} / {test_metrics['micro_f1']:.8f}",
        f"- Test evaluation count: {test_evaluation_count}",
        f"- Training seconds: {training_seconds:.2f}",
        f"- Peak allocated VRAM bytes: {peak_vram}",
        f"- NaN / Inf: {total_nan} / {total_inf}",
        "",
        "No class reaching its Validation-selected threshold means only that none of the five target classes reached threshold; it does not mean normal or No finding.",
        "",
    ]
    atomic_write_text(args.output_dir / "summary.md", "\n".join(summary_lines))
    checkpoint_hash_after = sha256_file(args.initialization_checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("Protected ROI initialization checkpoint changed during training")
    residual = list(args.output_dir.rglob("*.tmp")) + list(args.output_dir.rglob("*.writing"))
    if residual:
        raise RuntimeError(f"Temporary files remain: {residual}")
    final = {
        "status": "PASS",
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_validation_macro_auroc": best_macro_auroc,
        "thresholds": thresholds,
        "test_metrics": test_metrics,
        "per_class_test_metrics": per_class,
        "training_seconds": training_seconds,
        "peak_vram_bytes": peak_vram,
        "test_evaluation_count": test_evaluation_count,
        "checkpoint_sha256_unchanged": checkpoint_hash_after == checkpoint_hash_before,
        "temporary_files": 0,
    }
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
