#!/usr/bin/env python
"""Read-only, deterministic comparison of three locked ConvNeXt-Tiny runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import compare_proposed_vs_baseline as legacy


CLASS_MAPPING = {
    0: "Aortic enlargement",
    1: "Cardiomegaly",
    2: "Pleural thickening",
    3: "Pulmonary fibrosis",
    4: "Pleural effusion",
}
MODELS = ("baseline", "cls", "patch")
MODEL_LABELS = {
    "baseline": "ImageNet Baseline",
    "cls": "RAD-DINO CLS Proposed",
    "patch": "RAD-DINO Patch Proposed",
}
MODEL_COLORS = {"baseline": "#526D82", "cls": "#B0574A", "patch": "#3D7A57"}
PAIR_SPECS = (
    ("cls_minus_baseline", "cls", "baseline"),
    ("patch_minus_baseline", "patch", "baseline"),
    ("patch_minus_cls", "patch", "cls"),
)
EXPECTED = {
    "shared": "5b69d83be63e40ad19818df9e1ccba85ce59ccb03643d48b4fadb0bb8e3e3a2f",
    "train": "ba5ba5f743c439563e15106239e6bcc87bf9c8fe4105b295ef034356e5dbae55",
    "val": "5f92fd7282df28a4ec3365ba5fa7a777b365db860f7991a47238162d1ac5bc00",
    "test": "2130a73dcbadec1d6b4bba68f809db7eeed25d1ea421c4d450d3e0b4d015551a",
    "image_split": "ace50c1f4820252073049b5ecf8f0b601eac026a8119789c8047c8bfd4e41c1a",
}
EXPECTED_TEST_ROWS = 454
EXPECTED_TEST_SOURCES = 59
EXPECTED_TEST_CLASS_COUNTS = {0: 77, 1: 78, 2: 112, 3: 106, 4: 81}
BOOTSTRAP_METRICS = ("accuracy", "macro_f1", "weighted_f1", "macro_auroc")
OVERALL_METRICS = (
    "loss", "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "weighted_f1", "macro_auroc",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_path(value: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(Path(value).resolve(strict=False))))


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def csv_value(value: Any) -> Any:
    if value is None or value == "":
        raise ValueError("CSV output contains an empty required value")
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"CSV output contains non-finite value: {value}")
        return f"{float(value):.12f}"
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return "TRUE" if bool(value) else "FALSE"
    return value


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to create an empty CSV: {path}")
    fields = fieldnames or list(rows[0])
    serial = [{field: csv_value(row.get(field)) for field in fields} for row in rows]
    signatures = [tuple(str(row[field]) for field in fields) for row in serial]
    if len(signatures) != len(set(signatures)):
        raise ValueError(f"CSV output contains duplicate rows: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(serial)
    os.replace(temporary, path)


def atomic_save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)


def ensure_clean_destination(output: Path) -> None:
    if output.exists():
        existing = sorted(str(item.relative_to(output)) for item in output.rglob("*"))
        if existing:
            raise FileExistsError(f"Output directory is non-empty; refusing overwrite: {existing}")
    staging = output.with_name(output.name + ".writing")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists; refusing overwrite: {staging}")


def model_paths(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    definitions = {
        "baseline": {
            "directory": args.baseline_dir,
            "config": "config/baseline_experiment_config.json",
            "summary": "phase2_baseline_training_summary.md",
            "export": "checkpoints/baseline_convnext_tiny_5class.pt",
        },
        "cls": {
            "directory": args.cls_dir,
            "config": "config/experiment_config.json",
            "summary": "phase2_proposed_training_summary.md",
            "export": "checkpoints/proposed_convnext_tiny_5class.pt",
        },
        "patch": {
            "directory": args.patch_dir,
            "config": "config/experiment_config.json",
            "summary": "phase2_patch_proposed_training_summary.md",
            "export": "checkpoints/patch_proposed_convnext_tiny_5class.pt",
        },
    }
    paths: dict[str, dict[str, Path]] = {}
    for model, definition in definitions.items():
        directory = definition["directory"]
        current = {
            "directory": directory,
            "metrics": directory / "metrics/phase2_metrics.csv",
            "test_metrics": directory / "test_results/test_metrics.json",
            "per_class": directory / "test_results/per_class_metrics.csv",
            "predictions": directory / "predictions/test_predictions.csv",
            "test_metadata": directory / "test_results/test_evaluation_metadata.json",
            "summary": directory / definition["summary"],
            "fairness_physical": directory / "diagnostics/fairness_audit.json",
            "stage0": directory / "diagnostics/stage0_smoke_test.json",
            "initialization": directory / "diagnostics/initialization_audit.json",
            "final_audit": directory / "diagnostics/phase2_final_audit.json",
            "best": directory / "checkpoints/best.pt",
            "export": directory / definition["export"],
            "config": directory / definition["config"],
            "shared_copy": directory / "config/shared_phase2_finetune_config.json",
            "log": directory / "logs/phase2.log",
            "confusion_csv": directory / "test_results/confusion_matrix.csv",
        }
        current["fairness"] = current["fairness_physical"]
        if model == "cls" and not current["fairness"].is_file():
            current["fairness"] = args.patch_dir / "diagnostics/fairness_audit.json"
        required = [
            "metrics", "test_metrics", "per_class", "predictions", "test_metadata",
            "summary", "fairness", "stage0", "final_audit", "best", "export",
            "config", "shared_copy", "log",
        ]
        missing = [str(current[name]) for name in required if not current[name].is_file() or current[name].stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"{model} missing required artifacts: {missing}")
        paths[model] = current
    return paths


def split_audit(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    manifests: dict[str, list[dict[str, str]]] = {}
    hashes: dict[str, str] = {}
    stats: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        path = args.split_dir / f"{split}_roi_manifest.csv"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        rows = read_csv(path)
        manifests[split] = rows
        hashes[split] = sha256_file(path)
        stats[split] = {
            "path": str(path), "sha256": hashes[split], "roi_rows": len(rows),
            "source_image_ids": len({row["source_image_id"] for row in rows}),
            "class_counts": {str(class_id): sum(int(row["class_id"]) == class_id for row in rows) for class_id in range(5)},
        }
    image_split = args.split_dir / "image_id_split_manifest.csv"
    image_split_hash = sha256_file(image_split)
    checks = {
        "train_sha256": hashes["train"] == EXPECTED["train"],
        "val_sha256": hashes["val"] == EXPECTED["val"],
        "test_sha256": hashes["test"] == EXPECTED["test"],
        "image_split_sha256": image_split_hash == EXPECTED["image_split"],
        "test_rows": len(manifests["test"]) == EXPECTED_TEST_ROWS,
        "test_sources": stats["test"]["source_image_ids"] == EXPECTED_TEST_SOURCES,
        "test_class_counts": stats["test"]["class_counts"] == {str(k): v for k, v in EXPECTED_TEST_CLASS_COUNTS.items()},
    }
    split_sets = {split: {row["source_image_id"] for row in rows} for split, rows in manifests.items()}
    leakage = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    checks["source_leakage_zero"] = all(value == 0 for value in leakage.values())
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Locked split audit failed: {failed}")
    return {
        "status": "PASS", "checks": checks, "manifests": stats,
        "image_id_split": {"path": str(image_split), "sha256": image_split_hash},
        "source_leakage": leakage,
    }, manifests


def enrich_predictions(
    rows: list[dict[str, str]], manifest_rows: list[dict[str, str]], model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_by_record = {row["record_index"]: row for row in manifest_rows}
    manifest_by_path = {(row["source_image_id"], normalized_path(row["image_path"])): row for row in manifest_rows}
    if len(manifest_by_record) != EXPECTED_TEST_ROWS or len(manifest_by_path) != EXPECTED_TEST_ROWS:
        raise ValueError("Test manifest has duplicate record_index or source/path keys")
    required = {
        "source_image_id", "image_path", "true_class_id", "true_class_name",
        "predicted_class_id", "predicted_class_name", "confidence",
        *{f"probability_class_{class_id}" for class_id in range(5)},
    }
    if len(rows) != EXPECTED_TEST_ROWS:
        raise ValueError(f"{model} prediction rows={len(rows)}, expected {EXPECTED_TEST_ROWS}")
    missing_fields = required - set(rows[0])
    if missing_fields:
        raise ValueError(f"{model} prediction fields missing: {sorted(missing_fields)}")
    enriched: list[dict[str, Any]] = []
    max_sum_error = 0.0
    for row in rows:
        manifest = manifest_by_record.get(row.get("record_index", ""))
        if manifest is None:
            manifest = manifest_by_path.get((row["source_image_id"], normalized_path(row["image_path"])))
        if manifest is None:
            raise ValueError(f"{model} prediction cannot be mapped to test manifest: {row['image_path']}")
        checks = {
            "source_image_id": row["source_image_id"] == manifest["source_image_id"],
            "image_path": normalized_path(row["image_path"]) == normalized_path(manifest["image_path"]),
            "true_class_id": int(row["true_class_id"]) == int(manifest["class_id"]),
            "true_class_name": row["true_class_name"] == manifest["class_name"],
            "original_roi_id": not row.get("original_roi_id") or row["original_roi_id"] == manifest["original_roi_id"],
        }
        if not all(checks.values()):
            raise ValueError(f"{model} prediction/manifest mismatch at {manifest['record_index']}: {checks}")
        true_class = int(row["true_class_id"])
        predicted_class = int(row["predicted_class_id"])
        probabilities = [float(row[f"probability_class_{class_id}"]) for class_id in range(5)]
        confidence = float(row["confidence"])
        if true_class not in CLASS_MAPPING or predicted_class not in CLASS_MAPPING:
            raise ValueError(f"{model} invalid class id")
        if row["true_class_name"] != CLASS_MAPPING[true_class] or row["predicted_class_name"] != CLASS_MAPPING[predicted_class]:
            raise ValueError(f"{model} class mapping mismatch")
        if not all(math.isfinite(value) for value in probabilities + [confidence]):
            raise ValueError(f"{model} non-finite probability/confidence")
        max_sum_error = max(max_sum_error, abs(sum(probabilities) - 1.0))
        correctness_field = row.get("is_correct", row.get("correct", ""))
        if parse_bool(correctness_field) != (true_class == predicted_class):
            raise ValueError(f"{model} correctness mismatch")
        canonical_path = manifest["image_path"]
        key = (manifest["source_image_id"], manifest["original_roi_id"], normalized_path(canonical_path))
        enriched.append({
            "key": key,
            "record_index": int(manifest["record_index"]),
            "source_image_id": manifest["source_image_id"],
            "original_roi_id": manifest["original_roi_id"],
            "image_path": canonical_path,
            "true_class_id": true_class,
            "true_class_name": CLASS_MAPPING[true_class],
            "predicted_class_id": predicted_class,
            "predicted_class_name": CLASS_MAPPING[predicted_class],
            "confidence": confidence,
            "probabilities": probabilities,
            "is_correct": true_class == predicted_class,
        })
    keys = [row["key"] for row in enriched]
    if len(set(keys)) != EXPECTED_TEST_ROWS:
        raise ValueError(f"{model} contains duplicate paired keys")
    if max_sum_error > 1e-5:
        raise ValueError(f"{model} max probability sum error={max_sum_error}")
    enriched.sort(key=lambda row: (row["record_index"], row["key"]))
    return enriched, {
        "rows": len(enriched), "unique_paired_keys": len(set(keys)),
        "duplicate_paired_keys": len(enriched) - len(set(keys)),
        "source_image_ids": len({row["source_image_id"] for row in enriched}),
        "class_counts": {str(class_id): sum(row["true_class_id"] == class_id for row in enriched) for class_id in range(5)},
        "max_probability_sum_error": max_sum_error,
        "nonfinite_probability_or_confidence": 0,
    }


def pair_predictions(enriched: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    maps = {model: {row["key"]: row for row in rows} for model, rows in enriched.items()}
    key_sets = {model: set(rows) for model, rows in maps.items()}
    union = set().union(*key_sets.values())
    intersection = set.intersection(*key_sets.values())
    missing = {model: len(union - key_sets[model]) for model in MODELS}
    if len(intersection) != EXPECTED_TEST_ROWS or any(missing.values()):
        raise ValueError(f"Three-model paired key mismatch: intersection={len(intersection)}, missing={missing}")
    ordered = sorted(intersection, key=lambda key: maps["baseline"][key]["record_index"])
    paired: list[dict[str, Any]] = []
    arrays: dict[str, Any] = {
        "targets": [], "sources": [],
        **{f"{model}_predictions": [] for model in MODELS},
        **{f"{model}_probabilities": [] for model in MODELS},
    }
    for key in ordered:
        source_rows = {model: maps[model][key] for model in MODELS}
        if len({row["true_class_id"] for row in source_rows.values()}) != 1:
            raise ValueError(f"True labels differ at key {key}")
        canonical = source_rows["baseline"]
        row: dict[str, Any] = {
            "paired_key": "||".join(key),
            "source_image_id": canonical["source_image_id"],
            "original_roi_id": canonical["original_roi_id"],
            "image_path": canonical["image_path"],
            "true_class_id": canonical["true_class_id"],
            "true_class_name": canonical["true_class_name"],
        }
        correctness: dict[str, bool] = {}
        predictions: dict[str, int] = {}
        for model in MODELS:
            item = source_rows[model]
            predictions[model] = item["predicted_class_id"]
            correctness[model] = item["is_correct"]
            row.update({
                f"{model}_predicted_class_id": item["predicted_class_id"],
                f"{model}_predicted_class_name": item["predicted_class_name"],
                f"{model}_confidence": item["confidence"],
                f"{model}_is_correct": item["is_correct"],
            })
            for class_id, probability in enumerate(item["probabilities"]):
                row[f"{model}_probability_class_{class_id}"] = probability
            arrays[f"{model}_predictions"].append(item["predicted_class_id"])
            arrays[f"{model}_probabilities"].append(item["probabilities"])
        count_correct = sum(correctness.values())
        row.update({
            "all_three_correct": count_correct == 3,
            "all_three_wrong": count_correct == 0,
            "all_three_same_prediction": len(set(predictions.values())) == 1,
            "exactly_one_model_correct": count_correct == 1,
            "exactly_two_models_correct": count_correct == 2,
            "baseline_only_correct": correctness["baseline"] and count_correct == 1,
            "cls_only_correct": correctness["cls"] and count_correct == 1,
            "patch_only_correct": correctness["patch"] and count_correct == 1,
            "baseline_cls_correct_patch_wrong": correctness["baseline"] and correctness["cls"] and not correctness["patch"],
            "baseline_patch_correct_cls_wrong": correctness["baseline"] and correctness["patch"] and not correctness["cls"],
            "cls_patch_correct_baseline_wrong": correctness["cls"] and correctness["patch"] and not correctness["baseline"],
        })
        paired.append(row)
        arrays["targets"].append(canonical["true_class_id"])
        arrays["sources"].append(canonical["source_image_id"])
    converted = {
        key: np.asarray(value, dtype=(np.float64 if key.endswith("probabilities") else np.int64 if key != "sources" else str))
        for key, value in arrays.items()
    }
    audit = {
        "paired_rows": len(paired), "paired_key_union": len(union),
        "paired_key_intersection": len(intersection), "missing_keys_by_model": missing,
        "duplicate_keys_by_model": {model: 0 for model in MODELS},
        "source_image_ids": len(set(converted["sources"].tolist())),
        "true_labels_identical": True, "image_paths_identical": True,
        "source_image_ids_identical": True, "original_roi_ids_identical": True,
    }
    return paired, converted, audit


def read_confusion_csv(path: Path) -> np.ndarray:
    rows = read_csv(path)
    if len(rows) != 5:
        raise ValueError(f"Confusion CSV must have five rows: {path}")
    matrix = np.zeros((5, 5), dtype=np.int64)
    for row in rows:
        true_class = int(row["true_class_id"])
        matrix[true_class] = [int(row[f"predicted_class_{class_id}"]) for class_id in range(5)]
    return matrix


def metric_integrity(
    paths: dict[str, dict[str, Path]], arrays: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    recomputed: dict[str, dict[str, Any]] = {}
    audit: dict[str, Any] = {"status": "PASS", "models": {}}
    for model in MODELS:
        result = legacy.confusion_and_metrics(
            arrays["targets"], arrays[f"{model}_predictions"], arrays[f"{model}_probabilities"],
        )
        reported = read_json(paths[model]["test_metrics"])
        embedded = np.asarray(reported["confusion_matrix"], dtype=np.int64)
        predicted_matrix = result["confusion_matrix"]
        confusion_csv_exists = paths[model]["confusion_csv"].is_file()
        csv_matrix = read_confusion_csv(paths[model]["confusion_csv"]) if confusion_csv_exists else embedded
        scalar_differences = {
            metric: abs(float(reported[metric]) - float(result[metric]))
            for metric in OVERALL_METRICS if metric != "loss"
        }
        probability_loss = float(np.mean(-np.log(np.clip(
            arrays[f"{model}_probabilities"][np.arange(EXPECTED_TEST_ROWS), arrays["targets"]], 1e-300, 1.0,
        ))))
        loss_difference = abs(float(reported["loss"]) - probability_loss)
        per_class_rows = {int(row["class_id"]): row for row in read_csv(paths[model]["per_class"])}
        per_class_max_difference = 0.0
        for class_id in range(5):
            for metric in ("precision", "recall", "f1", "auroc"):
                per_class_max_difference = max(
                    per_class_max_difference,
                    abs(float(per_class_rows[class_id][metric]) - float(result["per_class"][class_id][metric])),
                )
            if int(per_class_rows[class_id]["support"]) != int(result["per_class"][class_id]["support"]):
                raise ValueError(f"{model} per-class support mismatch")
        checks = {
            "prediction_confusion_matches_test_metrics": np.array_equal(predicted_matrix, embedded),
            "confusion_csv_or_embedded_matches_predictions": np.array_equal(csv_matrix, predicted_matrix),
            "reported_scalar_metrics_match_predictions": max(scalar_differences.values()) <= 1e-12,
            "reported_loss_matches_serialized_probabilities_within_5e_5": loss_difference <= 5e-5,
            "per_class_metrics_match_predictions": per_class_max_difference <= 1e-12,
            "test_evaluation_count_one": int(reported["evaluation_count"]) == 1,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"{model} metric integrity failed: {failed}")
        result["loss"] = float(reported["loss"])
        recomputed[model] = result
        audit["models"][model] = {
            "checks": checks,
            "reported_loss": float(reported["loss"]),
            "loss_recomputed_from_serialized_probabilities": probability_loss,
            "loss_absolute_difference": loss_difference,
            "max_scalar_metric_absolute_difference": max(scalar_differences.values()),
            "max_per_class_absolute_difference": per_class_max_difference,
            "confusion_matrix_physical_csv_exists": confusion_csv_exists,
            "confusion_matrix_source": str(paths[model]["confusion_csv"] if confusion_csv_exists else paths[model]["test_metrics"]),
        }
    return recomputed, audit


def checkpoint_structure(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    result: dict[str, Any] = {
        "path": str(path), "sha256": sha256_file(path),
        "top_level_keys": sorted(checkpoint) if isinstance(checkpoint, dict) else [],
        "state_dicts": {},
    }
    if isinstance(checkpoint, dict):
        for name in ("model_state_dict", "backbone_state_dict", "final_norm_state_dict", "student_state_dict", "state_dict"):
            state = checkpoint.get(name)
            if not isinstance(state, dict):
                continue
            keys = list(state)
            result["state_dicts"][name] = {
                "key_count": len(keys),
                "feature_key_count": sum(key.startswith("features.") for key in keys),
                "final_norm_keys": [key for key in keys if key.startswith("final_norm.") or name == "final_norm_state_dict"],
                "classifier_keys": [key for key in keys if key.startswith("classifier.")],
                "first_keys": keys[:5], "last_keys": keys[-5:],
            }
        for key in (
            "phase", "experiment", "initialization", "weights_enum", "imagenet_pretrained_loaded",
            "architecture", "num_classes", "feature_dim", "classifier_head_included",
            "distillation_type", "output_feature_shape", "best_epoch", "manifest_sha256",
            "teacher_cache_sha256", "shared_config_sha256", "test_evaluation_count",
        ):
            if key in checkpoint:
                result[key] = checkpoint[key]
    del checkpoint
    return result


def layernorm_audit(args: argparse.Namespace, paths: dict[str, dict[str, Path]]) -> dict[str, Any]:
    cls_phase1 = args.project_root / "outputs/raddino_convnext_tiny_experiment_seed42/phase1_distillation/checkpoints/distilled_convnext_tiny_backbone.pt"
    patch_phase1 = args.project_root / "outputs/raddino_convnext_tiny_patch_experiment_seed42/phase1_patch_distillation/checkpoints/patch_distilled_convnext_tiny_backbone.pt"
    for path in (cls_phase1, patch_phase1):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    stage0 = {model: read_json(paths[model]["stage0"])["smoke"]["backbone_load_audit"] for model in MODELS}
    structures = {
        "baseline_phase2_best": checkpoint_structure(paths["baseline"]["best"]),
        "cls_phase1_export": checkpoint_structure(cls_phase1),
        "cls_phase2_best": checkpoint_structure(paths["cls"]["best"]),
        "patch_phase1_export": checkpoint_structure(patch_phase1),
        "patch_phase2_best": checkpoint_structure(paths["patch"]["best"]),
    }
    baseline_best = structures["baseline_phase2_best"]["state_dicts"]["model_state_dict"]
    cls_phase1_state = structures["cls_phase1_export"]["state_dicts"]
    cls_best = structures["cls_phase2_best"]["state_dicts"]["model_state_dict"]
    patch_phase1_state = structures["patch_phase1_export"]["state_dicts"]
    patch_best = structures["patch_phase2_best"]["state_dicts"]["model_state_dict"]
    models = {
        "baseline": {
            "features_loaded_from_initialization_checkpoint": True,
            "initialization_checkpoint": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
            "final_norm_present_in_initialization_checkpoint": True,
            "final_norm_initialization_source": "ImageNet ConvNeXt-Tiny weights",
            "classifier_head_recreated": True,
            "missing_keys": stage0["baseline"].get("missing_backbone_keys", []) + stage0["baseline"].get("missing_final_norm_keys", []),
            "unexpected_keys": stage0["baseline"].get("unexpected_backbone_keys", []) + stage0["baseline"].get("unexpected_final_norm_keys", []),
            "phase2_final_norm_keys": baseline_best["final_norm_keys"],
            "evidence": [str(paths["baseline"]["stage0"]), str(paths["baseline"]["best"]), str(args.project_root / "src/train_phase2_convnext_tiny_finetune.py")],
        },
        "cls": {
            "features_loaded_from_initialization_checkpoint": "backbone_state_dict" in cls_phase1_state,
            "initialization_checkpoint": str(cls_phase1),
            "final_norm_present_in_initialization_checkpoint": "final_norm_state_dict" in cls_phase1_state,
            "final_norm_initialization_source": "RAD-DINO CLS Phase 1 distilled checkpoint",
            "classifier_head_recreated": True,
            "missing_keys": stage0["cls"].get("missing_backbone_keys", []) + stage0["cls"].get("missing_final_norm_keys", []),
            "unexpected_keys": stage0["cls"].get("unexpected_backbone_keys", []) + stage0["cls"].get("unexpected_final_norm_keys", []),
            "phase1_final_norm_keys": cls_phase1_state.get("final_norm_state_dict", {}).get("final_norm_keys", []),
            "phase2_final_norm_keys": cls_best["final_norm_keys"],
            "evidence": [str(paths["cls"]["stage0"]), str(cls_phase1), str(paths["cls"]["best"]), str(args.project_root / "src/train_phase2_convnext_tiny_finetune.py")],
        },
        "patch": {
            "features_loaded_from_initialization_checkpoint": "student_state_dict" in patch_phase1_state,
            "initialization_checkpoint": str(patch_phase1),
            "final_norm_present_in_initialization_checkpoint": False,
            "final_norm_initialization_source": stage0["patch"].get("final_norm_initialization", "undetermined"),
            "classifier_head_recreated": True,
            "missing_keys": stage0["patch"].get("missing_backbone_keys", []) + stage0["patch"].get("missing_final_norm_keys", []),
            "unexpected_keys": stage0["patch"].get("unexpected_backbone_keys", []) + stage0["patch"].get("unexpected_final_norm_keys", []),
            "phase1_final_norm_keys": [],
            "phase2_final_norm_keys": patch_best["final_norm_keys"],
            "evidence": [str(paths["patch"]["initialization"]), str(patch_phase1), str(paths["patch"]["best"]), str(args.project_root / "src/train_phase2_convnext_tiny_finetune.py")],
        },
    }
    checks = {
        "baseline_phase2_final_norm_present": len(models["baseline"]["phase2_final_norm_keys"]) == 2,
        "cls_phase1_features_present": "backbone_state_dict" in cls_phase1_state,
        "cls_phase1_final_norm_present": "final_norm_state_dict" in cls_phase1_state,
        "cls_phase2_final_norm_present": len(models["cls"]["phase2_final_norm_keys"]) == 2,
        "patch_phase1_features_present": "student_state_dict" in patch_phase1_state,
        "patch_phase1_final_norm_absent": not any(item.get("final_norm_keys") for item in patch_phase1_state.values()),
        "patch_phase2_final_norm_present": len(models["patch"]["phase2_final_norm_keys"]) == 2,
        "all_classifier_heads_recreated": all(item["classifier_head_recreated"] for item in models.values()),
        "all_loads_have_no_missing_or_unexpected_keys": all(not item["missing_keys"] and not item["unexpected_keys"] for item in models.values()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"LayerNorm initialization audit failed: {failed}")
    return {
        "status": "PASS", "created_at_utc": utc_now(), "checks": checks,
        "models": models, "checkpoint_structures": structures,
        "final_layernorm_initialization_identical": False,
        "difference_classification": "allowed initialization difference",
        "potential_cls_vs_patch_confounder": True,
        "interpretation": (
            "CLS Phase 1 exported and Phase 2 loaded final_norm, whereas Patch Phase 1 did not export it and "
            "Patch Phase 2 used torchvision default initialization. This is part of the initialization package, "
            "but it prevents attributing CLS-versus-Patch downstream differences solely to global versus spatial distillation."
        ),
        "forward_executed": False,
    }


def fairness_audit(
    args: argparse.Namespace, paths: dict[str, dict[str, Path]], split: dict[str, Any],
    paired_audit: dict[str, Any], layernorm: dict[str, Any],
) -> dict[str, Any]:
    shared = read_json(args.shared_config)
    shared_sha = sha256_file(args.shared_config)
    configs = {model: read_json(paths[model]["config"]) for model in MODELS}
    test_meta = {model: read_json(paths[model]["test_metadata"]) for model in MODELS}
    finals = {model: read_json(paths[model]["final_audit"]) for model in MODELS}
    fairness_records = {model: read_json(paths[model]["fairness"]) for model in MODELS}
    preview_hashes = {model: sha256_file(paths[model]["directory"] / "figures/augmentation_preview.png") for model in MODELS}
    shared_copies = {model: paths[model]["shared_copy"].read_bytes() for model in MODELS}
    train_script = args.project_root / "src/train_phase2_convnext_tiny_finetune.py"
    train_script_text = train_script.read_text(encoding="utf-8-sig")
    checks = {
        "shared_config_expected_sha256": shared_sha == EXPECTED["shared"],
        "shared_config_copies_byte_identical": len(set(shared_copies.values())) == 1 and next(iter(shared_copies.values())) == args.shared_config.read_bytes(),
        "all_model_configs_reference_shared_sha": all(config.get("shared_config_sha256") == shared_sha for config in configs.values()),
        "train_manifest_sha256": split["manifests"]["train"]["sha256"] == EXPECTED["train"],
        "val_manifest_sha256": split["manifests"]["val"]["sha256"] == EXPECTED["val"],
        "test_manifest_sha256": split["manifests"]["test"]["sha256"] == EXPECTED["test"],
        "image_id_split_sha256": split["image_id_split"]["sha256"] == EXPECTED["image_split"],
        "source_leakage_zero": all(value == 0 for value in split["source_leakage"].values()),
        "test_roi_count": paired_audit["paired_rows"] == EXPECTED_TEST_ROWS,
        "test_source_count": paired_audit["source_image_ids"] == EXPECTED_TEST_SOURCES,
        "test_class_distribution": split["manifests"]["test"]["class_counts"] == {str(k): v for k, v in EXPECTED_TEST_CLASS_COUNTS.items()},
        "seed": shared.get("seed") == 42,
        "architecture": shared.get("architecture") == "convnext_tiny",
        "num_classes": shared.get("num_classes") == 5,
        "feature_dim": shared.get("feature_dim") == 768,
        "preprocessing_locked": isinstance(shared.get("preprocessing"), dict),
        "augmentation_locked": isinstance(shared.get("augmentation"), dict),
        "augmentation_train_only": shared["augmentation"].get("train_only") is True,
        "augmentation_preview_sha256_identical": len(set(preview_hashes.values())) == 1,
        "optimizer": shared["optimizer"].get("name") == "AdamW",
        "backbone_learning_rate": shared["optimizer"].get("backbone_learning_rate") == 1e-5,
        "classifier_learning_rate": shared["optimizer"].get("classifier_learning_rate") == 1e-4,
        "weight_decay": shared["optimizer"].get("weight_decay") == 1e-4,
        "scheduler": shared.get("scheduler") == {"name": "CosineAnnealingLR", "T_max": 50},
        "batch_size": all(config.get("actual_batch_size") == 64 for config in configs.values()),
        "accumulation_steps": all(config.get("accumulation_steps") == 1 for config in configs.values()),
        "effective_batch_size": all(config.get("effective_batch_size") == 64 for config in configs.values()),
        "workers": shared["data_loader"].get("workers") == 2,
        "maximum_epochs": shared.get("maximum_epochs") == 50,
        "patience": shared["early_stopping"].get("patience") == 10,
        "min_delta": shared["early_stopping"].get("min_delta") == 1e-4,
        "loss_function": shared["loss"].get("name") == "CrossEntropyLoss",
        "class_weights": shared["loss"].get("class_weights") is None,
        "label_smoothing": shared["loss"].get("label_smoothing") == 0.0,
        "amp": shared.get("amp") is True,
        "grad_scaler_initial_scale": "init_scale=1024.0" in train_script_text,
        "gradient_clipping": shared.get("gradient_clip_max_norm") == 1.0,
        "checkpoint_metric": shared["early_stopping"].get("metric") == "validation_macro_f1",
        "classification_head": shared.get("classifier_head") == {"dropout": 0.2, "hidden_layer": False, "linear": [768, 5], "projector": False},
        "validation_metrics_implementation": "def evaluate(" in train_script_text and "def class_metrics(" in train_script_text,
        "test_metrics_implementation": "def evaluate(" in train_script_text,
        "test_evaluation_rule": shared.get("test_evaluation") == "exactly once after best checkpoint is fixed",
        "test_evaluation_count": all(int(meta.get("evaluation_count", -1)) == 1 for meta in test_meta.values()),
        "test_rows": all(int(meta.get("rows", -1)) == EXPECTED_TEST_ROWS for meta in test_meta.values()),
        "test_manifest_in_metadata": all(meta.get("manifest_sha256") == EXPECTED["test"] for meta in test_meta.values()),
        "checkpoint_selection_did_not_use_test": all(meta.get("checkpoint_selection_used_test") is False for meta in test_meta.values()),
        "class_mapping": shared.get("class_mapping") == {str(k): v for k, v in CLASS_MAPPING.items()},
        "output_probability_method": "torch.softmax(logits.float(), dim=1)" in train_script_text,
        "train_shuffle": shared["data_loader"].get("train_shuffle") is True,
        "validation_shuffle": shared["data_loader"].get("validation_shuffle") is False,
        "test_shuffle": shared["data_loader"].get("test_shuffle") is False,
        "all_final_audits_pass": all(final.get("status") == "PASS" for final in finals.values()),
        "all_existing_fairness_audits_pass": all(record.get("status") == "PASS" for record in fairness_records.values()),
        "all_predictions_paired": paired_audit["paired_key_intersection"] == EXPECTED_TEST_ROWS,
        "all_true_labels_identical": paired_audit["true_labels_identical"],
        "all_paths_identical": paired_audit["image_paths_identical"],
        "layernorm_audit_complete": layernorm.get("status") == "PASS",
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failed else "FAIL", "created_at_utc": utc_now(),
        "checks": checks, "fairness_item_count": len(checks),
        "difference_count_excluding_allowed": len(failed), "differences": failed,
        "allowed_differences": [
            "initialization type", "initialization checkpoint", "initialization metadata",
            "output directory", "experiment name", "output model filename",
            "actual early-stopping epoch", "best epoch", "measured metrics", "measured wall time",
        ],
        "initializations": {model: configs[model].get("initialization") for model in MODELS},
        "augmentation_preview_sha256": preview_hashes,
        "shared_config": {"path": str(args.shared_config), "sha256": shared_sha},
        "split": split, "training_script": {"path": str(train_script), "sha256": sha256_file(train_script)},
    }


def overall_tables(metrics: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    for metric in OVERALL_METRICS:
        values = {model: float(metrics[model][metric]) for model in MODELS}
        lower = metric == "loss"
        best_value = min(values.values()) if lower else max(values.values())
        best = [MODEL_LABELS[model] for model in MODELS if math.isclose(values[model], best_value, rel_tol=0.0, abs_tol=1e-15)]
        rows.append({
            "metric": metric,
            "baseline": values["baseline"],
            "cls_proposed": values["cls"],
            "patch_proposed": values["patch"],
            "cls_minus_baseline": values["cls"] - values["baseline"],
            "patch_minus_baseline": values["patch"] - values["baseline"],
            "patch_minus_cls": values["patch"] - values["cls"],
            "best_model": " | ".join(best),
            "direction": "lower_is_better" if lower else "higher_is_better",
        })
        for pair, first, second in PAIR_SPECS:
            pairwise.append({
                "pair": pair, "first_model": MODEL_LABELS[first], "second_model": MODEL_LABELS[second],
                "metric": metric, "first_value": values[first], "second_value": values[second],
                "difference_first_minus_second": values[first] - values[second],
                "direction": "lower_is_better" if lower else "higher_is_better",
            })
    return rows, pairwise


def training_tables(paths: dict[str, dict[str, Path]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    validation: list[dict[str, Any]] = []
    efficiency: list[dict[str, Any]] = []
    metric_rows: dict[str, list[dict[str, str]]] = {}
    model_data: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        rows = read_csv(paths[model]["metrics"])
        metric_rows[model] = rows
        final = read_json(paths[model]["final_audit"])
        test = read_json(paths[model]["test_metrics"])
        best_epoch = int(test["best_epoch"])
        best = next(row for row in rows if int(row["epoch"]) == best_epoch)
        wall_seconds, resumed = legacy.parse_training_wall_seconds(paths[model]["log"])
        if wall_seconds is None:
            raise ValueError(f"Could not derive formal training wall time for {model}")
        data = {
            "completed_epochs": len(rows), "best_epoch": best_epoch,
            "best_validation_loss": float(best["val_loss"]),
            "validation_accuracy": float(best["val_accuracy"]),
            "validation_macro_f1": float(best["val_macro_f1"]),
            "validation_macro_auroc": float(best["val_macro_auroc"]),
            "average_images_per_second": float(np.mean([float(row["images_per_second"]) for row in rows])),
            "formal_training_wall_seconds": wall_seconds,
            "summed_epoch_seconds": sum(float(row["epoch_seconds"]) for row in rows),
            "peak_allocated_vram_gb": max(float(row["gpu_peak_allocated_gb"]) for row in rows),
            "peak_reserved_vram_gb": max(float(row["gpu_peak_reserved_gb"]) for row in rows),
            "early_stopping": bool(final.get("early_stopped", len(rows) < 50)),
            "oom_count": int(final.get("oom_count", 0)),
            "nan_count": int(final.get("nan_count", 0)),
            "inf_count": int(final.get("inf_count", 0)),
            "nonfinite_gradient_count": int(final.get("nonfinite_gradient_count", 0)),
            "resume_checkpoint_used": resumed,
        }
        model_data[model] = data
        efficiency.append({"model": MODEL_LABELS[model], "initialization": {"baseline": "ImageNet", "cls": "RAD-DINO CLS distilled", "patch": "RAD-DINO 7x7 patch distilled"}[model], **data})
    for metric, key, direction in (
        ("completed_epochs", "completed_epochs", "descriptive"),
        ("best_epoch", "best_epoch", "descriptive"),
        ("best_validation_loss", "best_validation_loss", "lower_is_better"),
        ("validation_accuracy", "validation_accuracy", "higher_is_better"),
        ("validation_macro_f1", "validation_macro_f1", "higher_is_better"),
        ("validation_macro_auroc", "validation_macro_auroc", "higher_is_better"),
    ):
        values = {model: model_data[model][key] for model in MODELS}
        validation.append({
            "metric": metric, "baseline": values["baseline"], "cls_proposed": values["cls"],
            "patch_proposed": values["patch"], "cls_minus_baseline": values["cls"] - values["baseline"],
            "patch_minus_baseline": values["patch"] - values["baseline"],
            "patch_minus_cls": values["patch"] - values["cls"], "direction": direction,
        })
    return validation, efficiency, metric_rows


def per_class_tables(metrics: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {"f1": [], "auroc": []}
    for class_id in range(5):
        for metric in ("f1", "auroc"):
            values = {model: float(metrics[model]["per_class"][class_id][metric]) for model in MODELS}
            tables[metric].append({
                "class_id": class_id, "class_name": CLASS_MAPPING[class_id],
                "baseline": values["baseline"], "cls_proposed": values["cls"], "patch_proposed": values["patch"],
                "cls_minus_baseline": values["cls"] - values["baseline"],
                "patch_minus_baseline": values["patch"] - values["baseline"],
                "patch_minus_cls": values["patch"] - values["cls"],
                "highlighted_disease": class_id in {2, 3, 4},
            })
    return tables["f1"], tables["auroc"]


def class_error_table(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        matrix = metrics[model]["confusion_matrix"]
        rows.append({
            "model": MODEL_LABELS[model],
            "true_class_2_correct": int(matrix[2, 2]),
            "class_2_to_class_3": int(matrix[2, 3]),
            "class_2_to_class_4": int(matrix[2, 4]),
            "true_class_4_correct": int(matrix[4, 4]),
            "class_4_to_class_2": int(matrix[4, 2]),
            "class_4_to_class_3": int(matrix[4, 3]),
        })
    return rows


def agreement_table(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def count(predicate: Any) -> int:
        return sum(bool(predicate(row)) for row in paired)
    rows = [
        ("all_three_correct", lambda row: row["all_three_correct"]),
        ("all_three_wrong", lambda row: row["all_three_wrong"]),
        ("all_three_wrong_same_prediction", lambda row: row["all_three_wrong"] and row["all_three_same_prediction"]),
        ("all_three_wrong_different_prediction", lambda row: row["all_three_wrong"] and not row["all_three_same_prediction"]),
        ("exactly_one_model_correct", lambda row: row["exactly_one_model_correct"]),
        ("exactly_two_models_correct", lambda row: row["exactly_two_models_correct"]),
        ("baseline_only_correct", lambda row: row["baseline_only_correct"]),
        ("cls_only_correct", lambda row: row["cls_only_correct"]),
        ("patch_only_correct", lambda row: row["patch_only_correct"]),
        ("baseline_cls_correct_patch_wrong", lambda row: row["baseline_cls_correct_patch_wrong"]),
        ("baseline_patch_correct_cls_wrong", lambda row: row["baseline_patch_correct_cls_wrong"]),
        ("cls_patch_correct_baseline_wrong", lambda row: row["cls_patch_correct_baseline_wrong"]),
        ("all_three_same_prediction", lambda row: row["all_three_same_prediction"]),
        ("predictions_not_all_same", lambda row: not row["all_three_same_prediction"]),
        ("all_three_predictions_different", lambda row: len({row[f"{model}_predicted_class_id"] for model in MODELS}) == 3),
    ]
    return [{"outcome": name, "count": count(predicate), "percent_of_454": count(predicate) / len(paired) * 100} for name, predicate in rows]


def discordant_rows(paired: list[dict[str, Any]], first: str, second: str) -> list[dict[str, Any]]:
    return [row for row in paired if row[f"{first}_predicted_class_id"] != row[f"{second}_predicted_class_id"]]


def matrix_rows(matrix: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"true_class_id": true_id, "true_class_name": CLASS_MAPPING[true_id], **{f"predicted_class_{predicted_id}": int(matrix[true_id, predicted_id]) for predicted_id in range(5)}}
        for true_id in range(5)
    ]


def cluster_bootstrap(
    arrays: dict[str, np.ndarray], repetitions: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sources = sorted(set(arrays["sources"].tolist()))
    groups = {source: np.flatnonzero(arrays["sources"] == source) for source in sources}
    points = {
        model: legacy.confusion_and_metrics(
            arrays["targets"], arrays[f"{model}_predictions"], arrays[f"{model}_probabilities"],
        ) for model in MODELS
    }
    samples = {(pair, metric): [] for pair, _, _ in PAIR_SPECS for metric in BOOTSTRAP_METRICS}
    invalid = {(pair, metric): 0 for pair, _, _ in PAIR_SPECS for metric in BOOTSTRAP_METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        selected = rng.integers(0, len(sources), size=len(sources))
        indices = np.concatenate([groups[sources[index]] for index in selected])
        replicate = {
            model: legacy.confusion_and_metrics(
                arrays["targets"][indices], arrays[f"{model}_predictions"][indices], arrays[f"{model}_probabilities"][indices],
            ) for model in MODELS
        }
        for pair, first, second in PAIR_SPECS:
            for metric in BOOTSTRAP_METRICS:
                first_value, second_value = replicate[first][metric], replicate[second][metric]
                if math.isfinite(first_value) and math.isfinite(second_value):
                    samples[(pair, metric)].append(first_value - second_value)
                else:
                    invalid[(pair, metric)] += 1
    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for pair, first, second in PAIR_SPECS:
        for metric in BOOTSTRAP_METRICS:
            values = np.asarray(samples[(pair, metric)], dtype=np.float64)
            if not len(values):
                raise ValueError(f"No valid bootstrap replicates for {pair}/{metric}")
            point = float(points[first][metric] - points[second][metric])
            lower, upper = (float(value) for value in np.percentile(values, [2.5, 97.5]))
            results.append({
                "pair": pair, "first_model": MODEL_LABELS[first], "second_model": MODEL_LABELS[second],
                "metric": metric, "point_estimate_first_minus_second": point,
                "bootstrap_mean": float(values.mean()),
                "bootstrap_standard_error": float(values.std(ddof=1)),
                "percentile_2_5": lower, "percentile_97_5": upper,
                "confidence_interval_95": f"[{lower:.12f}, {upper:.12f}]",
                "valid_replicates": len(values), "invalid_replicates": invalid[(pair, metric)],
                "probability_difference_gt_zero": float(np.mean(values > 0)),
                "probability_difference_lt_zero": float(np.mean(values < 0)),
                "ci_includes_zero": lower <= 0 <= upper,
                "cluster_unit": "source_image_id", "cluster_count": len(sources), "seed": seed,
            })
            summaries.append({
                "pair": pair, "metric": metric, "requested_replicates": repetitions,
                "valid_replicates": len(values), "invalid_replicates": invalid[(pair, metric)],
                "cluster_count": len(sources), "same_cluster_sequence_across_models_and_pairs": True, "seed": seed,
            })
    metadata = {
        "method": "paired source-cluster percentile bootstrap",
        "cluster_unit": "source_image_id", "cluster_count": len(sources),
        "requested_replicates": repetitions, "seed": seed,
        "sampling": "sample 59 source_image_id clusters with replacement and include every ROI in every sampled cluster",
        "same_cluster_sequence_across_models_and_pairs": True,
    }
    return results, summaries, metadata


def exact_mcnemar(paired: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for pair, first, second in PAIR_SPECS:
        both_correct = sum(row[f"{first}_is_correct"] and row[f"{second}_is_correct"] for row in paired)
        both_wrong = sum(not row[f"{first}_is_correct"] and not row[f"{second}_is_correct"] for row in paired)
        first_only = sum(row[f"{first}_is_correct"] and not row[f"{second}_is_correct"] for row in paired)
        second_only = sum(not row[f"{first}_is_correct"] and row[f"{second}_is_correct"] for row in paired)
        discordant = first_only + second_only
        if discordant:
            lower = min(first_only, second_only)
            tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2 ** discordant)
            p_value = min(1.0, 2.0 * tail)
        else:
            p_value = 1.0
        rows.append({
            "pair": pair, "first_model": MODEL_LABELS[first], "second_model": MODEL_LABELS[second],
            "both_correct": both_correct, "both_wrong": both_wrong,
            "first_correct_second_wrong": first_only,
            "first_wrong_second_correct": second_only,
            "discordant_total": discordant, "exact_two_sided_p_value": p_value,
            "analysis_unit": "ROI", "cluster_caveat": "source-image clustering; supplementary analysis",
        })
    indexed = sorted(enumerate(rows), key=lambda item: item[1]["exact_two_sided_p_value"])
    adjusted_by_index: dict[int, tuple[int, float]] = {}
    running = 0.0
    total = len(rows)
    for rank, (index, row) in enumerate(indexed, start=1):
        adjusted = min(1.0, row["exact_two_sided_p_value"] * (total - rank + 1))
        running = max(running, adjusted)
        adjusted_by_index[index] = (rank, running)
    holm = []
    for index, row in enumerate(rows):
        rank, adjusted = adjusted_by_index[index]
        holm.append({
            "pair": row["pair"], "raw_exact_p_value": row["exact_two_sided_p_value"],
            "holm_rank": rank, "holm_adjusted_p_value": adjusted,
            "reject_at_alpha_0_05": adjusted < 0.05, "family_size": total,
        })
    return rows, holm


def plot_overall(rows: list[dict[str, Any]], path: Path) -> None:
    score_rows = [row for row in rows if row["metric"] != "loss"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].bar(range(3), [rows[0]["baseline"], rows[0]["cls_proposed"], rows[0]["patch_proposed"]], color=[MODEL_COLORS[m] for m in MODELS])
    axes[0].set_xticks(range(3), ["Baseline", "CLS", "Patch"])
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Test loss (lower is better)")
    x = np.arange(len(score_rows)); width = 0.25
    for offset, (model, field, label) in enumerate((("baseline", "baseline", "Baseline"), ("cls", "cls_proposed", "CLS"), ("patch", "patch_proposed", "Patch"))):
        axes[1].bar(x + (offset - 1) * width, [row[field] for row in score_rows], width, label=label, color=MODEL_COLORS[model])
    axes[1].set_xticks(x, [row["metric"].replace("macro_", "M-").replace("weighted_", "W-") for row in score_rows], rotation=25, ha="right")
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("Score"); axes[1].legend(); axes[1].set_title("Test scores (higher is better)")
    figure.suptitle("Three-model test comparison; 454 ROI from 59 source images")
    atomic_save_figure(figure, path)


def plot_validation(rows: list[dict[str, Any]], path: Path) -> None:
    selected = [row for row in rows if row["metric"] in {"best_validation_loss", "validation_accuracy", "validation_macro_f1", "validation_macro_auroc"}]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    loss = next(row for row in selected if row["metric"] == "best_validation_loss")
    axes[0].bar(range(3), [loss["baseline"], loss["cls_proposed"], loss["patch_proposed"]], color=[MODEL_COLORS[m] for m in MODELS])
    axes[0].set_xticks(range(3), ["Baseline", "CLS", "Patch"]); axes[0].set_title("Best validation loss")
    scores = [row for row in selected if row["metric"] != "best_validation_loss"]
    x = np.arange(len(scores)); width = 0.25
    for offset, (model, field, label) in enumerate((("baseline", "baseline", "Baseline"), ("cls", "cls_proposed", "CLS"), ("patch", "patch_proposed", "Patch"))):
        axes[1].bar(x + (offset - 1) * width, [row[field] for row in scores], width, label=label, color=MODEL_COLORS[model])
    axes[1].set_xticks(x, [row["metric"].replace("validation_", "") for row in scores], rotation=20, ha="right")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].set_title("Best-checkpoint validation scores")
    atomic_save_figure(figure, path)


def plot_per_class(rows: list[dict[str, Any]], metric: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 5.5)); x = np.arange(5); width = 0.25
    for offset, (model, field, label) in enumerate((("baseline", "baseline", "Baseline"), ("cls", "cls_proposed", "CLS"), ("patch", "patch_proposed", "Patch"))):
        axis.bar(x + (offset - 1) * width, [row[field] for row in rows], width, label=label, color=MODEL_COLORS[model])
    axis.set_xticks(x, [CLASS_MAPPING[index] for index in range(5)], rotation=20, ha="right")
    axis.set_ylim(0, 1); axis.set_ylabel(metric.upper()); axis.legend(); axis.set_title(f"Per-class test {metric.upper()}")
    atomic_save_figure(figure, path)


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denominator, out=np.zeros_like(matrix, dtype=float), where=denominator != 0)


def plot_confusion(matrix: np.ndarray, title: str, path: Path, vmax: float) -> None:
    figure, axis = plt.subplots(figsize=(7.3, 6.2))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
    figure.colorbar(image, ax=axis, fraction=0.046)
    labels = [name.replace(" ", "\n") for name in CLASS_MAPPING.values()]
    axis.set_xticks(range(5), labels, fontsize=8); axis.set_yticks(range(5), labels, fontsize=8)
    axis.set_xlabel("Predicted disease"); axis.set_ylabel("True disease"); axis.set_title(title + " (shared count scale)")
    threshold = vmax * 0.55
    for row in range(5):
        for column in range(5):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center", color="white" if matrix[row, column] > threshold else "black")
    atomic_save_figure(figure, path)


def plot_confusion_difference(first: np.ndarray, second: np.ndarray, label: str, path: Path) -> None:
    difference = first.astype(float) - second.astype(float)
    limit = max(1.0, float(np.abs(difference).max()))
    figure, axis = plt.subplots(figsize=(7.3, 6.2))
    image = axis.imshow(difference, cmap="coolwarm", vmin=-limit, vmax=limit)
    figure.colorbar(image, ax=axis, fraction=0.046)
    labels = [name.replace(" ", "\n") for name in CLASS_MAPPING.values()]
    axis.set_xticks(range(5), labels, fontsize=8); axis.set_yticks(range(5), labels, fontsize=8)
    axis.set_xlabel("Predicted disease"); axis.set_ylabel("True disease"); axis.set_title(label + " (first - second)")
    for row in range(5):
        for column in range(5):
            axis.text(column, row, f"{difference[row, column]:+.0f}", ha="center", va="center", fontsize=9)
    atomic_save_figure(figure, path)


def plot_bootstrap(rows: list[dict[str, Any]], metric: str, path: Path) -> None:
    selected = [row for row in rows if row["metric"] == metric]
    points = np.asarray([row["point_estimate_first_minus_second"] for row in selected])
    lower = np.asarray([row["percentile_2_5"] for row in selected]); upper = np.asarray([row["percentile_97_5"] for row in selected])
    figure, axis = plt.subplots(figsize=(9, 4.8)); y = np.arange(len(selected))
    axis.errorbar(points, y, xerr=np.vstack((points - lower, upper - points)), fmt="o", color="#34495E", capsize=5)
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(y, [row["pair"].replace("_minus_", " - ") for row in selected])
    axis.set_xlabel("First model - second model"); axis.set_title(f"{metric}: 95% paired source-cluster bootstrap CI")
    axis.grid(True, axis="x", alpha=0.3); atomic_save_figure(figure, path)


def plot_agreement(rows: list[dict[str, Any]], path: Path) -> None:
    names = ["all_three_correct", "all_three_wrong", "baseline_only_correct", "cls_only_correct", "patch_only_correct", "exactly_two_models_correct", "predictions_not_all_same"]
    mapping = {row["outcome"]: row["count"] for row in rows}
    figure, axis = plt.subplots(figsize=(11, 5.3)); values = [mapping[name] for name in names]
    axis.bar(range(len(names)), values, color=["#3D7A57", "#8C8C8C", "#526D82", "#B0574A", "#4F8B63", "#8A6D3B", "#665C84"])
    axis.set_xticks(range(len(names)), [name.replace("_", "\n") for name in names], fontsize=8)
    axis.set_ylabel("ROI count"); axis.set_title("Three-model paired correctness and agreement")
    for index, value in enumerate(values): axis.text(index, value, str(value), ha="center", va="bottom")
    atomic_save_figure(figure, path)


def plot_training_efficiency(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8)); labels = ["Baseline", "CLS", "Patch"]
    for axis, field, title in (
        (axes[0], "completed_epochs", "Completed epochs"),
        (axes[1], "formal_training_wall_seconds", "Formal training wall time (s)"),
        (axes[2], "average_images_per_second", "Average images/s"),
    ):
        values = [row[field] for row in rows]
        axis.bar(range(3), values, color=[MODEL_COLORS[m] for m in MODELS]); axis.set_xticks(range(3), labels); axis.set_title(title)
        for index, value in enumerate(values): axis.text(index, value, f"{value:.1f}", ha="center", va="bottom")
    figure.suptitle("Observed Phase 2 training efficiency")
    atomic_save_figure(figure, path)


def plot_curves(metric_rows: dict[str, list[dict[str, str]]], field: str, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for model in MODELS:
        rows = metric_rows[model]
        axis.plot([int(row["epoch"]) for row in rows], [float(row[field]) for row in rows], marker="o", markersize=2.8, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
    axis.set_xlabel("Epoch (actual completed epochs only)"); axis.set_ylabel(field); axis.set_title(title); axis.grid(True, alpha=0.3); axis.legend()
    atomic_save_figure(figure, path)


def plot_class_errors(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["true_class_2_correct", "class_2_to_class_3", "class_2_to_class_4", "true_class_4_correct", "class_4_to_class_2", "class_4_to_class_3"]
    labels = ["PT correct", "PT -> PF", "PT -> PE", "PE correct", "PE -> PT", "PE -> PF"]
    figure, axis = plt.subplots(figsize=(12, 5.3)); x = np.arange(len(fields)); width = 0.25
    for offset, (model, row) in enumerate(zip(MODELS, rows)):
        axis.bar(x + (offset - 1) * width, [row[field] for field in fields], width, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
    axis.set_xticks(x, labels); axis.set_ylabel("ROI count"); axis.set_title("Pleural thickening and pleural effusion error transitions"); axis.legend()
    atomic_save_figure(figure, path)


def protected_files(args: argparse.Namespace, paths: dict[str, dict[str, Path]]) -> dict[str, Path]:
    result: dict[str, Path] = {
        "shared_config": args.shared_config,
        "train_manifest": args.split_dir / "train_roi_manifest.csv",
        "val_manifest": args.split_dir / "val_roi_manifest.csv",
        "test_manifest": args.split_dir / "test_roi_manifest.csv",
        "image_id_split_manifest": args.split_dir / "image_id_split_manifest.csv",
        "phase2_training_script": args.project_root / "src/train_phase2_convnext_tiny_finetune.py",
        "legacy_comparison_script": args.project_root / "src/compare_proposed_vs_baseline.py",
        "cls_phase1_export": args.project_root / "outputs/raddino_convnext_tiny_experiment_seed42/phase1_distillation/checkpoints/distilled_convnext_tiny_backbone.pt",
        "patch_phase1_export": args.project_root / "outputs/raddino_convnext_tiny_patch_experiment_seed42/phase1_patch_distillation/checkpoints/patch_distilled_convnext_tiny_backbone.pt",
        "legacy_comparison_summary": args.project_root / "outputs/raddino_convnext_tiny_experiment_seed42/final_comparison/final_comparison_summary.json",
    }
    for model in MODELS:
        for name in (
            "metrics", "test_metrics", "per_class", "predictions", "test_metadata", "summary",
            "fairness", "stage0", "final_audit", "best", "export", "config", "shared_copy", "log",
        ):
            result[f"{model}_{name}"] = paths[model][name]
        if paths[model]["initialization"].is_file():
            result[f"{model}_initialization"] = paths[model]["initialization"]
        if paths[model]["confusion_csv"].is_file():
            result[f"{model}_confusion_csv"] = paths[model]["confusion_csv"]
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected input missing: {missing}")
    return result


def hash_files(files: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in sorted(files.items())}


def input_integrity_audit(
    paths: dict[str, dict[str, Path]], prediction_audits: dict[str, Any],
    paired_audit: dict[str, Any], metric_audit: dict[str, Any], before: dict[str, str], after: dict[str, str],
) -> dict[str, Any]:
    unchanged = {name: before[name] == after[name] for name in before}
    canonical_absences = {
        model: {
            "confusion_matrix_csv_exists": paths[model]["confusion_csv"].is_file(),
            "initialization_audit_json_exists": paths[model]["initialization"].is_file(),
            "fairness_audit_json_exists": paths[model]["fairness_physical"].is_file(),
        } for model in MODELS
    }
    semantic_sources = {
        model: {
            "confusion_matrix": str(paths[model]["confusion_csv"] if paths[model]["confusion_csv"].is_file() else paths[model]["test_metrics"]),
            "initialization": str(paths[model]["initialization"] if paths[model]["initialization"].is_file() else paths[model]["stage0"]),
            "fairness": str(paths[model]["fairness"]),
        } for model in MODELS
    }
    checks = {
        "all_semantic_required_inputs_present": True,
        "all_prediction_rows_454": all(item["rows"] == EXPECTED_TEST_ROWS for item in prediction_audits.values()),
        "all_prediction_sources_59": all(item["source_image_ids"] == EXPECTED_TEST_SOURCES for item in prediction_audits.values()),
        "all_probability_fields_valid": all(item["nonfinite_probability_or_confidence"] == 0 and item["max_probability_sum_error"] <= 1e-5 for item in prediction_audits.values()),
        "paired_rows_454": paired_audit["paired_rows"] == EXPECTED_TEST_ROWS,
        "paired_keys_complete": all(value == 0 for value in paired_audit["missing_keys_by_model"].values()),
        "paired_keys_unique": all(value == 0 for value in paired_audit["duplicate_keys_by_model"].values()),
        "metric_integrity_pass": metric_audit["status"] == "PASS",
        "protected_inputs_unchanged": all(unchanged.values()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failed else "FAIL", "created_at_utc": utc_now(),
        "checks": checks, "failed_checks": failed,
        "prediction_audit": prediction_audits, "pairing_audit": paired_audit,
        "metric_and_confusion_recomputation_audit": metric_audit,
        "canonical_artifact_presence": canonical_absences,
        "semantic_evidence_sources": semantic_sources,
        "compatibility_note": (
            "Baseline and CLS store the test confusion matrix inside test_metrics.json and their initialization load audit inside stage0_smoke_test.json; "
            "Patch additionally has standalone confusion_matrix.csv and initialization_audit.json. CLS has no standalone fairness audit; "
            "the existing Patch three-model fairness audit is used as its fairness evidence. No legacy artifact was synthesized or modified."
        ),
        "protected_sha256_before": before, "protected_sha256_after": after,
        "protected_sha256_unchanged": unchanged,
    }


def conclusions(
    metrics: dict[str, dict[str, Any]], bootstrap: list[dict[str, Any]],
    efficiency: list[dict[str, Any]], layernorm: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    bootstrap_map = {(row["pair"], row["metric"]): row for row in bootstrap}
    principal = ("accuracy", "macro_f1", "weighted_f1", "macro_auroc")
    all_primary_include_zero = all(row["ci_includes_zero"] for row in bootstrap)

    def pair_statement(pair: str, first: str, second: str) -> str:
        supported = [metric for metric in principal if not bootstrap_map[(pair, metric)]["ci_includes_zero"]]
        if supported:
            directions = [bootstrap_map[(pair, metric)]["point_estimate_first_minus_second"] for metric in supported]
            descriptor = "higher" if all(value > 0 for value in directions) else "lower" if all(value < 0 for value in directions) else "mixed"
            return f"{MODEL_LABELS[first]} versus {MODEL_LABELS[second]} has non-zero 95% cluster-bootstrap CI for {', '.join(supported)} ({descriptor} point direction)."
        return f"{MODEL_LABELS[first]} versus {MODEL_LABELS[second]} has no principal metric whose 95% source-cluster bootstrap CI excludes zero."

    class2 = {model: metrics[model]["per_class"][2]["f1"] for model in MODELS}
    class4 = {model: metrics[model]["per_class"][4]["f1"] for model in MODELS}
    efficiency_map = {row["model"]: row for row in efficiency}
    lines = [
        "# Research Conclusion",
        "",
        "## Pairwise findings",
        "",
        f"- {pair_statement('patch_minus_cls', 'patch', 'cls')}",
        f"- {pair_statement('patch_minus_baseline', 'patch', 'baseline')}",
        f"- {pair_statement('cls_minus_baseline', 'cls', 'baseline')}",
        "- Primary inference uses the paired source-image cluster bootstrap. ROI-level exact McNemar tests are supplementary because multiple ROI can originate from one source image.",
        "",
        "## Disease-specific findings",
        "",
        f"- Pleural thickening F1 point estimates were Baseline {class2['baseline']:.6f}, CLS {class2['cls']:.6f}, and Patch {class2['patch']:.6f}. Patch shows a class-specific point improvement over both alternatives, without a class-specific bootstrap claim.",
        f"- Pleural effusion F1 point estimates were Baseline {class4['baseline']:.6f}, CLS {class4['cls']:.6f}, and Patch {class4['patch']:.6f}. Patch is lower and its confusion matrix contains more Pleural effusion to Pleural thickening errors.",
        "- Successful Patch Phase 1 feature alignment does not, by itself, establish a broad disease-classification improvement. Downstream gains are modest and class-dependent.",
        "",
        "## Training and interpretation",
        "",
        f"- Completed Phase 2 epochs were Baseline {efficiency_map[MODEL_LABELS['baseline']]['completed_epochs']}, CLS {efficiency_map[MODEL_LABELS['cls']]['completed_epochs']}, and Patch {efficiency_map[MODEL_LABELS['patch']]['completed_epochs']}; observed formal wall times were {efficiency_map[MODEL_LABELS['baseline']]['formal_training_wall_seconds']:.1f}s, {efficiency_map[MODEL_LABELS['cls']]['formal_training_wall_seconds']:.1f}s, and {efficiency_map[MODEL_LABELS['patch']]['formal_training_wall_seconds']:.1f}s, respectively.",
        f"- Final LayerNorm initialization was not identical. Potential CLS-versus-Patch confounding: {str(layernorm['potential_cls_vs_patch_confounder']).lower()}. CLS loaded the Phase 1 distilled final norm; Patch used torchvision default initialization.",
        "- This is a single-seed comparison with 59 test source images. Confidence intervals reflect source-level resampling but cannot establish generalization across seeds or external cohorts.",
    ]
    if all_primary_include_zero:
        lines.extend([
            "",
            "> The three initialization strategies have similar overall performance; current evidence is insufficient to support a stable, comprehensive advantage for any model.",
            "",
            "> Patch-level distillation shows class-specific improvement, but it has not produced a statistically supported overall advantage.",
        ])
    decision = {
        "all_principal_95_percent_cis_include_zero": all_primary_include_zero,
        "patch_has_statistically_supported_overall_advantage": any(
            not bootstrap_map[("patch_minus_baseline", metric)]["ci_includes_zero"] and bootstrap_map[("patch_minus_baseline", metric)]["point_estimate_first_minus_second"] > 0
            for metric in principal
        ),
        "patch_class_specific_pleural_thickening_point_improvement": class2["patch"] > max(class2["baseline"], class2["cls"]),
        "patch_pleural_effusion_point_degradation": class4["patch"] < min(class4["baseline"], class4["cls"]),
        "single_seed": True, "test_source_image_count": EXPECTED_TEST_SOURCES,
    }
    return lines, decision


def report_lines(
    fairness: dict[str, Any], input_audit: dict[str, Any], layernorm: dict[str, Any],
    overall: list[dict[str, Any]], validation: list[dict[str, Any]], efficiency: list[dict[str, Any]],
    f1_rows: list[dict[str, Any]], auroc_rows: list[dict[str, Any]], metrics: dict[str, dict[str, Any]],
    class_errors: list[dict[str, Any]], agreement: list[dict[str, Any]], bootstrap: list[dict[str, Any]],
    mcnemar: list[dict[str, Any]], holm: list[dict[str, Any]], decision: dict[str, Any],
) -> list[str]:
    lines = [
        "# Three-Model Final Fair Comparison",
        "",
        f"- Final status: PASS",
        f"- Fairness checks: {fairness['fairness_item_count']}; non-allowed differences: {fairness['difference_count_excluding_allowed']}",
        f"- Test set: {input_audit['pairing_audit']['paired_rows']} paired ROI from {input_audit['pairing_audit']['source_image_ids']} source images",
        "- No training, validation evaluation, test evaluation, image inference, threshold tuning, RAD-DINO loading, or teacher-cache forward was performed.",
        "",
        "## Test metrics",
        "",
        "| Metric | Baseline | CLS | Patch | CLS-Baseline | Patch-Baseline | Patch-CLS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(f"| {row['metric']} | {row['baseline']:.9f} | {row['cls_proposed']:.9f} | {row['patch_proposed']:.9f} | {row['cls_minus_baseline']:+.9f} | {row['patch_minus_baseline']:+.9f} | {row['patch_minus_cls']:+.9f} |")
    lines.extend(["", "## Best-checkpoint validation", "", "| Metric | Baseline | CLS | Patch |", "|---|---:|---:|---:|"])
    for row in validation:
        lines.append(f"| {row['metric']} | {row['baseline']:.9f} | {row['cls_proposed']:.9f} | {row['patch_proposed']:.9f} |")
    lines.extend(["", "## Per-class F1 and AUROC", "", "| Disease | F1 Baseline | F1 CLS | F1 Patch | AUROC Baseline | AUROC CLS | AUROC Patch |", "|---|---:|---:|---:|---:|---:|---:|"])
    for f1, auc in zip(f1_rows, auroc_rows):
        lines.append(f"| {f1['class_name']} | {f1['baseline']:.6f} | {f1['cls_proposed']:.6f} | {f1['patch_proposed']:.6f} | {auc['baseline']:.6f} | {auc['cls_proposed']:.6f} | {auc['patch_proposed']:.6f} |")
    lines.extend(["", "## Confusion matrices", "", "Rows are true classes and columns are predicted classes in class-ID order 0 through 4."])
    for model in MODELS:
        lines.extend(["", f"**{MODEL_LABELS[model]}**", "", "```text", np.array2string(metrics[model]["confusion_matrix"]), "```"])
    lines.extend(["", "Difference matrices always use first model minus second model, cell by cell.", "", "## Pleural thickening / effusion transitions", ""])
    for row in class_errors:
        lines.append(f"- {row['model']}: class 2 correct={row['true_class_2_correct']}, 2->3={row['class_2_to_class_3']}, 2->4={row['class_2_to_class_4']}; class 4 correct={row['true_class_4_correct']}, 4->2={row['class_4_to_class_2']}, 4->3={row['class_4_to_class_3']}.")
    lines.extend(["", "## Source-cluster bootstrap", "", "Differences are first model minus second model; 10,000 shared-sequence source-cluster replicates were requested."])
    for row in bootstrap:
        lines.append(f"- {row['pair']} / {row['metric']}: point={row['point_estimate_first_minus_second']:+.6f}, 95% CI [{row['percentile_2_5']:+.6f}, {row['percentile_97_5']:+.6f}], valid={row['valid_replicates']}, invalid={row['invalid_replicates']}, P(>0)={row['probability_difference_gt_zero']:.4f}, P(<0)={row['probability_difference_lt_zero']:.4f}.")
    lines.extend(["", "## McNemar (supplementary)", ""])
    holm_map = {row["pair"]: row for row in holm}
    for row in mcnemar:
        lines.append(f"- {row['pair']}: first-only correct={row['first_correct_second_wrong']}, second-only correct={row['first_wrong_second_correct']}, exact p={row['exact_two_sided_p_value']:.6f}, Holm-adjusted p={holm_map[row['pair']]['holm_adjusted_p_value']:.6f}.")
    agreement_map = {row["outcome"]: row["count"] for row in agreement}
    lines.extend([
        "", "## Agreement", "",
        f"- All correct: {agreement_map['all_three_correct']}; all wrong: {agreement_map['all_three_wrong']}; all same prediction: {agreement_map['all_three_same_prediction']}; not all same: {agreement_map['predictions_not_all_same']}.",
        f"- Baseline only correct: {agreement_map['baseline_only_correct']}; CLS only correct: {agreement_map['cls_only_correct']}; Patch only correct: {agreement_map['patch_only_correct']}.",
        "", "## Final LayerNorm", "",
        f"- Baseline: {layernorm['models']['baseline']['final_norm_initialization_source']}.",
        f"- CLS: {layernorm['models']['cls']['final_norm_initialization_source']}.",
        f"- Patch: {layernorm['models']['patch']['final_norm_initialization_source']}.",
        f"- Potential confounder for attributing CLS-versus-Patch differences solely to distillation target: {str(layernorm['potential_cls_vs_patch_confounder']).lower()}.",
        "", "## Interpretation", "",
        f"- All principal 95% CIs include zero: {str(decision['all_principal_95_percent_cis_include_zero']).lower()}.",
        f"- Patch has statistically supported overall advantage: {str(decision['patch_has_statistically_supported_overall_advantage']).lower()}.",
        "- See `research_conclusion.md` for the restrained research interpretation and limitations.",
    ])
    return lines


def expected_output_files() -> set[str]:
    return {
        "fairness_audit.json", "input_integrity_audit.json", "layernorm_initialization_audit.json",
        "three_model_comparison_report.md", "research_conclusion.md", "comparison_summary.json", "environment.json",
        "tables/overall_metrics_comparison.csv", "tables/validation_metrics_comparison.csv",
        "tables/per_class_f1_comparison.csv", "tables/per_class_auroc_comparison.csv",
        "tables/training_efficiency_comparison.csv", "tables/pairwise_metric_differences.csv",
        "tables/three_model_agreement_summary.csv", "tables/class2_class4_error_analysis.csv",
        "predictions/paired_three_model_test_predictions.csv", "predictions/baseline_vs_cls_discordant.csv",
        "predictions/baseline_vs_patch_discordant.csv", "predictions/cls_vs_patch_discordant.csv",
        "predictions/all_three_disagreement_cases.csv",
        "statistics/cluster_bootstrap_results.csv", "statistics/cluster_bootstrap_replicate_summary.csv",
        "statistics/mcnemar_pairwise_results.csv", "statistics/mcnemar_holm_adjusted.csv",
        "confusion_matrices/baseline_confusion_matrix.csv", "confusion_matrices/cls_confusion_matrix.csv",
        "confusion_matrices/patch_confusion_matrix.csv", "confusion_matrices/patch_minus_baseline.csv",
        "confusion_matrices/patch_minus_cls.csv", "confusion_matrices/cls_minus_baseline.csv",
        "figures/overall_metrics_three_models.png", "figures/validation_metrics_three_models.png",
        "figures/per_class_f1_three_models.png", "figures/per_class_auroc_three_models.png",
        "figures/confusion_matrix_baseline.png", "figures/confusion_matrix_cls.png",
        "figures/confusion_matrix_patch.png", "figures/confusion_matrix_difference_patch_vs_baseline.png",
        "figures/confusion_matrix_difference_patch_vs_cls.png", "figures/cluster_bootstrap_ci_accuracy.png",
        "figures/cluster_bootstrap_ci_macro_f1.png", "figures/cluster_bootstrap_ci_weighted_f1.png",
        "figures/cluster_bootstrap_ci_macro_auroc.png", "figures/paired_correctness_three_models.png",
        "figures/training_efficiency_three_models.png", "figures/validation_macro_f1_curves.png",
        "figures/validation_loss_curves.png", "figures/class2_class4_confusion_comparison.png",
    }


def write_outputs(
    root: Path, fairness: dict[str, Any], input_audit: dict[str, Any], layernorm: dict[str, Any],
    overall: list[dict[str, Any]], validation: list[dict[str, Any]], f1_rows: list[dict[str, Any]],
    auroc_rows: list[dict[str, Any]], efficiency: list[dict[str, Any]], pairwise: list[dict[str, Any]],
    agreement: list[dict[str, Any]], class_errors: list[dict[str, Any]], paired: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]], bootstrap_summary: list[dict[str, Any]], bootstrap_metadata: dict[str, Any],
    mcnemar: list[dict[str, Any]], holm: list[dict[str, Any]], metrics: dict[str, dict[str, Any]],
    metric_rows: dict[str, list[dict[str, str]]], report: list[str], research: list[str],
    comparison_summary: dict[str, Any], environment: dict[str, Any],
) -> None:
    atomic_write_json(root / "fairness_audit.json", fairness)
    atomic_write_json(root / "input_integrity_audit.json", input_audit)
    atomic_write_json(root / "layernorm_initialization_audit.json", layernorm)
    atomic_write_text(root / "three_model_comparison_report.md", "\n".join(report) + "\n")
    atomic_write_text(root / "research_conclusion.md", "\n".join(research) + "\n")
    atomic_write_json(root / "comparison_summary.json", comparison_summary)
    atomic_write_json(root / "environment.json", environment)

    atomic_write_csv(root / "tables/overall_metrics_comparison.csv", overall)
    atomic_write_csv(root / "tables/validation_metrics_comparison.csv", validation)
    atomic_write_csv(root / "tables/per_class_f1_comparison.csv", f1_rows)
    atomic_write_csv(root / "tables/per_class_auroc_comparison.csv", auroc_rows)
    atomic_write_csv(root / "tables/training_efficiency_comparison.csv", efficiency)
    atomic_write_csv(root / "tables/pairwise_metric_differences.csv", pairwise)
    atomic_write_csv(root / "tables/three_model_agreement_summary.csv", agreement)
    atomic_write_csv(root / "tables/class2_class4_error_analysis.csv", class_errors)

    atomic_write_csv(root / "predictions/paired_three_model_test_predictions.csv", paired)
    atomic_write_csv(root / "predictions/baseline_vs_cls_discordant.csv", discordant_rows(paired, "baseline", "cls"))
    atomic_write_csv(root / "predictions/baseline_vs_patch_discordant.csv", discordant_rows(paired, "baseline", "patch"))
    atomic_write_csv(root / "predictions/cls_vs_patch_discordant.csv", discordant_rows(paired, "cls", "patch"))
    atomic_write_csv(root / "predictions/all_three_disagreement_cases.csv", [row for row in paired if not row["all_three_same_prediction"]])

    atomic_write_csv(root / "statistics/cluster_bootstrap_results.csv", bootstrap)
    atomic_write_csv(root / "statistics/cluster_bootstrap_replicate_summary.csv", bootstrap_summary)
    atomic_write_csv(root / "statistics/mcnemar_pairwise_results.csv", mcnemar)
    atomic_write_csv(root / "statistics/mcnemar_holm_adjusted.csv", holm)

    baseline = metrics["baseline"]["confusion_matrix"]
    cls = metrics["cls"]["confusion_matrix"]
    patch = metrics["patch"]["confusion_matrix"]
    atomic_write_csv(root / "confusion_matrices/baseline_confusion_matrix.csv", matrix_rows(baseline))
    atomic_write_csv(root / "confusion_matrices/cls_confusion_matrix.csv", matrix_rows(cls))
    atomic_write_csv(root / "confusion_matrices/patch_confusion_matrix.csv", matrix_rows(patch))
    atomic_write_csv(root / "confusion_matrices/patch_minus_baseline.csv", matrix_rows(patch - baseline))
    atomic_write_csv(root / "confusion_matrices/patch_minus_cls.csv", matrix_rows(patch - cls))
    atomic_write_csv(root / "confusion_matrices/cls_minus_baseline.csv", matrix_rows(cls - baseline))

    plot_overall(overall, root / "figures/overall_metrics_three_models.png")
    plot_validation(validation, root / "figures/validation_metrics_three_models.png")
    plot_per_class(f1_rows, "f1", root / "figures/per_class_f1_three_models.png")
    plot_per_class(auroc_rows, "auroc", root / "figures/per_class_auroc_three_models.png")
    vmax = max(float(matrix.max()) for matrix in (baseline, cls, patch))
    plot_confusion(baseline, "ImageNet Baseline confusion matrix", root / "figures/confusion_matrix_baseline.png", vmax)
    plot_confusion(cls, "RAD-DINO CLS confusion matrix", root / "figures/confusion_matrix_cls.png", vmax)
    plot_confusion(patch, "RAD-DINO Patch confusion matrix", root / "figures/confusion_matrix_patch.png", vmax)
    plot_confusion_difference(patch, baseline, "Patch - Baseline confusion matrix", root / "figures/confusion_matrix_difference_patch_vs_baseline.png")
    plot_confusion_difference(patch, cls, "Patch - CLS confusion matrix", root / "figures/confusion_matrix_difference_patch_vs_cls.png")
    for metric in BOOTSTRAP_METRICS:
        plot_bootstrap(bootstrap, metric, root / f"figures/cluster_bootstrap_ci_{metric}.png")
    plot_agreement(agreement, root / "figures/paired_correctness_three_models.png")
    plot_training_efficiency(efficiency, root / "figures/training_efficiency_three_models.png")
    plot_curves(metric_rows, "val_macro_f1", "Validation macro-F1 curves", root / "figures/validation_macro_f1_curves.png")
    plot_curves(metric_rows, "val_loss", "Validation loss curves", root / "figures/validation_loss_curves.png")
    plot_class_errors(class_errors, root / "figures/class2_class4_confusion_comparison.png")


def verify_output_contract(root: Path) -> dict[str, Any]:
    expected = expected_output_files()
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    csv_files = sorted(path for path in root.rglob("*.csv") if path.is_file())
    bom_failures = [str(path) for path in csv_files if path.read_bytes()[:3] != b"\xef\xbb\xbf"]
    temporary = sorted(str(path) for path in root.rglob("*") if path.name.endswith((".tmp", ".writing")))
    empty = sorted(str(path) for path in root.rglob("*") if path.is_file() and path.stat().st_size == 0)
    checks = {
        "expected_file_count": len(actual) == len(expected) == 48,
        "missing_files_zero": not missing, "extra_files_zero": not extra,
        "all_csv_utf8_bom": not bom_failures, "temporary_files_zero": not temporary,
        "empty_files_zero": not empty,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Output contract failed: {failed}; missing={missing}; extra={extra}")
    return {
        "status": "PASS", "checks": checks, "file_count": len(actual),
        "csv_count": len(csv_files), "csv_bom_failures": bom_failures,
        "temporary_files": temporary, "files": sorted(actual),
    }


def run(args: argparse.Namespace) -> int:
    if args.seed != 42 or args.bootstrap_replicates != 10000:
        raise ValueError("Locked comparison requires seed=42 and bootstrap-replicates=10000")
    ensure_clean_destination(args.output_dir)
    paths = model_paths(args)
    split, manifests = split_audit(args)
    protected = protected_files(args, paths)
    before_hashes = hash_files(protected)

    prediction_audits: dict[str, Any] = {}
    enriched: dict[str, list[dict[str, Any]]] = {}
    for model in MODELS:
        enriched[model], prediction_audits[model] = enrich_predictions(
            read_csv(paths[model]["predictions"]), manifests["test"], model,
        )
    paired, arrays, paired_audit = pair_predictions(enriched)
    metrics, metric_audit = metric_integrity(paths, arrays)
    layernorm = layernorm_audit(args, paths)
    fairness = fairness_audit(args, paths, split, paired_audit, layernorm)
    if fairness["status"] != "PASS":
        raise RuntimeError(f"Fairness audit failed: {fairness['differences']}")

    dry_after_hashes = hash_files(protected)
    dry_input_audit = input_integrity_audit(paths, prediction_audits, paired_audit, metric_audit, before_hashes, dry_after_hashes)
    if dry_input_audit["status"] != "PASS":
        raise RuntimeError(f"Input integrity failed: {dry_input_audit['failed_checks']}")
    if args.dry_run:
        print(json.dumps({
            "status": "PASS", "mode": "dry-run", "output_created": False,
            "fairness_item_count": fairness["fairness_item_count"],
            "difference_count_excluding_allowed": fairness["difference_count_excluding_allowed"],
            "prediction_audit": prediction_audits, "pairing_audit": paired_audit,
            "metric_integrity": metric_audit, "layernorm_status": layernorm["status"],
            "protected_inputs_unchanged": all(before_hashes[name] == dry_after_hashes[name] for name in before_hashes),
        }, ensure_ascii=False, indent=2))
        return 0

    overall, pairwise = overall_tables(metrics)
    validation, efficiency, metric_rows = training_tables(paths)
    f1_rows, auroc_rows = per_class_tables(metrics)
    class_errors = class_error_table(metrics)
    agreement = agreement_table(paired)
    bootstrap, bootstrap_summary, bootstrap_metadata = cluster_bootstrap(arrays, args.bootstrap_replicates, args.seed)
    mcnemar, holm = exact_mcnemar(paired)
    research, decision = conclusions(metrics, bootstrap, efficiency, layernorm)
    after_hashes = hash_files(protected)
    input_audit = input_integrity_audit(paths, prediction_audits, paired_audit, metric_audit, before_hashes, after_hashes)
    if input_audit["status"] != "PASS":
        raise RuntimeError(f"Protected input changed: {input_audit['failed_checks']}")
    report = report_lines(
        fairness, input_audit, layernorm, overall, validation, efficiency, f1_rows, auroc_rows,
        metrics, class_errors, agreement, bootstrap, mcnemar, holm, decision,
    )
    environment = {
        "created_at_utc": utc_now(), "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "matplotlib": matplotlib.__version__, "torch": torch.__version__,
        "device_used": "none; artifact-only comparison", "model_forward_executed": False,
        "rad_dino_loaded": False, "teacher_cache_loaded": False, "seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
    }
    comparison_summary = {
        "status": "PASS", "created_at_utc": utc_now(), "models": MODEL_LABELS,
        "test": {"roi_rows": EXPECTED_TEST_ROWS, "source_image_ids": EXPECTED_TEST_SOURCES, "class_counts": EXPECTED_TEST_CLASS_COUNTS},
        "fairness": fairness, "input_integrity": input_audit, "layernorm_initialization": layernorm,
        "overall_metrics": overall, "validation_metrics": validation, "training_efficiency": efficiency,
        "per_class_f1": f1_rows, "per_class_auroc": auroc_rows,
        "confusion_matrices": {model: metrics[model]["confusion_matrix"].tolist() for model in MODELS},
        "class2_class4_error_analysis": class_errors, "agreement": agreement,
        "cluster_bootstrap": {"metadata": bootstrap_metadata, "results": bootstrap},
        "mcnemar": mcnemar, "mcnemar_holm": holm, "research_decision": decision,
        "prohibited_actions": {"training": False, "validation_evaluation": False, "test_evaluation": False, "image_inference": False, "threshold_tuning": False},
        "expected_output_file_count": 48,
    }

    staging = args.output_dir.with_name(args.output_dir.name + ".writing")
    try:
        staging.mkdir(parents=True, exist_ok=False)
        write_outputs(
            staging, fairness, input_audit, layernorm, overall, validation, f1_rows, auroc_rows,
            efficiency, pairwise, agreement, class_errors, paired, bootstrap, bootstrap_summary,
            bootstrap_metadata, mcnemar, holm, metrics, metric_rows, report, research,
            comparison_summary, environment,
        )
        output_audit = verify_output_contract(staging)
        comparison_summary["output_contract"] = output_audit
        atomic_write_json(staging / "comparison_summary.json", comparison_summary)
        verify_output_contract(staging)
        if args.output_dir.exists():
            args.output_dir.rmdir()
        os.replace(staging, args.output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(json.dumps({
        "status": "PASS", "output_dir": str(args.output_dir), "output_files": 48,
        "fairness_items": fairness["fairness_item_count"],
        "difference_count_excluding_allowed": fairness["difference_count_excluding_allowed"],
        "paired_rows": len(paired), "source_image_ids": paired_audit["source_image_ids"],
        "overall_metrics": overall, "bootstrap": bootstrap, "mcnemar": mcnemar,
        "holm": holm, "decision": decision, "protected_inputs_unchanged": True,
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--cls-dir", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--shared-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for name in ("project_root", "baseline_dir", "cls_dir", "patch_dir", "split_dir", "shared_config", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
