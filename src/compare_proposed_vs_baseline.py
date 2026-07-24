#!/usr/bin/env python
"""Compare completed Proposed and Baseline Phase 2 experiments without inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_TEST_SHA256 = "2130a73dcbadec1d6b4bba68f809db7eeed25d1ea421c4d450d3e0b4d015551a"
EXPECTED_TEST_ROWS = 454
EXPECTED_TEST_SOURCES = 59
CLASS_MAPPING = {
    0: "Aortic enlargement",
    1: "Cardiomegaly",
    2: "Pleural thickening",
    3: "Pulmonary fibrosis",
    4: "Pleural effusion",
}
PRIMARY_BOOTSTRAP_METRICS = ("accuracy", "macro_f1", "weighted_f1", "macro_auroc")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


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


def atomic_save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, dpi=170, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def require_files(directory: Path, experiment: str) -> dict[str, Path]:
    config_name = "experiment_config.json" if experiment == "proposed" else "baseline_experiment_config.json"
    summary_name = "phase2_proposed_training_summary.md" if experiment == "proposed" else "phase2_baseline_training_summary.md"
    relative = {
        "best_checkpoint": "checkpoints/best.pt",
        "metrics": "metrics/phase2_metrics.csv",
        "predictions": "predictions/test_predictions.csv",
        "test_metrics": "test_results/test_metrics.json",
        "per_class_metrics": "test_results/per_class_metrics.csv",
        "classification_report": "test_results/classification_report.csv",
        "test_metadata": "test_results/test_evaluation_metadata.json",
        "summary": summary_name,
        "experiment_config": f"config/{config_name}",
        "shared_config_copy": "config/shared_phase2_finetune_config.json",
        "final_audit": "diagnostics/phase2_final_audit.json",
        "log": "logs/phase2.log",
    }
    paths = {name: directory / value for name, value in relative.items()}
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"{experiment} required inputs are missing or empty: {missing}")
    return paths


def prediction_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["record_index"], row["source_image_id"], row["image_path"], row["true_class_id"])


def validate_predictions(rows: list[dict[str, str]], label: str) -> dict[str, Any]:
    required = {
        "record_index", "source_image_id", "image_path", "filename", "true_class_id",
        "true_class_name", "predicted_class_id", "predicted_class_name", "confidence", "correct",
        *{f"probability_class_{index}" for index in range(5)},
    }
    if len(rows) != EXPECTED_TEST_ROWS:
        raise ValueError(f"{label} predictions rows={len(rows)}, expected {EXPECTED_TEST_ROWS}")
    missing_fields = required - set(rows[0] if rows else {})
    if missing_fields:
        raise ValueError(f"{label} prediction fields missing: {sorted(missing_fields)}")
    keys = [prediction_key(row) for row in rows]
    record_indices = [row["record_index"] for row in rows]
    if len(set(keys)) != EXPECTED_TEST_ROWS or len(set(record_indices)) != EXPECTED_TEST_ROWS:
        raise ValueError(f"{label} predictions contain duplicate keys or record_index")
    max_sum_error = 0.0
    nonfinite = 0
    for row in rows:
        probabilities = [float(row[f"probability_class_{index}"]) for index in range(5)]
        confidence = float(row["confidence"])
        nonfinite += sum(not math.isfinite(value) for value in probabilities) + int(not math.isfinite(confidence))
        max_sum_error = max(max_sum_error, abs(sum(probabilities) - 1.0))
        class_id = int(row["true_class_id"])
        predicted = int(row["predicted_class_id"])
        if class_id not in CLASS_MAPPING or predicted not in CLASS_MAPPING:
            raise ValueError(f"{label} has invalid class id")
        if row["true_class_name"] != CLASS_MAPPING[class_id] or row["predicted_class_name"] != CLASS_MAPPING[predicted]:
            raise ValueError(f"{label} class mapping mismatch")
        if parse_bool(row["correct"]) != (class_id == predicted):
            raise ValueError(f"{label} correctness mismatch at record {row['record_index']}")
    if nonfinite or max_sum_error > 1e-5:
        raise ValueError(f"{label} probability audit failed: nonfinite={nonfinite}, max_sum_error={max_sum_error}")
    return {
        "rows": len(rows),
        "unique_record_index": len(set(record_indices)),
        "unique_keys": len(set(keys)),
        "source_image_ids": len({row["source_image_id"] for row in rows}),
        "max_probability_sum_error": max_sum_error,
        "nonfinite_values": nonfinite,
    }


def confusion_and_metrics(targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((5, 5), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1
    per_class: dict[int, dict[str, float | int]] = {}
    auroc_valid = True
    for class_id in range(5):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        tn = int(confusion.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        binary = (targets == class_id).astype(np.int64)
        positives = int(binary.sum())
        negatives = len(binary) - positives
        if positives == 0 or negatives == 0:
            auroc = float("nan")
            auroc_valid = False
        else:
            order = np.argsort(-probabilities[:, class_id], kind="stable")
            sorted_binary = binary[order]
            tpr = np.concatenate(([0.0], np.cumsum(sorted_binary) / positives))
            fpr = np.concatenate(([0.0], np.cumsum(1 - sorted_binary) / negatives))
            auroc = float(np.trapezoid(tpr, fpr))
        per_class[class_id] = {
            "precision": precision, "recall": recall, "f1": f1, "auroc": auroc,
            "support": int(confusion[class_id].sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
    total = int(confusion.sum())
    supports = [int(per_class[index]["support"]) for index in range(5)]
    return {
        "accuracy": float(np.trace(confusion) / total),
        "macro_precision": float(np.mean([per_class[index]["precision"] for index in range(5)])),
        "macro_recall": float(np.mean([per_class[index]["recall"] for index in range(5)])),
        "macro_f1": float(np.mean([per_class[index]["f1"] for index in range(5)])),
        "weighted_f1": float(sum(per_class[index]["f1"] * supports[index] for index in range(5)) / total),
        "macro_auroc": float(np.mean([per_class[index]["auroc"] for index in range(5)])) if auroc_valid else float("nan"),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def paired_rows(proposed: list[dict[str, str]], baseline: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    proposed_by_key = {prediction_key(row): row for row in proposed}
    baseline_by_key = {prediction_key(row): row for row in baseline}
    if set(proposed_by_key) != set(baseline_by_key):
        missing_proposed = sorted(set(baseline_by_key) - set(proposed_by_key))[:10]
        missing_baseline = sorted(set(proposed_by_key) - set(baseline_by_key))[:10]
        raise ValueError(f"Prediction key sets differ: missing proposed={missing_proposed}, missing baseline={missing_baseline}")
    ordered_keys = sorted(proposed_by_key, key=lambda key: (int(key[0]), key[1], key[2], int(key[3])))
    rows: list[dict[str, Any]] = []
    targets, baseline_predictions, proposed_predictions = [], [], []
    baseline_probabilities, proposed_probabilities, sources = [], [], []
    for key in ordered_keys:
        p = proposed_by_key[key]
        b = baseline_by_key[key]
        if p["true_class_name"] != b["true_class_name"]:
            raise ValueError(f"True class name differs at key {key}")
        target = int(p["true_class_id"])
        baseline_prediction = int(b["predicted_class_id"])
        proposed_prediction = int(p["predicted_class_id"])
        baseline_correct = target == baseline_prediction
        proposed_correct = target == proposed_prediction
        baseline_probability = [float(b[f"probability_class_{index}"]) for index in range(5)]
        proposed_probability = [float(p[f"probability_class_{index}"]) for index in range(5)]
        row: dict[str, Any] = {
            "record_index": p["record_index"], "source_image_id": p["source_image_id"],
            "image_path": p["image_path"], "true_class_id": target,
            "true_class_name": p["true_class_name"],
            "baseline_predicted_class_id": baseline_prediction,
            "baseline_predicted_class_name": b["predicted_class_name"],
            "baseline_confidence": float(b["confidence"]), "baseline_correct": baseline_correct,
            "proposed_predicted_class_id": proposed_prediction,
            "proposed_predicted_class_name": p["predicted_class_name"],
            "proposed_confidence": float(p["confidence"]), "proposed_correct": proposed_correct,
            "prediction_changed": baseline_prediction != proposed_prediction,
            "baseline_wrong_proposed_correct": (not baseline_correct) and proposed_correct,
            "baseline_correct_proposed_wrong": baseline_correct and (not proposed_correct),
            "both_correct": baseline_correct and proposed_correct,
            "both_wrong": (not baseline_correct) and (not proposed_correct),
            "confidence_difference_proposed_minus_baseline": float(p["confidence"]) - float(b["confidence"]),
        }
        for index in range(5):
            row[f"baseline_probability_class_{index}"] = baseline_probability[index]
            row[f"proposed_probability_class_{index}"] = proposed_probability[index]
        rows.append(row)
        targets.append(target)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction)
        baseline_probabilities.append(baseline_probability)
        proposed_probabilities.append(proposed_probability)
        sources.append(p["source_image_id"])
    arrays = {
        "targets": np.asarray(targets, dtype=np.int64),
        "baseline_predictions": np.asarray(baseline_predictions, dtype=np.int64),
        "proposed_predictions": np.asarray(proposed_predictions, dtype=np.int64),
        "baseline_probabilities": np.asarray(baseline_probabilities, dtype=np.float64),
        "proposed_probabilities": np.asarray(proposed_probabilities, dtype=np.float64),
        "sources": np.asarray(sources),
    }
    return rows, arrays


def validate_fairness(args: argparse.Namespace, paths: dict[str, dict[str, Path]], paired: list[dict[str, Any]]) -> dict[str, Any]:
    shared = read_json(args.shared_config)
    proposed_config = read_json(paths["proposed"]["experiment_config"])
    baseline_config = read_json(paths["baseline"]["experiment_config"])
    proposed_test_meta = read_json(paths["proposed"]["test_metadata"])
    baseline_test_meta = read_json(paths["baseline"]["test_metadata"])
    proposed_final = read_json(paths["proposed"]["final_audit"])
    baseline_final = read_json(paths["baseline"]["final_audit"])
    manifest_hashes = {
        split: sha256_file(args.split_dir / f"{split}_roi_manifest.csv")
        for split in ("train", "val", "test")
    }
    split_hash = sha256_file(args.split_dir / "image_id_split_manifest.csv")
    shared_sha = sha256_file(args.shared_config)
    checks = {
        "proposed_final_status_pass": proposed_final.get("status") == "PASS",
        "baseline_final_status_pass": baseline_final.get("status") == "PASS",
        "train_manifest_sha256": manifest_hashes["train"] == shared["manifests"]["train"]["sha256"],
        "val_manifest_sha256": manifest_hashes["val"] == shared["manifests"]["val"]["sha256"],
        "test_manifest_sha256": manifest_hashes["test"] == shared["manifests"]["test"]["sha256"] == EXPECTED_TEST_SHA256,
        "image_id_split_sha256": split_hash == shared["image_id_split_manifest"]["sha256"],
        "shared_config_sha256": proposed_config["shared_config_sha256"] == baseline_config["shared_config_sha256"] == shared_sha,
        "shared_config_copies": paths["proposed"]["shared_config_copy"].read_bytes() == paths["baseline"]["shared_config_copy"].read_bytes() == args.shared_config.read_bytes(),
        "seed": shared["seed"] == 42,
        "architecture": shared["architecture"] == "convnext_tiny",
        "num_classes": shared["num_classes"] == 5,
        "feature_dim": shared["feature_dim"] == 768,
        "preprocessing": isinstance(shared["preprocessing"], dict),
        "train_augmentation": isinstance(shared["augmentation"], dict),
        "validation_test_augmentation": shared["augmentation"]["train_only"] is True,
        "loss": shared["loss"]["name"] == "CrossEntropyLoss",
        "class_weights": shared["loss"]["class_weights"] is None,
        "label_smoothing": shared["loss"]["label_smoothing"] == 0.0,
        "optimizer": shared["optimizer"]["name"] == "AdamW",
        "backbone_learning_rate": shared["optimizer"]["backbone_learning_rate"] == 1e-5,
        "classifier_learning_rate": shared["optimizer"]["classifier_learning_rate"] == 1e-4,
        "weight_decay": shared["optimizer"]["weight_decay"] == 1e-4,
        "scheduler": shared["scheduler"] == {"name": "CosineAnnealingLR", "T_max": 50},
        "maximum_epochs": shared["maximum_epochs"] == 50,
        "patience": shared["early_stopping"]["patience"] == 10,
        "min_delta": shared["early_stopping"]["min_delta"] == 1e-4,
        "gradient_clipping": shared["gradient_clip_max_norm"] == 1.0,
        "amp": shared["amp"] is True,
        "batch_size": proposed_config["actual_batch_size"] == baseline_config["actual_batch_size"] == 64,
        "accumulation_steps": proposed_config["accumulation_steps"] == baseline_config["accumulation_steps"] == 1,
        "effective_batch_size": proposed_config["effective_batch_size"] == baseline_config["effective_batch_size"] == 64,
        "dataloader_workers": shared["data_loader"]["workers"] == 2,
        "checkpoint_selection": shared["early_stopping"]["metric"] == "validation_macro_f1",
        "test_evaluation_rule": shared["test_evaluation"] == "exactly once after best checkpoint is fixed",
        "test_evaluation_count": proposed_test_meta["evaluation_count"] == baseline_test_meta["evaluation_count"] == 1,
        "test_records": proposed_test_meta["rows"] == baseline_test_meta["rows"] == EXPECTED_TEST_ROWS,
        "test_manifest_metadata": proposed_test_meta["manifest_sha256"] == baseline_test_meta["manifest_sha256"] == EXPECTED_TEST_SHA256,
        "test_checkpoint_selection": proposed_test_meta["checkpoint_selection_used_test"] is False and baseline_test_meta["checkpoint_selection_used_test"] is False,
        "test_record_keys": len(paired) == EXPECTED_TEST_ROWS,
        "class_mapping": shared["class_mapping"] == {str(key): value for key, value in CLASS_MAPPING.items()},
        "allowed_initializations": proposed_config["initialization"] == "distilled" and baseline_config["initialization"] == "imagenet",
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failed else "FAIL",
        "created_at_utc": utc_now(),
        "checks": checks,
        "difference_count_excluding_allowed": len(failed),
        "differences": failed,
        "allowed_differences": ["initialization", "distilled backbone source", "output directory", "training and evaluation results"],
        "manifest_sha256": manifest_hashes,
        "image_id_split_sha256": split_hash,
        "shared_config_sha256": shared_sha,
        "test_records": len(paired),
        "test_source_image_ids": len({row["source_image_id"] for row in paired}),
    }


def overall_comparison(baseline: dict[str, Any], proposed: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = ("loss", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "macro_auroc")
    rows = []
    for metric in metrics:
        b = float(baseline[metric])
        p = float(proposed[metric])
        difference = p - b
        relative = difference / abs(b) * 100 if b else None
        if math.isclose(p, b, rel_tol=0.0, abs_tol=1e-12):
            better = "tie"
        elif metric == "loss":
            better = "proposed" if p < b else "baseline"
        else:
            better = "proposed" if p > b else "baseline"
        rows.append({"metric": metric, "baseline": b, "proposed": p, "absolute_difference": difference, "relative_change_percent": relative, "better_model": better})
    return rows


def per_class_comparisons(baseline: dict[str, Any], proposed: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    detail, f1_rows, auroc_rows = [], [], []
    for class_id in range(5):
        b = baseline["per_class"][class_id]
        p = proposed["per_class"][class_id]
        for metric in ("precision", "recall", "f1", "auroc", "support", "tp", "fp", "fn", "tn"):
            b_value, p_value = b[metric], p[metric]
            difference = p_value - b_value
            lower_better = metric in {"fp", "fn"}
            if p_value == b_value:
                better = "tie"
            elif lower_better:
                better = "proposed" if p_value < b_value else "baseline"
            else:
                better = "proposed" if p_value > b_value else "baseline"
            detail.append({"class_id": class_id, "class_name": CLASS_MAPPING[class_id], "metric": metric, "baseline_value": b_value, "proposed_value": p_value, "absolute_difference": difference, "better_model": better})
        f1_rows.append({"class_id": class_id, "class_name": CLASS_MAPPING[class_id], "baseline_f1": b["f1"], "proposed_f1": p["f1"], "absolute_difference": p["f1"] - b["f1"], "better_model": "proposed" if p["f1"] > b["f1"] else "baseline" if p["f1"] < b["f1"] else "tie"})
        auroc_rows.append({"class_id": class_id, "class_name": CLASS_MAPPING[class_id], "baseline_auroc": b["auroc"], "proposed_auroc": p["auroc"], "absolute_difference": p["auroc"] - b["auroc"], "better_model": "proposed" if p["auroc"] > b["auroc"] else "baseline" if p["auroc"] < b["auroc"] else "tie"})
    return detail, f1_rows, auroc_rows


def cluster_bootstrap(arrays: dict[str, np.ndarray], repetitions: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = sorted(set(arrays["sources"].tolist()))
    groups = {source: np.flatnonzero(arrays["sources"] == source) for source in sources}
    baseline_point = confusion_and_metrics(arrays["targets"], arrays["baseline_predictions"], arrays["baseline_probabilities"])
    proposed_point = confusion_and_metrics(arrays["targets"], arrays["proposed_predictions"], arrays["proposed_probabilities"])
    samples = {metric: [] for metric in PRIMARY_BOOTSTRAP_METRICS}
    invalid_auroc = 0
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        sampled = rng.integers(0, len(sources), size=len(sources))
        indices = np.concatenate([groups[sources[index]] for index in sampled])
        baseline = confusion_and_metrics(arrays["targets"][indices], arrays["baseline_predictions"][indices], arrays["baseline_probabilities"][indices])
        proposed = confusion_and_metrics(arrays["targets"][indices], arrays["proposed_predictions"][indices], arrays["proposed_probabilities"][indices])
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            samples[metric].append(proposed[metric] - baseline[metric])
        if math.isfinite(proposed["macro_auroc"]) and math.isfinite(baseline["macro_auroc"]):
            samples["macro_auroc"].append(proposed["macro_auroc"] - baseline["macro_auroc"])
        else:
            invalid_auroc += 1
    rows = []
    for metric in PRIMARY_BOOTSTRAP_METRICS:
        values = np.asarray(samples[metric], dtype=np.float64)
        point = proposed_point[metric] - baseline_point[metric]
        rows.append({
            "metric": metric,
            "point_estimate_proposed_minus_baseline": point,
            "bootstrap_mean_difference": float(values.mean()),
            "ci_2_5_percentile": float(np.percentile(values, 2.5)),
            "ci_97_5_percentile": float(np.percentile(values, 97.5)),
            "confidence_level": 0.95,
            "probability_difference_gt_zero": float(np.mean(values > 0)),
            "valid_replicates": int(len(values)),
            "invalid_replicates": int(repetitions - len(values)),
        })
    metadata = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "method": "paired cluster bootstrap by source_image_id",
        "seed": seed,
        "requested_repetitions": repetitions,
        "confidence_level": 0.95,
        "cluster_count": len(sources),
        "test_roi_count": len(arrays["targets"]),
        "sampling": "sample 59 source_image_id clusters with replacement; include every ROI in each sampled cluster",
        "macro_auroc_valid_replicates": len(samples["macro_auroc"]),
        "macro_auroc_invalid_replicates": invalid_auroc,
    }
    return rows, metadata


def mcnemar(paired: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts = Counter(
        "both_correct" if row["both_correct"] else
        "both_wrong" if row["both_wrong"] else
        "baseline_correct_proposed_wrong" if row["baseline_correct_proposed_wrong"] else
        "baseline_wrong_proposed_correct"
        for row in paired
    )
    table = [{"outcome": name, "count": counts[name]} for name in (
        "both_correct", "both_wrong", "baseline_correct_proposed_wrong", "baseline_wrong_proposed_correct"
    )]
    b = counts["baseline_correct_proposed_wrong"]
    c = counts["baseline_wrong_proposed_correct"]
    discordant = b + c
    if discordant:
        lower = min(b, c)
        tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2**discordant)
        exact_p = min(1.0, 2 * tail)
        chi_square_cc = (abs(b - c) - 1) ** 2 / discordant
    else:
        exact_p, chi_square_cc = 1.0, 0.0
    result = {
        "status": "PASS",
        "both_correct": counts["both_correct"],
        "both_wrong": counts["both_wrong"],
        "baseline_correct_proposed_wrong": b,
        "baseline_wrong_proposed_correct": c,
        "discordant_pairs": discordant,
        "exact_two_sided_binomial_p_value": exact_p,
        "continuity_corrected_chi_square": chi_square_cc,
        "interpretation_note": "ROI observations may be clustered within the same source image; McNemar is supplementary. Primary inference uses source-level cluster bootstrap.",
    }
    return table, result


def parse_training_wall_seconds(log_path: Path) -> tuple[float | None, bool]:
    lines = log_path.read_text(encoding="utf-8-sig").splitlines()
    resume_indices = [index for index, line in enumerate(lines) if "Formal training resume=" in line]
    if not resume_indices:
        return None, False
    start_index = resume_indices[-1]
    timestamp_pattern = re.compile(r"^\[([^]]+)\]")
    start_match = timestamp_pattern.match(lines[start_index])
    end_line = next((line for line in lines[start_index + 1:] if "Early stopping at epoch" in line), None)
    if end_line is None:
        epoch_lines = [line for line in lines[start_index + 1:] if " epoch=" in line]
        end_line = epoch_lines[-1] if epoch_lines else None
    if start_match is None or end_line is None or timestamp_pattern.match(end_line) is None:
        return None, True
    start = datetime.fromisoformat(start_match.group(1))
    end = datetime.fromisoformat(timestamp_pattern.match(end_line).group(1))
    return (end - start).total_seconds(), True


def training_efficiency(paths: dict[str, dict[str, Path]]) -> list[dict[str, Any]]:
    rows = []
    for experiment in ("baseline", "proposed"):
        metrics = read_csv(paths[experiment]["metrics"])
        test = read_json(paths[experiment]["test_metrics"])
        final = read_json(paths[experiment]["final_audit"])
        best_epoch = int(test["best_epoch"])
        best_row = next(row for row in metrics if int(row["epoch"]) == best_epoch)
        wall_seconds, resumed = parse_training_wall_seconds(paths[experiment]["log"])
        rows.append({
            "experiment": experiment,
            "initialization": "imagenet" if experiment == "baseline" else "distilled",
            "completed_epochs": len(metrics),
            "early_stopping_triggered": len(metrics) < 50,
            "best_epoch": best_epoch,
            "best_validation_loss": float(best_row["val_loss"]),
            "best_validation_accuracy": float(best_row["val_accuracy"]),
            "best_validation_macro_f1": float(best_row["val_macro_f1"]),
            "best_validation_macro_auroc": float(best_row["val_macro_auroc"]),
            "final_train_loss": float(metrics[-1]["train_loss"]),
            "final_validation_loss": float(metrics[-1]["val_loss"]),
            "peak_allocated_vram_gb": max(float(row["gpu_peak_allocated_gb"]) for row in metrics),
            "peak_reserved_vram_gb": max(float(row["gpu_peak_reserved_gb"]) for row in metrics),
            "average_images_per_second": float(np.mean([float(row["images_per_second"]) for row in metrics])),
            "summed_train_epoch_seconds": sum(float(row["epoch_seconds"]) for row in metrics),
            "formal_training_wall_seconds": wall_seconds,
            "resume_checkpoint_used": resumed,
            "training_interruption_recorded": (paths[experiment]["final_audit"].parent / "training_failure.json").is_file(),
            "oom": False,
            "nan_count": final.get("nan_count", 0),
            "inf_count": final.get("inf_count", 0),
            "nonfinite_gradient_count": final.get("nonfinite_gradient_count", 0),
        })
    return rows


def error_analysis(paired: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    baseline_wrong_proposed_correct = [row for row in paired if row["baseline_wrong_proposed_correct"]]
    baseline_correct_proposed_wrong = [row for row in paired if row["baseline_correct_proposed_wrong"]]
    both_wrong_same = [row for row in paired if row["both_wrong"] and row["baseline_predicted_class_id"] == row["proposed_predicted_class_id"]]
    both_wrong_different = [row for row in paired if row["both_wrong"] and row["baseline_predicted_class_id"] != row["proposed_predicted_class_id"]]
    increases = sorted(paired, key=lambda row: row["confidence_difference_proposed_minus_baseline"], reverse=True)[:25]
    decreases = sorted(paired, key=lambda row: row["confidence_difference_proposed_minus_baseline"])[:25]
    confidence_changes = []
    for direction, selected in (("largest_increase", increases), ("largest_decrease", decreases)):
        for rank, row in enumerate(selected, start=1):
            confidence_changes.append({"change_direction": direction, "rank": rank, **row})
    transition = Counter((row["baseline_predicted_class_id"], row["proposed_predicted_class_id"]) for row in paired)
    transition_rows = [
        {"baseline_predicted_class_id": baseline, "baseline_predicted_class_name": CLASS_MAPPING[baseline],
         "proposed_predicted_class_id": proposed, "proposed_predicted_class_name": CLASS_MAPPING[proposed],
         "count": transition[(baseline, proposed)]}
        for baseline in range(5) for proposed in range(5)
    ]
    return {
        "baseline_wrong_proposed_correct": baseline_wrong_proposed_correct,
        "baseline_correct_proposed_wrong": baseline_correct_proposed_wrong,
        "both_wrong_same_prediction": both_wrong_same,
        "both_wrong_different_prediction": both_wrong_different,
        "largest_confidence_changes": confidence_changes,
        "error_transition_matrix": transition_rows,
    }


def plot_overall(rows: list[dict[str, Any]], path: Path) -> None:
    score_rows = [row for row in rows if row["metric"] != "loss"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar([0, 1], [rows[0]["baseline"], rows[0]["proposed"]], color=["#5B6F8A", "#B5523B"])
    axes[0].set_xticks([0, 1], ["Baseline", "Proposed"])
    axes[0].set_ylim(0, max(rows[0]["baseline"], rows[0]["proposed"]) * 1.15)
    axes[0].set_title("Test Loss (lower is better)")
    x = np.arange(len(score_rows)); width = 0.36
    axes[1].bar(x - width / 2, [row["baseline"] for row in score_rows], width, label="Baseline", color="#5B6F8A")
    axes[1].bar(x + width / 2, [row["proposed"] for row in score_rows], width, label="Proposed", color="#B5523B")
    axes[1].set_xticks(x, [row["metric"].replace("macro_", "M-").replace("weighted_", "W-") for row in score_rows], rotation=25, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].set_title("Test Metrics (higher is better)")
    figure.suptitle("Test n=454; source images=59")
    atomic_save_figure(figure, path)


def plot_per_class(rows: list[dict[str, Any]], metric: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(5); width = 0.36
    axis.bar(x - width / 2, [row[f"baseline_{metric}"] for row in rows], width, label="Baseline", color="#5B6F8A")
    axis.bar(x + width / 2, [row[f"proposed_{metric}"] for row in rows], width, label="Proposed", color="#B5523B")
    axis.set_xticks(x, [f"Class {index}" for index in range(5)])
    axis.set_ylim(0, 1)
    axis.set_ylabel(metric.upper())
    axis.set_title(f"Per-class {metric.upper()} (Test n=454; source images=59)")
    axis.legend()
    atomic_save_figure(figure, path)


def plot_training_curves(paths: dict[str, dict[str, Path]], field: str, ylabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for experiment, color in (("baseline", "#5B6F8A"), ("proposed", "#B5523B")):
        metrics = read_csv(paths[experiment]["metrics"])
        axis.plot([int(row["epoch"]) for row in metrics], [float(row[field]) for row in metrics], marker="o", markersize=3, label=experiment.title(), color=color)
    axis.set_xlabel("Epoch"); axis.set_ylabel(ylabel); axis.grid(True, alpha=0.3); axis.legend()
    axis.set_title(f"Validation {ylabel}")
    atomic_save_figure(figure, path)


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    denominators = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denominators, out=np.zeros_like(matrix, dtype=float), where=denominators != 0)


def plot_confusion(matrix: np.ndarray, title: str, path: Path) -> None:
    normalized = row_normalize(matrix)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, values, subtitle, fmt in ((axes[0], matrix, "Raw count", "d"), (axes[1], normalized, "Row-normalized", ".2f")):
        image = axis.imshow(values, cmap="Blues", vmin=0)
        figure.colorbar(image, ax=axis, fraction=0.046)
        axis.set_title(subtitle); axis.set_xlabel("Predicted"); axis.set_ylabel("True")
        axis.set_xticks(range(5)); axis.set_yticks(range(5))
        for row in range(5):
            for column in range(5):
                text = format(int(values[row, column]), fmt) if fmt == "d" else format(values[row, column], fmt)
                axis.text(column, row, text, ha="center", va="center", fontsize=8)
    figure.suptitle(f"{title}; Test n=454, source images=59")
    atomic_save_figure(figure, path)


def plot_confusion_difference(baseline: np.ndarray, proposed: np.ndarray, path: Path) -> None:
    raw = proposed.astype(float) - baseline.astype(float)
    normalized = row_normalize(proposed) - row_normalize(baseline)
    limit_raw = max(1.0, float(np.abs(raw).max())); limit_norm = max(0.01, float(np.abs(normalized).max()))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, values, limit, title, fmt in ((axes[0], raw, limit_raw, "Raw count difference", ".0f"), (axes[1], normalized, limit_norm, "Row-normalized difference", ".2f")):
        image = axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit)
        figure.colorbar(image, ax=axis, fraction=0.046)
        axis.set_title(title); axis.set_xlabel("Predicted"); axis.set_ylabel("True")
        axis.set_xticks(range(5)); axis.set_yticks(range(5))
        for row in range(5):
            for column in range(5):
                axis.text(column, row, format(values[row, column], fmt), ha="center", va="center", fontsize=8)
    figure.suptitle("Proposed - Baseline confusion matrix; Test n=454, source images=59")
    atomic_save_figure(figure, path)


def plot_paired_correctness(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    labels = [row["outcome"].replace("_", "\n") for row in rows]
    counts = [row["count"] for row in rows]
    axis.bar(labels, counts, color=["#4C956C", "#8D99AE", "#5B6F8A", "#B5523B"])
    axis.set_ylim(0, max(counts) * 1.15); axis.set_ylabel("ROI count")
    axis.set_title("Paired correctness; Test n=454, source images=59")
    for index, value in enumerate(counts): axis.text(index, value, str(value), ha="center", va="bottom")
    atomic_save_figure(figure, path)


def plot_bootstrap(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    y = np.arange(len(rows)); points = np.asarray([row["point_estimate_proposed_minus_baseline"] for row in rows])
    lower = np.asarray([row["ci_2_5_percentile"] for row in rows]); upper = np.asarray([row["ci_97_5_percentile"] for row in rows])
    axis.errorbar(points, y, xerr=np.vstack((points - lower, upper - points)), fmt="o", color="#B5523B", capsize=5)
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(y, [row["metric"] for row in rows]); axis.set_xlabel("Proposed - Baseline")
    axis.set_title("95% paired source-cluster bootstrap CI (10,000 repetitions)")
    axis.grid(True, axis="x", alpha=0.3)
    atomic_save_figure(figure, path)


def plot_confidence(arrays: dict[str, np.ndarray], path: Path) -> None:
    baseline_confidence = arrays["baseline_probabilities"].max(axis=1)
    proposed_confidence = arrays["proposed_probabilities"].max(axis=1)
    figure, axis = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    axis.hist(baseline_confidence, bins=bins, alpha=0.55, label="Baseline", color="#5B6F8A")
    axis.hist(proposed_confidence, bins=bins, alpha=0.55, label="Proposed", color="#B5523B")
    axis.set_xlim(0, 1); axis.set_xlabel("Predicted-class confidence"); axis.set_ylabel("ROI count")
    axis.set_title("Confidence distributions; Test n=454, source images=59"); axis.legend()
    atomic_save_figure(figure, path)


def conclusion_from_results(overall: list[dict[str, Any]], bootstrap: list[dict[str, Any]], f1_rows: list[dict[str, Any]]) -> tuple[str, str]:
    differences = {row["metric"]: row["absolute_difference"] for row in overall}
    intervals = {row["metric"]: (row["ci_2_5_percentile"], row["ci_97_5_percentile"]) for row in bootstrap}
    positive_supported = [metric for metric in ("accuracy", "macro_f1", "macro_auroc") if differences[metric] > 0 and intervals[metric][0] > 0]
    negative_supported = [metric for metric in ("accuracy", "macro_f1", "macro_auroc") if differences[metric] < 0 and intervals[metric][1] < 0]
    if len(positive_supported) >= 2 and sum(row["absolute_difference"] >= 0 for row in f1_rows) >= 3:
        category = "proposed_clearly_better"
        text = "Proposed 顯示一致且由 cluster bootstrap 支持的優勢。"
    elif len(negative_supported) >= 2:
        category = "baseline_better"
        text = "Baseline 在主要指標上具有由 cluster bootstrap 支持的優勢；蒸餾後模型未優於直接 ImageNet fine-tuning。"
    elif sum(differences[metric] > 0 for metric in ("accuracy", "macro_f1", "macro_auroc")) >= 2:
        category = "proposed_uncertain_trend"
        text = "觀察到 Proposed 提升趨勢，但 95% cluster-bootstrap CI 未充分支持穩定優勢。"
    else:
        category = "similar_no_clear_distillation_benefit"
        text = "主要指標呈混合且幅度小的差異，95% cluster-bootstrap CI 未支持一致優勢；未觀察到明確蒸餾效益。"
    return category, text


def write_reports(
    output: Path, fairness: dict[str, Any], overall: list[dict[str, Any]],
    f1_rows: list[dict[str, Any]], auroc_rows: list[dict[str, Any]], bootstrap: list[dict[str, Any]],
    paired_table: list[dict[str, Any]], mcnemar_result: dict[str, Any], training: list[dict[str, Any]],
    conclusion_category: str, conclusion_text: str,
) -> None:
    overall_map = {row["metric"]: row for row in overall}
    bootstrap_map = {row["metric"]: row for row in bootstrap}
    counts = {row["outcome"]: row["count"] for row in paired_table}
    report_lines = [
        "# Proposed vs Baseline Final Comparison",
        "",
        f"- Fairness audit: {fairness['status']} (differences excluding allowed fields: {fairness['difference_count_excluding_allowed']})",
        "- Test set: 454 ROI from 59 source images",
        "- Test evaluation count: one per experiment; test did not select checkpoints",
        "- Difference convention: Proposed - Baseline",
        "",
        "## Overall Test Metrics",
        "",
        "| Metric | Baseline | Proposed | Difference | Better |",
        "|---|---:|---:|---:|---|",
    ]
    for row in overall:
        report_lines.append(f"| {row['metric']} | {row['baseline']:.6f} | {row['proposed']:.6f} | {row['absolute_difference']:+.6f} | {row['better_model']} |")
    report_lines.extend(["", "## Paired Source-Cluster Bootstrap", "", "| Metric | Point difference | 95% CI | P(diff > 0) |", "|---|---:|---:|---:|"])
    for row in bootstrap:
        report_lines.append(f"| {row['metric']} | {row['point_estimate_proposed_minus_baseline']:+.6f} | [{row['ci_2_5_percentile']:+.6f}, {row['ci_97_5_percentile']:+.6f}] | {row['probability_difference_gt_zero']:.4f} |")
    report_lines.extend([
        "", "## Paired Correctness",
        f"- Both correct: {counts['both_correct']}", f"- Both wrong: {counts['both_wrong']}",
        f"- Baseline correct / Proposed wrong: {counts['baseline_correct_proposed_wrong']}",
        f"- Baseline wrong / Proposed correct: {counts['baseline_wrong_proposed_correct']}",
        f"- Supplementary exact McNemar p-value: {mcnemar_result['exact_two_sided_binomial_p_value']:.6f}",
        "- McNemar is supplementary because ROI observations can cluster within source images; primary inference uses source-level cluster bootstrap.",
        "", "## Research Context and Limitations",
        "- Phase 1 used all 4,725 unlabeled ROI for transductive feature distillation.",
        "- Phase 2 used a grouped 8:1:1 split by source_image_id and identical fine-tuning settings.",
        "- The task is five-class classification of 224x224 ROI, not bounding-box detection.",
        "- Train includes fixed brightness-augmented ROI already present in the manifest; online Phase 1/2 augmentation used Gaussian blur and Gaussian noise, not brightness.",
        "- Results are limited to seed 42, 59 test source images, and multiple ROI potentially originating from one full image.",
        "", "## Conclusion", "", conclusion_text,
    ])
    atomic_write_text(output / "final_comparison_report.md", "\n".join(report_lines) + "\n")
    conclusion_lines = [
        "# Research Conclusion", "", f"**Category:** `{conclusion_category}`", "", conclusion_text, "",
        f"Accuracy difference: {overall_map['accuracy']['absolute_difference']:+.6f}, 95% cluster CI [{bootstrap_map['accuracy']['ci_2_5_percentile']:+.6f}, {bootstrap_map['accuracy']['ci_97_5_percentile']:+.6f}].",
        f"Macro-F1 difference: {overall_map['macro_f1']['absolute_difference']:+.6f}, 95% cluster CI [{bootstrap_map['macro_f1']['ci_2_5_percentile']:+.6f}, {bootstrap_map['macro_f1']['ci_97_5_percentile']:+.6f}].",
        f"Macro-AUROC difference: {overall_map['macro_auroc']['absolute_difference']:+.6f}, 95% cluster CI [{bootstrap_map['macro_auroc']['ci_2_5_percentile']:+.6f}, {bootstrap_map['macro_auroc']['ci_97_5_percentile']:+.6f}].",
        "", "This is a single-seed paired comparison. The confidence intervals account for source-image clustering but do not establish generalization beyond this test cohort.",
    ]
    atomic_write_text(output / "research_conclusion.md", "\n".join(conclusion_lines) + "\n")


def input_hashes(paths: dict[str, dict[str, Path]], args: argparse.Namespace) -> dict[str, Any]:
    files = {"shared_config": args.shared_config, "test_manifest": args.split_dir / "test_roi_manifest.csv", "image_id_split_manifest": args.split_dir / "image_id_split_manifest.csv"}
    for experiment in ("proposed", "baseline"):
        for name, path in paths[experiment].items():
            files[f"{experiment}_{name}"] = path
    return {name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for name, path in files.items()}


def create_directories(output: Path) -> dict[str, Path]:
    directories = {name: output / name for name in ("tables", "predictions", "statistics", "error_analysis", "figures", "config")}
    output.mkdir(parents=True, exist_ok=True)
    for directory in directories.values(): directory.mkdir(exist_ok=True)
    return directories


def run(args: argparse.Namespace) -> int:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        contents = [str(path) for path in sorted(args.output_dir.rglob("*"))[:100]]
        raise FileExistsError(f"Comparison output is non-empty and will not be overwritten: {contents}")
    paths = {
        "proposed": require_files(args.proposed_dir, "proposed"),
        "baseline": require_files(args.baseline_dir, "baseline"),
    }
    if sha256_file(args.split_dir / "test_roi_manifest.csv") != EXPECTED_TEST_SHA256:
        raise ValueError("Fixed Test Manifest SHA256 mismatch")
    proposed_predictions = read_csv(paths["proposed"]["predictions"])
    baseline_predictions = read_csv(paths["baseline"]["predictions"])
    prediction_audit = {
        "proposed": validate_predictions(proposed_predictions, "proposed"),
        "baseline": validate_predictions(baseline_predictions, "baseline"),
    }
    paired, arrays = paired_rows(proposed_predictions, baseline_predictions)
    if len(set(arrays["sources"].tolist())) != EXPECTED_TEST_SOURCES:
        raise ValueError("Expected 59 Test source_image_id clusters")
    fairness = validate_fairness(args, paths, paired)
    preflight = {
        "status": "PASS" if fairness["status"] == "PASS" else "FAIL",
        "prediction_audit": prediction_audit,
        "paired_keys": len(paired),
        "source_clusters": len(set(arrays["sources"].tolist())),
        "fairness": fairness,
    }
    if args.dry_run:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if fairness["status"] == "PASS" else 1
    if fairness["status"] != "PASS":
        raise RuntimeError(f"Fairness audit failed: {fairness['differences']}")

    directories = create_directories(args.output_dir)
    atomic_write_json(args.output_dir / "fairness_audit.json", fairness)
    fairness_lines = ["Proposed vs Baseline Fairness Audit", f"Status: {fairness['status']}", f"Difference count excluding allowed fields: {fairness['difference_count_excluding_allowed']}"]
    fairness_lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in fairness["checks"].items())
    atomic_write_text(args.output_dir / "fairness_audit.txt", "\n".join(fairness_lines) + "\n")
    atomic_write_csv(directories["predictions"] / "paired_test_predictions.csv", paired)

    baseline_metrics_json = read_json(paths["baseline"]["test_metrics"])
    proposed_metrics_json = read_json(paths["proposed"]["test_metrics"])
    baseline_metrics = confusion_and_metrics(arrays["targets"], arrays["baseline_predictions"], arrays["baseline_probabilities"])
    proposed_metrics = confusion_and_metrics(arrays["targets"], arrays["proposed_predictions"], arrays["proposed_probabilities"])
    for name, computed, source in (("baseline", baseline_metrics, baseline_metrics_json), ("proposed", proposed_metrics, proposed_metrics_json)):
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "macro_auroc"):
            if not math.isclose(computed[metric], float(source[metric]), rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(f"{name} recomputed {metric} differs from saved Test metrics")
        computed["loss"] = float(source["loss"])

    overall = overall_comparison(baseline_metrics, proposed_metrics)
    per_class_detail, f1_rows, auroc_rows = per_class_comparisons(baseline_metrics, proposed_metrics)
    training = training_efficiency(paths)
    atomic_write_csv(directories["tables"] / "overall_metrics_comparison.csv", overall)
    atomic_write_csv(directories["tables"] / "per_class_metrics_comparison.csv", per_class_detail)
    atomic_write_csv(directories["tables"] / "per_class_f1_summary.csv", f1_rows)
    atomic_write_csv(directories["tables"] / "per_class_auroc_summary.csv", auroc_rows)
    atomic_write_csv(directories["tables"] / "training_efficiency_comparison.csv", training)

    bootstrap_rows, bootstrap_metadata = cluster_bootstrap(arrays, args.bootstrap_repetitions, args.seed)
    paired_table, mcnemar_result = mcnemar(paired)
    atomic_write_csv(directories["statistics"] / "cluster_bootstrap_results.csv", bootstrap_rows)
    atomic_write_json(directories["statistics"] / "cluster_bootstrap_metadata.json", bootstrap_metadata)
    atomic_write_csv(directories["statistics"] / "paired_correctness_table.csv", paired_table)
    atomic_write_json(directories["statistics"] / "mcnemar_supplementary.json", mcnemar_result)

    errors = error_analysis(paired)
    for name, rows in errors.items():
        fieldnames = list(rows[0]) if rows else list(paired[0])
        atomic_write_csv(directories["error_analysis"] / f"{name}.csv", rows, fieldnames)

    plot_overall(overall, directories["figures"] / "overall_metrics_comparison.png")
    plot_per_class(f1_rows, "f1", directories["figures"] / "per_class_f1_comparison.png")
    plot_per_class(auroc_rows, "auroc", directories["figures"] / "per_class_auroc_comparison.png")
    plot_training_curves(paths, "val_macro_f1", "Macro-F1", directories["figures"] / "validation_macro_f1_comparison.png")
    plot_training_curves(paths, "val_loss", "Loss", directories["figures"] / "validation_loss_comparison.png")
    plot_confusion(baseline_metrics["confusion_matrix"], "Baseline Test Confusion Matrix", directories["figures"] / "test_confusion_matrix_baseline.png")
    plot_confusion(proposed_metrics["confusion_matrix"], "Proposed Test Confusion Matrix", directories["figures"] / "test_confusion_matrix_proposed.png")
    plot_confusion_difference(baseline_metrics["confusion_matrix"], proposed_metrics["confusion_matrix"], directories["figures"] / "confusion_matrix_difference.png")
    plot_paired_correctness(paired_table, directories["figures"] / "paired_correctness_comparison.png")
    plot_bootstrap(bootstrap_rows, directories["figures"] / "cluster_bootstrap_difference_ci.png")
    plot_confidence(arrays, directories["figures"] / "confidence_distribution_comparison.png")

    conclusion_category, conclusion_text = conclusion_from_results(overall, bootstrap_rows, f1_rows)
    write_reports(args.output_dir, fairness, overall, f1_rows, auroc_rows, bootstrap_rows, paired_table, mcnemar_result, training, conclusion_category, conclusion_text)
    summary = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "fairness": fairness,
        "prediction_audit": prediction_audit,
        "overall_metrics": overall,
        "per_class_f1": f1_rows,
        "per_class_auroc": auroc_rows,
        "cluster_bootstrap": bootstrap_rows,
        "cluster_bootstrap_metadata": bootstrap_metadata,
        "paired_correctness": paired_table,
        "mcnemar_supplementary": mcnemar_result,
        "training_efficiency": training,
        "error_analysis_counts": {name: len(rows) for name, rows in errors.items()},
        "conclusion_category": conclusion_category,
        "research_conclusion": conclusion_text,
        "models_retrained": False,
        "test_reexecuted": False,
        "model_checkpoints_loaded": False,
        "source_experiment_outputs_modified": False,
    }
    atomic_write_json(args.output_dir / "final_comparison_summary.json", summary)
    config = {
        "seed": args.seed,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "confidence_level": 0.95,
        "cluster_unit": "source_image_id",
        "difference_definition": "Proposed - Baseline",
        "proposed_dir": str(args.proposed_dir), "baseline_dir": str(args.baseline_dir),
        "split_dir": str(args.split_dir), "shared_config": str(args.shared_config),
        "output_dir": str(args.output_dir),
    }
    atomic_write_json(directories["config"] / "comparison_config.json", config)
    atomic_write_json(directories["config"] / "input_file_hashes.json", input_hashes(paths, args))
    atomic_write_json(directories["config"] / "environment.json", {
        "created_at_utc": utc_now(), "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "matplotlib": matplotlib.__version__, "gpu_used": False,
    })
    print(json.dumps({"status": "PASS", "output_dir": str(args.output_dir), "conclusion": conclusion_text, "overall": overall, "bootstrap": bootstrap_rows}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(r"C:\Users\09688\thoracic-cxr-project-3")
    experiment = root / "outputs" / "raddino_convnext_tiny_experiment_seed42"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--proposed-dir", type=Path, default=experiment / "phase2_proposed_distilled")
    parser.add_argument("--baseline-dir", type=Path, default=experiment / "phase2_baseline_imagenet")
    parser.add_argument("--split-dir", type=Path, default=experiment / "phase2_split")
    parser.add_argument("--shared-config", type=Path, default=experiment / "shared_phase2_finetune_config.json")
    parser.add_argument("--output-dir", type=Path, default=experiment / "final_comparison")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.bootstrap_repetitions < 1 or args.seed != 42:
        raise ValueError("bootstrap repetitions must be positive and the locked seed is 42")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
