#!/usr/bin/env python
"""Deterministic Patch Proposed Validation demo for classes 2 and 4."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from infer_patch_proposed_single_roi import (
    CLASS_MAPPING,
    NUM_CLASSES,
    Phase2Transform,
    environment_info,
    inspect_image,
    load_export_model,
    resolve_device,
    sha256_file,
    utc_now,
)


EXPECTED_MODEL_SHA256 = "8a68d68b901d721c63a38b5e75ee3291a8c06d13195572d20f29fd34a56485e5"
EXPECTED_VAL_MANIFEST_SHA256 = "5f92fd7282df28a4ec3365ba5fa7a777b365db860f7991a47238162d1ac5bc00"
EXPECTED_VAL_ROWS = 454
EXPECTED_CLASS_ROWS = {2: 112, 4: 81}
EXPECTED_CLASSES = [2, 4]
DISCLAIMER = "Research validation subset only. Not for clinical diagnosis."
REQUIRED_MANIFEST_FIELDS = {
    "source_image_id",
    "split",
    "image_path",
    "class_id",
    "class_name",
    "original_roi_id",
    "image_sha256",
    "is_brightness_augmented",
}
SELECTION_FIELDS = [
    "selection_order",
    "class_id",
    "class_name",
    "source_image_id",
    "original_roi_id",
    "image_path",
    "image_sha256",
    "selection_seed",
    "unique_source_preferred",
    "paired_key",
]
PREDICTION_FIELDS = [
    "selection_order",
    "paired_key",
    "source_image_id",
    "original_roi_id",
    "image_path",
    "image_sha256",
    "true_class_id",
    "true_class_name",
    "predicted_class_id",
    "predicted_class_name",
    "confidence",
    "probability_aortic_enlargement",
    "probability_cardiomegaly",
    "probability_pleural_thickening",
    "probability_pulmonary_fibrosis",
    "probability_pleural_effusion",
    "true_class_probability",
    "class2_vs_class4_margin",
    "is_correct",
    "inference_seconds",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_path(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve()))


def paired_key(row: dict[str, str]) -> str:
    return "||".join(
        [
            row["source_image_id"],
            row["original_roi_id"],
            normalized_path(Path(row["image_path"])),
        ]
    )


def validate_manifest(path: Path, classes: list[int]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Validation manifest is missing or empty: {path}")
    manifest_sha256 = sha256_file(path)
    if manifest_sha256 != EXPECTED_VAL_MANIFEST_SHA256:
        raise ValueError(
            f"Validation manifest SHA256 mismatch: {manifest_sha256} != {EXPECTED_VAL_MANIFEST_SHA256}"
        )
    rows = read_csv(path)
    if len(rows) != EXPECTED_VAL_ROWS:
        raise ValueError(f"Validation manifest rows={len(rows)}, expected {EXPECTED_VAL_ROWS}")
    if not rows or not REQUIRED_MANIFEST_FIELDS.issubset(rows[0]):
        missing = sorted(REQUIRED_MANIFEST_FIELDS - set(rows[0] if rows else []))
        raise ValueError(f"Validation manifest is missing required fields: {missing}")

    keys: list[str] = []
    missing_images: list[str] = []
    invalid_rows: list[int] = []
    class_counts: Counter[int] = Counter()
    unique_sources: dict[int, set[str]] = defaultdict(set)
    for position, row in enumerate(rows):
        try:
            class_id = int(row["class_id"])
        except (TypeError, ValueError):
            invalid_rows.append(position)
            continue
        expected_name = CLASS_MAPPING.get(class_id)
        image_path = Path(row["image_path"])
        if (
            row["split"] != "val"
            or expected_name is None
            or row["class_name"] != expected_name
            or not row["source_image_id"]
            or not row["original_roi_id"]
            or row["is_brightness_augmented"].strip().lower() not in {"false", "0", "no"}
        ):
            invalid_rows.append(position)
        if not image_path.is_file() or image_path.stat().st_size <= 0:
            missing_images.append(str(image_path))
        class_counts[class_id] += 1
        unique_sources[class_id].add(row["source_image_id"])
        keys.append(paired_key(row))

    duplicate_paired_keys = len(keys) - len(set(keys))
    expected_counts_ok = all(class_counts[class_id] == EXPECTED_CLASS_ROWS[class_id] for class_id in classes)
    if invalid_rows or missing_images or duplicate_paired_keys or not expected_counts_ok:
        raise ValueError(
            "Validation manifest audit failed: "
            f"invalid_rows={len(invalid_rows)}, missing_images={len(missing_images)}, "
            f"duplicate_paired_keys={duplicate_paired_keys}, class_counts={dict(class_counts)}"
        )
    audit = {
        "status": "PASS",
        "path": str(path),
        "sha256": manifest_sha256,
        "rows": len(rows),
        "class_counts": {str(key): class_counts[key] for key in classes},
        "unique_source_counts": {str(key): len(unique_sources[key]) for key in classes},
        "missing_images": 0,
        "invalid_rows": 0,
        "duplicate_paired_keys": 0,
        "brightness_augmented_rows": 0,
        "test_manifest_read": False,
        "test_images_read_count": 0,
    }
    return rows, audit


def deterministic_sample(
    rows: list[dict[str, str]], classes: list[int], samples_per_class: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    globally_used_sources: set[str] = set()

    for class_id in classes:
        class_rows = [row for row in rows if int(row["class_id"]) == class_id]
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in class_rows:
            grouped[row["source_image_id"]].append(row)
        for source_rows in grouped.values():
            source_rows.sort(key=lambda row: (row["original_roi_id"], normalized_path(Path(row["image_path"]))))

        sources = sorted(grouped)
        rng.shuffle(sources)
        preferred_sources = [source for source in sources if source not in globally_used_sources]
        fallback_sources = [source for source in sources if source in globally_used_sources]
        ordered_sources = preferred_sources + fallback_sources
        if len(ordered_sources) < samples_per_class:
            raise ValueError(
                f"Class {class_id} has only {len(ordered_sources)} unique sources; "
                f"cannot select {samples_per_class} unique-source ROI"
            )

        for source in ordered_sources[:samples_per_class]:
            candidates = grouped[source]
            row = candidates[rng.randrange(len(candidates))]
            actual_hash = sha256_file(Path(row["image_path"]))
            if actual_hash != row["image_sha256"]:
                raise ValueError(
                    f"Selected image SHA256 mismatch for {row['image_path']}: "
                    f"{actual_hash} != {row['image_sha256']}"
                )
            order = len(selected) + 1
            selected.append(
                {
                    "selection_order": order,
                    "class_id": class_id,
                    "class_name": CLASS_MAPPING[class_id],
                    "source_image_id": row["source_image_id"],
                    "original_roi_id": row["original_roi_id"],
                    "image_path": str(Path(row["image_path"]).resolve()),
                    "image_sha256": actual_hash,
                    "selection_seed": seed,
                    "unique_source_preferred": source not in globally_used_sources,
                    "paired_key": paired_key(row),
                }
            )
            globally_used_sources.add(source)

    expected_total = len(classes) * samples_per_class
    if len(selected) != expected_total:
        raise RuntimeError(f"Selected {len(selected)} rows, expected {expected_total}")
    if len({row["paired_key"] for row in selected}) != expected_total:
        raise RuntimeError("Selected paired keys are not unique")
    for class_id in classes:
        if sum(int(row["class_id"]) == class_id for row in selected) != samples_per_class:
            raise RuntimeError(f"Class {class_id} sample count mismatch")
    return selected


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + stream.getvalue()).encode("utf-8")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def figure_bytes(figure: plt.Figure) -> bytes:
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return stream.getvalue()


def individual_visualization(image: Image.Image, row: dict[str, Any]) -> bytes:
    probabilities = [row[f"probability_class_{index}"] for index in range(NUM_CLASSES)]
    colors = ["#176b87" if index == row["predicted_class_id"] else "#9aa9b2" for index in range(NUM_CLASSES)]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1, 1.35]})
    axes[0].imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Validation ROI")
    axes[0].axis("off")
    positions = np.arange(NUM_CLASSES)
    axes[1].barh(positions, probabilities, color=colors)
    axes[1].set_yticks(positions, [CLASS_MAPPING[index] for index in range(NUM_CLASSES)])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Softmax probability")
    axes[1].grid(axis="x", alpha=0.22)
    for index, probability in enumerate(probabilities):
        axes[1].text(min(probability + 0.012, 0.94), index, f"{probability:.4f}", va="center", fontsize=9)
    correctness = "Correct" if row["is_correct"] else "Incorrect"
    figure.suptitle(
        f"GT: {row['true_class_id']} - {row['true_class_name']}  |  "
        f"Pred: {row['predicted_class_id']} - {row['predicted_class_name']}  |  "
        f"{correctness}\nConfidence: {row['confidence']:.4f}  |  "
        f"P(class 2): {row['probability_class_2']:.4f}  |  "
        f"P(class 4): {row['probability_class_4']:.4f}  |  "
        f"Margin P2-P4: {row['class2_vs_class4_margin']:.4f}",
        fontsize=12,
    )
    figure.text(0.5, 0.015, DISCLAIMER, ha="center", fontsize=8, color="#7a2f2a")
    figure.tight_layout(rect=[0, 0.05, 1, 0.88])
    return figure_bytes(figure)


def confusion_matrix_rows(predictions: list[dict[str, Any]], classes: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for true_class in classes:
        counts = Counter(
            int(row["predicted_class_id"])
            for row in predictions
            if int(row["true_class_id"]) == true_class
        )
        rows.append(
            {
                "true_class_id": true_class,
                "true_class_name": CLASS_MAPPING[true_class],
                **{f"predicted_class_{index}": counts[index] for index in range(NUM_CLASSES)},
            }
        )
    return rows


def confusion_figure(matrix_rows: list[dict[str, Any]]) -> bytes:
    matrix = np.asarray(
        [[row[f"predicted_class_{index}"] for index in range(NUM_CLASSES)] for row in matrix_rows],
        dtype=int,
    )
    figure, axis = plt.subplots(figsize=(9, 4.6))
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    axis.set_xticks(range(NUM_CLASSES), [f"{index}\n{CLASS_MAPPING[index]}" for index in range(NUM_CLASSES)], fontsize=8)
    axis.set_yticks(range(len(matrix_rows)), [f"True {row['true_class_id']}\n{row['true_class_name']}" for row in matrix_rows])
    axis.set_xlabel("Predicted class")
    axis.set_title("Patch Proposed: sampled Validation class 2/4 confusion matrix")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.03, pad=0.03)
    figure.tight_layout()
    return figure_bytes(figure)


def probability_figure(predictions: list[dict[str, Any]]) -> bytes:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for axis, true_class in zip(axes, EXPECTED_CLASSES, strict=True):
        subset = [row for row in predictions if row["true_class_id"] == true_class]
        x = np.arange(1, len(subset) + 1)
        axis.plot(x, [row["probability_class_2"] for row in subset], marker="o", label="P(Pleural thickening)", color="#176b87")
        axis.plot(x, [row["probability_class_4"] for row in subset], marker="s", label="P(Pleural effusion)", color="#a33b32")
        axis.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--")
        axis.set_title(f"True class {true_class}: {CLASS_MAPPING[true_class]}")
        axis.set_xlabel("Sample within class")
        axis.set_xticks(x)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Softmax probability")
    axes[1].legend(loc="best")
    figure.suptitle("Class 2 vs class 4 probability comparison on sampled Validation ROI")
    figure.tight_layout()
    return figure_bytes(figure)


def confidence_figure(predictions: list[dict[str, Any]]) -> bytes:
    correct = [row["confidence"] for row in predictions if row["is_correct"]]
    incorrect = [row["confidence"] for row in predictions if not row["is_correct"]]
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    groups = [("Correct", correct, "#287d72"), ("Incorrect", incorrect, "#a33b32")]
    for group_index, (label, values, color) in enumerate(groups):
        if values:
            offsets = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
            axis.scatter(np.full(len(values), group_index) + offsets, values, color=color, s=55, alpha=0.85, label=f"{label} ROI")
            axis.hlines(np.mean(values), group_index - 0.22, group_index + 0.22, color="black", linewidth=2)
    axis.axhline(0.8, color="#a33b32", linestyle="--", linewidth=1, label="High confidence = 0.80")
    axis.axhline(0.6, color="#6b7280", linestyle=":", linewidth=1, label="Low confidence = 0.60")
    axis.set_xticks([0, 1], [f"Correct\n(n={len(correct)})", f"Incorrect\n(n={len(incorrect)})"])
    axis.set_ylim(0.0, 1.03)
    axis.set_ylabel("Prediction confidence")
    axis.set_title("Confidence of correct and incorrect sampled Validation predictions")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    return figure_bytes(figure)


def montage_figure(predictions: list[dict[str, Any]], image_by_order: dict[int, Image.Image]) -> bytes:
    figure, axes = plt.subplots(4, 5, figsize=(15, 11.5))
    for axis, row in zip(axes.flat, predictions, strict=True):
        axis.imshow(image_by_order[row["selection_order"]].convert("L"), cmap="gray", vmin=0, vmax=255)
        correctness = "Correct" if row["is_correct"] else "Incorrect"
        color = "#155d52" if row["is_correct"] else "#a33b32"
        axis.set_title(
            f"#{row['selection_order']} GT {row['true_class_id']} -> Pred {row['predicted_class_id']}\n"
            f"Conf {row['confidence']:.3f} | {correctness}",
            fontsize=9,
            color=color,
        )
        axis.axis("off")
    figure.suptitle(
        "Patch Proposed sampled Validation ROI: class 2 first, class 4 second\n" + DISCLAIMER,
        fontsize=14,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    return figure_bytes(figure)


def summarize(predictions: list[dict[str, Any]], classes: list[int]) -> dict[str, Any]:
    class_summaries: dict[str, Any] = {}
    representatives: dict[str, Any] = {}
    for class_id in classes:
        subset = [row for row in predictions if row["true_class_id"] == class_id]
        correct = [row for row in subset if row["is_correct"]]
        incorrect = [row for row in subset if not row["is_correct"]]
        predicted_counts = Counter(row["predicted_class_id"] for row in subset)
        class_summaries[str(class_id)] = {
            "class_name": CLASS_MAPPING[class_id],
            "selected_count": len(subset),
            "correct_count": len(correct),
            "incorrect_count": len(incorrect),
            "sample_accuracy": len(correct) / len(subset),
            "predicted_as_class_0": predicted_counts[0],
            "predicted_as_class_1": predicted_counts[1],
            "predicted_as_class_2": predicted_counts[2],
            "predicted_as_class_3": predicted_counts[3],
            "predicted_as_class_4": predicted_counts[4],
            "mean_confidence": float(np.mean([row["confidence"] for row in subset])),
            "mean_true_class_probability": float(np.mean([row["true_class_probability"] for row in subset])),
            "mean_class2_vs_class4_margin": float(np.mean([row["class2_vs_class4_margin"] for row in subset])),
        }
        representatives[str(class_id)] = {
            "selection_rule": "Highest-confidence rows within correctness stratum; no sample was added or removed.",
            "correct": [
                {"selection_order": row["selection_order"], "paired_key": row["paired_key"], "confidence": row["confidence"]}
                for row in sorted(correct, key=lambda item: (-item["confidence"], item["selection_order"]))[:2]
            ],
            "incorrect": [
                {"selection_order": row["selection_order"], "paired_key": row["paired_key"], "confidence": row["confidence"]}
                for row in sorted(incorrect, key=lambda item: (-item["confidence"], item["selection_order"]))[:2]
            ],
        }
    overall = {
        "selected_count": len(predictions),
        "correct_count": sum(row["is_correct"] for row in predictions),
        "incorrect_count": sum(not row["is_correct"] for row in predictions),
        "class_2_predicted_as_class_4": sum(
            row["true_class_id"] == 2 and row["predicted_class_id"] == 4 for row in predictions
        ),
        "class_4_predicted_as_class_2": sum(
            row["true_class_id"] == 4 and row["predicted_class_id"] == 2 for row in predictions
        ),
        "high_confidence_incorrect_count": sum(
            (not row["is_correct"]) and row["confidence"] >= 0.80 for row in predictions
        ),
        "low_confidence_prediction_count": sum(row["confidence"] < 0.60 for row in predictions),
    }
    return {"classes": class_summaries, "overall": overall, "representative_examples": representatives}


def summary_markdown(summary: dict[str, Any], selected: list[dict[str, Any]]) -> str:
    lines = [
        "# Patch Proposed Class 2/4 Validation Demo",
        "",
        "> 本報告只描述 seed 42 固定抽樣的 20 張 Validation ROI，不代表完整 Validation 或 Test 表現。",
        "",
        "## 抽樣設計",
        "",
        "- 類別：class 2 Pleural thickening、class 4 Pleural effusion。",
        "- 每類 10 張，總計 20 張；抽樣不使用模型預測結果。",
        f"- Unique source_image_id：{len({row['source_image_id'] for row in selected})} / 20。",
        "- Test manifest 與 Test images 均未讀取。",
        "",
        "## 子集結果",
        "",
        "| True class | Correct | Incorrect | Sample accuracy | Pred 2 | Pred 3 | Pred 4 | Mean confidence | Mean true probability | Mean P2-P4 margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for class_id in EXPECTED_CLASSES:
        item = summary["classes"][str(class_id)]
        lines.append(
            f"| {class_id} {item['class_name']} | {item['correct_count']} | {item['incorrect_count']} | "
            f"{item['sample_accuracy']:.4f} | {item['predicted_as_class_2']} | "
            f"{item['predicted_as_class_3']} | {item['predicted_as_class_4']} | "
            f"{item['mean_confidence']:.4f} | {item['mean_true_class_probability']:.4f} | "
            f"{item['mean_class2_vs_class4_margin']:.4f} |"
        )
    overall = summary["overall"]
    lines.extend(
        [
            "",
            "## 錯誤方向",
            "",
            f"- Total correct / incorrect：{overall['correct_count']} / {overall['incorrect_count']}。",
            f"- Class 2 -> class 4：{overall['class_2_predicted_as_class_4']}。",
            f"- Class 4 -> class 2：{overall['class_4_predicted_as_class_2']}。",
            f"- High-confidence incorrect (confidence >= 0.80)：{overall['high_confidence_incorrect_count']}。",
            f"- Low-confidence predictions (confidence < 0.60)：{overall['low_confidence_prediction_count']}。",
            "",
            "## 解讀限制",
            "",
            "這是固定抽樣的質性示範，不重新計算完整 Validation/Test Accuracy、Macro-F1 或 AUROC，也不能取代既有 Test 與 source-cluster bootstrap 結論。錯誤案例只能用於理解 class 2/4 的可能混淆方向，不能據此挑選 threshold、checkpoint 或宣稱整體模型改善。",
            "",
        ]
    )
    for class_id in EXPECTED_CLASSES:
        reps = summary["representative_examples"][str(class_id)]
        lines.append(f"## Class {class_id} 代表案例")
        lines.append("")
        for label in ("correct", "incorrect"):
            values = reps[label]
            rendered = ", ".join(
                f"#{item['selection_order']} (confidence {item['confidence']:.4f})" for item in values
            ) or "本次固定樣本中不足 2 張，不補抽、不更換樣本"
            lines.append(f"- {label.title()}：{rendered}。")
        lines.append("")
    return "\n".join(lines)


def atomic_write_tree(output_dir: Path, payloads: dict[Path, bytes]) -> list[Path]:
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing:
            raise FileExistsError(f"Output directory is non-empty; refusing to overwrite: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    temporary_paths: list[Path] = []
    final_paths: list[Path] = []
    try:
        for relative_path, content in payloads.items():
            destination = output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp")
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
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Sample and infer Patch Proposed Validation ROI for Pleural thickening/effusion."
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--classes", type=int, nargs="+", default=EXPECTED_CLASSES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda:0"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline_started = time.perf_counter()
    project_root = args.project_root.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")
    classes = list(args.classes)
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"This controlled demo requires --classes 2 4, received {classes}")
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive")
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
    val_manifest = (
        args.val_manifest.expanduser().resolve()
        if args.val_manifest
        else project_root
        / "outputs"
        / "raddino_convnext_tiny_experiment_seed42"
        / "phase2_split"
        / "val_roi_manifest.csv"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else project_root / "outputs" / "patch_proposed_class2_class4_validation_demo"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is non-empty; refusing to overwrite: {output_dir}")
    model_sha256 = sha256_file(model_path)
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise ValueError(f"Patch Proposed model SHA256 mismatch: {model_sha256}")

    manifest_rows, manifest_audit = validate_manifest(val_manifest, classes)
    selected = deterministic_sample(manifest_rows, classes, args.samples_per_class, args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device.index if device.index is not None else 0)
    model, checkpoint, load_audit, model_load_seconds = load_export_model(model_path, device)
    preprocessing = Phase2Transform(training=False).preprocessing_config()

    dry_run = {
        "status": "PASS",
        "dry_run": True,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "architecture": checkpoint["architecture"],
        "initialization": checkpoint["initialization_description"],
        "strict_load": load_audit["strict_load"],
        "missing_keys": load_audit["missing_keys"],
        "unexpected_keys": load_audit["unexpected_keys"],
        "validation_manifest": manifest_audit,
        "classes": classes,
        "samples_per_class": args.samples_per_class,
        "selection_seed": args.seed,
        "selected_rows": len(selected),
        "unique_source_image_ids": len({row["source_image_id"] for row in selected}),
        "selected_paired_keys": [row["paired_key"] for row in selected],
        "preprocessing": preprocessing,
        "planned_output_directory": str(output_dir),
        "forward_executed": False,
        "files_written": 0,
        "test_manifest_read": False,
        "test_images_read_count": 0,
    }
    if args.dry_run:
        print(json.dumps(dry_run, ensure_ascii=False, indent=2))
        return 0

    transform = Phase2Transform(training=False)
    predictions: list[dict[str, Any]] = []
    image_by_order: dict[int, Image.Image] = {}
    visualization_payloads: dict[Path, bytes] = {}
    total_preprocessing_seconds = 0.0
    total_inference_seconds = 0.0
    numerical_nan_count = 0
    numerical_inf_count = 0

    for selection in selected:
        image, image_audit = inspect_image(Path(selection["image_path"]))
        image_by_order[selection["selection_order"]] = image.copy()
        preprocessing_started = time.perf_counter()
        tensor = transform(image).unsqueeze(0).to(device)
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        total_preprocessing_seconds += preprocessing_seconds
        if list(tensor.shape) != [1, 3, 224, 224]:
            raise RuntimeError(f"Unexpected input shape for selection {selection['selection_order']}: {list(tensor.shape)}")

        inference_started = time.perf_counter()
        with torch.inference_mode():
            logits = model(tensor)
            probabilities_tensor = torch.softmax(logits.float(), dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_started
        total_inference_seconds += inference_seconds
        if list(logits.shape) != [1, 5] or list(probabilities_tensor.shape) != [1, 5]:
            raise RuntimeError(
                f"Unexpected output shapes for selection {selection['selection_order']}: "
                f"logits={list(logits.shape)}, probabilities={list(probabilities_tensor.shape)}"
            )
        numerical_nan_count += int(torch.isnan(logits).sum().item()) + int(torch.isnan(probabilities_tensor).sum().item())
        numerical_inf_count += int(torch.isinf(logits).sum().item()) + int(torch.isinf(probabilities_tensor).sum().item())
        probability_sum = float(probabilities_tensor.sum().item())
        if numerical_nan_count or numerical_inf_count or abs(probability_sum - 1.0) > 1e-6:
            raise RuntimeError(
                f"Numerical audit failed at selection {selection['selection_order']}: "
                f"NaN={numerical_nan_count}, Inf={numerical_inf_count}, sum={probability_sum}"
            )
        probabilities = [float(value) for value in probabilities_tensor[0].detach().cpu().tolist()]
        predicted_class_id = int(torch.argmax(probabilities_tensor, dim=1).item())
        true_class_id = int(selection["class_id"])
        confidence = probabilities[predicted_class_id]
        row = {
            **selection,
            "true_class_id": true_class_id,
            "true_class_name": CLASS_MAPPING[true_class_id],
            "predicted_class_id": predicted_class_id,
            "predicted_class_name": CLASS_MAPPING[predicted_class_id],
            "confidence": confidence,
            **{f"probability_class_{index}": probabilities[index] for index in range(NUM_CLASSES)},
            "probability_aortic_enlargement": probabilities[0],
            "probability_cardiomegaly": probabilities[1],
            "probability_pleural_thickening": probabilities[2],
            "probability_pulmonary_fibrosis": probabilities[3],
            "probability_pleural_effusion": probabilities[4],
            "true_class_probability": probabilities[true_class_id],
            "class2_vs_class4_margin": probabilities[2] - probabilities[4],
            "is_correct": predicted_class_id == true_class_id,
            "inference_seconds": inference_seconds,
            "original_width": image_audit["width"],
            "original_height": image_audit["height"],
            "original_mode": image_audit["mode"],
        }
        predictions.append(row)
        visualization_name = f"{selection['selection_order']:02d}_{Path(selection['image_path']).stem}.png"
        visualization_payloads[Path("visualizations") / visualization_name] = individual_visualization(image, row)

    matrix_rows = confusion_matrix_rows(predictions, classes)
    summary_stats = summarize(predictions, classes)
    environment = environment_info(device)
    gpu_metrics = {
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
    }
    if device.type == "cuda":
        gpu_metrics = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    total_pipeline_seconds = time.perf_counter() - pipeline_started
    summary_document = {
        "status": "PASS",
        "scope": "Deterministic sampled Validation subset only; not full Validation or Test metrics.",
        "timestamp": utc_now(),
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "validation_manifest_path": str(val_manifest),
        "validation_manifest_sha256": manifest_audit["sha256"],
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
        "classes": classes,
        "selected_rows": len(selected),
        "unique_source_image_ids": len({row["source_image_id"] for row in selected}),
        "statistics": summary_stats,
        "confusion_matrix": matrix_rows,
        "preprocessing": preprocessing,
        "test_manifest_read": False,
        "test_images_read_count": 0,
        "threshold_used": False,
        "full_validation_metrics_recomputed": False,
        "test_metrics_recomputed": False,
        "disclaimer": DISCLAIMER,
    }
    inference_audit = {
        "status": "PASS",
        "model_load_seconds": model_load_seconds,
        "total_preprocessing_seconds": total_preprocessing_seconds,
        "total_inference_seconds": total_inference_seconds,
        "average_inference_seconds_per_roi": total_inference_seconds / len(predictions),
        "total_pipeline_seconds_before_output_serialization": total_pipeline_seconds,
        "device": str(device),
        "device_name": environment["gpu"]["name"] if environment["gpu"] else "CPU",
        "cuda_used": device.type == "cuda",
        "gpu_memory": gpu_metrics,
        "strict_load": load_audit["strict_load"],
        "missing_keys": load_audit["missing_keys"],
        "unexpected_keys": load_audit["unexpected_keys"],
        "model_eval": load_audit["model_eval"],
        "trainable_parameter_count": load_audit["trainable_parameter_count"],
        "optimizer_created": False,
        "backward_called": False,
        "rad_dino_loaded": False,
        "teacher_cache_loaded": False,
        "patch_phase1_checkpoint_loaded": False,
        "cls_checkpoint_loaded": False,
        "baseline_checkpoint_loaded": False,
        "test_manifest_read": False,
        "test_images_read_count": 0,
        "validation_manifest_rows": manifest_audit["rows"],
        "selected_rows": len(selected),
        "unique_paired_keys": len({row["paired_key"] for row in predictions}),
        "unique_source_image_ids": len({row["source_image_id"] for row in predictions}),
        "per_roi_input_shape": [1, 3, 224, 224],
        "per_roi_logits_shape": [1, 5],
        "per_roi_probabilities_shape": [1, 5],
        "probability_sum_max_absolute_error": max(
            abs(sum(row[f"probability_class_{index}"] for index in range(NUM_CLASSES)) - 1.0)
            for row in predictions
        ),
        "nan_count": numerical_nan_count,
        "inf_count": numerical_inf_count,
        "threshold_used": False,
        "full_validation_metrics_recomputed": False,
        "test_metrics_recomputed": False,
        "source_roi_modified": False,
        "protected_artifacts_modified": False,
    }

    confusion_fields = [
        "true_class_id",
        "true_class_name",
        *[f"predicted_class_{index}" for index in range(NUM_CLASSES)],
    ]
    payloads: dict[Path, bytes] = {
        Path("selected_validation_samples.csv"): csv_bytes(selected, SELECTION_FIELDS),
        Path("predictions.csv"): csv_bytes(predictions, PREDICTION_FIELDS),
        Path("summary.json"): json_bytes(summary_document),
        Path("summary.md"): summary_markdown(summary_stats, selected).encode("utf-8"),
        Path("class2_class4_confusion_matrix.csv"): csv_bytes(matrix_rows, confusion_fields),
        Path("class2_class4_confusion_matrix.png"): confusion_figure(matrix_rows),
        Path("probability_comparison.png"): probability_figure(predictions),
        Path("confidence_correct_vs_incorrect.png"): confidence_figure(predictions),
        Path("prediction_montage.png"): montage_figure(predictions, image_by_order),
        Path("inference_audit.json"): json_bytes(inference_audit),
        Path("environment.json"): json_bytes(environment),
        **visualization_payloads,
    }
    written = atomic_write_tree(output_dir, payloads)
    residual = [
        str(path)
        for path in output_dir.rglob("*")
        if path.is_file() and (path.name.endswith(".tmp") or path.name.endswith(".writing"))
    ]
    if residual:
        raise RuntimeError(f"Temporary output files remain: {residual}")
    result = {
        "status": "PASS",
        "output_directory": str(output_dir),
        "files_written": len(written),
        "root_files": len([path for path in written if path.parent == output_dir]),
        "visualizations": len(visualization_payloads),
        "selected_rows": len(selected),
        "unique_source_image_ids": len({row["source_image_id"] for row in selected}),
        "statistics": summary_stats,
        "model_load_seconds": model_load_seconds,
        "total_preprocessing_seconds": total_preprocessing_seconds,
        "total_inference_seconds": total_inference_seconds,
        "total_pipeline_seconds_before_output_serialization": total_pipeline_seconds,
        "device": str(device),
        "test_images_read_count": 0,
        "temporary_files_remaining": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
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
