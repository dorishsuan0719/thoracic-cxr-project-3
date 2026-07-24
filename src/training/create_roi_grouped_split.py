from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))

from common import CLASS_ORDER, SPLITS, metadata_dir, reports_dir, write_csv_rows  # noqa: E402

RANDOM_SEED = 42
SPLIT_RATIO = {"train": 0.8, "val": 0.1, "test": 0.1}
SOURCE_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "roi_224_source_split_seed42.csv"
ROI_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "roi_224_roi_split_seed42.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stable_hash(value: str) -> str:
    return hashlib.sha256(f"{RANDOM_SEED}:{value}".encode("utf-8")).hexdigest()


def source_labels(master_rows: list[dict[str, str]]) -> dict[str, set[int]]:
    labels: dict[str, set[int]] = defaultdict(set)
    for row in master_rows:
        labels[row["source_image_id"]].add(int(row["class_id"]))
    return labels


def targets(total: int) -> dict[str, int]:
    train = round(total * SPLIT_RATIO["train"])
    val = round(total * SPLIT_RATIO["val"])
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def class_targets(labels_by_source: dict[str, set[int]]) -> dict[str, dict[int, int]]:
    class_totals = Counter()
    for labels in labels_by_source.values():
        for class_id in labels:
            class_totals[class_id] += 1
    result = {split: {} for split in SPLITS}
    for _, class_id in CLASS_ORDER:
        split_targets = targets(class_totals[class_id])
        for split in SPLITS:
            result[split][class_id] = split_targets[split]
    return result


def assign_splits(labels_by_source: dict[str, set[int]]) -> dict[str, str]:
    total_targets = targets(len(labels_by_source))
    target_by_class = class_targets(labels_by_source)
    total_counts = Counter()
    class_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    assignments: dict[str, str] = {}

    ordered_sources = sorted(
        labels_by_source,
        key=lambda source_id: (-len(labels_by_source[source_id]), stable_hash(source_id)),
    )

    for source_id in ordered_sources:
        labels = labels_by_source[source_id]
        best_split = None
        best_score = None
        for split in SPLITS:
            over_total = max(0, total_counts[split] + 1 - total_targets[split])
            label_need_before = sum(target_by_class[split][class_id] - class_counts[split][class_id] for class_id in labels)
            label_over_after = sum(max(0, class_counts[split][class_id] + 1 - target_by_class[split][class_id]) for class_id in labels)
            total_fill_ratio = (total_counts[split] + 1) / max(1, total_targets[split])
            score = (
                over_total * 10000
                + label_over_after * 100
                - label_need_before * 10
                + total_fill_ratio
                + {"train": 0.0, "val": 0.01, "test": 0.02}[split]
            )
            if best_score is None or score < best_score:
                best_score = score
                best_split = split
        assert best_split is not None
        assignments[source_id] = best_split
        total_counts[best_split] += 1
        for class_id in labels:
            class_counts[best_split][class_id] += 1

    return assignments


def write_split_manifests(master_rows: list[dict[str, str]], assignments: dict[str, str], labels_by_source: dict[str, set[int]]) -> None:
    source_rows = []
    for source_id in sorted(labels_by_source):
        labels = labels_by_source[source_id]
        source_rows.append(
            {
                "source_image_id": source_id,
                "split": assignments[source_id],
                "class_0_present": int(0 in labels),
                "class_1_present": int(1 in labels),
                "class_2_present": int(2 in labels),
                "class_3_present": int(3 in labels),
                "class_4_present": int(4 in labels),
                "random_seed": RANDOM_SEED,
            }
        )
    write_csv_rows(
        SOURCE_SPLIT_PATH,
        ["source_image_id", "split", "class_0_present", "class_1_present", "class_2_present", "class_3_present", "class_4_present", "random_seed"],
        source_rows,
    )

    roi_rows = []
    for row in master_rows:
        roi_rows.append(
            {
                "master_image_path": row["master_image_path"],
                "source_image_id": row["source_image_id"],
                "annotation_index": row["annotation_index"],
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "split": assignments[row["source_image_id"]],
                "random_seed": RANDOM_SEED,
            }
        )
    write_csv_rows(
        ROI_SPLIT_PATH,
        ["master_image_path", "source_image_id", "annotation_index", "class_id", "class_name", "split", "random_seed"],
        roi_rows,
    )


def audit(master_rows: list[dict[str, str]], assignments: dict[str, str], labels_by_source: dict[str, set[int]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    source_split_sets: dict[str, set[str]] = defaultdict(set)
    for source_id, split in assignments.items():
        source_split_sets[split].add(source_id)

    leakages = {}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = source_split_sets[left] & source_split_sets[right]
        leakages[f"{left}_vs_{right}"] = len(overlap)
        if overlap:
            errors.append({"error_type": "source_image_id_leakage", "split_pair": f"{left}_vs_{right}", "detail": ";".join(sorted(overlap)[:100])})

    by_source_class: list[dict[str, Any]] = []
    by_roi_class: list[dict[str, Any]] = []
    target_by_class = class_targets(labels_by_source)
    for split in SPLITS:
        split_sources = {sid for sid, assigned in assignments.items() if assigned == split}
        for class_name, class_id in CLASS_ORDER:
            unique_source_count = sum(1 for sid in split_sources if class_id in labels_by_source[sid])
            roi_count = sum(1 for row in master_rows if assignments[row["source_image_id"]] == split and int(row["class_id"]) == class_id)
            target_count = target_by_class[split][class_id]
            by_source_class.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "unique_source_image_count": unique_source_count,
                    "target_unique_source_image_count": target_count,
                    "difference_from_target": unique_source_count - target_count,
                }
            )
            by_roi_class.append({"split": split, "class_id": class_id, "class_name": class_name, "roi_count": roi_count})

    roi_split_by_source = defaultdict(set)
    for row in master_rows:
        roi_split_by_source[row["source_image_id"]].add(assignments[row["source_image_id"]])
    for source_id, splits in roi_split_by_source.items():
        if len(splits) > 1:
            errors.append({"error_type": "roi_source_assigned_to_multiple_splits", "split_pair": ",".join(sorted(splits)), "detail": source_id})

    source_targets = targets(len(labels_by_source))
    metrics = {
        "master_roi_count": len(master_rows),
        "unique_source_images": len(labels_by_source),
        "train_unique_source_images": len(source_split_sets["train"]),
        "val_unique_source_images": len(source_split_sets["val"]),
        "test_unique_source_images": len(source_split_sets["test"]),
        "target_train_unique_source_images": source_targets["train"],
        "target_val_unique_source_images": source_targets["val"],
        "target_test_unique_source_images": source_targets["test"],
        "train_roi_count": sum(1 for row in master_rows if assignments[row["source_image_id"]] == "train"),
        "val_roi_count": sum(1 for row in master_rows if assignments[row["source_image_id"]] == "val"),
        "test_roi_count": sum(1 for row in master_rows if assignments[row["source_image_id"]] == "test"),
        "train_vs_val_leakage": leakages["train_vs_val"],
        "train_vs_test_leakage": leakages["train_vs_test"],
        "val_vs_test_leakage": leakages["val_vs_test"],
        "legacy_split_used_for_new_split": False,
    }
    return metrics, by_source_class, by_roi_class, errors


def write_version(metrics: dict[str, Any]) -> None:
    version = {
        "dataset_name": "thoracic_cxr_roi_classification_master",
        "version": "roi_224_master_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_classes": 5,
        "total_roi_images": metrics["master_roi_count"],
        "unique_source_images": metrics["unique_source_images"],
        "class_mapping": {str(class_id): class_name for class_name, class_id in CLASS_ORDER},
        "image_size": [224, 224],
        "resize_method": "keep_aspect_ratio_letterbox",
        "padding_value": 0,
        "master_manifest": str(metadata_dir() / "roi_224_master_manifest.csv"),
        "source_split_manifest": str(SOURCE_SPLIT_PATH),
        "roi_split_manifest": str(ROI_SPLIT_PATH),
        "split_ratio": SPLIT_RATIO,
        "random_seed": RANDOM_SEED,
        "split_unit": "source_image_id",
        "leakage_check": {
            "train_vs_val": metrics["train_vs_val_leakage"],
            "train_vs_test": metrics["train_vs_test_leakage"],
            "val_vs_test": metrics["val_vs_test_leakage"],
        },
        "legacy_dataset_version": "roi_224_v1",
    }
    (metadata_dir() / "dataset_version_roi_224_master_v1.json").write_text(json.dumps(version, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    master_rows = read_csv_rows(metadata_dir() / "roi_224_master_manifest.csv")
    labels_by_source = source_labels(master_rows)
    assignments = assign_splits(labels_by_source)
    write_split_manifests(master_rows, assignments, labels_by_source)
    metrics, by_source_class, by_roi_class, errors = audit(master_rows, assignments, labels_by_source)

    reports_dir().mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        reports_dir() / "roi_224_grouped_split_by_source_class.csv",
        ["split", "class_id", "class_name", "unique_source_image_count", "target_unique_source_image_count", "difference_from_target"],
        by_source_class,
    )
    write_csv_rows(
        reports_dir() / "roi_224_grouped_split_by_roi_class.csv",
        ["split", "class_id", "class_name", "roi_count"],
        by_roi_class,
    )
    write_csv_rows(
        reports_dir() / "roi_224_grouped_split_errors.csv",
        ["error_type", "split_pair", "detail"],
        errors,
    )
    with (reports_dir() / "roi_224_grouped_split_audit.txt").open("w", encoding="utf-8") as f:
        f.write("ROI 224 grouped split audit\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write(f"error_rows: {len(errors)}\n")
        f.write("split_method: deterministic greedy multilabel source_image_id grouped split\n")

    write_version(metrics)
    print("ROI 224 grouped split completed.")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"error_rows: {len(errors)}")
    print(f"Source split: {SOURCE_SPLIT_PATH}")
    print(f"ROI split: {ROI_SPLIT_PATH}")
    return 0 if not errors and metrics["master_roi_count"] == 2343 else 1


if __name__ == "__main__":
    sys.exit(main())

