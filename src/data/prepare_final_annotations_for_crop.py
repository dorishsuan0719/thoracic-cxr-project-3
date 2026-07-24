from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import CLASS_ORDER, SPLITS, crop_manifest_path, metadata_dir, reports_dir, validate_bbox, write_csv_rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_status(value: str) -> str:
    return value.strip().lower()


def annotation_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["split"], row["image_id"], str(row["annotation_index"]))


def bbox_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (row["image_id"], row["x_min"], row["y_min"], row["x_max"], row["y_max"])


def write_manual_review_summary(review_rows: list[dict[str, str]]) -> None:
    reports_dir().mkdir(parents=True, exist_ok=True)
    status_counts = Counter(normalize_status(row.get("review_status", "")) for row in review_rows)
    issue_counts = Counter(normalize_status(row.get("technical_issue", "")) for row in review_rows)
    split_status_counts = Counter(
        (row.get("split", ""), normalize_status(row.get("review_status", ""))) for row in review_rows
    )

    summary_rows: list[dict[str, Any]] = []
    for status in sorted(status_counts):
        summary_rows.append({"summary_type": "review_status", "split": "", "name": status or "(blank)", "count": status_counts[status]})
    for issue in sorted(issue_counts):
        summary_rows.append({"summary_type": "technical_issue", "split": "", "name": issue or "(blank)", "count": issue_counts[issue]})
    for split in SPLITS:
        for status in ("pass", "review", "fail", ""):
            count = split_status_counts[(split, status)]
            if count:
                summary_rows.append({"summary_type": "split_review_status", "split": split, "name": status or "(blank)", "count": count})

    write_csv_rows(reports_dir() / "manual_bbox_review_summary.csv", ["summary_type", "split", "name", "count"], summary_rows)

    with (reports_dir() / "manual_bbox_review_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Manual BBox overlay review summary\n")
        f.write("\nreview_status counts\n")
        for status in ("pass", "review", "fail", ""):
            if status_counts[status]:
                f.write(f"{status or '(blank)'}: {status_counts[status]}\n")
        f.write("\ntechnical_issue counts\n")
        for issue, count in sorted(issue_counts.items()):
            f.write(f"{issue or '(blank)'}: {count}\n")
        f.write("\nsplit x review_status counts\n")
        for split in SPLITS:
            for status in ("pass", "review", "fail", ""):
                count = split_status_counts[(split, status)]
                if count:
                    f.write(f"{split},{status or '(blank)'}: {count}\n")


def find_same_bbox_conflicts(valid_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], set[tuple[str, str, str]]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in valid_rows:
        groups[bbox_key(row)].append(row)

    conflict_rows: list[dict[str, Any]] = []
    conflict_keys: set[tuple[str, str, str]] = set()
    for rows in groups.values():
        class_pairs = {(row["class_id"], row["class_name"]) for row in rows}
        if len(class_pairs) <= 1:
            continue
        for row in rows:
            conflict_keys.add(annotation_key(row))
            for other in rows:
                if other is row:
                    continue
                if other["class_id"] == row["class_id"] and other["class_name"] == row["class_name"]:
                    continue
                conflict_rows.append(
                    {
                        "split": row["split"],
                        "image_id": row["image_id"],
                        "class_id": row["class_id"],
                        "class_name": row["class_name"],
                        "rad_id": row["rad_id"],
                        "annotation_index": row["annotation_index"],
                        "x_min": row["x_min"],
                        "y_min": row["y_min"],
                        "x_max": row["x_max"],
                        "y_max": row["y_max"],
                        "conflicting_class_id": other["class_id"],
                        "conflicting_class_name": other["class_name"],
                        "conflicting_annotation_index": other["annotation_index"],
                    }
                )
    return conflict_rows, conflict_keys


def add_exclusion(
    exclusions: dict[tuple[str, str, str, str], dict[str, Any]],
    row: dict[str, str],
    reason: str,
    source_file: str,
) -> None:
    key = (row["split"], row["image_id"], str(row["annotation_index"]), reason)
    exclusions[key] = {
        "split": row["split"],
        "image_id": row["image_id"],
        "class_id": row["class_id"],
        "class_name": row["class_name"],
        "annotation_index": row["annotation_index"],
        "x_min": row["x_min"],
        "y_min": row["y_min"],
        "x_max": row["x_max"],
        "y_max": row["y_max"],
        "exclusion_reason": reason,
        "source_file": source_file,
    }


def main() -> None:
    manual_path = metadata_dir() / "manual_bbox_review.csv"
    valid_path = metadata_dir() / "valid_annotations.csv"
    audited_path = metadata_dir() / "audited_annotations.csv"
    conflict_path = metadata_dir() / "same_bbox_different_class_conflicts.csv"
    excluded_path = metadata_dir() / "excluded_annotations.csv"
    final_path = metadata_dir() / "final_annotations_for_crop.csv"

    manual_rows = read_csv(manual_path)
    valid_rows = read_csv(valid_path)
    audited_rows = read_csv(audited_path)

    write_manual_review_summary(manual_rows)

    conflict_rows, conflict_keys = find_same_bbox_conflicts(valid_rows)
    write_csv_rows(
        conflict_path,
        [
            "split",
            "image_id",
            "class_id",
            "class_name",
            "rad_id",
            "annotation_index",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "conflicting_class_id",
            "conflicting_class_name",
            "conflicting_annotation_index",
        ],
        conflict_rows,
    )

    manual_fail_image_ids = {
        row["image_id"] for row in manual_rows if normalize_status(row.get("review_status", "")) == "fail"
    }

    exclusions: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in audited_rows:
        if row["image_id"] in manual_fail_image_ids:
            add_exclusion(exclusions, row, "manual_review_fail", str(manual_path))
        if annotation_key(row) in conflict_keys:
            add_exclusion(exclusions, row, "same_bbox_different_class", str(conflict_path))
        if str(row.get("is_duplicate", "")).strip().lower() == "true":
            add_exclusion(exclusions, row, "duplicate_annotation", str(audited_path))
        elif str(row.get("is_valid", "")).strip().lower() != "true":
            add_exclusion(exclusions, row, "invalid_bbox", str(audited_path))

    write_csv_rows(
        excluded_path,
        [
            "split",
            "image_id",
            "class_id",
            "class_name",
            "annotation_index",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "exclusion_reason",
            "source_file",
        ],
        exclusions.values(),
    )

    final_rows: list[dict[str, str]] = []
    duplicate_count_in_valid = 0
    manual_fail_count = 0
    conflict_count = 0
    missing_source_count = 0
    non_positive_count = 0
    for row in valid_rows:
        if str(row.get("is_valid", "")).strip().lower() != "true":
            continue
        if str(row.get("is_duplicate", "")).strip().lower() == "true":
            duplicate_count_in_valid += 1
            continue
        if row["image_id"] in manual_fail_image_ids:
            manual_fail_count += 1
            continue
        if annotation_key(row) in conflict_keys:
            conflict_count += 1
            continue
        if not row.get("raw_image_path") or not Path(row["raw_image_path"]).exists():
            missing_source_count += 1
            continue
        width = int(float(row["original_width"]))
        height = int(float(row["original_height"]))
        valid, _, _, crop_box = validate_bbox(
            {
                "x_min": float(row["x_min"]),
                "y_min": float(row["y_min"]),
                "x_max": float(row["x_max"]),
                "y_max": float(row["y_max"]),
            },
            width,
            height,
            margin_ratio=0.0,
        )
        if not valid or crop_box["crop_width"] <= 0 or crop_box["crop_height"] <= 0:
            non_positive_count += 1
            continue
        final_rows.append(row)

    fieldnames = list(valid_rows[0].keys()) if valid_rows else []
    write_csv_rows(final_path, fieldnames, final_rows)

    final_summary_rows: list[dict[str, Any]] = []
    final_counts = Counter((row["split"], row["class_id"], row["class_name"]) for row in final_rows)
    for split in SPLITS:
        for class_name, class_id in CLASS_ORDER:
            final_summary_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "final_annotation_count": final_counts[(split, str(class_id), class_name)],
                }
            )
    write_csv_rows(
        metadata_dir() / "final_annotations_summary_by_split_class.csv",
        ["split", "class_id", "class_name", "final_annotation_count"],
        final_summary_rows,
    )

    exclusion_counts = Counter(row["exclusion_reason"] for row in exclusions.values())
    with (reports_dir() / "final_annotation_preparation_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Final annotation preparation summary\n")
        f.write(f"original_valid_annotations: {len(valid_rows)}\n")
        f.write(f"excluded_duplicate_from_valid: {duplicate_count_in_valid}\n")
        f.write(f"excluded_manual_fail_from_valid: {manual_fail_count}\n")
        f.write(f"excluded_same_bbox_different_class_from_valid: {conflict_count}\n")
        f.write(f"excluded_missing_source_from_valid: {missing_source_count}\n")
        f.write(f"excluded_non_positive_bbox_from_valid: {non_positive_count}\n")
        f.write(f"final_annotations_for_crop: {len(final_rows)}\n")
        f.write(f"same_bbox_different_class_conflict_rows: {len(conflict_rows)}\n")
        f.write(f"same_bbox_different_class_conflict_annotations: {len(conflict_keys)}\n")
        f.write(f"excluded_annotations_total_rows: {len(exclusions)}\n")
        for reason, count in sorted(exclusion_counts.items()):
            f.write(f"excluded_{reason}: {count}\n")

    print("Final annotation preparation completed.")
    print(f"Manual review summary: {reports_dir() / 'manual_bbox_review_summary.txt'}")
    print(f"Conflict rows: {len(conflict_rows)}")
    print(f"Conflict annotations: {len(conflict_keys)}")
    print(f"Excluded annotation rows: {len(exclusions)}")
    print(f"Original valid annotations: {len(valid_rows)}")
    print(f"Final annotations for crop: {len(final_rows)}")
    print(f"Final annotation file: {final_path}")


if __name__ == "__main__":
    main()

