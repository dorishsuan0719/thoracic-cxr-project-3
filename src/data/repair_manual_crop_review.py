from __future__ import annotations

import csv
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import CLASS_ORDER, SPLITS, crop_manifest_path, metadata_dir, write_csv_rows

RANDOM_SEED = 42
SAMPLES_PER_SPLIT_CLASS = 10
REVIEW_FIELDS = ["review_status", "technical_issue", "review_note"]
OUTPUT_FIELDS = [
    "crop_path",
    "source_image_id",
    "split",
    "class_id",
    "class_name",
    "annotation_index",
    "review_status",
    "technical_issue",
    "review_note",
]
BBOX_PATTERN = re.compile(r"_bbox(\d+)\.png$", re.IGNORECASE)


def read_csv_with_fallback(path: Path) -> tuple[list[dict[str, str]], list[str], str]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp950", "mbcs"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                return list(reader), list(reader.fieldnames or []), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error or UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode CSV")


def normalize_path(value: str) -> str:
    if not value:
        return ""
    return str(Path(value).resolve()).casefold()


def bbox_index_from_path(crop_path: str) -> int | None:
    match = BBOX_PATTERN.search(Path(crop_path).name)
    return int(match.group(1)) if match else None


def backup_manual_file(manual_path: Path, backup_path: Path) -> None:
    if backup_path.exists():
        raise SystemExit(f"Backup already exists; refusing to overwrite: {backup_path}")
    shutil.copy2(manual_path, backup_path)


def build_clean_template(manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    valid_rows = [row for row in manifest_rows if str(row.get("is_valid", "")).strip().lower() == "true"]
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in valid_rows:
        grouped[(row["split"], int(row["class_id"]))].append(row)

    rng = random.Random(RANDOM_SEED)
    template_rows: list[dict[str, str]] = []
    for split in SPLITS:
        for class_name, class_id in CLASS_ORDER:
            candidates = sorted(
                grouped[(split, class_id)],
                key=lambda row: (row["source_image_id"], int(row["annotation_index"])),
            )
            selected = candidates[:] if len(candidates) <= SAMPLES_PER_SPLIT_CLASS else rng.sample(candidates, SAMPLES_PER_SPLIT_CLASS)
            selected = sorted(selected, key=lambda row: (row["source_image_id"], int(row["annotation_index"])))
            for row in selected:
                template_rows.append(
                    {
                        "crop_path": row["crop_path"],
                        "source_image_id": row["source_image_id"],
                        "split": row["split"],
                        "class_id": row["class_id"],
                        "class_name": row["class_name"],
                        "annotation_index": row["annotation_index"],
                        "review_status": "",
                        "technical_issue": "",
                        "review_note": "",
                    }
                )
    return template_rows


def review_tuple(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(str(row.get(field, "") or "").strip() for field in REVIEW_FIELDS)


def repair_reviews(
    template_rows: list[dict[str, str]],
    backup_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    template_keys = {normalize_path(row["crop_path"]) for row in template_rows}
    backup_by_path: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in backup_rows:
        backup_by_path[normalize_path(row.get("crop_path", ""))].append(row)

    merge_conflicts: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    missing_reviews: list[dict[str, Any]] = []
    repaired_rows: list[dict[str, str]] = []

    for key, rows in backup_by_path.items():
        if key not in template_keys:
            for row in rows:
                unmatched_rows.append({**row, "unmatched_reason": "crop_path_not_in_clean_template"})

    for template in template_rows:
        key = normalize_path(template["crop_path"])
        matching_rows = backup_by_path.get(key, [])
        repaired = template.copy()
        if not matching_rows:
            missing_reviews.append({**template, "missing_reason": "no_manual_review_for_template_crop"})
            repaired_rows.append(repaired)
            continue

        distinct_reviews = {review_tuple(row) for row in matching_rows}
        if len(distinct_reviews) == 1:
            status, issue, note = next(iter(distinct_reviews))
            repaired["review_status"] = status
            repaired["technical_issue"] = issue
            repaired["review_note"] = note
        else:
            for row in matching_rows:
                merge_conflicts.append(
                    {
                        "crop_path": template["crop_path"],
                        "source_image_id": template["source_image_id"],
                        "split": template["split"],
                        "class_id": template["class_id"],
                        "class_name": template["class_name"],
                        "annotation_index_from_manifest": template["annotation_index"],
                        "backup_annotation_index": row.get("annotation_index", ""),
                        "backup_review_status": row.get("review_status", ""),
                        "backup_technical_issue": row.get("technical_issue", ""),
                        "backup_review_note": row.get("review_note", ""),
                        "conflict_reason": "same_crop_path_different_manual_review_values",
                    }
                )
        repaired_rows.append(repaired)
    return repaired_rows, merge_conflicts, unmatched_rows, missing_reviews


def validate_repaired(repaired_rows: list[dict[str, str]], manifest_by_path: dict[str, dict[str, str]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    crop_path_counts = Counter(normalize_path(row["crop_path"]) for row in repaired_rows)
    duplicate_crop_path_rows = sum(count for count in crop_path_counts.values() if count > 1)
    annotation_mismatch = 0
    filename_mismatch = 0
    manifest_missing = 0

    for row in repaired_rows:
        key = normalize_path(row["crop_path"])
        manifest = manifest_by_path.get(key)
        if manifest is None:
            manifest_missing += 1
            errors.append({**row, "error_type": "crop_path_missing_from_manifest", "error_detail": row["crop_path"]})
            continue
        manifest_fields = ["source_image_id", "split", "class_id", "class_name", "annotation_index"]
        mismatched = [field for field in manifest_fields if str(row[field]) != str(manifest[field])]
        if mismatched:
            annotation_mismatch += 1
            errors.append({**row, "error_type": "manifest_field_mismatch", "error_detail": ";".join(mismatched)})
        bbox_index = bbox_index_from_path(row["crop_path"])
        if bbox_index is None or int(row["annotation_index"]) != bbox_index:
            filename_mismatch += 1
            errors.append(
                {
                    **row,
                    "error_type": "annotation_index_filename_mismatch",
                    "error_detail": f"annotation_index={row['annotation_index']}, filename_bbox={bbox_index}",
                }
            )

    metrics = {
        "repaired_rows": len(repaired_rows),
        "repaired_unique_crop_paths": len(crop_path_counts),
        "duplicate_crop_path_rows": duplicate_crop_path_rows,
        "manifest_annotation_mismatch": annotation_mismatch,
        "filename_annotation_mismatch": filename_mismatch,
        "crop_path_missing_from_manifest": manifest_missing,
    }
    return metrics, errors


def write_summary(path: Path, values: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for key, value in values.items():
            f.write(f"{key}: {value}\n")


def main() -> int:
    manual_path = metadata_dir() / "manual_crop_review.csv"
    backup_path = metadata_dir() / "manual_crop_review_backup_before_repair.csv"
    clean_template_path = metadata_dir() / "manual_crop_review_clean_template.csv"
    repaired_path = metadata_dir() / "manual_crop_review_repaired.csv"
    merge_conflicts_path = metadata_dir() / "manual_crop_review_merge_conflicts.csv"
    unmatched_path = metadata_dir() / "manual_crop_review_unmatched_rows.csv"
    missing_reviews_path = metadata_dir() / "manual_crop_review_missing_reviews.csv"
    validation_errors_path = metadata_dir() / "manual_crop_review_repair_validation_errors.csv"
    summary_path = metadata_dir() / "manual_crop_review_repair_summary.txt"

    original_rows, _, original_encoding = read_csv_with_fallback(manual_path)
    original_unique_crop_paths = len({normalize_path(row.get("crop_path", "")) for row in original_rows})
    original_duplicate_removed = len(original_rows) - original_unique_crop_paths

    backup_manual_file(manual_path, backup_path)

    manifest_rows, _, _ = read_csv_with_fallback(crop_manifest_path())
    manifest_by_path = {normalize_path(row["crop_path"]): row for row in manifest_rows}
    clean_template = build_clean_template(manifest_rows)
    write_csv_rows(clean_template_path, OUTPUT_FIELDS, clean_template)

    backup_rows, _, backup_encoding = read_csv_with_fallback(backup_path)
    repaired_rows, merge_conflicts, unmatched_rows, missing_reviews = repair_reviews(clean_template, backup_rows)
    write_csv_rows(repaired_path, OUTPUT_FIELDS, repaired_rows)

    write_csv_rows(
        merge_conflicts_path,
        [
            "crop_path",
            "source_image_id",
            "split",
            "class_id",
            "class_name",
            "annotation_index_from_manifest",
            "backup_annotation_index",
            "backup_review_status",
            "backup_technical_issue",
            "backup_review_note",
            "conflict_reason",
        ],
        merge_conflicts,
    )
    unmatched_fields = list(backup_rows[0].keys()) + ["unmatched_reason"] if backup_rows else OUTPUT_FIELDS + ["unmatched_reason"]
    write_csv_rows(unmatched_path, unmatched_fields, unmatched_rows)
    write_csv_rows(missing_reviews_path, OUTPUT_FIELDS + ["missing_reason"], missing_reviews)

    repair_metrics, validation_errors = validate_repaired(repaired_rows, manifest_by_path)
    write_csv_rows(validation_errors_path, OUTPUT_FIELDS + ["error_type", "error_detail"], validation_errors)

    annotation_index_corrections = 0
    repaired_by_path = {normalize_path(row["crop_path"]): row for row in repaired_rows}
    for row in original_rows:
        repaired = repaired_by_path.get(normalize_path(row.get("crop_path", "")))
        if repaired and str(row.get("annotation_index", "")) != str(repaired["annotation_index"]):
            annotation_index_corrections += 1

    blocking = {
        "merge_conflicts": len(merge_conflicts),
        "missing_reviews": len(missing_reviews),
        "annotation_index_mismatch": repair_metrics["manifest_annotation_mismatch"] + repair_metrics["filename_annotation_mismatch"],
        "duplicate_crop_path": repair_metrics["duplicate_crop_path_rows"],
        "row_count_not_150": 0 if repair_metrics["repaired_rows"] == 150 else 1,
    }
    can_apply = all(value == 0 for value in blocking.values())

    summary = {
        "original_manual_csv_rows": len(original_rows),
        "original_unique_crop_paths": original_unique_crop_paths,
        "original_csv_encoding_read_as": original_encoding,
        "backup_csv_encoding_read_as": backup_encoding,
        "duplicate_rows_removed_by_repair": original_duplicate_removed,
        "annotation_index_corrections_from_original_rows": annotation_index_corrections,
        "merge_conflict_rows": len(merge_conflicts),
        "merge_conflict_crop_paths": len({normalize_path(row["crop_path"]) for row in merge_conflicts}),
        "missing_review_rows": len(missing_reviews),
        "unmatched_rows": len(unmatched_rows),
        "clean_template_rows": len(clean_template),
        "repaired_rows": repair_metrics["repaired_rows"],
        "repaired_unique_crop_paths": repair_metrics["repaired_unique_crop_paths"],
        "repaired_duplicate_crop_path_rows": repair_metrics["duplicate_crop_path_rows"],
        "repaired_manifest_annotation_mismatch": repair_metrics["manifest_annotation_mismatch"],
        "repaired_filename_annotation_mismatch": repair_metrics["filename_annotation_mismatch"],
        "can_apply_repaired_to_manual_crop_review": can_apply,
    }
    write_summary(summary_path, summary)

    print("Manual crop review repair completed.")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Backup: {backup_path}")
    print(f"Clean template: {clean_template_path}")
    print(f"Repaired: {repaired_path}")
    print(f"Merge conflicts: {merge_conflicts_path}")
    print(f"Unmatched rows: {unmatched_path}")
    print(f"Missing reviews: {missing_reviews_path}")
    print(f"Validation errors: {validation_errors_path}")

    if not can_apply:
        print("Stopped before overwriting manual_crop_review.csv and before creating final_crops_for_model.csv.")
        return 1

    shutil.copy2(repaired_path, manual_path)
    print(f"Applied repaired file to: {manual_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

