from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from common import CLASS_ORDER, SPLITS, crop_manifest_path, metadata_dir, reports_dir, write_csv_rows


def read_manifest() -> list[dict[str, Any]]:
    with crop_manifest_path().open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_final_annotations() -> list[dict[str, Any]]:
    with (metadata_dir() / "final_annotations_for_crop.csv").open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = read_manifest()
    final_rows = read_final_annotations()
    reports_dir().mkdir(parents=True, exist_ok=True)

    checked_rows: list[dict[str, Any]] = []
    stats = defaultdict(
        lambda: {
            "source_images": set(),
            "annotation_count": 0,
            "crop_count": 0,
            "width_sum": 0,
            "height_sum": 0,
            "min_width": None,
            "min_height": None,
            "max_width": 0,
            "max_height": 0,
            "missing_crop_count": 0,
            "unreadable_crop_count": 0,
            "zero_size_crop_count": 0,
        }
    )
    split_sources = defaultdict(set)
    crop_path_to_annotations = defaultdict(list)

    for row in rows:
        split = row["split"]
        class_id = int(row["class_id"])
        key = (split, class_id)
        source_image_id = row["source_image_id"]
        split_sources[split].add(source_image_id)
        stats[key]["source_images"].add(source_image_id)
        stats[key]["annotation_count"] += 1

        crop_path = Path(row["crop_path"])
        crop_path_to_annotations[str(crop_path)].append(
            {
                "crop_id": row["crop_id"],
                "split": split,
                "source_image_id": source_image_id,
                "class_id": class_id,
                "annotation_index": row["annotation_index"],
            }
        )
        exists = crop_path.exists()
        readable = False
        width = 0
        height = 0
        reason = ""
        if not exists:
            reason = "missing_crop"
            stats[key]["missing_crop_count"] += 1
        else:
            try:
                with Image.open(crop_path) as image:
                    image.verify()
                with Image.open(crop_path) as image:
                    width, height = image.size
                readable = True
                if width <= 0 or height <= 0:
                    reason = "zero_size_crop"
                    stats[key]["zero_size_crop_count"] += 1
                else:
                    stats[key]["crop_count"] += 1
                    stats[key]["width_sum"] += width
                    stats[key]["height_sum"] += height
                    stats[key]["min_width"] = width if stats[key]["min_width"] is None else min(stats[key]["min_width"], width)
                    stats[key]["min_height"] = height if stats[key]["min_height"] is None else min(stats[key]["min_height"], height)
                    stats[key]["max_width"] = max(stats[key]["max_width"], width)
                    stats[key]["max_height"] = max(stats[key]["max_height"], height)
            except Exception as exc:  # noqa: BLE001
                reason = f"unreadable_crop:{exc}"
                stats[key]["unreadable_crop_count"] += 1

        checked_rows.append(
            {
                "crop_id": row["crop_id"],
                "split": split,
                "source_image_id": source_image_id,
                "crop_path": str(crop_path),
                "class_id": class_id,
                "class_name": row["class_name"],
                "exists": exists,
                "readable": readable,
                "actual_crop_width": width,
                "actual_crop_height": height,
                "error_reason": reason,
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for class_name, class_id in CLASS_ORDER:
            value = stats[(split, class_id)]
            crop_count = value["crop_count"]
            summary_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "source_image_count": len(value["source_images"]),
                    "annotation_count": value["annotation_count"],
                    "crop_count": crop_count,
                    "min_crop_width": value["min_width"] or 0,
                    "min_crop_height": value["min_height"] or 0,
                    "max_crop_width": value["max_width"],
                    "max_crop_height": value["max_height"],
                    "mean_crop_width": round(value["width_sum"] / crop_count, 2) if crop_count else 0,
                    "mean_crop_height": round(value["height_sum"] / crop_count, 2) if crop_count else 0,
                    "missing_crop_count": value["missing_crop_count"],
                    "unreadable_crop_count": value["unreadable_crop_count"],
                    "zero_size_crop_count": value["zero_size_crop_count"],
                }
            )

    leakage_rows: list[dict[str, Any]] = []
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sorted(split_sources[left] & split_sources[right])
        leakage_rows.append({"split_pair": f"{left}_vs_{right}", "leakage_count": len(overlap), "source_image_ids": ";".join(overlap[:100])})

    duplicate_path_rows: list[dict[str, Any]] = []
    same_path_different_annotation_count = 0
    for crop_path, annotations in crop_path_to_annotations.items():
        if len(annotations) <= 1:
            continue
        unique_annotation_keys = {
            (item["split"], item["source_image_id"], str(item["class_id"]), str(item["annotation_index"]))
            for item in annotations
        }
        if len(unique_annotation_keys) > 1:
            same_path_different_annotation_count += 1
        duplicate_path_rows.append(
            {
                "crop_path": crop_path,
                "manifest_row_count": len(annotations),
                "unique_annotation_count": len(unique_annotation_keys),
                "crop_ids": ";".join(item["crop_id"] for item in annotations),
            }
        )

    valid_manifest_rows = [row for row in rows if str(row.get("is_valid", "")).strip().lower() == "true"]
    existing_valid_crop_paths = {row["crop_path"] for row in checked_rows if row["exists"] is True and row["readable"] is True and int(row["actual_crop_width"]) > 0 and int(row["actual_crop_height"]) > 0}
    manifest_actual_consistency = len(existing_valid_crop_paths) == len(valid_manifest_rows) and len(duplicate_path_rows) == 0

    write_csv_rows(metadata_dir() / "cropped_dataset_checked_files.csv", ["crop_id", "split", "source_image_id", "crop_path", "class_id", "class_name", "exists", "readable", "actual_crop_width", "actual_crop_height", "error_reason"], checked_rows)
    write_csv_rows(
        metadata_dir() / "cropped_dataset_summary_by_split_class.csv",
        [
            "split",
            "class_id",
            "class_name",
            "source_image_count",
            "annotation_count",
            "crop_count",
            "min_crop_width",
            "min_crop_height",
            "max_crop_width",
            "max_crop_height",
            "mean_crop_width",
            "mean_crop_height",
            "missing_crop_count",
            "unreadable_crop_count",
            "zero_size_crop_count",
        ],
        summary_rows,
    )
    write_csv_rows(metadata_dir() / "split_leakage.csv", ["split_pair", "leakage_count", "source_image_ids"], leakage_rows)
    write_csv_rows(metadata_dir() / "duplicate_crop_paths.csv", ["crop_path", "manifest_row_count", "unique_annotation_count", "crop_ids"], duplicate_path_rows)

    with (reports_dir() / "final_crop_dataset_audit.txt").open("w", encoding="utf-8") as f:
        f.write("Final cropped dataset audit\n")
        f.write(f"final_annotations_for_crop_rows: {len(final_rows)}\n")
        f.write(f"crop_manifest_rows: {len(rows)}\n")
        f.write(f"valid_manifest_rows: {len(valid_manifest_rows)}\n")
        f.write(f"checked_crops: {len(checked_rows)}\n")
        f.write(f"missing_crops: {sum(row['missing_crop_count'] for row in summary_rows)}\n")
        f.write(f"unreadable_crops: {sum(row['unreadable_crop_count'] for row in summary_rows)}\n")
        f.write(f"zero_size_crops: {sum(row['zero_size_crop_count'] for row in summary_rows)}\n")
        f.write(f"crop_count_equals_final_annotations: {len(valid_manifest_rows) == len(final_rows)}\n")
        f.write(f"manifest_actual_files_consistent: {manifest_actual_consistency}\n")
        f.write(f"duplicate_crop_path_count: {len(duplicate_path_rows)}\n")
        f.write(f"same_crop_path_different_annotation_count: {same_path_different_annotation_count}\n")
        for row in leakage_rows:
            f.write(f"{row['split_pair']}_leakage: {row['leakage_count']}\n")

    print("Cropped dataset audit completed.")
    print(f"Checked crops: {len(checked_rows)}")
    print(f"Missing crops: {sum(row['missing_crop_count'] for row in summary_rows)}")
    print(f"Unreadable crops: {sum(row['unreadable_crop_count'] for row in summary_rows)}")
    print(f"Zero-size crops: {sum(row['zero_size_crop_count'] for row in summary_rows)}")
    print(f"Final annotations: {len(final_rows)}")
    print(f"Valid manifest crops: {len(valid_manifest_rows)}")
    print(f"Crop count equals final annotations: {len(valid_manifest_rows) == len(final_rows)}")
    print(f"Manifest/actual file consistency: {manifest_actual_consistency}")
    print(f"Duplicate crop paths: {len(duplicate_path_rows)}")
    print(f"Same crop path with different annotation: {same_path_different_annotation_count}")
    print("Leakage:")
    for row in leakage_rows:
        print(f"  {row['split_pair']}: {row['leakage_count']}")
    print(f"Final report: {reports_dir() / 'final_crop_dataset_audit.txt'}")


if __name__ == "__main__":
    main()
