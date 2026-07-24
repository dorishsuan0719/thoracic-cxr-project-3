from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from common import CLASS_ORDER, metadata_dir, reports_dir, write_csv_rows

AUDIT_FIELDS = [
    "image_id",
    "image_path",
    "unique_class_count",
    "class_ids",
    "class_names",
    "original_bbox_count",
    "valid_bbox_count",
    "duplicate_bbox_count",
    "invalid_bbox_count",
    "image_exists",
    "is_single_label",
    "final_status",
    "exclusion_reason",
]
MASTER_FIELDS = [
    "image_id",
    "image_path",
    "class_id",
    "class_name",
    "annotation_index",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
]
MULTILABEL_FIELDS = ["image_id", "image_path", "class_ids", "class_names", "bbox_count", "exclusion_reason"]
INVALID_FIELDS = ["image_id", "image_path", "original_bbox_count", "valid_bbox_count", "exclusion_reason"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def bool_csv(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def class_name_for(class_id: str) -> str:
    mapping = {str(cid): name for name, cid in CLASS_ORDER}
    return mapping.get(str(class_id), "")


def image_readable(path: str) -> bool:
    if not path or not Path(path).exists():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def grouped_annotations(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["image_id"]].append(row)
    return grouped


def load_image_paths() -> dict[str, str]:
    path = metadata_dir() / "image_manifest.csv"
    rows = read_csv_rows(path)
    return {row["source_image_id"]: row.get("raw_image_path", "") for row in rows}


def main() -> None:
    audited_path = metadata_dir() / "audited_annotations.csv"
    rows = read_csv_rows(audited_path)
    image_paths = load_image_paths()
    by_image = grouped_annotations(rows)

    audit_rows: list[dict[str, Any]] = []
    master_rows: list[dict[str, Any]] = []
    multilabel_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    total_original_bbox = 0
    total_valid_bbox = 0
    total_duplicate_bbox = 0
    total_invalid_bbox = 0

    for image_id in sorted(by_image):
        image_rows = by_image[image_id]
        image_path = image_paths.get(image_id) or image_rows[0].get("raw_image_path", "")
        exists = image_readable(image_path)
        original_bbox_count = len(image_rows)
        duplicate_rows = [row for row in image_rows if bool_csv(row.get("is_duplicate"))]
        invalid_bbox_rows = [
            row for row in image_rows
            if not bool_csv(row.get("is_duplicate")) and not bool_csv(row.get("is_valid"))
        ]
        valid_rows = [
            row for row in image_rows
            if bool_csv(row.get("is_valid")) and not bool_csv(row.get("is_duplicate"))
        ]

        class_ids = sorted({row["class_id"] for row in valid_rows}, key=lambda value: int(value))
        class_names = [class_name_for(class_id) for class_id in class_ids]
        unique_class_count = len(class_ids)
        is_single_label = unique_class_count == 1

        if not exists:
            final_status = "excluded"
            exclusion_reason = "missing_or_unreadable_image"
        elif len(valid_rows) == 0:
            final_status = "excluded"
            exclusion_reason = "no_valid_bbox"
        elif not is_single_label:
            final_status = "excluded"
            exclusion_reason = "multi_label_image"
        else:
            final_status = "included"
            exclusion_reason = ""

        audit_rows.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "unique_class_count": unique_class_count,
                "class_ids": ";".join(class_ids),
                "class_names": ";".join(class_names),
                "original_bbox_count": original_bbox_count,
                "valid_bbox_count": len(valid_rows),
                "duplicate_bbox_count": len(duplicate_rows),
                "invalid_bbox_count": len(invalid_bbox_rows),
                "image_exists": exists,
                "is_single_label": is_single_label,
                "final_status": final_status,
                "exclusion_reason": exclusion_reason,
            }
        )

        total_original_bbox += original_bbox_count
        total_valid_bbox += len(valid_rows)
        total_duplicate_bbox += len(duplicate_rows)
        total_invalid_bbox += len(invalid_bbox_rows)

        if final_status == "included":
            for row in valid_rows:
                master_rows.append(
                    {
                        "image_id": image_id,
                        "image_path": image_path,
                        "class_id": row["class_id"],
                        "class_name": row["class_name"],
                        "annotation_index": row["annotation_index"],
                        "bbox_xmin": row["x_min"],
                        "bbox_ymin": row["y_min"],
                        "bbox_xmax": row["x_max"],
                        "bbox_ymax": row["y_max"],
                    }
                )
        elif exclusion_reason == "multi_label_image":
            multilabel_rows.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "class_ids": ";".join(class_ids),
                    "class_names": ";".join(class_names),
                    "bbox_count": len(valid_rows),
                    "exclusion_reason": exclusion_reason,
                }
            )
        else:
            invalid_rows.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "original_bbox_count": original_bbox_count,
                    "valid_bbox_count": len(valid_rows),
                    "exclusion_reason": exclusion_reason,
                }
            )

    write_csv_rows(metadata_dir() / "full_image_single_label_audit.csv", AUDIT_FIELDS, audit_rows)
    write_csv_rows(metadata_dir() / "full_image_single_label_master.csv", MASTER_FIELDS, master_rows)
    write_csv_rows(metadata_dir() / "excluded_multilabel_images.csv", MULTILABEL_FIELDS, multilabel_rows)
    write_csv_rows(metadata_dir() / "excluded_invalid_images.csv", INVALID_FIELDS, invalid_rows)
    write_summary(
        audit_rows=audit_rows,
        master_rows=master_rows,
        multilabel_rows=multilabel_rows,
        invalid_rows=invalid_rows,
        totals={
            "original_bbox": total_original_bbox,
            "valid_bbox": total_valid_bbox,
            "duplicate_bbox": total_duplicate_bbox,
            "invalid_bbox": total_invalid_bbox,
        },
    )

    print("Full-image strict single-label audit completed.")
    print(f"Unique full images: {len(audit_rows)}")
    print(f"Included strict single-label images: {sum(1 for row in audit_rows if row['final_status'] == 'included')}")
    print(f"Master BBox rows: {len(master_rows)}")
    print(f"Excluded multi-label images: {len(multilabel_rows)}")
    print(f"Excluded invalid images: {len(invalid_rows)}")
    print(f"Duplicate annotation rows: {total_duplicate_bbox}")
    print(f"Invalid BBox rows: {total_invalid_bbox}")


def write_summary(
    audit_rows: list[dict[str, Any]],
    master_rows: list[dict[str, Any]],
    multilabel_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    totals: dict[str, int],
) -> None:
    reports_dir().mkdir(parents=True, exist_ok=True)
    included_images = [row for row in audit_rows if row["final_status"] == "included"]
    class_image_sets: dict[str, set[str]] = defaultdict(set)
    class_bbox_counts: Counter[str] = Counter()
    for row in master_rows:
        class_image_sets[str(row["class_id"])].add(str(row["image_id"]))
        class_bbox_counts[str(row["class_id"])] += 1

    summary_path = reports_dir() / "full_image_single_label_audit_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Full Image Strict Single-Label Audit Summary\n\n")
        f.write("This audit is for the YOLO full-image object-detection main line. It does not train models, crop ROIs, resize images, normalize images, or create train/val/test splits.\n\n")
        f.write("## Totals\n\n")
        f.write(f"- Unique full images with annotations: {len(audit_rows)}\n")
        f.write(f"- Strict single-label included full images: {len(included_images)}\n")
        f.write(f"- Excluded multi-label images: {len(multilabel_rows)}\n")
        f.write(f"- Excluded invalid/no-valid-BBox images: {len(invalid_rows)}\n")
        f.write(f"- Original BBox annotation rows: {totals['original_bbox']}\n")
        f.write(f"- Valid non-duplicate BBox rows: {totals['valid_bbox']}\n")
        f.write(f"- Duplicate annotation rows: {totals['duplicate_bbox']}\n")
        f.write(f"- Invalid BBox rows: {totals['invalid_bbox']}\n")
        f.write(f"- Master BBox rows: {len(master_rows)}\n\n")

        f.write("## Class Distribution In Strict Single-Label Master\n\n")
        f.write("| class_id | class_name | unique_full_image_count | bbox_count | target_350_delta |\n")
        f.write("|---:|---|---:|---:|---:|\n")
        for class_name, class_id in CLASS_ORDER:
            image_count = len(class_image_sets[str(class_id)])
            bbox_count = class_bbox_counts[str(class_id)]
            f.write(f"| {class_id} | {class_name} | {image_count} | {bbox_count} | {image_count - 350} |\n")

        status_counts = Counter(row["exclusion_reason"] or "included" for row in audit_rows)
        f.write("\n## Final Status Counts\n\n")
        for status, count in sorted(status_counts.items()):
            f.write(f"- {status}: {count}\n")

        f.write("\n## Strict Rules Applied\n\n")
        f.write("- Include only images with at least one valid non-duplicate BBox.\n")
        f.write("- Include only images whose valid non-duplicate BBoxes all share exactly one class_id.\n")
        f.write("- Exclude image_id with valid BBoxes from more than one class as multi_label_image.\n")
        f.write("- Exclude missing/unreadable images and images with no valid BBox.\n")
        f.write("- Exact duplicate annotations are counted and removed from the master rows.\n")


if __name__ == "__main__":
    main()
