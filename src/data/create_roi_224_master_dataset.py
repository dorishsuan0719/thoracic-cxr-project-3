from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from common import CLASS_DIR_NAMES, CLASS_ORDER, PROJECT_ROOT, metadata_dir, reports_dir, write_csv_rows

MASTER_FIELDS = [
    "master_image_path",
    "legacy_output_224_path",
    "source_crop_path",
    "source_image_id",
    "annotation_index",
    "class_id",
    "class_name",
    "legacy_split",
    "manual_crop_review_status",
    "manual_224_review_status",
    "include_for_model",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create split-agnostic master 224 ROI dataset.")
    parser.add_argument("--model-input-manifest", type=Path, default=metadata_dir() / "model_input_224_manifest.csv")
    parser.add_argument("--final-model-csv", type=Path, default=metadata_dir() / "final_crops_for_model.csv")
    parser.add_argument("--manual-224-review", type=Path, default=metadata_dir() / "manual_224_review.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "bbox_crops_224_master")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp950", "mbcs"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error or UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode CSV")


def norm_path(value: str) -> str:
    return str(Path(value).resolve()).casefold() if value else ""


def bool_true(value: Any) -> bool:
    return str(value).strip().upper() == "TRUE"


def create_master_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows = read_csv_rows(args.model_input_manifest)
    final_rows = read_csv_rows(args.final_model_csv)
    manual_224_rows = read_csv_rows(args.manual_224_review)

    final_by_crop = {norm_path(row["crop_path"]): row for row in final_rows if bool_true(row.get("include_for_model"))}
    manual_224_by_output = {norm_path(row["output_224_path"]): row for row in manual_224_rows}
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    destination_counter: Counter[str] = Counter()

    for row in manifest_rows:
        legacy_path = Path(row["output_224_path"])
        final = final_by_crop.get(norm_path(row["source_crop_path"]))
        manual_224 = manual_224_by_output.get(norm_path(row["output_224_path"]))
        class_id = int(row["class_id"])
        master_path = args.output_dir / CLASS_DIR_NAMES[class_id] / legacy_path.name
        destination_counter[norm_path(str(master_path))] += 1
        rows.append(
            {
                "master_image_path": str(master_path),
                "legacy_output_224_path": str(legacy_path),
                "source_crop_path": row["source_crop_path"],
                "source_image_id": row["source_image_id"],
                "annotation_index": row["annotation_index"],
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "legacy_split": row["split"],
                "manual_crop_review_status": final.get("manual_review_status", "") if final else "",
                "manual_224_review_status": manual_224.get("review_status", "not_sampled") if manual_224 else "not_sampled",
                "include_for_model": "TRUE" if final else "FALSE",
            }
        )
        if not final:
            errors.append(error_row(row, str(master_path), "missing_from_final_crops_for_model", row["source_crop_path"]))
        if row.get("conversion_status") != "success":
            errors.append(error_row(row, str(master_path), "legacy_conversion_not_success", row.get("error_reason", "")))

    for row in rows:
        if destination_counter[norm_path(row["master_image_path"])] > 1:
            errors.append(
                {
                    "master_image_path": row["master_image_path"],
                    "legacy_output_224_path": row["legacy_output_224_path"],
                    "source_image_id": row["source_image_id"],
                    "annotation_index": row["annotation_index"],
                    "class_id": row["class_id"],
                    "error_type": "duplicate_master_image_path",
                    "error_detail": row["master_image_path"],
                }
            )
    return rows, errors


def error_row(row: dict[str, str], master_path: str, error_type: str, detail: str) -> dict[str, Any]:
    return {
        "master_image_path": master_path,
        "legacy_output_224_path": row.get("output_224_path", ""),
        "source_image_id": row.get("source_image_id", ""),
        "annotation_index": row.get("annotation_index", ""),
        "class_id": row.get("class_id", ""),
        "error_type": error_type,
        "error_detail": detail,
    }


def copy_master_images(rows: list[dict[str, Any]], overwrite: bool) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for row in rows:
        src = Path(row["legacy_output_224_path"])
        dst = Path(row["master_image_path"])
        try:
            if not src.exists():
                raise FileNotFoundError(f"legacy 224 image missing: {src}")
            if dst.exists() and not overwrite:
                raise FileExistsError(f"master image exists and --overwrite was not set: {dst}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except Exception as exc:  # noqa: BLE001
            errors.append({**row, "error_type": "copy_error", "error_detail": str(exc)})
    return errors


def audit_master(rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    missing = unreadable = wrong_size = empty = 0
    duplicate_source_annotation = 0
    source_annotation_counts = Counter((row["source_image_id"], row["annotation_index"]) for row in rows)
    duplicate_source_annotation = sum(1 for count in source_annotation_counts.values() if count > 1)

    for row in rows:
        path = Path(row["master_image_path"])
        if not path.exists():
            missing += 1
            errors.append({**row, "error_type": "missing_master_image", "error_detail": str(path)})
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                gray = image.convert("L")
                if gray.size != (224, 224):
                    wrong_size += 1
                    errors.append({**row, "error_type": "wrong_size", "error_detail": f"{gray.size[0]}x{gray.size[1]}"})
                if gray.getextrema() == (0, 0):
                    empty += 1
                    errors.append({**row, "error_type": "empty_all_zero", "error_detail": str(path)})
        except Exception as exc:  # noqa: BLE001
            unreadable += 1
            errors.append({**row, "error_type": "unreadable_master_image", "error_detail": str(exc)})

    master_paths = [norm_path(row["master_image_path"]) for row in rows]
    metrics = {
        "master_roi_count": len(rows),
        "master_unique_master_image_paths": len(set(master_paths)),
        "missing_master_images": missing,
        "unreadable_master_images": unreadable,
        "wrong_size_master_images": wrong_size,
        "empty_master_images": empty,
        "duplicate_master_image_paths": len(master_paths) - len(set(master_paths)),
        "duplicate_source_image_id_annotation_index_pairs": duplicate_source_annotation,
    }
    return metrics


def write_class_counts(rows: list[dict[str, Any]]) -> None:
    source_sets: dict[int, set[str]] = defaultdict(set)
    roi_counts: Counter[int] = Counter()
    for row in rows:
        class_id = int(row["class_id"])
        source_sets[class_id].add(row["source_image_id"])
        roi_counts[class_id] += 1

    out_rows = []
    for class_name, class_id in CLASS_ORDER:
        out_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "unique_source_image_count": len(source_sets[class_id]),
                "roi_count": roi_counts[class_id],
            }
        )
    write_csv_rows(
        reports_dir() / "roi_master_class_counts.csv",
        ["class_id", "class_name", "unique_source_image_count", "roi_count"],
        out_rows,
    )


def main() -> int:
    args = parse_args()
    rows, errors = create_master_rows(args)
    if not errors:
        errors.extend(copy_master_images(rows, overwrite=args.overwrite))
    metrics = audit_master(rows, errors)
    reports_dir().mkdir(parents=True, exist_ok=True)

    write_csv_rows(metadata_dir() / "roi_224_master_manifest.csv", MASTER_FIELDS, rows)
    write_class_counts(rows)
    write_csv_rows(
        reports_dir() / "roi_224_master_errors.csv",
        [
            "master_image_path",
            "legacy_output_224_path",
            "source_image_id",
            "annotation_index",
            "class_id",
            "error_type",
            "error_detail",
        ],
        errors,
    )
    with (reports_dir() / "roi_224_master_audit.txt").open("w", encoding="utf-8") as f:
        f.write("ROI 224 master audit\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write(f"error_rows: {len(errors)}\n")

    print("ROI 224 master dataset creation completed.")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"error_rows: {len(errors)}")
    print(f"Master manifest: {metadata_dir() / 'roi_224_master_manifest.csv'}")
    return 0 if not errors and metrics["master_roi_count"] == 2343 else 1


if __name__ == "__main__":
    sys.exit(main())
