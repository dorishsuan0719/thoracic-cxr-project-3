from __future__ import annotations

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from common import CLASS_ORDER, SPLITS, metadata_dir, reports_dir, write_csv_rows

EXPECTED_COUNT = 2343
IMAGE_SIZE = 224
RANDOM_SEED = 42


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm_path(value: str) -> str:
    return str(Path(value).resolve()).casefold() if value else ""


def bool_true(value: Any) -> bool:
    return str(value).strip().upper() == "TRUE"


def audit_outputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    final_rows = [row for row in read_csv_rows(metadata_dir() / "final_crops_for_model.csv") if bool_true(row.get("include_for_model"))]
    manifest_rows = read_csv_rows(metadata_dir() / "model_input_224_manifest.csv")
    errors: list[dict[str, Any]] = []
    checked_rows: list[dict[str, Any]] = []

    manifest_by_output = {}
    duplicate_output_groups = 0
    for output_path, rows in defaultdict(list, {}).items():
        pass
    grouped_outputs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        grouped_outputs[norm_path(row["output_224_path"])].append(row)
    for key, rows in grouped_outputs.items():
        if len(rows) > 1:
            duplicate_output_groups += 1
            for row in rows:
                errors.append(error_row(row, "duplicate_output_path", f"count={len(rows)}"))
        else:
            manifest_by_output[key] = rows[0]

    source_missing = 0
    unreadable = 0
    wrong_size = 0
    empty_image = 0
    missing_output = 0
    status_failed = 0

    for row in manifest_rows:
        output_path = Path(row["output_224_path"])
        source_path = Path(row["source_crop_path"])
        checked = {
            "output_224_path": str(output_path),
            "source_crop_path": str(source_path),
            "source_image_id": row["source_image_id"],
            "split": row["split"],
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "annotation_index": row["annotation_index"],
            "exists": False,
            "readable": False,
            "width": 0,
            "height": 0,
            "is_empty_all_zero": False,
            "error_reason": "",
        }
        if not source_path.exists():
            source_missing += 1
            errors.append(error_row(row, "source_crop_missing", str(source_path)))
        if row.get("conversion_status") != "success":
            status_failed += 1
            errors.append(error_row(row, "conversion_status_failed", row.get("error_reason", "")))
        if not output_path.exists():
            missing_output += 1
            checked["error_reason"] = "missing_output"
            errors.append(error_row(row, "missing_output", str(output_path)))
            checked_rows.append(checked)
            continue

        checked["exists"] = True
        try:
            with Image.open(output_path) as image:
                image.verify()
            with Image.open(output_path) as image:
                gray = image.convert("L")
                width, height = gray.size
                checked["width"] = width
                checked["height"] = height
                checked["readable"] = True
                extrema = gray.getextrema()
                if extrema == (0, 0):
                    empty_image += 1
                    checked["is_empty_all_zero"] = True
                    errors.append(error_row(row, "empty_image_all_zero", str(output_path)))
                if (width, height) != (IMAGE_SIZE, IMAGE_SIZE):
                    wrong_size += 1
                    errors.append(error_row(row, "wrong_image_size", f"{width}x{height}"))
        except Exception as exc:  # noqa: BLE001
            unreadable += 1
            checked["error_reason"] = f"unreadable:{exc}"
            errors.append(error_row(row, "unreadable_output", str(exc)))
        checked_rows.append(checked)

    final_output_keys = {
        norm_path(expected_output_from_final(row))
        for row in final_rows
    }
    manifest_output_keys = {norm_path(row["output_224_path"]) for row in manifest_rows}
    manifest_missing_expected = sorted(final_output_keys - manifest_output_keys)
    manifest_unexpected = sorted(manifest_output_keys - final_output_keys)
    for path in manifest_missing_expected:
        errors.append(blank_error("manifest_missing_expected_output", path))
    for path in manifest_unexpected:
        errors.append(blank_error("manifest_unexpected_output", path))

    split_sources: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        if row.get("conversion_status") == "success":
            split_sources[row["split"]].add(row["source_image_id"])
    leakage_rows = []
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sorted(split_sources[left] & split_sources[right])
        leakage_rows.append({"split_pair": f"{left}_vs_{right}", "leakage_count": len(overlap), "source_image_ids": ";".join(overlap[:100])})

    summary_rows = summarize_by_split_class(manifest_rows)
    metrics = {
        "expected_outputs": EXPECTED_COUNT,
        "final_include_true_rows": len(final_rows),
        "manifest_rows": len(manifest_rows),
        "successful_manifest_rows": sum(1 for row in manifest_rows if row.get("conversion_status") == "success"),
        "actual_png_files": count_actual_pngs(),
        "missing_output": missing_output,
        "unreadable_output": unreadable,
        "wrong_size": wrong_size,
        "empty_image_all_zero": empty_image,
        "duplicate_output_path_groups": duplicate_output_groups,
        "source_crop_missing": source_missing,
        "conversion_status_failed": status_failed,
        "manifest_missing_expected_count": len(manifest_missing_expected),
        "manifest_unexpected_count": len(manifest_unexpected),
        "train_vs_val_leakage": leakage_rows[0]["leakage_count"],
        "train_vs_test_leakage": leakage_rows[1]["leakage_count"],
        "val_vs_test_leakage": leakage_rows[2]["leakage_count"],
    }
    return errors, checked_rows, summary_rows, {"metrics": metrics, "leakage_rows": leakage_rows, "manifest_rows": manifest_rows}


def expected_output_from_final(row: dict[str, str]) -> Path:
    from common import CLASS_DIR_NAMES, PROJECT_ROOT

    return PROJECT_ROOT / "data" / "processed" / "bbox_crops_224" / row["split"] / CLASS_DIR_NAMES[int(row["class_id"])] / Path(row["crop_path"]).name


def count_actual_pngs() -> int:
    from common import PROJECT_ROOT

    root = PROJECT_ROOT / "data" / "processed" / "bbox_crops_224"
    return len(list(root.rglob("*.png"))) if root.exists() else 0


def error_row(row: dict[str, str], error_type: str, detail: str) -> dict[str, Any]:
    return {
        "output_224_path": row.get("output_224_path", ""),
        "source_crop_path": row.get("source_crop_path", ""),
        "source_image_id": row.get("source_image_id", ""),
        "split": row.get("split", ""),
        "class_id": row.get("class_id", ""),
        "class_name": row.get("class_name", ""),
        "annotation_index": row.get("annotation_index", ""),
        "error_type": error_type,
        "error_detail": detail,
    }


def blank_error(error_type: str, detail: str) -> dict[str, Any]:
    return error_row({}, error_type, detail)


def summarize_by_split_class(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts = Counter((row["split"], row["class_id"], row["class_name"]) for row in manifest_rows if row.get("conversion_status") == "success")
    rows = []
    for split in SPLITS:
        for class_name, class_id in CLASS_ORDER:
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "output_count": counts[(split, str(class_id), class_name)],
                }
            )
    return rows


def create_manual_review_sample(manifest_rows: list[dict[str, str]]) -> int:
    output_path = metadata_dir() / "manual_224_review.csv"
    successful = [row for row in manifest_rows if row.get("conversion_status") == "success"]
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in successful:
        grouped[(int(row["class_id"]), row["split"])].append(row)

    rng = random.Random(RANDOM_SEED)
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for class_name, class_id in CLASS_ORDER:
        for split in SPLITS:
            candidates = sorted(
                grouped[(class_id, split)],
                key=lambda row: (
                    0 if row.get("manual_review_status") == "review" else 1,
                    row["source_image_id"],
                    int(row["annotation_index"]),
                ),
            )
            if not candidates:
                continue
            chosen = candidates[0]
            selected.append(chosen)
            seen.add(norm_path(chosen["output_224_path"]))

        class_selected_count = sum(1 for row in selected if int(row["class_id"]) == class_id)
        remaining = [
            row for row in successful
            if int(row["class_id"]) == class_id and norm_path(row["output_224_path"]) not in seen
        ]
        remaining = sorted(
            remaining,
            key=lambda row: (
                0 if row.get("manual_review_status") == "review" else 1,
                row["split"],
                row["source_image_id"],
                int(row["annotation_index"]),
            ),
        )
        while class_selected_count < 6 and remaining:
            pool_review = [row for row in remaining if row.get("manual_review_status") == "review"]
            if pool_review:
                chosen = pool_review[0]
            else:
                chosen = rng.choice(remaining)
            selected.append(chosen)
            seen.add(norm_path(chosen["output_224_path"]))
            remaining = [row for row in remaining if norm_path(row["output_224_path"]) not in seen]
            class_selected_count += 1

    selected = selected[:30]
    rows = [
        {
            "output_224_path": row["output_224_path"],
            "source_crop_path": row["source_crop_path"],
            "source_image_id": row["source_image_id"],
            "split": row["split"],
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "annotation_index": row["annotation_index"],
            "manual_crop_review_status": row.get("manual_review_status", ""),
            "previous_technical_issue": row.get("technical_issue", ""),
            "review_status": "",
            "review_note": "",
        }
        for row in selected
    ]
    write_csv_rows(
        output_path,
        [
            "output_224_path",
            "source_crop_path",
            "source_image_id",
            "split",
            "class_id",
            "class_name",
            "annotation_index",
            "manual_crop_review_status",
            "previous_technical_issue",
            "review_status",
            "review_note",
        ],
        rows,
    )
    return len(rows)


def main() -> int:
    errors, checked_rows, summary_rows, extra = audit_outputs()
    reports_dir().mkdir(parents=True, exist_ok=True)

    write_csv_rows(
        reports_dir() / "model_input_224_errors.csv",
        [
            "output_224_path",
            "source_crop_path",
            "source_image_id",
            "split",
            "class_id",
            "class_name",
            "annotation_index",
            "error_type",
            "error_detail",
        ],
        errors,
    )
    write_csv_rows(
        reports_dir() / "model_input_224_summary_by_split_class.csv",
        ["split", "class_id", "class_name", "output_count"],
        summary_rows,
    )
    metrics = extra["metrics"]
    with (reports_dir() / "model_input_224_audit.txt").open("w", encoding="utf-8") as f:
        f.write("Model input 224 audit\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    blocking = (
        metrics["expected_outputs"] != EXPECTED_COUNT
        or metrics["final_include_true_rows"] != EXPECTED_COUNT
        or metrics["manifest_rows"] != EXPECTED_COUNT
        or metrics["successful_manifest_rows"] != EXPECTED_COUNT
        or metrics["actual_png_files"] != EXPECTED_COUNT
        or metrics["missing_output"] != 0
        or metrics["unreadable_output"] != 0
        or metrics["wrong_size"] != 0
        or metrics["empty_image_all_zero"] != 0
        or metrics["duplicate_output_path_groups"] != 0
        or metrics["source_crop_missing"] != 0
        or metrics["conversion_status_failed"] != 0
        or metrics["manifest_missing_expected_count"] != 0
        or metrics["manifest_unexpected_count"] != 0
        or metrics["train_vs_val_leakage"] != 0
        or metrics["train_vs_test_leakage"] != 0
        or metrics["val_vs_test_leakage"] != 0
        or bool(errors)
    )

    sample_count = 0
    if not blocking:
        sample_count = create_manual_review_sample(extra["manifest_rows"])
        with (reports_dir() / "model_input_224_audit.txt").open("a", encoding="utf-8") as f:
            f.write(f"manual_224_review_rows: {sample_count}\n")

    print("224 model input audit completed.")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"error_rows: {len(errors)}")
    print(f"manual_224_review_rows: {sample_count}")
    print(f"Audit report: {reports_dir() / 'model_input_224_audit.txt'}")
    return 0 if not blocking else 1


if __name__ == "__main__":
    sys.exit(main())

