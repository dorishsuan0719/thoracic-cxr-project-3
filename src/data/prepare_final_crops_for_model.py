from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import CLASS_ORDER, SPLITS, crop_manifest_path, metadata_dir, reports_dir, write_csv_rows

REQUIRED_MANUAL_COLUMNS = [
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
ALLOWED_REVIEW_STATUS = {"pass", "review", "fail"}
BBOX_PATTERN = re.compile(r"_bbox(\d+)\.png$", re.IGNORECASE)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp950", "mbcs"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                return list(reader), list(reader.fieldnames or [])
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise last_error or UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode CSV")


def normalize(value: Any) -> str:
    return str(value or "").strip()


def normalize_status(value: Any) -> str:
    return normalize(value).lower()


def parse_bbox_index_from_path(crop_path: str) -> int | None:
    match = BBOX_PATTERN.search(Path(crop_path).name)
    if not match:
        return None
    return int(match.group(1))


def parse_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def add_error(errors: list[dict[str, Any]], row_number: int, row: dict[str, str], error_type: str, detail: str) -> None:
    errors.append(
        {
            "row_number": row_number,
            "crop_path": row.get("crop_path", ""),
            "source_image_id": row.get("source_image_id", ""),
            "split": row.get("split", ""),
            "class_id": row.get("class_id", ""),
            "class_name": row.get("class_name", ""),
            "annotation_index": row.get("annotation_index", ""),
            "review_status": row.get("review_status", ""),
            "technical_issue": row.get("technical_issue", ""),
            "error_type": error_type,
            "error_detail": detail,
        }
    )


def write_review_summary(rows: list[dict[str, str]], errors: list[dict[str, Any]]) -> None:
    reports_dir().mkdir(parents=True, exist_ok=True)
    status_counts = Counter(normalize_status(row.get("review_status")) for row in rows)
    split_status_counts = Counter((row.get("split", ""), normalize_status(row.get("review_status"))) for row in rows)
    class_status_counts = Counter(
        (row.get("class_id", ""), row.get("class_name", ""), normalize_status(row.get("review_status"))) for row in rows
    )
    issue_counts = Counter(normalize_status(row.get("technical_issue")) for row in rows)
    error_counts = Counter(row["error_type"] for row in errors)

    summary_rows: list[dict[str, Any]] = []
    for status in sorted(status_counts):
        summary_rows.append({"summary_type": "review_status", "split": "", "class_id": "", "class_name": "", "name": status or "(blank)", "count": status_counts[status]})
    for split in SPLITS:
        for status in ("pass", "review", "fail", ""):
            count = split_status_counts[(split, status)]
            if count:
                summary_rows.append({"summary_type": "split_review_status", "split": split, "class_id": "", "class_name": "", "name": status or "(blank)", "count": count})
    for class_name, class_id in CLASS_ORDER:
        for status in ("pass", "review", "fail", ""):
            count = class_status_counts[(str(class_id), class_name, status)]
            if count:
                summary_rows.append({"summary_type": "class_review_status", "split": "", "class_id": class_id, "class_name": class_name, "name": status or "(blank)", "count": count})
    for issue in sorted(issue_counts):
        summary_rows.append({"summary_type": "technical_issue", "split": "", "class_id": "", "class_name": "", "name": issue or "(blank)", "count": issue_counts[issue]})
    for error_type, count in sorted(error_counts.items()):
        summary_rows.append({"summary_type": "validation_error", "split": "", "class_id": "", "class_name": "", "name": error_type, "count": count})

    write_csv_rows(
        reports_dir() / "manual_crop_review_summary.csv",
        ["summary_type", "split", "class_id", "class_name", "name", "count"],
        summary_rows,
    )
    write_csv_rows(
        reports_dir() / "manual_crop_review_errors.csv",
        [
            "row_number",
            "crop_path",
            "source_image_id",
            "split",
            "class_id",
            "class_name",
            "annotation_index",
            "review_status",
            "technical_issue",
            "error_type",
            "error_detail",
        ],
        errors,
    )

    with (reports_dir() / "manual_crop_review_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Manual crop review summary\n")
        f.write(f"manual_review_rows: {len(rows)}\n")
        f.write("\nreview_status counts\n")
        for status in ("pass", "review", "fail", ""):
            if status_counts[status]:
                f.write(f"{status or '(blank)'}: {status_counts[status]}\n")
        f.write("\nsplit x review_status counts\n")
        for split in SPLITS:
            for status in ("pass", "review", "fail", ""):
                count = split_status_counts[(split, status)]
                if count:
                    f.write(f"{split},{status or '(blank)'}: {count}\n")
        f.write("\nclass x review_status counts\n")
        for class_name, class_id in CLASS_ORDER:
            for status in ("pass", "review", "fail", ""):
                count = class_status_counts[(str(class_id), class_name, status)]
                if count:
                    f.write(f"class{class_id},{class_name},{status or '(blank)'}: {count}\n")
        f.write("\ntechnical_issue counts\n")
        for issue, count in sorted(issue_counts.items()):
            f.write(f"{issue or '(blank)'}: {count}\n")
        f.write("\nvalidation errors\n")
        if not errors:
            f.write("none: 0\n")
        else:
            for error_type, count in sorted(error_counts.items()):
                f.write(f"{error_type}: {count}\n")


def validate_manual_review(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    missing_columns = [column for column in REQUIRED_MANUAL_COLUMNS if column not in fieldnames]
    if missing_columns:
        placeholder = {column: "" for column in REQUIRED_MANUAL_COLUMNS}
        add_error(errors, 0, placeholder, "missing_required_column", ";".join(missing_columns))
        return errors

    crop_path_rows: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    annotation_key_rows: dict[tuple[str, str, str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)

    for index, row in enumerate(rows, start=2):
        for column in REQUIRED_MANUAL_COLUMNS:
            if column == "review_note":
                continue
            if normalize(row.get(column)) == "":
                add_error(errors, index, row, f"missing_{column}", f"{column} is blank")

        status = normalize_status(row.get("review_status"))
        if status and status not in ALLOWED_REVIEW_STATUS:
            add_error(errors, index, row, "invalid_review_status", f"review_status={row.get('review_status')}")

        annotation_index = parse_int(row.get("annotation_index"))
        bbox_index = parse_bbox_index_from_path(row.get("crop_path", ""))
        if bbox_index is None:
            add_error(errors, index, row, "missing_bbox_index_in_filename", Path(row.get("crop_path", "")).name)
        elif annotation_index is None:
            add_error(errors, index, row, "invalid_annotation_index", str(row.get("annotation_index")))
        elif annotation_index != bbox_index:
            add_error(errors, index, row, "annotation_index_filename_mismatch", f"annotation_index={annotation_index}, filename_bbox={bbox_index}")

        crop_path = normalize(row.get("crop_path"))
        if crop_path:
            crop_path_rows[crop_path].append((index, row))
        annotation_key = (
            normalize(row.get("split")),
            normalize(row.get("source_image_id")),
            normalize(row.get("class_id")),
            normalize(row.get("annotation_index")),
        )
        if all(annotation_key):
            annotation_key_rows[annotation_key].append((index, row))

    for crop_path, duplicate_rows in crop_path_rows.items():
        if len(duplicate_rows) > 1:
            row_numbers = ",".join(str(item[0]) for item in duplicate_rows)
            for row_number, row in duplicate_rows:
                add_error(errors, row_number, row, "duplicate_crop_path", f"rows={row_numbers}")

    for annotation_key, duplicate_rows in annotation_key_rows.items():
        if len(duplicate_rows) > 1:
            distinct_crop_paths = {normalize(row.get("crop_path")) for _, row in duplicate_rows}
            if len(distinct_crop_paths) > 1:
                row_numbers = ",".join(str(item[0]) for item in duplicate_rows)
                for row_number, row in duplicate_rows:
                    add_error(errors, row_number, row, "duplicate_annotation_index", f"rows={row_numbers}; key={annotation_key}")

    return errors


def create_final_crops_for_model(manual_rows: list[dict[str, str]]) -> dict[str, Any]:
    manifest_rows, _ = read_csv(crop_manifest_path())
    manual_by_crop_path = {normalize(row["crop_path"]): row for row in manual_rows}
    final_rows: list[dict[str, Any]] = []

    for manifest in manifest_rows:
        crop_path = normalize(manifest["crop_path"])
        manual = manual_by_crop_path.get(crop_path)
        if manual is None:
            manual_status = "not_sampled"
            technical_issue = ""
            review_note = ""
            include_for_model = "TRUE"
            exclusion_reason = ""
        else:
            manual_status = normalize_status(manual["review_status"])
            technical_issue = manual["technical_issue"]
            review_note = manual.get("review_note", "")
            include_for_model = "FALSE" if manual_status == "fail" else "TRUE"
            exclusion_reason = "manual_review_fail" if manual_status == "fail" else ""

        final_rows.append(
            {
                "crop_path": crop_path,
                "source_image_id": manifest["source_image_id"],
                "split": manifest["split"],
                "class_id": manifest["class_id"],
                "class_name": manifest["class_name"],
                "annotation_index": manifest["annotation_index"],
                "crop_width": manifest["crop_width"],
                "crop_height": manifest["crop_height"],
                "manual_review_status": manual_status,
                "technical_issue": technical_issue,
                "review_note": review_note,
                "include_for_model": include_for_model,
                "exclusion_reason": exclusion_reason,
            }
        )

    output_path = metadata_dir() / "final_crops_for_model.csv"
    write_csv_rows(
        output_path,
        [
            "crop_path",
            "source_image_id",
            "split",
            "class_id",
            "class_name",
            "annotation_index",
            "crop_width",
            "crop_height",
            "manual_review_status",
            "technical_issue",
            "review_note",
            "include_for_model",
            "exclusion_reason",
        ],
        final_rows,
    )

    include_true = [row for row in final_rows if row["include_for_model"] == "TRUE"]
    include_false = [row for row in final_rows if row["include_for_model"] == "FALSE"]
    split_class_counts = Counter((row["split"], row["class_id"], row["class_name"]) for row in include_true)
    review_issues = Counter(row["technical_issue"] for row in final_rows if row["manual_review_status"] == "review")
    fail_reasons = Counter(row["exclusion_reason"] for row in include_false)
    split_sources: dict[str, set[str]] = defaultdict(set)
    for row in include_true:
        split_sources[row["split"]].add(row["source_image_id"])

    leakage = {}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        leakage[f"{left}_vs_{right}"] = len(split_sources[left] & split_sources[right])

    summary_rows = []
    for split in SPLITS:
        for class_name, class_id in CLASS_ORDER:
            summary_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "included_count": split_class_counts[(split, str(class_id), class_name)],
                }
            )
    write_csv_rows(
        metadata_dir() / "final_crops_for_model_summary_by_split_class.csv",
        ["split", "class_id", "class_name", "included_count"],
        summary_rows,
    )

    with (reports_dir() / "final_crops_for_model_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Final crops for model summary\n")
        f.write(f"include_for_model_TRUE: {len(include_true)}\n")
        f.write(f"include_for_model_FALSE: {len(include_false)}\n")
        f.write("\nsplit x class included counts\n")
        for row in summary_rows:
            f.write(f"{row['split']},class{row['class_id']},{row['class_name']}: {row['included_count']}\n")
        f.write("\nreview technical_issue counts\n")
        for issue, count in sorted(review_issues.items()):
            f.write(f"{issue}: {count}\n")
        f.write("\nfail exclusion reasons\n")
        if fail_reasons:
            for reason, count in sorted(fail_reasons.items()):
                f.write(f"{reason}: {count}\n")
        else:
            f.write("none: 0\n")
        f.write("\nleakage\n")
        for pair, count in leakage.items():
            f.write(f"{pair}: {count}\n")

    return {
        "output_path": str(output_path),
        "include_true": len(include_true),
        "include_false": len(include_false),
        "leakage": leakage,
    }


def main() -> None:
    manual_path = metadata_dir() / "manual_crop_review.csv"
    manual_rows, fieldnames = read_csv(manual_path)
    errors = validate_manual_review(manual_rows, fieldnames)
    write_review_summary(manual_rows, errors)

    print("Manual crop review validation completed.")
    print(f"Manual review rows: {len(manual_rows)}")
    print(f"Validation errors: {len(errors)}")
    print(f"Summary: {reports_dir() / 'manual_crop_review_summary.txt'}")
    print(f"Errors: {reports_dir() / 'manual_crop_review_errors.csv'}")

    if errors:
        print("Stopped before creating final_crops_for_model.csv because manual crop review has blocking errors.")
        return sys.exit(1)

    result = create_final_crops_for_model(manual_rows)
    print("Final crops for model created.")
    print(f"Output: {result['output_path']}")
    print(f"include_for_model TRUE: {result['include_true']}")
    print(f"include_for_model FALSE: {result['include_false']}")
    for pair, count in result["leakage"].items():
        print(f"{pair} leakage: {count}")


if __name__ == "__main__":
    main()
