from __future__ import annotations

import csv
import random
from collections import defaultdict
from typing import Any

from common import CLASS_ORDER, SPLITS, crop_manifest_path, metadata_dir, write_csv_rows

RANDOM_SEED = 42
SAMPLES_PER_SPLIT_CLASS = 10


def read_manifest() -> list[dict[str, Any]]:
    with crop_manifest_path().open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    output_path = metadata_dir() / "manual_crop_review.csv"
    rows = [
        row for row in read_manifest()
        if str(row.get("is_valid", "")).strip().lower() == "true"
    ]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], int(row["class_id"]))].append(row)

    rng = random.Random(RANDOM_SEED)
    review_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for class_name, class_id in CLASS_ORDER:
            candidates = sorted(grouped[(split, class_id)], key=lambda row: (row["source_image_id"], int(row["annotation_index"])))
            selected = candidates[:] if len(candidates) <= SAMPLES_PER_SPLIT_CLASS else rng.sample(candidates, SAMPLES_PER_SPLIT_CLASS)
            selected = sorted(selected, key=lambda row: (row["source_image_id"], int(row["annotation_index"])))
            for row in selected:
                review_rows.append(
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

    write_csv_rows(
        output_path,
        [
            "crop_path",
            "source_image_id",
            "split",
            "class_id",
            "class_name",
            "annotation_index",
            "review_status",
            "technical_issue",
            "review_note",
        ],
        review_rows,
    )
    print("Manual crop review template created.")
    print(f"Rows: {len(review_rows)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

