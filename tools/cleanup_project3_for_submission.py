#!/usr/bin/env python
"""Safely clean thoracic-cxr-project-3 for submission.

The default mode is audit-only and must not write, move, or delete files.
Use --apply only after reviewing the candidate lists printed by audit-only.
"""

from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_FULL_MODEL_SHA256 = "0287fe36d3623ccdb5aa43857db1168a1598788071ebdecbc43324a6953f426f"
EXPECTED_FULL_THRESHOLDS_SHA256 = "73a54c9b6a3de2b2f63479b0bd918836cedb577a29255b0f6e0b30dac310e9d5"
EXPECTED_ROI_PATCH_SHA256 = "8a68d68b901d721c63a38b5e75ee3291a8c06d13195572d20f29fd34a56485e5"

FULL_MODEL_REL = Path(
    "outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/checkpoints/"
    "full_image_multilabel_patch_transfer.pt"
)
FULL_THRESHOLDS_REL = Path(
    "outputs/full_image_224_multilabel_seed42/phase2_patch_transfer/"
    "validation_selected_thresholds.json"
)

PROTECTED_FILES = {
    Path("app_full_image_multilabel_ollama_gradio.py"),
    Path("src/full_image_multilabel_inference_service.py"),
    Path("src/full_image_multilabel_report_prompt.py"),
    Path("src/full_image_multilabel_ollama_service.py"),
    Path("README.md"),
    Path("requirements.txt"),
    Path(".gitignore"),
    Path("final_research_report.html"),
}

PROTECTED_DIRS = {
    Path(".git"),
    Path("data/raw"),
    Path("data/processed/bbox_crops"),
    Path("data/processed/bbox_crops_224"),
    Path("data/processed/bbox_overlay"),
    Path("data/metadata"),
    Path("data/final"),
    Path("docs"),
    Path("src"),
    Path("tools"),
    Path("outputs/full_image_224_multilabel_seed42"),
    Path("outputs/raddino_convnext_tiny_patch_experiment_seed42"),
    Path("outputs/raddino_convnext_tiny_three_model_comparison_seed42"),
}

FORMAL_OUTPUT_DIRS = {
    Path("outputs/full_image_224_multilabel_seed42"),
    Path("outputs/raddino_convnext_tiny_patch_experiment_seed42"),
    Path("outputs/raddino_convnext_tiny_three_model_comparison_seed42"),
}

SKIP_SCAN_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "_cleanup_quarantine",
}

CACHE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".coverage_cache",
    "htmlcov",
}

CACHE_FILE_PATTERNS = {
    "*.pyc",
    "*.pyo",
    ".coverage",
    "coverage.xml",
}

BACKUP_FILE_PATTERNS = {
    "*_backup.py",
    "*_old.py",
    "*_fixed.py",
    "*_fixed.py.py",
    "*_copy.py",
    "*_before_css_fix.py",
    "*_before_cleanup.py",
    "*_before_*.py",
    "* (1).py",
    "* (2).py",
    "Pasted code*.py",
    "untitled*.py",
    "temp_*.py",
    "tmp_*.py",
    "已貼上文字*.txt",
}

LOG_FILE_NAMES = {
    "ui_error.log",
    "app_startup_audit.json",
    "diagnosis_report_pdf_error.txt",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}


@dataclass
class Action:
    kind: str
    relative_path: str
    target_path: str = ""
    size_bytes: int = 0
    sha256: str = ""
    reason: str = ""
    status: str = "planned"


@dataclass
class CleanupPlan:
    cache_deletes: list[Action] = field(default_factory=list)
    smoke_session_deletes: list[Action] = field(default_factory=list)
    quarantines: list[Action] = field(default_factory=list)
    empty_dir_deletes: list[Action] = field(default_factory=list)
    manual_review: list[Action] = field(default_factory=list)
    retained_smoke_audit: str = ""
    retained_general_sessions: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform the planned cleanup")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="audit without writing, moving, or deleting files (default)",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def normalize_rel(path: Path) -> Path:
    return Path(*path.parts)


def rel_path(root: Path, path: Path) -> Path:
    return normalize_rel(path.resolve().relative_to(root.resolve()))


def rel_str(path: Path) -> str:
    return str(normalize_rel(path)).replace("/", "\\")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def is_under_any(path: Path, parents: Iterable[Path]) -> bool:
    return any(path == parent or is_relative_to(path, parent) for parent in parents)


def is_protected(rel: Path) -> bool:
    rel = normalize_rel(rel)
    return rel in PROTECTED_FILES or is_under_any(rel, PROTECTED_DIRS)


def is_formal_output(rel: Path) -> bool:
    rel = normalize_rel(rel)
    return is_under_any(rel, FORMAL_OUTPUT_DIRS)


def should_skip_dir(rel: Path) -> bool:
    return any(part in SKIP_SCAN_DIR_NAMES for part in rel.parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += file_size(item)
    return total


def snapshot(root: Path) -> dict[str, Any]:
    files = 0
    dirs = 0
    size = 0
    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        rel = rel_path(root, current_path)
        if should_skip_dir(rel):
            dirnames[:] = []
            continue
        dirs += len(dirnames)
        files += len(filenames)
        for filename in filenames:
            path = current_path / filename
            try:
                size += path.stat().st_size
            except OSError:
                pass
    return {"files": files, "dirs": dirs, "size_bytes": size}


def iter_project_files(root: Path) -> Iterable[Path]:
    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        rel = rel_path(root, current_path)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not should_skip_dir(normalize_rel(rel / dirname))
        ]
        for filename in filenames:
            yield current_path / filename


def iter_project_dirs(root: Path) -> Iterable[Path]:
    for current, dirnames, _filenames in os.walk(root):
        current_path = Path(current)
        rel = rel_path(root, current_path)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not should_skip_dir(normalize_rel(rel / dirname))
        ]
        for dirname in dirnames:
            yield current_path / dirname


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def action_for_path(root: Path, kind: str, path: Path, reason: str) -> Action:
    rel = rel_path(root, path)
    if path.is_dir():
        size = dir_size(path)
        digest = ""
    else:
        size = file_size(path)
        digest = sha256_file(path)
    return Action(kind=kind, relative_path=rel_str(rel), size_bytes=size, sha256=digest, reason=reason)


def collect_cache_candidates(root: Path) -> list[Action]:
    actions: list[Action] = []
    for directory in iter_project_dirs(root):
        rel = rel_path(root, directory)
        if directory.name in CACHE_DIR_NAMES and not any(part in {".venv", "venv", "env"} for part in rel.parts):
            actions.append(action_for_path(root, "delete_cache_dir", directory, "auto-rebuildable cache directory"))
    for path in iter_project_files(root):
        rel = rel_path(root, path)
        if any(part in {".venv", "venv", "env"} for part in rel.parts):
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in CACHE_FILE_PATTERNS):
            actions.append(action_for_path(root, "delete_cache_file", path, "auto-rebuildable cache file"))
    return dedupe_actions(actions)


def session_sort_key(path: Path) -> tuple[str, float]:
    audit = read_json(path / "session_audit.json")
    timestamp = str(audit.get("timestamp", "")) if audit else ""
    return timestamp, path.stat().st_mtime


def collect_demo_sessions(root: Path, plan: CleanupPlan) -> None:
    sessions_root = root / "outputs/full_image_multilabel_gradio_demo/sessions"
    if not sessions_root.is_dir():
        return

    general_sessions: list[Path] = []
    for session_dir in sorted(item for item in sessions_root.iterdir() if item.is_dir()):
        audit_path = session_dir / "session_audit.json"
        if not audit_path.is_file():
            plan.manual_review.append(
                action_for_path(root, "manual_review_session", session_dir, "session has no session_audit.json")
            )
            continue
        audit = read_json(audit_path)
        if audit is None:
            plan.manual_review.append(
                action_for_path(root, "manual_review_session", session_dir, "session_audit.json is unreadable")
            )
            continue
        if audit.get("smoke_test") is True:
            plan.smoke_session_deletes.append(
                action_for_path(root, "delete_smoke_session", session_dir, "session_audit.json has smoke_test=true")
            )
        elif audit.get("smoke_test") is False:
            general_sessions.append(session_dir)
        else:
            plan.manual_review.append(
                action_for_path(root, "manual_review_session", session_dir, "session smoke_test field is missing")
            )

    general_sessions.sort(key=session_sort_key, reverse=True)
    keep = general_sessions[:3]
    plan.retained_general_sessions = [rel_str(rel_path(root, item)) for item in keep]
    for session_dir in general_sessions[3:]:
        plan.quarantines.append(
            action_for_path(root, "quarantine_demo_session", session_dir, "older non-smoke demo session; keep latest 3")
        )


def collect_smoke_audits(root: Path, plan: CleanupPlan) -> None:
    demo_root = root / "outputs/full_image_multilabel_gradio_demo"
    if not demo_root.is_dir():
        return
    audits = sorted(demo_root.glob("smoke_test_audit*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    pass_audits = [path for path in audits if (read_json(path) or {}).get("status") == "PASS"]
    retain = pass_audits[0] if pass_audits else (audits[0] if audits else None)
    if retain:
        plan.retained_smoke_audit = rel_str(rel_path(root, retain))
    for path in audits:
        if retain and path == retain:
            continue
        plan.quarantines.append(
            action_for_path(root, "quarantine_old_smoke_audit", path, "old smoke test audit; keep newest PASS when available")
        )


def matches_backup_pattern(path: Path) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in BACKUP_FILE_PATTERNS)


def collect_backup_files(root: Path) -> list[Action]:
    actions: list[Action] = []
    formal_app = Path("app_full_image_multilabel_ollama_gradio.py")
    for path in iter_project_files(root):
        rel = rel_path(root, path)
        if rel == formal_app:
            continue
        if is_protected(rel) or is_formal_output(rel):
            continue
        if matches_backup_pattern(path):
            actions.append(action_for_path(root, "quarantine_backup_file", path, "backup/copy/temp filename pattern"))
    return dedupe_actions(actions)


def collect_duplicate_files(root: Path) -> list[Action]:
    formal_app = root / "app_full_image_multilabel_ollama_gradio.py"
    if not formal_app.is_file():
        return []
    formal_sha = sha256_file(formal_app)
    actions: list[Action] = []
    for path in iter_project_files(root):
        rel = rel_path(root, path)
        if rel == Path("app_full_image_multilabel_ollama_gradio.py"):
            continue
        if is_protected(rel) or is_formal_output(rel):
            continue
        if not matches_backup_pattern(path):
            continue
        if file_size(path) == file_size(formal_app) and sha256_file(path) == formal_sha:
            actions.append(action_for_path(root, "quarantine_duplicate_backup", path, "exact duplicate of canonical Gradio app"))
    return dedupe_actions(actions)


def collect_old_logs(root: Path) -> list[Action]:
    by_name: dict[str, list[Path]] = {name: [] for name in LOG_FILE_NAMES}
    for path in iter_project_files(root):
        if path.name not in LOG_FILE_NAMES:
            continue
        rel = rel_path(root, path)
        if is_formal_output(rel):
            continue
        by_name[path.name].append(path)

    actions: list[Action] = []
    for name, paths in by_name.items():
        if len(paths) <= 1:
            continue
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths[1:]:
            actions.append(action_for_path(root, "quarantine_old_log", path, f"old duplicate {name}; keep newest"))
    return actions


def collect_empty_dirs(root: Path) -> list[Action]:
    actions: list[Action] = []
    for directory in sorted(iter_project_dirs(root), key=lambda p: len(p.parts), reverse=True):
        rel = rel_path(root, directory)
        if is_protected(rel) or rel.parts[:1] in {(Path("data"),)}:
            continue
        if directory.name in {"outputs", "docs", "data", "src", "tools"}:
            continue
        try:
            if any(directory.iterdir()):
                continue
        except OSError:
            continue
        actions.append(action_for_path(root, "delete_empty_dir", directory, "safe empty directory outside protected structures"))
    return actions


def collect_manual_reviews(root: Path) -> list[Action]:
    actions: list[Action] = []
    for directory_name in (".venv", "venv", "env"):
        path = root / directory_name
        if path.exists():
            actions.append(action_for_path(root, "manual_review_venv", path, "virtual environment is reported only; not auto-cleaned"))
    return actions


def dedupe_actions(actions: list[Action]) -> list[Action]:
    seen: set[tuple[str, str]] = set()
    result: list[Action] = []
    for action in actions:
        key = (action.kind, action.relative_path)
        if key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result


def build_plan(root: Path) -> CleanupPlan:
    plan = CleanupPlan()
    plan.cache_deletes = collect_cache_candidates(root)
    collect_demo_sessions(root, plan)
    collect_smoke_audits(root, plan)
    plan.quarantines.extend(collect_backup_files(root))
    plan.quarantines.extend(collect_duplicate_files(root))
    plan.quarantines.extend(collect_old_logs(root))
    plan.quarantines = dedupe_actions(plan.quarantines)
    plan.empty_dir_deletes = collect_empty_dirs(root)
    plan.manual_review.extend(collect_manual_reviews(root))
    plan.manual_review = dedupe_actions(plan.manual_review)
    return plan


def find_sha_in_dirs(root: Path, expected_sha: str, dirs: Iterable[Path], suffixes: set[str]) -> str:
    for rel_dir in dirs:
        base = root / rel_dir
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if sha256_file(path) == expected_sha:
                return rel_str(rel_path(root, path))
    return ""


def count_raw_images(root: Path) -> int:
    raw = root / "data/raw"
    if not raw.is_dir():
        return 0
    return sum(1 for path in raw.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def count_annotation_files(root: Path) -> int:
    data = root / "data"
    if not data.is_dir():
        return 0
    tokens = ("annotation", "annotations", "manifest", "split", "label", "manual_review")
    return sum(
        1
        for path in data.rglob("*")
        if path.is_file()
        and (path.suffix.lower() in {".csv", ".json", ".xlsx", ".txt"})
        and any(token in str(path).lower() for token in tokens)
    )


def paths_exist(root: Path, paths: Iterable[Path]) -> dict[str, bool]:
    return {rel_str(path): (root / path).exists() for path in paths}


def ast_parse_errors(root: Path) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for path in iter_project_files(root):
        rel = rel_path(root, path)
        if path.suffix != ".py":
            continue
        if "_cleanup_quarantine" in rel.parts:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append({"path": rel_str(rel), "error": f"{type(exc).__name__}: {exc}"})
    return errors


def validation_state(root: Path) -> dict[str, Any]:
    full_model = root / FULL_MODEL_REL
    full_thresholds = root / FULL_THRESHOLDS_REL
    train_val_test = [
        Path("outputs/full_image_224_multilabel_seed42/phase0_dataset/train_manifest.csv"),
        Path("outputs/full_image_224_multilabel_seed42/phase0_dataset/val_manifest.csv"),
        Path("outputs/full_image_224_multilabel_seed42/phase0_dataset/test_manifest.csv"),
    ]
    formal_test_candidates = list((root / "outputs/full_image_224_multilabel_seed42").rglob("*test*")) if (root / "outputs/full_image_224_multilabel_seed42").is_dir() else []
    roi_metrics = list((root / "outputs/raddino_convnext_tiny_patch_experiment_seed42").rglob("*metric*")) if (root / "outputs/raddino_convnext_tiny_patch_experiment_seed42").is_dir() else []
    roi_figures = list((root / "outputs/raddino_convnext_tiny_patch_experiment_seed42").rglob("*.png")) if (root / "outputs/raddino_convnext_tiny_patch_experiment_seed42").is_dir() else []
    final_report = next(root.rglob("final_research_report.html"), None)
    return {
        "full_model_path": rel_str(FULL_MODEL_REL),
        "full_model_exists": full_model.is_file(),
        "full_model_sha256": sha256_file(full_model) if full_model.is_file() else "",
        "full_thresholds_path": rel_str(FULL_THRESHOLDS_REL),
        "full_thresholds_exists": full_thresholds.is_file(),
        "full_thresholds_sha256": sha256_file(full_thresholds) if full_thresholds.is_file() else "",
        "roi_patch_checkpoint_path": find_sha_in_dirs(
            root,
            EXPECTED_ROI_PATCH_SHA256,
            [
                Path("outputs/raddino_convnext_tiny_patch_experiment_seed42"),
                Path("outputs/raddino_convnext_tiny_three_model_comparison_seed42"),
            ],
            {".pt", ".pth"},
        ),
        "app_exists": (root / "app_full_image_multilabel_ollama_gradio.py").is_file(),
        "service_files": paths_exist(
            root,
            [
                Path("src/full_image_multilabel_inference_service.py"),
                Path("src/full_image_multilabel_report_prompt.py"),
                Path("src/full_image_multilabel_ollama_service.py"),
            ],
        ),
        "raw_image_count": count_raw_images(root),
        "annotation_file_count": count_annotation_files(root),
        "manifests": paths_exist(root, train_val_test),
        "formal_test_file_count": len([path for path in formal_test_candidates if path.is_file()]),
        "roi_metric_file_count": len([path for path in roi_metrics if path.is_file()]),
        "roi_figure_file_count": len([path for path in roi_figures if path.is_file()]),
        "final_research_report": rel_str(rel_path(root, final_report)) if final_report else "",
        "python_ast_errors": ast_parse_errors(root),
    }


def validation_passes(state: dict[str, Any]) -> dict[str, bool]:
    return {
        "full_model_sha_ok": state["full_model_sha256"] == EXPECTED_FULL_MODEL_SHA256,
        "full_thresholds_sha_ok": state["full_thresholds_sha256"] == EXPECTED_FULL_THRESHOLDS_SHA256,
        "roi_patch_sha_found": bool(state["roi_patch_checkpoint_path"]),
        "app_exists": bool(state["app_exists"]),
        "service_files_exist": all(state["service_files"].values()),
        "manifests_exist": all(state["manifests"].values()),
        "formal_test_files_exist": state["formal_test_file_count"] > 0,
        "roi_results_exist": state["roi_metric_file_count"] > 0 and state["roi_figure_file_count"] > 0,
        "final_report_exists": bool(state["final_research_report"]),
        "python_ast_parse_ok": not state["python_ast_errors"],
    }


def ensure_safe_plan(root: Path, plan: CleanupPlan) -> None:
    risky: list[str] = []
    all_actions = (
        plan.cache_deletes
        + plan.smoke_session_deletes
        + plan.quarantines
        + plan.empty_dir_deletes
    )
    for action in all_actions:
        rel = normalize_rel(Path(action.relative_path))
        if rel in PROTECTED_FILES or is_formal_output(rel):
            risky.append(f"{action.kind}: {action.relative_path}")
        if is_under_any(rel, {Path("data/raw"), Path("data/processed/bbox_crops"), Path("data/processed/bbox_crops_224")}):
            risky.append(f"{action.kind}: {action.relative_path}")
    if risky:
        raise RuntimeError("Refusing unsafe cleanup candidates:\n" + "\n".join(risky))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def action_rows(actions: list[Action]) -> list[dict[str, Any]]:
    return [action.__dict__ for action in actions]


def markdown_summary(title: str, snapshot_data: dict[str, Any], state: dict[str, Any], checks: dict[str, bool], plan: CleanupPlan | None = None) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Files: {snapshot_data['files']}",
        f"- Directories: {snapshot_data['dirs']}",
        f"- Size bytes: {snapshot_data['size_bytes']}",
        "",
        "## Protected validation",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: {'PASS' if value else 'WARN'}")
    lines.extend(
        [
            "",
            "## Important paths",
            f"- Full-image model: {state['full_model_path']}",
            f"- Full-image thresholds: {state['full_thresholds_path']}",
            f"- ROI patch checkpoint found: {state['roi_patch_checkpoint_path'] or 'MISSING'}",
            f"- Final report: {state['final_research_report'] or 'MISSING'}",
            f"- Raw image count: {state['raw_image_count']}",
            f"- Annotation-like file count: {state['annotation_file_count']}",
            f"- Formal Test file count: {state['formal_test_file_count']}",
            f"- ROI metric files: {state['roi_metric_file_count']}",
            f"- ROI figure files: {state['roi_figure_file_count']}",
        ]
    )
    if plan:
        lines.extend(
            [
                "",
                "## Planned / applied cleanup",
                f"- Cache deletes: {len(plan.cache_deletes)}",
                f"- Smoke sessions deletes: {len(plan.smoke_session_deletes)}",
                f"- Quarantine actions: {len(plan.quarantines)}",
                f"- Empty dirs deletes: {len(plan.empty_dir_deletes)}",
                f"- Retained smoke audit: {plan.retained_smoke_audit or 'none'}",
                f"- Retained general demo sessions: {len(plan.retained_general_sessions)}",
                f"- Manual review candidates: {len(plan.manual_review)}",
            ]
        )
    if state["python_ast_errors"]:
        lines.append("")
        lines.append("## Python AST parse warnings")
        for error in state["python_ast_errors"][:20]:
            lines.append(f"- {error['path']}: {error['error']}")
    return "\n".join(lines) + "\n"


def write_restore_files(quarantine_root: Path) -> None:
    readme = """# Cleanup quarantine restore

Files in this folder were moved here by tools/cleanup_project3_for_submission.py.

Run:

python restore_cleanup.py

The restore script refuses to overwrite existing files and verifies SHA256 before restoring.
Cache files and smoke-test sessions are not quarantined and are intentionally not restored.
"""
    (quarantine_root / "README_RESTORE.md").write_text(readme, encoding="utf-8")
    restore = r'''#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


quarantine_root = Path(__file__).resolve().parent
project_root = quarantine_root.parents[1]
manifest = quarantine_root / "quarantine_manifest.csv"
rows = []
restored = []
skipped = []

with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        rows.append(row)

for row in rows:
    source = quarantine_root / row["quarantined_path"]
    target = project_root / row["original_path"]
    expected = row.get("sha256", "")
    if not source.is_file():
        skipped.append({"path": row["original_path"], "reason": "quarantined file missing"})
        continue
    if target.exists():
        skipped.append({"path": row["original_path"], "reason": "target already exists; not overwriting"})
        continue
    if expected and sha256_file(source) != expected:
        skipped.append({"path": row["original_path"], "reason": "sha256 mismatch"})
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    restored.append(row["original_path"])

report = {"restored": restored, "skipped": skipped}
(quarantine_root / "restore_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
'''
    (quarantine_root / "restore_cleanup.py").write_text(restore, encoding="utf-8")


def quarantine_path_for(rel: str) -> Path:
    return normalize_rel(Path(rel))


def quarantine_action(root: Path, quarantine_root: Path, action: Action, manifest_rows: list[dict[str, Any]]) -> None:
    source = root / Path(action.relative_path)
    if not source.exists():
        action.status = "missing"
        return
    destination_rel = quarantine_path_for(action.relative_path)
    destination = quarantine_root / destination_rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    files_to_record = [source] if source.is_file() else [path for path in source.rglob("*") if path.is_file()]
    for path in files_to_record:
        file_rel = rel_path(root, path)
        quarantined_rel = destination_rel / path.relative_to(source) if source.is_dir() else destination_rel
        manifest_rows.append(
            {
                "original_path": rel_str(file_rel),
                "quarantined_path": rel_str(quarantined_rel),
                "sha256": sha256_file(path),
                "size_bytes": file_size(path),
                "reason": action.reason,
            }
        )
    if destination.exists():
        raise RuntimeError(f"Quarantine destination already exists: {destination}")
    shutil.move(str(source), str(destination))
    action.target_path = rel_str(Path("_cleanup_quarantine") / quarantine_root.name / destination_rel)
    action.status = "quarantined"


def delete_action(root: Path, action: Action) -> None:
    path = root / Path(action.relative_path)
    if not path.exists():
        action.status = "missing"
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    action.status = "deleted"


def apply_cleanup(root: Path, plan: CleanupPlan, before: dict[str, Any], before_state: dict[str, Any], before_checks: dict[str, bool]) -> dict[str, Any]:
    timestamp = now_slug()
    audit_dir = root / "docs/project_cleanup_audit"
    quarantine_root = root / "_cleanup_quarantine" / timestamp
    audit_dir.mkdir(parents=True, exist_ok=True)
    quarantine_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []

    for action in plan.cache_deletes:
        delete_action(root, action)
    for action in plan.smoke_session_deletes:
        delete_action(root, action)
    for action in plan.quarantines:
        quarantine_action(root, quarantine_root, action, manifest_rows)
    for action in plan.empty_dir_deletes:
        delete_action(root, action)

    write_restore_files(quarantine_root)
    write_csv(
        quarantine_root / "quarantine_manifest.csv",
        manifest_rows,
        ["original_path", "quarantined_path", "sha256", "size_bytes", "reason"],
    )

    after = snapshot(root)
    after_state = validation_state(root)
    after_checks = validation_passes(after_state)

    all_actions = (
        plan.cache_deletes
        + plan.smoke_session_deletes
        + plan.quarantines
        + plan.empty_dir_deletes
    )
    write_csv(
        audit_dir / "cleanup_actions.csv",
        action_rows(all_actions),
        ["kind", "relative_path", "target_path", "size_bytes", "sha256", "reason", "status"],
    )
    write_csv(
        audit_dir / "deleted_cache_files.csv",
        action_rows(plan.cache_deletes),
        ["kind", "relative_path", "target_path", "size_bytes", "sha256", "reason", "status"],
    )
    write_csv(
        audit_dir / "deleted_smoke_sessions.csv",
        action_rows(plan.smoke_session_deletes),
        ["kind", "relative_path", "target_path", "size_bytes", "sha256", "reason", "status"],
    )
    write_csv(
        audit_dir / "quarantined_files.csv",
        action_rows(plan.quarantines),
        ["kind", "relative_path", "target_path", "size_bytes", "sha256", "reason", "status"],
    )
    write_csv(
        audit_dir / "manual_review_candidates.csv",
        action_rows(plan.manual_review),
        ["kind", "relative_path", "target_path", "size_bytes", "sha256", "reason", "status"],
    )
    (audit_dir / "cleanup_before.md").write_text(
        markdown_summary("Cleanup Before", before, before_state, before_checks, plan),
        encoding="utf-8",
    )
    (audit_dir / "cleanup_after.md").write_text(
        markdown_summary("Cleanup After", after, after_state, after_checks, plan),
        encoding="utf-8",
    )
    return {
        "audit_dir": str(audit_dir),
        "quarantine_dir": str(quarantine_root),
        "before": before,
        "after": after,
        "before_validation": before_state,
        "after_validation": after_state,
        "before_checks": before_checks,
        "after_checks": after_checks,
        "actions": {
            "cache_deletes": len(plan.cache_deletes),
            "smoke_session_deletes": len(plan.smoke_session_deletes),
            "quarantines": len(plan.quarantines),
            "empty_dir_deletes": len(plan.empty_dir_deletes),
            "manual_review": len(plan.manual_review),
        },
        "retained_smoke_audit": plan.retained_smoke_audit,
        "retained_general_sessions": plan.retained_general_sessions,
        "released_size_bytes_estimate": before["size_bytes"] - after["size_bytes"],
    }


def audit_summary(root: Path, plan: CleanupPlan, before: dict[str, Any], state: dict[str, Any], checks: dict[str, bool]) -> dict[str, Any]:
    protected_candidate_hits = []
    for action in plan.cache_deletes + plan.smoke_session_deletes + plan.quarantines + plan.empty_dir_deletes:
        rel = normalize_rel(Path(action.relative_path))
        if rel in PROTECTED_FILES or is_formal_output(rel) or is_under_any(rel, {Path("data/raw")}):
            protected_candidate_hits.append(action.relative_path)
    return {
        "mode": "audit-only",
        "note": "No files were written, moved, or deleted.",
        "root": str(root),
        "before": before,
        "validation_checks": checks,
        "validation_state": state,
        "candidate_counts": {
            "cache_deletes": len(plan.cache_deletes),
            "smoke_session_deletes": len(plan.smoke_session_deletes),
            "quarantines": len(plan.quarantines),
            "empty_dir_deletes": len(plan.empty_dir_deletes),
            "manual_review": len(plan.manual_review),
        },
        "candidate_size_bytes": {
            "cache_deletes": sum(action.size_bytes for action in plan.cache_deletes),
            "smoke_session_deletes": sum(action.size_bytes for action in plan.smoke_session_deletes),
            "quarantines": sum(action.size_bytes for action in plan.quarantines),
            "empty_dir_deletes": 0,
        },
        "retained_smoke_audit": plan.retained_smoke_audit,
        "retained_general_sessions": plan.retained_general_sessions,
        "protected_candidate_hits": protected_candidate_hits,
        "sample_cache_candidates": [action.relative_path for action in plan.cache_deletes[:20]],
        "sample_smoke_sessions": [action.relative_path for action in plan.smoke_session_deletes[:20]],
        "sample_quarantine_candidates": [action.relative_path for action in plan.quarantines[:30]],
        "manual_review_candidates": [action.relative_path for action in plan.manual_review[:30]],
    }


def main() -> int:
    args = parse_args()
    root = project_root()
    before = snapshot(root)
    before_state = validation_state(root)
    before_checks = validation_passes(before_state)
    plan = build_plan(root)
    ensure_safe_plan(root, plan)

    if not args.apply:
        print(json.dumps(audit_summary(root, plan, before, before_state, before_checks), ensure_ascii=False, indent=2))
        return 0

    result = apply_cleanup(root, plan, before, before_state, before_checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(result["after_checks"].values()):
        print("WARNING: Some post-cleanup validation checks are not PASS.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
