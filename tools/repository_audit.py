#!/usr/bin/env python3
"""Read-only structural audit for the NGC-work competition archive."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {
    ".csv",
    ".html",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}
CORE_MARKDOWN = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "repository-guide.md",
    ROOT / "docs" / "postmortem" / "2026-neurogolf-retrospective.md",
    *[ROOT / f"workplace {owner}" / "readme.md" for owner in "ABCDEF"],
]
REQUIRED = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "assignments" / "task_assignment_400.csv",
    ROOT / "assignments" / "task_assignment_summary.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "repository-guide.md",
    ROOT / "docs" / "workplace-index.csv",
    ROOT / "docs" / "evidence" / "final-results-20260716.json",
    ROOT / "docs" / "postmortem" / "2026-neurogolf-retrospective.md",
]
FORBIDDEN_CREDENTIAL_NAMES = {
    ".env",
    "access_token",
    "kaggle.json",
}
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def audit_assignments(errors: list[str]) -> None:
    path = ROOT / "assignments" / "task_assignment_400.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    primaries = [row for row in rows if row["assignment_type"] == "primary"]
    primary_tasks = [row["task"] for row in primaries]
    slots = Counter(row["owner"] for row in rows)

    if len(rows) != 402:
        errors.append(f"assignment slots: expected 402, found {len(rows)}")
    if len(primaries) != 400 or len(set(primary_tasks)) != 400:
        errors.append(
            "primary assignments: expected 400 rows and 400 unique tasks, "
            f"found {len(primaries)} rows and {len(set(primary_tasks))} unique"
        )
    expected_tasks = {f"task{i:03d}" for i in range(1, 401)}
    if set(primary_tasks) != expected_tasks:
        errors.append("primary task IDs do not exactly cover task001..task400")
    if slots != Counter({owner: 67 for owner in "ABCDEF"}):
        errors.append(f"owner slot counts differ from 67 each: {dict(slots)}")


def audit_utf8(files: list[Path], errors: list[str]) -> None:
    for path in files:
        if path.suffix.lower() not in TEXT_EXTENSIONS or not path.exists():
            continue
        try:
            path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            errors.append(f"non-UTF-8 tracked text: {path.relative_to(ROOT)} ({exc})")


def audit_markdown_links(errors: list[str]) -> None:
    for path in CORE_MARKDOWN:
        text = path.read_text(encoding="utf-8-sig")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target or re.match(r"^[A-Za-z]:[\\/]", target):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(
                    f"broken relative link in {path.relative_to(ROOT)}: {raw_target}"
                )


def audit_credentials(files: list[Path], errors: list[str]) -> None:
    for path in files:
        name = path.name.lower()
        if name in FORBIDDEN_CREDENTIAL_NAMES or "access_token" in name:
            errors.append(f"tracked credential filename: {path.relative_to(ROOT)}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    errors: list[str] = []
    warnings: list[str] = []

    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing required path: {path.relative_to(ROOT)}")

    files = tracked_files()
    audit_assignments(errors)
    audit_utf8(files, errors)
    audit_markdown_links(errors)
    audit_credentials(files, errors)

    binary_counts = Counter(path.suffix.lower() for path in files)
    for suffix in (".onnx", ".zip"):
        if binary_counts[suffix]:
            warnings.append(
                f"historical tracked {suffix} files: {binary_counts[suffix]} "
                "(preserved; new generated binaries are ignored)"
            )

    absolute_path_hits = 0
    drive_pattern = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
    for path in files:
        if path.suffix.lower() not in {".md", ".py", ".ps1"}:
            continue
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig")
        absolute_path_hits += len(drive_pattern.findall(text))
    if absolute_path_hits:
        warnings.append(
            f"historical Windows absolute-path references: {absolute_path_hits}; "
            "new code should use repository-relative paths or environment variables"
        )

    print("NGC-work repository audit")
    print(f"tracked files: {len(files)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"FAILED: {len(errors)} error(s)")
        return 1
    print("PASS: repository structure, assignment invariants, UTF-8, links, credentials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
