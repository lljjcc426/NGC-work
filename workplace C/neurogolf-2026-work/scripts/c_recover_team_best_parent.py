from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path


COMPETITION = "neurogolf-2026"
DEFAULT_KAGGLEGOLF = Path("E:/kagglegolf")
DEFAULT_TEAM_ID = "16252365"
KNOWN_KERNEL_SLUGS = [
    "muelsyse111/neurogolf-submit-current",
    "muelsyse111/neurogolf-7113-franksunp-payload-submit",
    "muelsyse111/neurogolf-submit-prvsiyan-7266-72-repro",
]


def run(args: list[str], cwd: Path, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kaggle", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def onnx_count(zip_path: Path) -> int:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            return sum(1 for name in archive.namelist() if name.endswith(".onnx"))
    except zipfile.BadZipFile:
        return 0


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(out_dir)


def best_submission(cwd: Path, page_size: int) -> tuple[dict, str]:
    result = run(
        ["competitions", "submissions", COMPETITION, "--page-size", str(page_size), "--format", "json"],
        cwd,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    rows = json.loads(result.stdout or "[]")
    complete = [row for row in rows if "COMPLETE" in str(row.get("status", "")) and row.get("publicScore")]
    if not complete:
        raise RuntimeError("no complete scored submissions returned by Kaggle CLI")
    complete.sort(key=lambda row: float(row["publicScore"]), reverse=True)
    return complete[0], result.stdout


def team_submission(team_id: str, cwd: Path) -> tuple[list[dict], str]:
    result = run(["competitions", "team-submissions", team_id, "--format", "json"], cwd, timeout=120)
    if result.returncode != 0:
        return [], result.stderr or result.stdout
    return json.loads(result.stdout or "[]"), result.stdout


def download_kernel_outputs(cwd: Path, target_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for slug in KNOWN_KERNEL_SLUGS:
        safe_slug = slug.replace("/", "__")
        out = target_dir / safe_slug
        out.mkdir(parents=True, exist_ok=True)
        result = run(
            ["kernels", "output", slug, "-p", str(out), "-o", "--file-pattern", "submission.zip"],
            cwd,
            timeout=180,
        )
        zip_path = out / "submission.zip"
        rows.append(
            {
                "source": f"kaggle kernels output {slug}",
                "path": str(zip_path) if zip_path.exists() else "",
                "returncode": result.returncode,
                "sha256": sha256(zip_path) if zip_path.exists() else "",
                "onnx_count": onnx_count(zip_path) if zip_path.exists() else 0,
                "stdout_tail": (result.stdout or "")[-500:],
                "stderr_tail": (result.stderr or "")[-500:],
            }
        )
    return rows


def scan_local_zips(root: Path, wanted_prefix: str) -> list[dict]:
    rows: list[dict] = []
    for zip_path in root.rglob("submission.zip"):
        try:
            digest = sha256(zip_path)
        except OSError:
            continue
        if wanted_prefix and not digest.startswith(wanted_prefix):
            continue
        rows.append(
            {
                "source": "local_scan",
                "path": str(zip_path),
                "sha256": digest,
                "onnx_count": onnx_count(zip_path),
            }
        )
    return rows


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# C Team Best Parent Recovery",
        "",
        f"captured_at: {payload['captured_at']}",
        f"competition: {payload['competition']}",
        f"team_id: {payload['team_id']}",
        f"best_ref: {payload['best_submission'].get('ref')}",
        f"best_score: {payload['best_submission'].get('publicScore')}",
        f"best_description: {payload['best_submission'].get('description')}",
        f"wanted_sha_prefix: {payload['wanted_sha_prefix']}",
        f"ready_for_rebase: {str(payload['ready_for_rebase']).lower()}",
        f"recovered_zip: {payload.get('recovered_zip', '')}",
        f"recovered_onnx_dir: {payload.get('recovered_onnx_dir', '')}",
        "",
        "## Team CLI",
        "",
        "```json",
        json.dumps(payload["team_submissions"], indent=2),
        "```",
        "",
        "## Attempts",
        "",
        "| source | onnx_count | sha256 | path |",
        "| --- | ---: | --- | --- |",
    ]
    for row in payload["attempts"]:
        lines.append(
            f"| {row.get('source', '')} | {row.get('onnx_count', '')} | "
            f"{row.get('sha256', '')} | `{row.get('path', '')}` |"
        )
    if payload.get("blocker"):
        lines.extend(["", "## Blocker", "", payload["blocker"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kagglegolf-root", default=str(DEFAULT_KAGGLEGOLF))
    parser.add_argument("--team-id", default=DEFAULT_TEAM_ID)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--download-kernel-outputs", action="store_true")
    args = parser.parse_args()

    root = Path(args.kagglegolf_root)
    out_root = root / "submissions" / "downloaded_best" / "team_best_recovery"
    out_root.mkdir(parents=True, exist_ok=True)
    best, raw_history = best_submission(root, args.page_size)
    team_rows, raw_team = team_submission(args.team_id, root)
    description = str(best.get("description", ""))
    wanted_prefix = ""
    if " sha " in description:
        wanted_prefix = description.rsplit(" sha ", 1)[1].split()[0].strip()

    attempts: list[dict] = []
    if args.download_kernel_outputs:
        attempts.extend(download_kernel_outputs(root, out_root / "kernel_outputs"))
    attempts.extend(scan_local_zips(root, wanted_prefix))

    recovered = next(
        (
            row
            for row in attempts
            if row.get("sha256", "").startswith(wanted_prefix)
            and int(row.get("onnx_count") or 0) == 400
        ),
        None,
    )
    recovered_dir = ""
    if recovered:
        recovered_dir = str(out_root / "onnx")
        extract_zip(Path(recovered["path"]), Path(recovered_dir))

    payload = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "competition": COMPETITION,
        "team_id": args.team_id,
        "best_submission": best,
        "team_submissions": team_rows,
        "wanted_sha_prefix": wanted_prefix,
        "ready_for_rebase": bool(recovered),
        "recovered_zip": recovered.get("path", "") if recovered else "",
        "recovered_onnx_dir": recovered_dir,
        "attempts": attempts,
        "blocker": ""
        if recovered
        else (
            "Kaggle CLI exposes the team best score and submission id, but does not expose "
            "a competition-submission-file download command. No local submission.zip matched "
            f"the best description SHA prefix `{wanted_prefix}`."
        ),
        "raw_history": raw_history,
        "raw_team": raw_team,
    }
    json_path = out_root / "recovery.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = Path(__file__).resolve().parents[2] / "score_docs" / "32_TEAM_BEST_REBASE_RECOVERY.md"
    write_report(report_path, payload)
    print(json_path)
    print(report_path)
    print("ready_for_rebase", str(payload["ready_for_rebase"]).lower())
    if not payload["ready_for_rebase"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
