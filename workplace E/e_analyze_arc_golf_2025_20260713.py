from __future__ import annotations

import argparse
import ast
import csv
import hashlib
from pathlib import Path
import zlib


def expanded_source(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    text = raw.decode("latin1")
    try:
        tree = ast.parse(text)
        decompress = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "decompress"
        )
        bytes_call = decompress.args[0]
        payload = ast.literal_eval(bytes_call.args[0]).encode("latin1")
        wbits = ast.literal_eval(decompress.args[1]) if len(decompress.args) > 1 else zlib.MAX_WBITS
        return zlib.decompress(payload, wbits).decode("latin1"), "zlib_expanded"
    except (StopIteration, AttributeError, TypeError, ValueError, SyntaxError, zlib.error):
        return text, "plain"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-csv", type=Path, required=True)
    parser.add_argument("--assignment-csv", type=Path, required=True)
    parser.add_argument("--solutions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.assignment_csv.open(newline="", encoding="utf-8") as handle:
        assigned = {
            int(row["task"].replace("task", ""))
            for row in csv.DictReader(handle)
            if row["owner"] == "E"
        }
    with args.score_csv.open(newline="", encoding="utf-8") as handle:
        scores = {
            int(row["task"]): row
            for row in csv.DictReader(handle)
            if int(row["task"]) in assigned
        }

    rows: list[dict[str, object]] = []
    for task in sorted(assigned):
        solution = args.solutions_dir / f"task{task:03d}.py"
        source, source_kind = expanded_source(solution)
        compact = " ".join(source.split())
        score = scores.get(task, {})
        rows.append(
            {
                "task": task,
                "cost": score.get("cost", ""),
                "points": score.get("points", ""),
                "source_chars": len(source),
                "source_lines": len(source.splitlines()),
                "source_kind": source_kind,
                "source_sha256": hashlib.sha256(solution.read_bytes()).hexdigest(),
                "source_path": str(solution),
                "preview": compact[:240],
            }
        )

    rows.sort(key=lambda row: (int(row["source_chars"]), -int(row["cost"] or 0)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    for row in rows[:20]:
        print(
            f"task{int(row['task']):03d}: source_chars={row['source_chars']}, "
            f"cost={row['cost']}, preview={row['preview']}"
        )


if __name__ == "__main__":
    main()
