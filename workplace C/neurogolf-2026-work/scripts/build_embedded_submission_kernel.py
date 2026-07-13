from __future__ import annotations

import argparse
import base64
import hashlib
import json
import textwrap
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained Kaggle notebook around a submission ZIP."
    )
    parser.add_argument("--submission-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--changed-tasks", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = args.submission_zip.read_bytes()
    package_sha = hashlib.sha256(payload).hexdigest()
    with zipfile.ZipFile(args.submission_zip) as archive:
        names = archive.namelist()
    root_onnx = [
        name for name in names if "/" not in name and name.endswith(".onnx")
    ]
    if len(names) != 400 or len(root_onnx) != 400:
        raise RuntimeError(
            f"expected 400 root ONNX files, got files={len(names)} root={len(root_onnx)}"
        )

    encoded = base64.b64encode(payload).decode("ascii")
    chunks = textwrap.wrap(encoded, 100)
    source = [
        "from pathlib import Path\n",
        "import base64\n",
        "import hashlib\n",
        "import json\n",
        "import zipfile\n",
        "\n",
        f"EXP_ID = {args.exp_id!r}\n",
        f"SOURCE_ID = {args.source_id!r}\n",
        f"CHANGED_TASKS = {args.changed_tasks!r}\n",
        f"EXPECTED_SHA256 = {package_sha!r}\n",
        "PAYLOAD_B64 = (\n",
    ]
    source.extend(f"    {chunk!r}\n" for chunk in chunks)
    source.extend(
        [
            ")\n",
            "out = Path('/kaggle/working/submission.zip')\n",
            "payload = base64.b64decode(PAYLOAD_B64, validate=True)\n",
            "if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:\n",
            "    raise RuntimeError('embedded package SHA256 mismatch')\n",
            "out.write_bytes(payload)\n",
            "with zipfile.ZipFile(out) as archive:\n",
            "    names = archive.namelist()\n",
            "    root_onnx = [name for name in names if '/' not in name and name.endswith('.onnx')]\n",
            "if len(names) != 400 or len(root_onnx) != 400:\n",
            "    raise RuntimeError(f'Expected 400 root ONNX files, got {len(names)} / {len(root_onnx)}')\n",
            "print(json.dumps({'exp_id': EXP_ID, 'source_id': SOURCE_ID, 'changed_tasks': CHANGED_TASKS, 'package_sha256': EXPECTED_SHA256, 'file_count': len(root_onnx)}, indent=2))\n",
            "print('submission.zip is ready at', out)\n",
        ]
    )

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# NeuroGolf submit\n", f"exp_id: `{args.exp_id}`\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": args.kernel_id,
        "title": args.kernel_id.split("/", 1)[-1],
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "competition_sources": ["neurogolf-2026"],
        "dataset_sources": [],
        "model_sources": [],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "notebook.ipynb").write_text(
        json.dumps(notebook, indent=2), encoding="utf-8"
    )
    (args.output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "kernel_id": args.kernel_id,
                "package_sha256": package_sha,
                "root_onnx_count": len(root_onnx),
                "notebook_bytes": (args.output_dir / "notebook.ipynb").stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
