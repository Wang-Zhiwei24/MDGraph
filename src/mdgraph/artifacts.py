from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def atomic_json(path: Path, data) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_frozen_task(root: Path) -> tuple[dict, dict]:
    """Load the exact task bytes that match the run manifest."""
    manifest = read_json(root / "run_manifest.json")
    payload = (root / "task.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["task_sha256"]:
        raise ValueError("Frozen task configuration changed; create a new run")
    return manifest, json.loads(payload.decode("utf-8"))


def verify_files(folder: Path, files: dict[str, str]) -> None:
    for name, digest in files.items():
        if Path(name).name != name or sha256(folder / name) != digest:
            raise ValueError(f"Prepared artifact changed: {name}")
