"""Detached local worker: executes argv without a shell and persists the exit result."""

import os
import signal
import subprocess
import sys
from pathlib import Path

from .artifacts import atomic_json, read_json


def main():
    folder = Path(sys.argv[1])
    spec = read_json(folder / "worker.json")
    result = {}
    try:
        with (folder / "runner.log").open("wb") as f:
            proc = subprocess.Popen(
                spec["command"],
                cwd=folder,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name != "nt"),
            )
            try:
                result["returncode"] = proc.wait(timeout=spec["timeout"])
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True
                    )
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                result = {"returncode": 124, "reason": "Local task timeout"}
    except Exception as e:
        result = {"returncode": 127, "reason": str(e)}
    atomic_json(folder / "exit.json", result)


if __name__ == "__main__":
    main()
