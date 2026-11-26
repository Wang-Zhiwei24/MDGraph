import io

import pytest

from mdgraph.backends import AmbiguousSubmission, SSHBackend
from mdgraph.config import Execution


class RemoteFiles:
    """In-memory SSH contract fixture; does not submit any real cluster jobs."""

    def __init__(self, files):
        self.files = files
        self.uploads = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def open(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return io.BytesIO(self.files[path])

    def stat(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return object()

    def put(self, source, dest):
        self.uploads.append(dest)


class Client:
    def __init__(self, ftp):
        self.ftp = ftp

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def open_sftp(self):
        return self.ftp


def backend():
    return SSHBackend(
        Execution(
            backend="ssh", scheduler="slurm", host="test", username="test", remote_root="/work"
        )
    )


def test_remote_receipt_recovered_after_controller_crash(tmp_path, monkeypatch):
    folder = tmp_path / "attempt_00"
    folder.mkdir()
    ftp = RemoteFiles({"/work/run123/attempt_00/.job_id": b"12345;cluster\n"})
    engine = backend()
    monkeypatch.setattr(engine, "connect", lambda: Client(ftp))
    calls = []

    def command(client, text):
        calls.append(text)
        return 0, "", ""

    monkeypatch.setattr(engine, "command", command)
    receipt = engine.submit(folder, "run123")
    assert receipt["job_id"] == "12345"
    assert not ftp.uploads
    assert not any("sbatch" in text for text in calls)


def test_remote_ambiguous_submission_never_repeated(tmp_path, monkeypatch):
    folder = tmp_path / "attempt_00"
    folder.mkdir()
    ftp = RemoteFiles({"/work/run123/attempt_00/.submit_lock": b""})
    engine = backend()
    monkeypatch.setattr(engine, "connect", lambda: Client(ftp))
    monkeypatch.setattr(engine, "command", lambda *args: (0, "", ""))
    with pytest.raises(AmbiguousSubmission):
        engine.submit(folder, "run123")
    assert not ftp.uploads


@pytest.mark.parametrize(
    "scheduler_state,retryable",
    [("NODE_FAIL", True), ("TIMEOUT", False), ("OUT_OF_MEMORY", False), ("COMPLETED", False)],
)
def test_slurm_terminal_classification(tmp_path, monkeypatch, scheduler_state, retryable):
    engine = backend()
    monkeypatch.setattr(engine, "connect", lambda: Client(None))

    def command(client, text):
        if text.startswith("cat"):
            return 1, "", "not found"
        if text.startswith("squeue"):
            return 0, "", ""
        return 0, scheduler_state + "|\n", ""

    monkeypatch.setattr(engine, "command", command)
    result = engine.poll(tmp_path, {"job_id": "123", "remote_dir": "/work/run"})
    assert result["status"] == "failed" and result["retryable"] is retryable


def test_failed_program_exit_not_success(tmp_path, monkeypatch):
    engine = backend()
    monkeypatch.setattr(engine, "connect", lambda: Client(None))
    monkeypatch.setattr(engine, "command", lambda *args: (0, "1", ""))
    assert engine.poll(tmp_path, {"job_id": "123", "remote_dir": "/work/run"})["status"] == "failed"
