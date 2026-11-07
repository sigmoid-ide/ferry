"""The helper that runs on the machine: moving code over, checked, and rebuilding it exactly."""

import io
import json
import sys
import tarfile
import zlib

import pytest
from hunch.snapshot import store_object

from ferry import box


def stdin(monkeypatch, data: bytes):
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data), encoding="utf-8"))


def tar_of(objects, shas) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for sha in shas:
            tar.add(str(objects / sha[:2] / sha[2:]), arcname=f"{sha[:2]}/{sha[2:]}")
    return buf.getvalue()


def test_only_missing_objects_are_asked_for(tmp_path, monkeypatch, capsys):
    here, there = tmp_path / "here", tmp_path / "there"
    a = store_object(here, b"print('a')\n")
    b = store_object(here, b"print('b')\n")
    store_object(there, b"print('a')\n")
    stdin(monkeypatch, f"{a}\n{b}\nnot-a-sha\n".encode())
    box.cmd_missing(str(there))
    assert capsys.readouterr().out.split() == [b]


def test_unpacked_objects_rebuild_the_exact_files(tmp_path, monkeypatch, capsys):
    here, there, work = tmp_path / "here", tmp_path / "there", tmp_path / "work"
    files = {"train.py": b"print('hi')\r\n", "src/model.py": b"class M: pass\n"}
    manifest = {"files": {path: {"sha": store_object(here, data), "size": len(data)} for path, data in files.items()}}
    stdin(monkeypatch, tar_of(here, [e["sha"] for e in manifest["files"].values()]))
    box.cmd_unpack(str(there))
    (tmp_path / "files.json").write_text(json.dumps(manifest))
    box.cmd_checkout(str(there), str(tmp_path / "files.json"), str(work))
    assert (work / "train.py").read_bytes() == files["train.py"]
    assert (work / "src" / "model.py").read_bytes() == files["src/model.py"]
    capsys.readouterr()


def test_a_damaged_object_is_refused(tmp_path, monkeypatch):
    here = tmp_path / "here"
    sha = store_object(here, b"good\n")
    (here / sha[:2] / sha[2:]).write_bytes(zlib.compress(b"blob 4\0evil"))
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(tar_of(here, [sha]))))
    with pytest.raises(SystemExit, match="damaged"):
        box.cmd_unpack(str(tmp_path / "there"))


def test_unsafe_paths_in_a_snapshot_are_refused(tmp_path):
    here = tmp_path / "here"
    sha = store_object(here, b"x\n")
    (tmp_path / "files.json").write_text(json.dumps({"files": {"../escape.py": {"sha": sha, "size": 2}}}))
    with pytest.raises(SystemExit, match="unsafe"):
        box.cmd_checkout(str(here), str(tmp_path / "files.json"), str(tmp_path / "work"))
    assert not (tmp_path / "escape.py").exists()
