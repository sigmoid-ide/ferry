"""Data sent once and linked into every run; files a run wrote, listed and pulled home."""

import io
import json
import os
import shutil
import sys
from types import SimpleNamespace

import paramiko
import pytest

from ferry import box, data
from ferry.config import FerryError


def can_symlink(tmp_path) -> bool:
    try:
        os.symlink(tmp_path, tmp_path / "probe")
    except OSError:
        return False
    os.unlink(tmp_path / "probe")
    return True


def test_data_already_on_the_machine_is_not_sent_again(tmp_path, monkeypatch, capsys):
    shared = tmp_path / "shared"
    (shared / "data").mkdir(parents=True)
    (shared / "data" / "same.bin").write_bytes(b"1234")
    os.utime(shared / "data" / "same.bin", (1000, 1000))
    (shared / "data" / "edited.bin").write_bytes(b"1234")
    os.utime(shared / "data" / "edited.bin", (1000, 1000))
    entries = [["data/same.bin", 4, 1000.4], ["data/edited.bin", 4, 2000], ["data/new.bin", 1, 1000]]
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(entries)))
    box.cmd_want(str(shared))
    assert json.loads(capsys.readouterr().out) == ["data/edited.bin", "data/new.bin"]
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps([["../outside", 1, 1]])))
    with pytest.raises(SystemExit, match="unsafe"):
        box.cmd_want(str(shared))


def test_shared_data_appears_in_each_run_at_the_same_path_and_the_code_wins(tmp_path):
    if not can_symlink(tmp_path):
        pytest.skip("this system does not allow symlinks")
    shared, work = tmp_path / "shared", tmp_path / "work"
    (shared / "data").mkdir(parents=True)
    (shared / "data" / "train.bin").write_bytes(b"data")
    (shared / "weights.pt").write_bytes(b"w")
    (shared / "train.py").write_text("print('stale copy')")
    (shared / "big.bin.ferry-part").write_bytes(b"half")
    (work / "data").mkdir(parents=True)
    (work / "data" / "prepare.py").write_text("# code")
    (work / "train.py").write_text("print('the code')")
    box._link_shared(str(shared), str(work))
    assert (work / "data" / "train.bin").read_bytes() == b"data"  # into a folder the code also has
    assert (work / "weights.pt").is_symlink()
    assert (work / "train.py").read_text() == "print('the code')"
    assert not (work / "big.bin.ferry-part").exists()


def test_outputs_are_what_the_run_wrote(tmp_path, capsys):
    work = tmp_path / "work"
    (work / "checkpoints").mkdir(parents=True)
    (work / "__pycache__").mkdir()
    (work / "train.py").write_text("code")
    (work / "checkpoints" / "best.pt").write_bytes(b"x" * 10)
    (work / "samples.png").write_bytes(b"png")
    (work / "__pycache__" / "train.cpython-311.pyc").write_bytes(b"pyc")
    if can_symlink(tmp_path):
        os.symlink(tmp_path, work / "data")
    (tmp_path / "files.json").write_text(json.dumps({"files": {"train.py": {"sha": "0" * 40, "size": 4}}}))
    box.cmd_outputs(str(work), str(tmp_path / "files.json"))
    outputs = json.loads(capsys.readouterr().out)
    assert outputs == [["samples.png", 3], ["checkpoints/best.pt", 10]]
    assert data.summary(outputs) == ["checkpoints/ (1 file, 10 B)", "samples.png (3 B)"]
    assert data.summary(outputs, "checkpoints") == ["best.pt (10 B)", "../samples.png (3 B)"]  # as seen from there


def test_only_data_inside_the_project_is_collected(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "data" / ".git").mkdir(parents=True)
    (project / "data" / "env").mkdir()
    (project / "data" / "env" / "pyvenv.cfg").write_text("")
    (project / "data" / "train.bin").write_bytes(b"1")
    (project / "data" / ".git" / "HEAD").write_text("ref")
    (tmp_path / "elsewhere.bin").write_bytes(b"2")
    monkeypatch.chdir(project)
    assert [f.rel for f in data.collect(project, ["data"])] == ["data/train.bin"]
    with pytest.raises(FerryError, match="inside the project"):
        data.collect(project, ["../elsewhere.bin"])
    with pytest.raises(FerryError, match="not the whole project"):
        data.collect(project, ["."])
    with pytest.raises(FerryError, match="no such file"):
        data.collect(project, ["missing"])


class LocalSFTP:
    """The part of paramiko's SFTP client that pulling uses, over a local folder."""

    def stat(self, path):
        return paramiko.SFTPAttributes.from_stat(os.stat(path))

    def listdir_attr(self, path):
        return [paramiko.SFTPAttributes.from_stat(os.lstat(os.path.join(path, n)), n) for n in os.listdir(path)]

    def get(self, remote, local, callback=None):
        shutil.copyfile(remote, local)


def test_pull_brings_files_home_by_name_and_never_overwrites_by_accident(tmp_path):
    work, home = tmp_path / "work", tmp_path / "home"
    (work / "checkpoints").mkdir(parents=True)
    (work / "checkpoints" / "best.pt").write_bytes(b"best")
    (work / "checkpoints" / "last.pt").write_bytes(b"last")
    (work / "samples.png").write_bytes(b"png")
    if can_symlink(tmp_path):
        os.symlink(tmp_path, work / "checkpoints" / "data")
    remote = SimpleNamespace(box=SimpleNamespace(sftp=LocalSFTP, label="box"))
    assert data.pull(remote, str(work), ["checkpoints", "samples.png"], home, force=False) == 3
    assert (home / "checkpoints" / "best.pt").read_bytes() == b"best"
    assert (home / "samples.png").read_bytes() == b"png"
    assert not (home / "checkpoints" / "data").exists()  # linked data is not the run's to give back
    with pytest.raises(FerryError, match="already there"):
        data.pull(remote, str(work), ["samples.png"], home, force=False)
    assert data.pull(remote, str(work), ["samples.png"], home, force=True) == 1
    with pytest.raises(FerryError, match="inside the run's folder"):
        data.pull(remote, str(work), ["../secrets"], home, force=False)
    with pytest.raises(FerryError, match="not in the run's folder"):
        data.pull(remote, str(work), ["nothing.pt"], home, force=False)
    assert data.pull(remote, str(work), ["best.pt"], tmp_path / "from-there", force=False, where="checkpoints") == 1
    assert (tmp_path / "from-there" / "best.pt").read_bytes() == b"best"
