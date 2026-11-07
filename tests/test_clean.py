"""`ferry clean`: idle environments go with uv's cache, ended runs' outputs go only when
asked, and runs marked keep, runs going, and runs another journal knows always stay."""

import io
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest
from hunch.capture import start_run
from hunch.store import Store, append_jsonl, finish_run, timestamp

from ferry import box, launch
from ferry.remote import project_key


def make_env(root, name, days_ago):
    venv = root / "envs" / name
    (venv / "lib").mkdir(parents=True)
    (venv / ".ready").touch()
    used = time.time() - days_ago * 86400
    os.utime(venv / ".ready", (used, used))
    return venv


def test_only_files_nothing_else_links_to_count_as_freed(tmp_path):
    root = tmp_path / ".ferry"
    cache = root / "cache" / "uv"
    cache.mkdir(parents=True)
    idle, kept = make_env(root, "idle", 30), make_env(root, "kept", 1)
    (cache / "shared.whl").write_bytes(b"x" * 5000)
    os.link(cache / "shared.whl", idle / "lib" / "shared.py")
    os.link(cache / "shared.whl", kept / "lib" / "shared.py")  # still used by kept: not freed
    (cache / "only-idle.whl").write_bytes(b"y" * 7000)
    os.link(cache / "only-idle.whl", idle / "lib" / "only.py")
    freed = box._freed_by([str(idle), str(cache)])
    assert 7000 <= freed < 12000 + 8192  # only-idle, counted once, in whole disk blocks


def test_the_plan_names_idle_environments_and_ended_runs(tmp_path, monkeypatch, capsys):
    root = tmp_path / ".ferry"
    make_env(root, "py3.11-old", 30)
    make_env(root, "py3.11-new", 2)
    base = root / "projects" / "proj"
    for run_id, ended in (("r-ended", True), ("r-going", False)):
        (base / "work" / run_id).mkdir(parents=True)
        (base / "work" / run_id / "model.pt").write_bytes(b"w" * 3000)
        (base / ".hunch" / "runs" / run_id).mkdir(parents=True)
        if ended:
            (base / ".hunch" / "runs" / run_id / "end.json").write_text("{}")
    box.cmd_clean_plan(str(root), "proj")
    plan = json.loads(capsys.readouterr().out)
    assert plan["envs"] == ["py3.11-old"] and plan["kept"] == 1
    assert [run["id"] for run in plan["runs"]] == ["r-ended"] and plan["runs"][0]["size"] >= 3000


def test_cleaning_checks_again_and_removes_only_what_is_idle(tmp_path, monkeypatch, capsys):
    pytest.importorskip("fcntl")  # the machine side runs on Linux
    root = tmp_path / ".ferry"
    make_env(root, "py3.11-old", 30)
    make_env(root, "py3.11-new", 2)
    base = root / "projects" / "proj"
    (base / "work" / "r-ended").mkdir(parents=True)
    (base / ".hunch" / "runs" / "r-ended").mkdir(parents=True)
    (base / ".hunch" / "runs" / "r-ended" / "end.json").write_text("{}")
    (base / "work" / "r-going").mkdir(parents=True)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"envs": ["py3.11-old", "py3.11-new", "../.."],
                                                               "runs": ["r-ended", "r-going"]})))
    box.cmd_clean(str(root), "proj")
    done = json.loads(capsys.readouterr().out)
    assert done["envs"] == 1 and done["runs"] == 1
    assert done["cache"] is False  # no uv here to clear it
    assert sorted(p.name for p in (root / "envs").iterdir() if p.is_dir()) == ["py3.11-new"]
    assert [p.name for p in (base / "work").iterdir()] == ["r-going"]


class CleanRemote:
    def __init__(self, plan):
        self.plan = plan
        self.removed = None
        self.box = SimpleNamespace(close=lambda: None)

    def clean_plan(self, project):
        self.project = project
        return self.plan

    def clean(self, project, envs, runs):
        self.removed = (envs, runs)
        return {"envs": len(envs), "runs": len(runs), "cache": True, "freed": 6 * 1024 ** 3}


def test_outputs_go_only_when_asked_and_never_for_runs_marked_keep(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path)
    ids = {}
    for why in ("best so far", "a dud"):
        run_dir = start_run(store, command=["python", "train.py"], cwd=tmp_path, why=why, parent=None, source="wrapper", python=None)
        finish_run(run_dir, status="finished", exit_code=0, duration=60.0)
        ids[why] = run_dir.name
    append_jsonl(tmp_path / ".hunch" / "runs" / ids["best so far"] / "notes.jsonl", {"at": timestamp(), "kind": "verdict", "value": "keep"})
    plan = {"envs": ["py3.11-old"], "kept": 1, "freed": 5 * 1024 ** 3, "free": 40 * 1024 ** 3,
            "runs": [{"id": ids["best so far"], "size": 2 * 1024 ** 3}, {"id": ids["a dud"], "size": 1024 ** 3},
                     {"id": "20261101-0000-another", "size": 1024 ** 3}]}

    remote = CleanRemote(plan)
    monkeypatch.setattr(launch, "connect", lambda name: remote)
    launch.clean("box", outputs=False, yes=True)
    out = capsys.readouterr().out
    assert remote.project == project_key(tmp_path)
    assert remote.removed == (["py3.11-old"], [])
    assert "frees about 5.0 GB" in out and "keep 1.0 GB there" in out

    remote = CleanRemote(plan)
    monkeypatch.setattr(launch, "connect", lambda name: remote)
    launch.clean("box", outputs=True, yes=True)
    assert remote.removed == (["py3.11-old"], [ids["a dud"]])  # not the kept run, not another journal's
    assert "a dud" in capsys.readouterr().out
