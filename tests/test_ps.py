"""`ferry ps` asks each machine how its runs are, and `ferry forget` lets a run whose
machine is gone show as lost."""

import io
import json
import sys
from types import SimpleNamespace

import pytest
from hunch.capture import start_run
from hunch.store import Store

from ferry import box, launch, mirror
from ferry.config import FerryError


def away_run(store, why, machine="box"):
    run_dir = start_run(store, command=["python", "train.py"], cwd=store.root, why=why, parent=None, source="wrapper", python=None)
    mirror.write_state(run_dir, {"machine": machine, "dir": f"/box/runs/{run_dir.name}", "pidfile": f"/box/pids/{run_dir.name}"})
    return run_dir


class StatesRemote:
    def __init__(self, answers):
        self.answers = answers
        self.box = SimpleNamespace(close=lambda: None)

    def states(self, runs):
        return [self.answers[run_dir.rsplit("/", 1)[1]] for run_dir, _ in runs]


def test_ps_says_how_each_run_is_and_what_to_do(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path)
    runs = {why: away_run(store, why) for why in ("going", "done", "died", "wiped")}
    lost = away_run(store, "far away", machine="oldbox")
    answers = {runs["going"].name: "running", runs["done"].name: "ended", runs["died"].name: "gone", runs["wiped"].name: "missing"}

    def connect(name, wake=True):
        assert wake is False  # ps never starts a rented machine that stopped itself
        if name == "oldbox":
            raise FerryError("could not reach oldbox (10.0.0.9:22): timed out")
        return StatesRemote(answers)

    monkeypatch.setattr(launch, "connect", connect)
    assert launch.ps() == 0
    out = capsys.readouterr().out
    short = {why: run_dir.name[-4:] for why, run_dir in runs.items()}
    assert f"{short['going']}  running on box  going\n      ferry attach {short['going']} copies it home" in out
    assert f"{short['done']}  ended on box  done\n      ferry attach {short['done']} brings the rest home" in out
    assert f"{short['died']}  stopped on box without recording an end  died" in out
    assert f"ferry forget {short['wiped']} marks it lost" in out
    assert f"{lost.name[-4:]}  on oldbox, which cannot be reached (could not reach oldbox (10.0.0.9:22): timed out)" in out


def test_a_forgotten_run_shows_as_lost_until_it_is_copied_again(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path)
    run_dir = away_run(store, "on a box that burned down")
    (run_dir / "heartbeat").unlink(missing_ok=True)
    assert store.resolve(run_dir.name).status == "running"  # remote: this computer may just have slept

    launch.forget(run_dir.name[-4:])
    assert Store(tmp_path).resolve(run_dir.name).status == "lost"
    with pytest.raises(FerryError, match="nothing to forget"):
        launch.forget(run_dir.name[-4:])

    state = mirror.read_state(run_dir)  # `ferry attach` still knows where it went
    assert state["machine"] == "box"
    mirror.write_state(run_dir, state)  # and its first copy puts it back
    assert not (run_dir / mirror.GONE_FILE).exists()
    assert Store(tmp_path).resolve(run_dir.name).status == "running"


def test_the_machine_tells_running_ended_gone_and_missing_runs_apart(tmp_path, monkeypatch, capsys):
    ended, gone = tmp_path / "ended", tmp_path / "gone"
    ended.mkdir()
    gone.mkdir()
    (ended / "end.json").write_text("{}")
    pairs = [[str(ended), str(tmp_path / "p1")], [str(gone), str(tmp_path / "p2")], [str(tmp_path / "wiped"), str(tmp_path / "p3")]]
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(pairs)))
    box.cmd_states()
    assert json.loads(capsys.readouterr().out) == ["ended", "gone", "missing"]
