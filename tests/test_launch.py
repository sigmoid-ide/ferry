"""Launching: a launch cut short must not leave a run looking alive, and what the run is
given on the machine (variables, GPUs) is checked here first."""

import json
import subprocess
from types import SimpleNamespace

import pytest
from hunch.store import Store

from ferry import box, launch
from ferry.config import FerryError
from ferry.remote import RunPaths


class FakeRemote:
    def __init__(self, fail_with):
        self.fail_with = fail_with
        self.box = SimpleNamespace(label="box", close=lambda: None)

    def gpus(self):
        return []

    def install_uv(self):
        pass

    def build_env(self, spec, on_line):
        return "/box/envs/py3.11-x"

    def shared(self, project):
        return f"/box/projects/{project}/shared"

    def paths(self, project, run_id):
        return RunPaths(project, "/box/objects", f"/box/runs/{run_id}", f"/box/work/{run_id}", "/box/shared", f"/box/pids/{run_id}")

    def missing(self, objects, shas):
        return shas

    def send_objects(self, objects, local, shas):
        raise self.fail_with


@pytest.mark.parametrize("error, status", [(KeyboardInterrupt(), "killed"), (OSError("connection reset"), "crashed")])
def test_a_launch_cut_short_is_recorded_as_ended(tmp_path, monkeypatch, error, status):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('hi')\n")
    monkeypatch.setattr(launch, "connect", lambda name: FakeRemote(error))
    monkeypatch.setattr(launch, "capture_env", lambda python: {"python": "3.11.9", "packages": {"numpy": "2.1.0"}})
    with pytest.raises(type(error)):
        launch.run("box", ["python", "train.py"], why="test", parent_ref=None)
    (run,) = Store(tmp_path).runs()
    assert run.status == status
    assert not (run.dir / "remote.json").exists()
    assert run.notes[-1]["kind"] == "ferry"


def test_variables_for_the_run_come_from_here_or_the_command_line(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert launch.run_env(["HF_TOKEN", "WANDB_MODE=offline", "EMPTY="]) == {
        "HF_TOKEN": "hf_secret", "WANDB_MODE": "offline", "EMPTY": ""}
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(FerryError, match="HF_TOKEN is not set here"):
        launch.run_env(["HF_TOKEN"])
    with pytest.raises(FerryError, match="Ferry sets PATH") as refused:
        launch.run_env(["PATH=/tmp/mine"])
    assert "/tmp/mine" not in str(refused.value)
    with pytest.raises(FerryError) as bad:
        launch.run_env(["MY-TOKEN=hf_secret"])
    assert "hf_secret" not in str(bad.value)  # errors never show a value


def test_gpus_are_numbered_as_nvidia_smi_shows_them():
    assert launch.run_env([], "1") == {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    with pytest.raises(FerryError, match="such as 1 or 0,1"):
        launch.run_env([], "0, 1")
    with pytest.raises(FerryError, match="not both"):
        launch.run_env(["CUDA_VISIBLE_DEVICES=0"], "1")


GPUS = [{"index": 0, "name": "RTX 3090", "memory": 24576, "busy": 11264},
        {"index": 1, "name": "RTX 3090", "memory": 24576, "busy": 0}]


def test_a_gpu_the_machine_lacks_is_refused_and_a_busy_one_is_pointed_out(capsys):
    with pytest.raises(FerryError, match=r"no GPU 2. Its GPUs: 0 \(RTX 3090\), 1"):
        launch.check_gpus("box", GPUS, "2")  # CUDA would quietly run it on the CPU
    with pytest.raises(FerryError, match="no NVIDIA GPU"):
        launch.check_gpus("box", [], "0")
    launch.check_gpus("box", [], None)  # no GPU and none asked for: fine

    launch.check_gpus("box", GPUS, None)
    assert "GPU 0 on box is already in use by another program (11.0 of 24.0 GB); GPU 1 is free: --gpus 1" in capsys.readouterr().err
    launch.check_gpus("box", GPUS, "1")
    launch.check_gpus("box", GPUS, "GPU-5f1c")  # ids are left to CUDA
    assert capsys.readouterr().err == ""


def test_the_machine_reports_its_gpus_and_what_other_programs_use(monkeypatch, capsys):
    answers = {
        "--query-gpu=index,uuid,memory.total,name": "0, GPU-a, 24576, NVIDIA GeForce RTX 3090\n1, GPU-b, 24576, NVIDIA GeForce RTX 3090\n",
        "--query-compute-apps=gpu_uuid,used_memory": "GPU-a, 9000\nGPU-a, 2264\nGPU-b, [N/A]\n",
    }
    monkeypatch.setattr(box.subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 0, answers[args[1]].encode()))
    box.cmd_gpus()
    assert json.loads(capsys.readouterr().out) == [
        {"index": 0, "name": "NVIDIA GeForce RTX 3090", "memory": 24576, "busy": 11264},
        {"index": 1, "name": "NVIDIA GeForce RTX 3090", "memory": 24576, "busy": 0}]

    def no_nvidia_smi(args, **kw):
        raise FileNotFoundError(args[0])

    monkeypatch.setattr(box.subprocess, "run", no_nvidia_smi)
    box.cmd_gpus()
    assert json.loads(capsys.readouterr().out) == []
