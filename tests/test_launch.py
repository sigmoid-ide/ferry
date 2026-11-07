"""A launch cut short after the run was recorded must not leave it looking alive."""

from types import SimpleNamespace

import pytest
from hunch.store import Store

from ferry import launch
from ferry.remote import RunPaths


class FakeRemote:
    def __init__(self, fail_with):
        self.fail_with = fail_with
        self.box = SimpleNamespace(label="box", close=lambda: None)

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
