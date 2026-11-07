"""What the machine installs, and the machines Ferry remembers."""

import hunch
import pytest

from ferry.config import FerryError, Machine, get_machine, load_machines, parse_target, save_machine
from ferry.envspec import env_spec
from ferry.remote import project_key


def test_a_projects_requirements_win(tmp_path):
    (tmp_path / "requirements.txt").write_text("torch==2.5.1\nsigmoid-hunch>=0.1\nnumpy\n")
    spec = env_spec(tmp_path, {"python": "3.12.4", "packages": {"pywin32": "306"}})
    assert spec.python == "3.12"
    assert spec.source == "requirements.txt"
    lines = spec.requirements.splitlines()
    assert lines[:2] == ["torch==2.5.1", "numpy"]
    assert lines[-1].startswith("sigmoid-hunch")  # the project's own pin is replaced by this Hunch


def test_without_requirements_the_recorded_packages_are_used(tmp_path):
    env = {"python": "3.11.9", "packages": {"torch": "2.5.1+cpu", "pywin32": "306", "Numpy": "2.1.0", "sigmoid-ferry": "0.1"}}
    spec = env_spec(tmp_path, env)
    lines = spec.requirements.splitlines()
    assert "Numpy==2.1.0" in lines and "torch==2.5.1" in lines  # local build tags dropped, torch is picked on the machine
    assert not any(line.startswith(("pywin32", "sigmoid-ferry")) for line in lines)


def test_the_same_spec_names_the_same_environment(tmp_path):
    a = env_spec(tmp_path, {"python": "3.11.2", "packages": {"numpy": "2.1.0"}})
    b = env_spec(tmp_path, {"python": "3.11.9", "packages": {"numpy": "2.1.0"}})
    c = env_spec(tmp_path, {"python": "3.11.9", "packages": {"numpy": "2.2.0"}})
    assert a.key == b.key != c.key
    assert hunch.__version__ in a.requirements or "sigmoid-hunch" in a.requirements


def test_machines_are_saved_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("FERRY_HOME", str(tmp_path))
    save_machine(Machine(name="box", host="10.0.0.5", user="me", port=2222, gpu="RTX 3090"))
    assert load_machines()["box"].port == 2222
    assert get_machine("box").gpu == "RTX 3090"
    assert "password" not in (tmp_path / "machines.json").read_text()
    with pytest.raises(FerryError, match="ferry add other"):
        get_machine("other")


def test_targets_and_project_keys():
    assert parse_target("me@gpu.lab:2222") == ("gpu.lab", "me", 2222)
    assert parse_target("gpu-box") == ("gpu-box", None, None)
    assert project_key(__import__("pathlib").Path("My Project")).startswith("My-Project-")


def test_the_projects_own_python_is_read_never_ferrys(tmp_path, monkeypatch):
    from ferry.envspec import local_python

    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    assert local_python(tmp_path, ["python", "train.py"]) == str(venv_python)
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "activated"))  # an activated environment wins
    monkeypatch.setattr("shutil.which", lambda name: str(tmp_path / "activated" / "bin" / name))
    assert local_python(tmp_path, ["python", "train.py"]) == str(tmp_path / "activated" / "bin" / "python")


def test_an_unreadable_python_without_requirements_says_what_to_do(tmp_path):
    with pytest.raises(FerryError, match="requirements.txt"):
        env_spec(tmp_path, {"executable": "python", "error": "not found"})
