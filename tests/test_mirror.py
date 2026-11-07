"""Copying a run home: new lines only, whole JSON files swapped in, nothing duplicated."""

import json
import os
import time

import paramiko

from ferry.mirror import pull


class LocalSFTP:
    """Just enough of paramiko's SFTP client, over a local folder."""

    def listdir_attr(self, path):
        out = []
        for name in os.listdir(path):
            attrs = paramiko.SFTPAttributes.from_stat(os.stat(os.path.join(path, name)), name)
            out.append(attrs)
        return out

    def open(self, path, mode="rb"):
        return open(path, mode)


def setup(tmp_path):
    box, home = tmp_path / "box", tmp_path / "home"
    box.mkdir()
    home.mkdir()
    state = {"machine": "box", "dir": str(box)}
    return box, home, state


def test_only_complete_lines_come_home_and_never_twice(tmp_path):
    box, home, state = setup(tmp_path)
    sftp = LocalSFTP()
    (box / "metrics.jsonl").write_bytes(b'{"_step": 0}\n{"_step": 1}\n{"_ste')
    assert pull(sftp, state, home, 0.0) is False
    assert (home / "metrics.jsonl").read_bytes() == b'{"_step": 0}\n{"_step": 1}\n'
    assert pull(sftp, state, home, 0.0) is False  # nothing new: nothing appended
    with open(box / "metrics.jsonl", "ab") as f:
        f.write(b'p": 2}\n')
    pull(sftp, state, home, 0.0)
    steps = [json.loads(line)["_step"] for line in (home / "metrics.jsonl").read_text().splitlines()]
    assert steps == [0, 1, 2]


def test_a_new_copy_after_a_restart_continues_where_it_stopped(tmp_path):
    box, home, state = setup(tmp_path)
    (box / "metrics.jsonl").write_bytes(b'{"_step": 0}\n')
    pull(LocalSFTP(), state, home, 0.0)
    fresh_state = {"machine": "box", "dir": str(box)}  # remote.json lost: offsets come from the file at home
    with open(box / "metrics.jsonl", "ab") as f:
        f.write(b'{"_step": 1}\n')
    pull(LocalSFTP(), fresh_state, home, 0.0)
    assert (home / "metrics.jsonl").read_text().count("_step") == 2


def test_json_files_are_replaced_whole_and_the_end_brings_the_rest(tmp_path):
    box, home, state = setup(tmp_path)
    (box / "env.json").write_text('{"python": "3.11"}')
    (box / "output.log").write_bytes(b"epoch 1\npartial")
    pull(LocalSFTP(), state, home, 0.0)
    assert json.loads((home / "env.json").read_text()) == {"python": "3.11"}
    assert (home / "output.log").read_bytes() == b"epoch 1\n"
    (box / "end.json").write_text('{"status": "finished"}')
    assert pull(LocalSFTP(), state, home, 0.0) is True
    assert (home / "output.log").read_bytes() == b"epoch 1\npartial"  # at the end, the last partial line too


def test_a_fresh_heartbeat_on_the_box_keeps_the_run_alive_at_home(tmp_path):
    box, home, state = setup(tmp_path)
    (box / "heartbeat").touch()
    (home / "heartbeat").touch()
    old = time.time() - 3600
    os.utime(home / "heartbeat", (old, old))
    pull(LocalSFTP(), state, home, 0.0)
    assert time.time() - (home / "heartbeat").stat().st_mtime < 60
    assert json.loads((home / "remote.json").read_text())["heartbeat_age"] < 60
