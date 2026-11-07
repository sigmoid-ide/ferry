"""`ferry run`: sends a run to a machine and follows it home."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from hunch.capture import capture_env, start_run
from hunch.store import Store, append_jsonl, finish_run, read_json, timestamp, write_json

from . import mirror
from .config import FerryError, get_machine
from .envspec import env_spec, local_python
from .remote import Remote, project_key
from .ssh import Box

RUN_FILES = ["run.json", "files.json", "config.json", "env.json"]


def say(text: str) -> None:
    print(f"ferry: {text}", file=sys.stderr)


def connect(name: str) -> Remote:
    """Connects and puts the current helper on the machine, so every command it knows is there."""
    machine = get_machine(name)
    remote = Remote(Box(machine).connect())
    try:
        remote.install_helper()
    except BaseException:
        remote.box.close()
        raise
    return remote


def run(name: str, command: list[str], why: str | None, parent_ref: str | None) -> int:
    store = Store.find()
    parent = store.resolve(parent_ref).id if parent_ref else None
    remote = connect(name)
    try:
        say(f"getting {name} ready")
        remote.install_uv()
        env = capture_env(local_python(store.root, command))
        spec = env_spec(store.root, env)
        say(f"environment from {spec.source}")
        venv = remote.build_env(spec, on_line=lambda line: print(f"  {line}", file=sys.stderr))

        # The run is recorded here, on this computer: its code, git, line and parent.
        run_dir = start_run(store, command=command, cwd=Path.cwd(), why=why, parent=parent, source="wrapper", python=None)
        write_json(run_dir / "env.json", env)  # this computer's environment, until the machine records its own
        paths = remote.paths(project_key(store.root), run_dir.name)
        mirror.write_state(run_dir, {"machine": name, "dir": paths.run_dir, "pidfile": paths.pidfile, "venv": venv})
        started = time.monotonic()
        try:
            files = read_json(run_dir / "files.json", {}).get("files", {})
            shas = sorted({entry["sha"] for entry in files.values()})
            missing = remote.missing(paths.objects, shas)
            remote.send_objects(paths.objects, store.dir / "objects", missing)
            say(f"sent {len(missing)} of {len(shas)} files ({len(shas) - len(missing)} already on {name})")
            remote.send_files(paths.run_dir, run_dir, RUN_FILES)
            remote.checkout(paths)
            remote.start(paths, venv, command)
        except FerryError as e:
            # The launch failed after the run was recorded: keep it, with the reason.
            append_jsonl(run_dir / "notes.jsonl", {"at": timestamp(), "kind": "ferry", "value": str(e)})
            (run_dir / mirror.REMOTE_FILE).unlink(missing_ok=True)
            finish_run(run_dir, status="crashed", exit_code=None, duration=time.monotonic() - started)
            raise
        say(f"run {run_dir.name} started on {name}. Ctrl+C stops copying, not the run")
        return mirror.follow(remote, run_dir)
    finally:
        remote.box.close()


def attach(ref: str) -> int:
    store = Store.find()
    run = store.resolve(ref)
    state = mirror.read_state(run.dir)
    remote = connect(state["machine"])
    try:
        say(f"copying {run.id} from {state['machine']}")
        return mirror.follow(remote, run.dir)
    finally:
        remote.box.close()


def stop(ref: str) -> int:
    store = Store.find()
    run = store.resolve(ref)
    state = mirror.read_state(run.dir)
    remote = connect(state["machine"])
    try:
        say(f"{run.id} on {state['machine']}: {remote.stop(state['pidfile'])}")
        return 0
    finally:
        remote.box.close()
