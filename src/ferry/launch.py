"""`ferry run`: sends a run to a machine and follows it home."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from hunch.capture import capture_env, start_run
from hunch.store import Store, append_jsonl, finish_run, read_json, timestamp, write_json

from . import data, mirror
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


def run(name: str, command: list[str], why: str | None, parent_ref: str | None, data_paths: list[str] = ()) -> int:
    store = Store.find()
    parent = store.resolve(parent_ref).id if parent_ref else None
    # Runs start in the same project folder as here, so relative paths in the command still work.
    cwd = Path.cwd().resolve().relative_to(store.root.resolve()).as_posix()
    data_files = data.collect(store.root, list(data_paths))  # bad paths fail before anything is sent
    remote = connect(name)
    try:
        say(f"getting {name} ready")
        remote.install_uv()
        env = capture_env(local_python(store.root, command))
        spec = env_spec(store.root, env)
        say(f"environment from {spec.source}")
        venv = remote.build_env(spec, on_line=lambda line: print(f"  {line}", file=sys.stderr))
        project = project_key(store.root)
        data.send(remote, remote.shared(project), data_files)

        # The run is recorded here, on this computer: its code, git, line and parent.
        run_dir = start_run(store, command=command, cwd=Path.cwd(), why=why, parent=parent, source="wrapper", python=None)
        write_json(run_dir / "env.json", env)  # this computer's environment, until the machine records its own
        paths = remote.paths(project, run_dir.name)
        mirror.write_state(run_dir, {"machine": name, "dir": paths.run_dir, "work": paths.work, "pidfile": paths.pidfile,
                                     "venv": venv})
        started = time.monotonic()
        may_be_running = False
        try:
            snapshot = read_json(run_dir / "files.json", {})
            shas = sorted({entry["sha"] for entry in snapshot.get("files", {}).values()})
            missing = remote.missing(paths.objects, shas)
            remote.send_objects(paths.objects, store.dir / "objects", missing)
            say(f"sent {len(missing)} of {len(shas)} files ({len(shas) - len(missing)} already on {name})")
            data.warn_missing(remote, paths.shared, store.root, snapshot.get("skipped", []))
            remote.send_files(paths.run_dir, run_dir, RUN_FILES)
            remote.checkout(paths)
            may_be_running = True
            remote.start(paths, venv, command, cwd)
        except BaseException as e:
            if may_be_running and not isinstance(e, FerryError):
                # Cut off while the machine was starting it: it may well be going. Keep it, and say how to find out.
                say(f"lost {name} while the run was starting; it may be going there. "
                    f"`ferry attach {run_dir.name[-4:]}` follows it, `ferry stop {run_dir.name[-4:]}` stops it")
                raise
            # The launch failed after the run was recorded: keep it, with the reason.
            interrupted = isinstance(e, KeyboardInterrupt)
            reason = "the launch was interrupted" if interrupted else (str(e) or type(e).__name__)
            append_jsonl(run_dir / "notes.jsonl", {"at": timestamp(), "kind": "ferry", "value": reason})
            (run_dir / mirror.REMOTE_FILE).unlink(missing_ok=True)
            finish_run(run_dir, status="killed" if interrupted else "crashed", exit_code=None,
                       duration=time.monotonic() - started)
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


def pull(ref: str, paths: list[str], to: str, force: bool) -> int:
    store = Store.find()
    run = store.resolve(ref)
    state = mirror.read_state(run.dir)
    remote = connect(state["machine"])
    try:
        work = data.work_dir(state)
        if not paths:
            outputs = remote.outputs(work, state["dir"])
            if not outputs:
                say(f"run {run.id[-4:]} has not written any files on {state['machine']}")
                return 0
            print(f"run {run.id[-4:]} wrote on {state['machine']} ({work}):")
            for line in data.summary(outputs, data.here(store.root)):
                print(f"  {line}")
            print(f"bring one home with: ferry pull {run.id[-4:]} PATH [--to DIR]")
            return 0
        count = data.pull(remote, work, paths, Path(to), force, data.here(store.root))
        say(f"copied {data.count_files(count)} from {state['machine']} into {Path(to).resolve()}")
        return 0
    finally:
        remote.box.close()
