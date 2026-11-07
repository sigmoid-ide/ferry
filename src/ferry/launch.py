"""`ferry run`: sends a run to a machine and follows it home."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import paramiko
from hunch.capture import capture_env, start_run
from hunch.store import LOST_AFTER_SECONDS, Store, append_jsonl, finish_run, read_json, timestamp, write_json

from . import box, data, mirror
from .config import FerryError, can_ask, get_machine
from .envspec import env_spec, local_python
from .remote import Remote, project_key
from .ssh import Box

RUN_FILES = ["run.json", "files.json", "config.json", "env.json"]
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
GPU_LIST = re.compile(r"\d+(,\d+)*")
FERRY_SETS = ("PATH", "VIRTUAL_ENV", "HUNCH_RUN_DIR", "FERRY_SHARED")  # the run's environment Ferry sets itself


def say(text: str) -> None:
    print(f"ferry: {text}", file=sys.stderr)


def run_env(specs: list[str], gpus: str | None = None) -> dict[str, str]:
    """The variables the run gets on the machine, from --env NAME[=VALUE] and --gpus.
    NAME alone takes the value from this computer, so a token is never typed out.
    Errors name the variable, never its value."""
    env: dict[str, str] = {}
    for spec in specs:
        name, has_value, value = spec.partition("=")
        if not ENV_NAME.fullmatch(name):
            raise FerryError(f"--env {name!r}: give NAME, to send its value from here, or NAME=VALUE")
        if name in FERRY_SETS:
            raise FerryError(f"--env {name}: Ferry sets {name} for the run itself")
        if not has_value:
            if name not in os.environ:
                raise FerryError(f"--env {name}: {name} is not set here. Set it first, or give --env {name}=VALUE")
            value = os.environ[name]
        env[name] = value
    if gpus is not None:
        if not GPU_LIST.fullmatch(gpus):
            raise FerryError(f"--gpus {gpus!r}: give GPU numbers as nvidia-smi shows them, such as 1 or 0,1")
        if "CUDA_VISIBLE_DEVICES" in env:
            raise FerryError("give --gpus or --env CUDA_VISIBLE_DEVICES, not both")
        env["CUDA_VISIBLE_DEVICES"] = gpus
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # so CUDA numbers the GPUs as nvidia-smi does
    return env


def check_gpus(name: str, gpus: list[dict], wanted: str | None) -> None:
    """Refuses GPU numbers the machine does not have, which CUDA would quietly answer by
    running on the CPU, and warns when another program already computes on a GPU the
    run gets."""
    by_index = {gpu["index"]: gpu for gpu in gpus}
    if wanted is not None and not GPU_LIST.fullmatch(wanted):
        return  # GPU ids, or none at all: nothing to check against these numbers
    if wanted is not None:
        if not gpus:
            raise FerryError(f"{name} has no NVIDIA GPU that nvidia-smi can see, so there is nothing to pick with --gpus")
        for index in map(int, wanted.split(",")):
            if index not in by_index:
                have = ", ".join(f"{gpu['index']} ({gpu['name']})" for gpu in gpus)
                raise FerryError(f"{name} has no GPU {index}. Its GPUs: {have}")
        chosen = [by_index[int(i)] for i in wanted.split(",")]
    else:
        chosen = gpus
    free = [gpu["index"] for gpu in gpus if not gpu["busy"]]
    for gpu in chosen:
        if gpu["busy"]:
            hint = f"; GPU {free[0]} is free: --gpus {free[0]}" if free else "; the run may run out of memory or go slower"
            say(f"GPU {gpu['index']} on {name} is already in use by another program "
                f"({gpu['busy'] / 1024:.1f} of {gpu['memory'] / 1024:.1f} GB){hint}")


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


def run(name: str, command: list[str], why: str | None, parent_ref: str | None, data_paths: list[str] = (),
        env_vars: dict[str, str] | None = None) -> int:
    env_vars = env_vars or {}
    store = Store.find()
    parent = store.resolve(parent_ref).id if parent_ref else None
    # Runs start in the same project folder as here, so relative paths in the command still work.
    cwd = Path.cwd().resolve().relative_to(store.root.resolve()).as_posix()
    data_files = data.collect(store.root, list(data_paths))  # bad paths fail before anything is sent
    remote = connect(name)
    try:
        say(f"getting {name} ready")
        check_gpus(name, remote.gpus(), env_vars.get("CUDA_VISIBLE_DEVICES"))
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
            remote.start(paths, venv, command, cwd, env_vars)
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


STATE_TEXT = {
    "ended": ("ended on {machine}", "ferry attach {short} brings the rest home"),
    "gone": ("stopped on {machine} without recording an end", "ferry attach {short} marks it crashed"),
    "missing": ("no longer on {machine}", "if it is gone for good, ferry forget {short} marks it lost"),
}


def ps() -> int:
    """Asks each machine how its runs are doing."""
    store = Store.find()
    away: dict[str, list] = {}
    for run in store.runs():
        if run.end is None and (run.dir / mirror.REMOTE_FILE).exists():
            state = mirror.read_state(run.dir)
            away.setdefault(state.get("machine", "?"), []).append((run, state))
    if not away:
        print("no runs going on other machines")
        return 0
    for machine, runs in away.items():
        states, problem = None, None
        try:
            remote = connect(machine)
            try:
                states = remote.states([[state["dir"], state["pidfile"]] for _, state in runs])
            finally:
                remote.box.close()
        except (FerryError, OSError, paramiko.SSHException) as e:
            problem = e
        for i, (run, _) in enumerate(runs):
            short = run.id[-4:]
            if states is None:
                what, hint = f"on {machine}, which cannot be reached ({problem})", f"if it is gone for good, ferry forget {short} marks it lost"
            elif states[i] == "running":
                copied = mirror.being_copied(run.dir)
                what = f"running on {machine}" + (", being copied home" if copied else "")
                hint = None if copied else f"ferry attach {short} copies it home"
            else:
                what, hint = (text.format(machine=machine, short=short) for text in STATE_TEXT[states[i]])
            print(f"{short}  {what}" + (f"  {run.why}" if run.why else ""))
            if hint:
                print(f"      {hint}")
    return 0


def forget(ref: str) -> int:
    """For a run whose machine is gone for good: it stops counting as running and shows
    as lost, as Hunch shows any run that never recorded an end."""
    store = Store.find()
    run = store.resolve(ref)
    short = run.id[-4:]
    if run.end is not None or not (run.dir / mirror.REMOTE_FILE).exists():
        raise FerryError(f"run {short} is not going on another machine; there is nothing to forget")
    if mirror.being_copied(run.dir):
        raise FerryError(f"a ferry is copying run {short} right now, so its machine is there. Stop that one first")
    machine = mirror.read_state(run.dir).get("machine")
    os.replace(run.dir / mirror.REMOTE_FILE, run.dir / mirror.GONE_FILE)
    # The last sign of life copied here no longer counts: the machine is gone.
    beat = run.dir / "heartbeat"
    beat.touch()
    stale = time.time() - LOST_AFTER_SECONDS - 1
    os.utime(beat, (stale, stale))
    say(f"run {short} shows as lost now. If {machine} comes back, `ferry attach {short}` picks it up again")
    return 0


def clean(name: str, outputs: bool, yes: bool) -> int:
    """Frees disk space on a machine: environments no run has asked for in a while, and
    with --outputs what this project's ended runs wrote there, except runs marked keep."""
    store = Store.find()
    remote = connect(name)
    try:
        project = project_key(store.root)
        plan = remote.clean_plan(project)
        known = {run.id: run for run in store.runs()}
        # Runs this journal does not know, and runs marked keep, are left alone.
        runs = [(known[entry["id"]], entry["size"]) for entry in plan["runs"]
                if entry["id"] in known and known[entry["id"]].verdict != "keep"]
        envs = plan["envs"]
        print(f"{name} has {data.human(plan['free'])} free.")
        if envs:
            print(f"  {len(envs)} environment{'s' if len(envs) != 1 else ''} no run has asked for in "
                  f"{box.CLEAN_AFTER_DAYS} days: removing {'them' if len(envs) != 1 else 'it'} and uv's download cache "
                  f"frees about {data.human(plan['freed'])}. New environments download their packages again.")
        total = sum(size for _, size in runs)
        if runs and outputs:
            print(f"  what {len(runs)} ended run{'s' if len(runs) != 1 else ''} of this project wrote there, "
                  f"{data.human(total)} (runs you marked keep stay):")
            for run, size in runs:
                print(f"    {run.id[-4:]}  {data.human(size):>9}  {run.why or ''}".rstrip())
        elif runs:
            print(f"  ended runs of this project keep {data.human(total)} there (checkpoints and the like); "
                  f"ferry clean {name} --outputs removes them, except runs you marked keep.")
        if not envs and not (outputs and runs):
            print("nothing to remove.")
            return 0
        if not yes:
            if not can_ask():
                raise FerryError("nothing removed: run again with --yes to remove these")
            if input("Remove these? [y/N] ").strip().lower() not in ("y", "yes"):
                print("nothing removed.")
                return 0
        done = remote.clean(project, envs, [run.id for run, _ in runs] if outputs else [])
        say(f"freed {data.human(max(done['freed'], 0))} on {name}")
        if done["cache"] is False:
            say("uv's download cache is in use, so it stayed; ferry clean again later clears it")
        return 0
    finally:
        remote.box.close()
