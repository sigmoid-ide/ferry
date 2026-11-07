"""The `ferry` command."""

from __future__ import annotations

import argparse
import os
import re
import sys

import paramiko
from hunch.store import HunchError, Store

from . import __version__, launch, mirror
from .config import FerryError, Machine, can_ask, get_machine, load_machines, save_machine
from .remote import Remote
from .ssh import Box

WHY_ENV = "HUNCH_WHY"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ferry", description="Run your Hunch experiments on a GPU machine and bring every run home.")
    parser.add_argument("--version", action="version", version=f"ferry {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    add = commands.add_parser("add", help="add a machine you can reach over SSH")
    add.add_argument("name", help="what to call it, for example mybox")
    add.add_argument("target", help="user@host, user@host:port, or a Host from ~/.ssh/config")
    add.add_argument("--key", help="the SSH key to use, when not your default")
    add.add_argument("--yes", action="store_true", help="trust the machine's host key without asking")
    add.set_defaults(handler=cmd_add)

    machines = commands.add_parser("machines", help="list your machines")
    machines.set_defaults(handler=cmd_machines)

    run = commands.add_parser("run", help="run a command on a machine and bring the run home")
    run.add_argument("--on", required=True, metavar="MACHINE", help="the machine to run on")
    run.add_argument("-m", "--why", help="why you are running this (default: $HUNCH_WHY, else asked for in a terminal)")
    run.add_argument("--from", dest="parent", metavar="RUN", help="the run this one builds on (default: the last run on this branch)")
    run.add_argument("--data", action="append", default=[], metavar="PATH",
                     help="a file or folder the run needs besides its code, such as a dataset; sent once, kept on the "
                          "machine for later runs, and found at the same path there (repeatable)")
    run.add_argument("command", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    run.set_defaults(handler=cmd_run)

    attach = commands.add_parser("attach", help="keep copying a run that is still going on its machine")
    attach.add_argument("run")
    attach.set_defaults(handler=cmd_attach)

    stop = commands.add_parser("stop", help="stop a run on its machine")
    stop.add_argument("run")
    stop.set_defaults(handler=cmd_stop)

    pull = commands.add_parser("pull", help="bring files a run wrote on its machine home, such as checkpoints")
    pull.add_argument("run")
    pull.add_argument("paths", nargs="*", metavar="PATH", help="files or folders in the run's folder (none: list them)")
    pull.add_argument("--to", default=".", metavar="DIR", help="where to put them (default: here)")
    pull.add_argument("--force", action="store_true", help="replace files that are already there")
    pull.set_defaults(handler=cmd_pull)

    ps = commands.add_parser("ps", help="runs that are going on other machines")
    ps.set_defaults(handler=cmd_ps)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (FerryError, HunchError) as e:
        print(f"ferry: {e}", file=sys.stderr)
        return 1
    except (OSError, paramiko.SSHException) as e:
        print(f"ferry: the connection failed: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nferry: stopped", file=sys.stderr)
        return 130


def _parse_target(target: str) -> tuple[str, str | None, int | None]:
    match = re.fullmatch(r"(?:(?P<user>[^@]+)@)?(?P<host>[^:@]+)(?::(?P<port>\d+))?", target)
    if match is None:
        raise FerryError(f"could not read {target!r}; use user@host or user@host:port")
    return match["host"], match["user"], int(match["port"]) if match["port"] else None


def cmd_add(args) -> int:
    host, user, port = _parse_target(args.target)
    machine = Machine(name=args.name, host=host, user=user, port=port, key=args.key)

    def trust(fp: str, kind: str) -> bool:
        print(f"{args.name}: first time connecting. Its {kind} host key is\n  {fp}")
        if args.yes:
            return True
        if not can_ask():
            raise FerryError("check the key above, then run again with --yes to trust it")
        return input("Trust this key? [y/N] ").strip().lower() in ("y", "yes")

    box = Box(machine).connect(trust=trust)
    try:
        info = Remote(box).probe()
    finally:
        box.close()
    machine.hostname = info.get("hostname") or None
    machine.arch = info.get("arch") or None
    machine.gpu = info.get("gpu") or None
    machine.driver = info.get("driver") or None
    machine.python = info.get("python") or None
    save_machine(machine)
    gpu = f"{machine.gpu}, driver {machine.driver}" if machine.gpu else "no NVIDIA GPU found"
    print(f"added {machine.name}: {machine.hostname} ({machine.arch}), {gpu}, Python {machine.python or 'not found'}")
    if not machine.python:
        print("  Ferry needs python3 on the machine to start runs.")
    return 0


def cmd_machines(args) -> int:
    machines = load_machines()
    if not machines:
        print("no machines yet. Add one with: ferry add mybox user@host")
        return 0
    for m in machines.values():
        where = f"{m.user + '@' if m.user else ''}{m.host}{':' + str(m.port) if m.port else ''}"
        print(f"{m.name:<12} {where:<28} {m.gpu or 'no GPU'}")
    return 0


def _why(args) -> str | None:
    if args.why is not None:
        return args.why
    if WHY_ENV in os.environ:
        return os.environ[WHY_ENV].strip() or None
    if sys.stdin.isatty() and sys.stderr.isatty():
        try:
            return input("why? (Enter to skip) ").strip() or None
        except EOFError:
            return None
    return None


def cmd_run(args) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise FerryError("nothing to run. Usage: ferry run --on mybox -- python train.py")
    get_machine(args.on)  # fail before asking why
    return launch.run(args.on, command, _why(args), args.parent, args.data)


def cmd_attach(args) -> int:
    return launch.attach(args.run)


def cmd_stop(args) -> int:
    return launch.stop(args.run)


def cmd_pull(args) -> int:
    return launch.pull(args.run, args.paths, args.to, args.force)


def cmd_ps(args) -> int:
    store = Store.find()
    found = 0
    for run in store.runs():
        path = run.dir / mirror.REMOTE_FILE
        if not path.exists() or run.end is not None:
            continue
        state = mirror.read_state(run.dir)
        found += 1
        print(f"{run.id[-4:]}  on {state.get('machine')}  synced {state.get('synced_at') or 'never'}  {run.why or ''}".rstrip())
    if not found:
        print("no runs going on other machines")
    return 0
