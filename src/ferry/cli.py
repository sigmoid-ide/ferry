"""The `ferry` command."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time

import paramiko
from hunch.store import HunchError

from . import __version__, launch, rent, vast
from .config import FerryError, Machine, can_ask, get_machine, load_machines, parse_target, save_machine
from .remote import Remote
from .ssh import Box

WHY_ENV = "HUNCH_WHY"
GPU_ROWS = 20


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
    machines.add_argument("--json", action="store_true", help="as JSON, for editors and scripts")
    machines.set_defaults(handler=cmd_machines)

    run = commands.add_parser("run", help="run a command on a machine and bring the run home")
    run.add_argument("--on", metavar="MACHINE", help="the machine to run on")
    run.add_argument("--rent", metavar="OFFER",
                     help="rent a GPU on Vast.ai for this run instead: an offer from `ferry gpus`, or a GPU name such as "
                          '"RTX 4090" for its cheapest offer. Given back once the run is home')
    run.add_argument("--for", dest="for_", metavar="TIME",
                     help="with --rent: the most time the machine may run, such as 2h or 90m; it destroys itself then")
    run.add_argument("--disk", type=int, default=vast.DISK_GB, metavar="GB", help=f"with --rent: its disk (default: {vast.DISK_GB})")
    run.add_argument("--yes", action="store_true", help="with --rent: rent without asking")
    run.add_argument("-m", "--why", help="why you are running this (default: $HUNCH_WHY, else asked for in a terminal)")
    run.add_argument("--from", dest="parent", metavar="RUN", help="the run this one builds on (default: the last run on this branch)")
    run.add_argument("--data", action="append", default=[], metavar="PATH",
                     help="a file or folder the run needs besides its code, such as a dataset; sent once, kept on the "
                          "machine for later runs, and found at the same path there (repeatable)")
    run.add_argument("--env", action="append", default=[], metavar="NAME[=VALUE]",
                     help="an environment variable for the run, such as a token. NAME alone sends its value from here, "
                          "so it is never typed out. Handed to the run in memory, never saved in a file (repeatable)")
    run.add_argument("--gpus", metavar="LIST",
                     help="the machine's GPUs the run may use, numbered as nvidia-smi shows them, such as 1 or 0,1 "
                          "(default: all)")
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

    ps = commands.add_parser("ps", help="runs that are going on other machines, and how they are doing there")
    ps.set_defaults(handler=cmd_ps)

    forget = commands.add_parser("forget", help="for a run whose machine is gone for good: show it as lost")
    forget.add_argument("run")
    forget.set_defaults(handler=cmd_forget)

    clean = commands.add_parser("clean", help="free disk space on a machine")
    clean.add_argument("machine")
    clean.add_argument("--outputs", action="store_true",
                       help="also remove what this project's ended runs wrote there, such as checkpoints, "
                            "except runs you marked keep in Hunch")
    clean.add_argument("--yes", action="store_true", help="remove without asking")
    clean.set_defaults(handler=cmd_clean)

    login = commands.add_parser("login", help="keep your Vast.ai API key in the OS keychain")
    login.add_argument("provider", choices=["vast"])
    login.set_defaults(handler=cmd_login)

    logout = commands.add_parser("logout", help="remove your Vast.ai API key from the OS keychain")
    logout.add_argument("provider", choices=["vast"])
    logout.set_defaults(handler=cmd_logout)

    gpus = commands.add_parser("gpus", help="GPUs you can rent on Vast.ai now, cheapest first")
    gpus.add_argument("--gpu", metavar="NAME", help='only this GPU, such as "RTX 4090", 4090 or A100')
    gpus.add_argument("--max-price", type=float, metavar="DOLLARS", help="at most this much per hour")
    gpus.add_argument("--any-host", action="store_true",
                      help="also hosts Vast.ai has not verified (they can see your code and data)")
    gpus.add_argument("--json", action="store_true", help="as JSON, for editors and scripts")
    gpus.set_defaults(handler=cmd_gpus)

    rent_ = commands.add_parser("rent", help="rent a GPU on Vast.ai and add it as a machine")
    rent_.add_argument("offer", help='an offer from `ferry gpus`, or a GPU name such as "RTX 4090" for its cheapest offer')
    rent_.add_argument("--for", dest="for_", required=True, metavar="TIME",
                       help="the most time it may run, such as 3h or 90m; it destroys itself then")
    rent_.add_argument("--disk", type=int, default=vast.DISK_GB, metavar="GB",
                       help=f"its disk, fixed for its life (default: {vast.DISK_GB})")
    rent_.add_argument("--yes", action="store_true", help="rent without asking")
    rent_.set_defaults(handler=cmd_rent)

    release = commands.add_parser("release", help="give a rented machine back now")
    release.add_argument("machine")
    release.add_argument("--yes", action="store_true", help="also when a run on it has not come home")
    release.set_defaults(handler=cmd_release)

    args = parser.parse_args(argv)
    if args.command != "release":
        _remind_time_up()
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


def _remind_time_up() -> None:
    """A rented machine past its time is destroyed, or stopped with a run that never came
    home, and then its disk still costs until someone gives it back."""
    for m in load_machines().values():
        if m.provider and m.deadline and m.deadline < time.time():
            print(f"ferry: the time you rented {m.name} for is up. If a run on it never came home, it stands stopped and "
                  f"its disk still costs: `ferry ps` says. `ferry release {m.name}` gives it back", file=sys.stderr)


def cmd_add(args) -> int:
    host, user, port = parse_target(args.target)
    machine = Machine(name=args.name, host=host, user=user, port=port, key=args.key)

    def trust(label: str, fp: str, kind: str) -> bool:
        print(f"{label}: first time connecting. Its {kind} host key is\n  {fp}")
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
    if args.json:
        fields = ("name", "host", "user", "port", "hostname", "arch", "gpu", "driver", "python",
                  "provider", "instance", "price", "tier", "place", "rented_at", "deadline", "once")
        print(json.dumps([{k: getattr(m, k) for k in fields if getattr(m, k) is not None} for m in machines.values()]))
        return 0
    if not machines:
        print("no machines yet. Add one with: ferry add mybox user@host, or rent one: ferry gpus")
        return 0
    for m in machines.values():
        where = f"{m.user + '@' if m.user else ''}{m.host}{':' + str(m.port) if m.port else ''}"
        line = f"{m.name:<14} {where:<28} {m.gpu or 'no GPU'}"
        if m.provider:
            left = (m.deadline or 0) - time.time()
            line += (f", ${m.price:.3f}/h, {rent.duration(left) + ' left' if left > 0 else 'time is up'}, "
                     f"{rent.money(rent.cost_so_far(m))} so far")
        print(line)
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
    if (args.on is None) == (args.rent is None):
        raise FerryError("say where to run: --on MACHINE, or --rent GPU --for 2h to rent one on Vast.ai")
    env = launch.run_env(args.env, args.gpus)
    if args.rent is not None:
        if args.for_ is None:
            raise FerryError("--rent needs --for, the most time the machine may run, such as --for 2h")
        seconds = rent.parse_duration(args.for_)
        return launch.run_rented(args.rent, seconds, args.disk, args.yes, command, _why(args), args.parent, args.data, env)
    get_machine(args.on)  # fail before asking why
    return launch.run(args.on, command, _why(args), args.parent, args.data, env)


def cmd_attach(args) -> int:
    return launch.attach(args.run)


def cmd_stop(args) -> int:
    return launch.stop(args.run)


def cmd_pull(args) -> int:
    return launch.pull(args.run, args.paths, args.to, args.force)


def cmd_ps(args) -> int:
    return launch.ps()


def cmd_forget(args) -> int:
    return launch.forget(args.run)


def cmd_clean(args) -> int:
    get_machine(args.machine)
    return launch.clean(args.machine, args.outputs, args.yes)


def cmd_login(args) -> int:
    if not can_ask():
        raise FerryError(f"run ferry login in a terminal to type the key, or set {vast.KEY_ENV}")
    print("Paste an API key from the Keys page of your Vast.ai account. It is not shown as you type.")
    key = getpass.getpass("Vast.ai API key: ").strip()
    if not key:
        raise FerryError("no key given; nothing changed")
    balance = vast.Vast(key).balance()  # refuses a wrong key before it is kept
    vast.store_key(key)
    print(f"logged in to Vast.ai: your balance is ${balance:.2f}. The key is in your system's keychain, and nowhere else.")
    if os.environ.get(vast.KEY_ENV):
        print(f"  {vast.KEY_ENV} is set here, so Ferry uses that key while it is set.")
    return 0


def cmd_logout(args) -> int:
    removed = vast.forget_key()
    print("removed your Vast.ai key from the keychain." if removed else "there was no Vast.ai key in the keychain.")
    if os.environ.get(vast.KEY_ENV):
        print(f"  {vast.KEY_ENV} is still set here, and Ferry uses it while it is.")
    return 0


def cmd_gpus(args) -> int:
    offers = vast.Vast(vast.stored_key()).search(args.gpu, args.max_price, args.any_host)
    if args.json:
        print(json.dumps([offer.to_json() for offer in offers]))
        return 0
    if not offers:
        wanted = f" {args.gpu}" if args.gpu else ""
        cap = f" at ${args.max_price:g}/h or less" if args.max_price is not None else ""
        print(f"no{wanted} offers on Vast.ai{cap} right now that Ferry can use"
              f"{'' if args.any_host else ' from verified hosts'}.")
        return 0
    print(f"Offers on Vast.ai now, cheapest first. Prices include {vast.DISK_GB} GB of disk.")
    print(f"{'OFFER':<10}{'GPU':<22}{'MEMORY':>8}{'$/HOUR':>9}  {'RELIABLE':>8}  {'HOST':<11}{'WHERE':<22}{'DOWNLOAD':>10}")
    shown = offers[:GPU_ROWS]
    for o in shown:
        gpu = f"{o.count}x {o.gpu}" if o.count > 1 else o.gpu
        print(f"{o.id:<10}{gpu[:21]:<22}{o.memory:>5g} GB{o.price:>9.3f}  {o.reliability:>7.1%}  {o.tier:<11}"
              f"{o.place[:21]:<22}{o.download:>5.0f} Mb/s")
    if len(offers) > len(shown):
        print(f"and {len(offers) - len(shown)} more (--json lists them all, --gpu or --max-price narrows them)")
    if any(o.tier == "unverified" for o in shown):
        print("Unverified hosts have not been checked by Vast.ai; like any host, they can see your code and data.")
    print("Rent one with: ferry rent OFFER --for 2h   or run on one with: ferry run --rent OFFER --for 2h -- python train.py")
    return 0


def cmd_rent(args) -> int:
    machine = rent.rent(args.offer, rent.parse_duration(args.for_), args.disk, args.yes)
    print(f"run on it with: ferry run --on {machine.name} -- python train.py")
    print(f"give it back with: ferry release {machine.name} (it destroys itself at "
          f"{time.strftime('%H:%M', time.localtime(machine.deadline))} anyway)")
    return 0


def cmd_release(args) -> int:
    rent.release(args.machine, args.yes)
    return 0
