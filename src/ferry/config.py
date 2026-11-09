"""Ferry's settings: the machines you added and their pinned host keys.

Never a password, token or private key: SSH keys stay where SSH keeps them.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

HOME_ENV = "FERRY_HOME"


class FerryError(Exception):
    pass


def can_ask() -> bool:
    """Whether someone is at a terminal to answer. Both ends must be terminals: on
    Windows, input from the null device (as an editor gives) also counts as one."""
    return sys.stdin.isatty() and sys.stderr.isatty()


def parse_target(target: str) -> tuple[str, str | None, int | None]:
    """user@host:port, where only the host is needed, as (host, user, port)."""
    match = re.fullmatch(r"(?:(?P<user>[^@]+)@)?(?P<host>[^:@]+)(?::(?P<port>\d+))?", target)
    if match is None:
        raise FerryError(f"could not read {target!r}; use user@host or user@host:port")
    return match["host"], match["user"], int(match["port"]) if match["port"] else None


def home() -> Path:
    if os.environ.get(HOME_ENV):
        return Path(os.environ[HOME_ENV])
    if os.name == "nt":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "ferry"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "ferry"


def known_hosts_path() -> Path:
    return home() / "known_hosts"


@dataclass
class Machine:
    name: str
    host: str  # a hostname, an IP or an alias from ~/.ssh/config
    user: str | None = None
    port: int | None = None
    key: str | None = None  # an identity file, when not SSH's defaults
    hostname: str | None = None  # what the box calls itself
    arch: str | None = None
    gpu: str | None = None
    driver: str | None = None
    python: str | None = None
    # A machine rented with `ferry rent`: which instance, what it costs, and when it destroys itself.
    provider: str | None = None
    instance: int | None = None
    price: float | None = None  # $ per hour, its disk included
    disk_rate: float | None = None  # $ per hour its disk costs alone, which it still pays while stopped
    tier: str | None = None
    place: str | None = None
    rented_at: float | None = None  # Unix time
    deadline: float | None = None  # Unix time
    stopped_seconds: float | None = None  # time it stood stopped, paying for its disk only
    once: bool | None = None  # rented for one run, and released once that run is home


def rented_dir(name: str) -> Path:
    """A rented machine's own SSH key and pinned host keys, deleted when it is released."""
    return home() / "rented" / name


def known_hosts_for(machine: Machine) -> Path:
    """Where a machine's host keys are pinned. A rented machine's are its own, because
    Vast hands the same address and port to one instance after another."""
    return rented_dir(machine.name) / "known_hosts" if machine.provider else known_hosts_path()


def _machines_path() -> Path:
    return home() / "machines.json"


def load_machines() -> dict[str, Machine]:
    try:
        data = json.loads(_machines_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    known = {f.name for f in fields(Machine)}
    return {name: Machine(**{k: v for k, v in entry.items() if k in known}) for name, entry in data.get("machines", {}).items()}


def save_machine(machine: Machine) -> None:
    machines = load_machines()
    machines[machine.name] = machine
    _save(machines)


def remove_machine(name: str) -> None:
    machines = load_machines()
    if machines.pop(name, None) is not None:
        _save(machines)


def _save(machines: dict[str, Machine]) -> None:
    path = _machines_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {"machines": {name: {k: v for k, v in asdict(m).items() if v is not None} for name, m in sorted(machines.items())}}
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def get_machine(name: str) -> Machine:
    machines = load_machines()
    if name not in machines:
        known = ", ".join(sorted(machines)) or "none yet"
        raise FerryError(f"no machine named {name!r} (known: {known}). Add it with: ferry add {name} user@host")
    return machines[name]
