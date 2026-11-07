"""Ferry's settings: the machines you added and their pinned host keys.

Never a password, token or private key: SSH keys stay where SSH keeps them.
"""

from __future__ import annotations

import json
import os
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
