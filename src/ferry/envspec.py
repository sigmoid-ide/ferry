"""What the machine installs for a run: the project's requirements.txt when it has
one, otherwise the packages Hunch recorded on this computer, minus the ones that
only make sense here. Torch is pinned by version and picked on the machine to
match its driver."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import hunch

# Packages that exist only on Windows or macOS, or that are Ferry's own and never needed by the run.
SKIP = {
    "pywin32", "pywin32-ctypes", "pywinpty", "winpty", "wmi", "comtypes", "pyreadline3", "windows-curses",
    "pyobjc", "appnope", "pip", "sigmoid-ferry", "sigmoid-hunch", "paramiko", "bcrypt", "pynacl",
}


@dataclass(frozen=True)
class EnvSpec:
    python: str  # major.minor
    requirements: str
    source: str

    @property
    def key(self) -> str:
        """Names the environment on the machine, so the same spec is built once."""
        digest = hashlib.sha256(f"{self.python}\n{self.requirements}\ntorch-backend=auto".encode()).hexdigest()
        return f"py{self.python}-{digest[:16]}"


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def env_spec(root: Path, env: dict) -> EnvSpec:
    python = ".".join(str(env.get("python") or "3.11").split(".")[:2])
    requirements_file = root / "requirements.txt"
    if requirements_file.is_file():
        lines = requirements_file.read_text(encoding="utf-8").splitlines()
        source = "requirements.txt"
        lines = [line for line in lines if _normalise(re.split(r"[<>=!~;\[ ]", line.strip(), maxsplit=1)[0]) != "sigmoid-hunch"]
    else:
        packages = env.get("packages") or {}
        lines = [
            f"{name}=={str(version).split('+')[0]}"
            for name, version in sorted(packages.items(), key=lambda item: _normalise(item[0]))
            if _normalise(name) not in SKIP and version
        ]
        source = "the packages Hunch recorded on this computer"
    version = getattr(hunch, "__version__", "")
    lines.append(f"sigmoid-hunch=={version}" if re.fullmatch(r"\d+(\.\d+)*", version) else "sigmoid-hunch")
    return EnvSpec(python=python, requirements="\n".join(lines).strip() + "\n", source=source)
