"""Everything Ferry does on the machine, through the helper in box.py.

Layout on the machine, under ~/.ferry:
  box.py                                 the helper
  bin/uv                                 uv, pinned and checksummed
  envs/<key>/                            one Python environment per requirements set
  projects/<project>/objects/            code, one file per content, shared by runs
  projects/<project>/.hunch/runs/<id>/   the run, as Hunch writes it
  projects/<project>/work/<id>/          the code the run ran in
  projects/<project>/shared/             kept between runs, for data and checkpoints ($FERRY_SHARED)
  projects/<project>/pids/<id>           the run's supervisor
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shlex
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .config import FerryError
from .envspec import EnvSpec
from .ssh import Box

UV_VERSION = "0.13.7"
UV_SHA256 = {
    "x86_64": "c5221ff41b7892bfc223543883600125f6a47fe845093e0f5bdff4f359196d83",
    "aarch64": "71279f8d3c67241e807b9d8db872d1bf541579fb9f08b7d1eb1682196d1e57dc",
}
ENV_MARK = "FERRY-ENV: "


def project_key(root: Path) -> str:
    """Names a project on the machine: its folder name plus a hash of where it lives here."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", root.name).strip("-") or "project"
    return f"{name}-{hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:8]}"


@dataclass(frozen=True)
class RunPaths:
    project: str
    objects: str
    run_dir: str
    work: str
    shared: str
    pidfile: str


class Remote:
    def __init__(self, box: Box):
        self.box = box
        code, out, err = box.run('printf "%s\\n%s\\n" "$HOME" "$(uname -m)"')
        if code != 0:
            raise FerryError(f"could not run commands on {box.label}: {err.decode(errors='replace').strip()}")
        self.home, self.arch = out.decode().split("\n")[:2]
        self.root = f"{self.home}/.ferry"
        self.helper = f"{self.root}/box.py"

    def paths(self, project: str, run_id: str) -> RunPaths:
        base = f"{self.root}/projects/{project}"
        return RunPaths(
            project=project,
            objects=f"{base}/objects",
            run_dir=f"{base}/.hunch/runs/{run_id}",
            work=f"{base}/work/{run_id}",
            shared=f"{base}/shared",
            pidfile=f"{base}/pids/{run_id}",
        )

    def call(self, *args: str, stdin: bytes | None = None, python: str = "python3",
             on_line: Callable[[str], None] | None = None, timeout: float = 300) -> str:
        command = " ".join([shlex.quote(python), shlex.quote(self.helper), *map(shlex.quote, args)])
        code, out, err = self.box.run(command, stdin=stdin, on_line=on_line, timeout=timeout)
        if code != 0:
            detail = (err or out).decode("utf-8", "replace").strip().splitlines()
            if code == -1 and not detail:
                raise FerryError(f"the connection to {self.box.label} closed before {args[0]} finished")
            raise FerryError(f"{args[0]} failed on {self.box.label}: {detail[-1] if detail else f'exit {code}'}")
        return out.decode("utf-8", "replace")

    def install_helper(self) -> None:
        """Copies box.py over; it is small, so it goes every time and is never stale."""
        data = resources.files("ferry").joinpath("box.py").read_bytes()
        code, _, err = self.box.run(f"mkdir -p {shlex.quote(self.root)}")
        if code != 0:
            raise FerryError(f"could not create {self.root} on {self.box.label}: {err.decode(errors='replace').strip()}")
        sftp = self.box.sftp()
        tmp = f"{self.helper}.tmp"
        with sftp.open(tmp, "wb") as f:
            f.write(data)
        sftp.posix_rename(tmp, self.helper)

    def install_uv(self) -> None:
        sha = UV_SHA256.get(self.arch)
        if sha is None:
            raise FerryError(f"{self.box.label} is {self.arch}; Ferry supports x86_64 and aarch64 Linux machines")
        url = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-{self.arch}-unknown-linux-gnu.tar.gz"
        self.call("bootstrap-uv", self.root, url, sha)

    def build_env(self, spec: EnvSpec, on_line: Callable[[str], None]) -> str:
        """Builds the environment once per spec and returns its path on the machine."""
        found: list[str] = []

        def line(text: str) -> None:
            if text.startswith(ENV_MARK):
                found.append(text[len(ENV_MARK):])
            elif text.strip():
                on_line(text)

        self.call("env", self.root, spec.key, spec.python, stdin=spec.requirements.encode(), on_line=line, timeout=1800)
        if not found:
            raise FerryError(f"building the environment on {self.box.label} did not finish")
        return found[-1]

    def missing(self, objects: str, shas: list[str]) -> list[str]:
        if not shas:
            return []
        return self.call("missing", objects, stdin="\n".join(shas).encode()).split()

    def send_objects(self, objects: str, local_objects: Path, shas: list[str]) -> None:
        if not shas:
            return
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for sha in shas:
                tar.add(str(local_objects / sha[:2] / sha[2:]), arcname=f"{sha[:2]}/{sha[2:]}")
        self.call("unpack", objects, stdin=buf.getvalue())

    def send_files(self, remote_dir: str, local_dir: Path, names: list[str]) -> None:
        self.call("mkdirs", remote_dir)
        sftp = self.box.sftp()
        for name in names:
            path = local_dir / name
            if path.is_file():
                sftp.put(str(path), f"{remote_dir}/{name}")

    def checkout(self, paths: RunPaths) -> None:
        self.call("checkout", paths.objects, f"{paths.run_dir}/files.json", paths.work)

    def start(self, paths: RunPaths, venv: str, command: list[str], cwd: str = ".") -> int:
        """Starts the run in `cwd`, the project-relative folder it was launched from here."""
        workdir = paths.work if cwd in ("", ".") else f"{paths.work}/{cwd}"
        out = self.call("start", paths.run_dir, workdir, venv, paths.shared, paths.pidfile, "--", *command)
        return int(out.strip().splitlines()[-1])

    def stop(self, pidfile: str) -> str:
        return self.call("stop", pidfile).strip()

    def reap(self, venv: str, run_dir: str, pidfile: str) -> str:
        """Ends the run on the machine if its supervisor died without an end: alive, ended or reaped."""
        return self.call("reap", run_dir, pidfile, python=f"{venv}/bin/python").strip()

    def alive(self, pidfile: str) -> bool:
        return self.call("alive", pidfile).strip() == "yes"

    def hashes(self, run_dir: str) -> dict[str, str]:
        return json.loads(self.call("hash", run_dir))

    def now(self) -> float:
        return float(self.call("now").strip())

    def probe(self) -> dict:
        """What the machine is: its name, architecture, GPU, driver and Python."""
        script = (
            'PATH="$PATH:/usr/lib/wsl/lib"; '
            'echo "hostname=$(hostname)"; echo "arch=$(uname -m)"; '
            'echo "python=$(python3 --version 2>/dev/null | cut -d" " -f2)"; '
            'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 | sed "s/^/gpu=/"'
        )
        code, out, _ = self.box.run(script)
        info: dict[str, str] = {}
        for line in out.decode("utf-8", "replace").splitlines():
            key, _, value = line.partition("=")
            info[key.strip()] = value.strip()
        if "gpu" in info and "," in info["gpu"]:
            info["gpu"], info["driver"] = (part.strip() for part in info["gpu"].rsplit(",", 1))
        return info
