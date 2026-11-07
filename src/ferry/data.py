"""Files a run needs that are not its code, and files a run writes.

Data sent with --data goes into the project's shared folder on the machine, at
the same paths as in the project here, and every run's folder links to it, so
`open("data/train.bin")` works there as it does here. It goes once: a file is
sent again only when its size or modification time changed.

Files a run writes (checkpoints, plots) stay in its folder on the machine until
`ferry pull` brings them home.
"""

from __future__ import annotations

import os
import posixpath
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .box import PART_SUFFIX  # a file still arriving, on either side
from .config import FerryError
from .remote import Remote

SKIP_DIRS = {".git", ".hunch", "__pycache__", "node_modules"}
LIST_NAMES = 5


@dataclass(frozen=True)
class LocalFile:
    rel: str  # relative to the project, with /
    path: Path
    size: int
    mtime: int


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def count_files(count: int) -> str:
    return f"{count} file{'' if count == 1 else 's'}"


def _names(names: list[str]) -> str:
    shown = ", ".join(names[:LIST_NAMES])
    return shown + (f" and {len(names) - LIST_NAMES} more" if len(names) > LIST_NAMES else "")


def _local(root: Path, path: Path) -> LocalFile:
    st = path.stat()
    return LocalFile(path.relative_to(root).as_posix(), path, st.st_size, int(st.st_mtime))


def collect(root: Path, given: list[str]) -> list[LocalFile]:
    """The files under each given path. They must be inside the project, so they have
    a place in the run's folder on the machine."""
    root = root.resolve()
    found: dict[str, LocalFile] = {}
    for name in given:
        path = Path(name).resolve()
        if path == root:
            raise FerryError("--data takes the files or folders that hold data, not the whole project")
        try:
            path.relative_to(root)
        except ValueError:
            raise FerryError(f"--data {name}: only files inside the project ({root}) can be sent") from None
        if not path.exists():
            raise FerryError(f"--data {name}: no such file or folder")
        if path.is_file():
            found[path.relative_to(root).as_posix()] = _local(root, path)
            continue
        for folder, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not (Path(folder, d) / "pyvenv.cfg").exists()]
            for file in files:
                item = Path(folder, file)
                if item.is_file() and not file.endswith(PART_SUFFIX):
                    entry = _local(root, item)
                    found[entry.rel] = entry
    return sorted(found.values(), key=lambda f: f.rel)


class _Progress:
    """One line that counts up while files move, on a terminal; silent otherwise."""

    def __init__(self, verb: str, total: int):
        self.verb, self.total, self.done = verb, total, 0
        self.live = sys.stderr.isatty()

    def file_done(self, size: int) -> None:
        self.done += size
        self.show(0)

    def show(self, partial: int) -> None:
        if self.live:
            print(f"\r  {self.verb} {human(self.done + partial)} of {human(self.total)}   ", end="", file=sys.stderr, flush=True)

    def end(self) -> None:
        if self.live:
            print(file=sys.stderr)


def send(remote: Remote, shared: str, files: list[LocalFile]) -> None:
    """Sends the data files the machine lacks or has in another version."""
    if not files:
        return
    wanted = set(remote.want(shared, [[f.rel, f.size, f.mtime] for f in files]))
    todo = [f for f in files if f.rel in wanted]
    total = sum(f.size for f in todo)
    kept = len(files) - len(todo)
    print(f"ferry: data: sending {len(todo)} of {count_files(len(files))} ({human(total)})"
          + (f", {kept} already on {remote.box.label}" if kept else ""), file=sys.stderr)
    if not todo:
        return
    remote.mkdirs(sorted({posixpath.dirname(f"{shared}/{f.rel}") for f in todo}))
    sftp = remote.box.sftp()
    progress = _Progress("sent", total)
    for f in todo:
        dest = f"{shared}/{f.rel}"
        part = dest + PART_SUFFIX
        sftp.put(str(f.path), part, callback=lambda done, _size: progress.show(done))
        sftp.utime(part, (f.mtime, f.mtime))  # the quick check next time compares these
        sftp.posix_rename(part, dest)  # a run never sees half a file
        progress.file_done(f.size)
    progress.end()


def warn_missing(remote: Remote, shared: str, root: Path, skipped: list[str]) -> None:
    """Says which files the code snapshot left out (binary, too big) that the machine does not have either."""
    files = [_local(root, root / rel) for rel in skipped if (root / rel).is_file()]
    if not files:
        return
    missing = remote.want(shared, [[f.rel, f.size, f.mtime] for f in files])
    if missing:
        print(f"ferry: {count_files(len(missing))} left out of the code (binary, over 1 MB or past the limits) "
              f"{'is' if len(missing) == 1 else 'are'} not on {remote.box.label} either: {_names(missing)}. "
              f"If the run needs them, add --data PATH", file=sys.stderr)


def here(root: Path) -> str:
    """Where you are in the project, as a project-relative folder ("" at its top). Paths
    you type for a run are relative to it, as they would be for a run on this computer."""
    rel = Path.cwd().resolve().relative_to(root.resolve()).as_posix()
    return "" if rel == "." else rel


def work_dir(state: dict) -> str:
    """The folder the run ran in on the machine."""
    return state.get("work") or state["dir"].replace("/.hunch/runs/", "/work/")


def summary(outputs: list[list], where: str = "") -> list[str]:
    """What a run wrote, one entry per file or folder at the level of `where` (as from
    `here()`), largest first."""
    groups: dict[str, list] = {}
    for rel, size in outputs:
        parts = posixpath.relpath(rel, where or ".").split("/")
        depth = 0
        while depth < len(parts) - 1 and parts[depth] == "..":
            depth += 1
        name = "/".join(parts[:depth + 1]) + ("/" if len(parts) > depth + 1 else "")
        group = groups.setdefault(name, [0, 0])
        group[0] += 1
        group[1] += size
    ordered = sorted(groups.items(), key=lambda item: -item[1][1])
    return [f"{name} ({count_files(count)}, {human(size)})" if name.endswith("/") else f"{name} ({human(size)})"
            for name, (count, size) in ordered]


def pull(remote: Remote, work: str, given: list[str], dest: Path, force: bool, where: str = "") -> int:
    """Copies files or folders from the run's folder on the machine into `dest`, each
    under its own name, as scp does. Paths are relative to `where` in the project.
    Returns the number of files."""
    sftp = remote.box.sftp()
    plan: list[tuple[str, Path, int]] = []

    def walk(remote_path: str, local: Path) -> None:
        for entry in sftp.listdir_attr(remote_path):
            mode = entry.st_mode or 0
            if stat.S_ISLNK(mode):
                continue  # linked data, not something the run wrote
            if stat.S_ISDIR(mode):
                walk(f"{remote_path}/{entry.filename}", local / entry.filename)
            elif stat.S_ISREG(mode):
                plan.append((f"{remote_path}/{entry.filename}", local / entry.filename, entry.st_size or 0))

    for name in given:
        typed = name.replace("\\", "/")
        rel = PurePosixPath(posixpath.normpath(posixpath.join(where, typed)))
        if typed.startswith("/") or ".." in rel.parts or str(rel) in ("", "."):
            raise FerryError(f"{name}: name a file or folder inside the run's folder, such as checkpoints/best.pt")
        remote_path = f"{work}/{rel.as_posix()}"
        try:
            attrs = sftp.stat(remote_path)
        except FileNotFoundError:
            raise FerryError(f"{name} is not in the run's folder on {remote.box.label}; `ferry pull RUN` lists what is") from None
        if stat.S_ISDIR(attrs.st_mode or 0):
            walk(remote_path, dest / rel.name)
        else:
            plan.append((remote_path, dest / rel.name, attrs.st_size or 0))

    existing = [str(local) for _, local, _ in plan if local.exists()]
    if existing and not force:
        raise FerryError(f"{_names(existing)} {'is' if len(existing) == 1 else 'are'} already there; "
                         "pull into another folder with --to DIR, or replace with --force")
    progress = _Progress("copied", sum(size for _, _, size in plan))
    for remote_path, local, size in plan:
        local.parent.mkdir(parents=True, exist_ok=True)
        part = local.with_name(local.name + PART_SUFFIX)
        sftp.get(remote_path, str(part), callback=lambda done, _size: progress.show(done))
        os.replace(part, local)
        progress.file_done(size)
    progress.end()
    return len(plan)
