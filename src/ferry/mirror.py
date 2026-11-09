"""Brings a run home while it trains. The machine writes the run folder as Hunch
always does; this side copies it into the local .hunch/ so `hunch log` and the
editor plugins see it like any other run.

Append-only files are read from where the last copy stopped, complete lines
only, so a dropped connection never duplicates or loses a line. Small JSON files
are copied whole when they change and swapped in atomically. The copy's state
lives in remote.json, next to the run's files.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import paramiko
from hunch.store import LOST_AFTER_SECONDS, read_json, write_json

from . import data, rent
from .config import FerryError
from .remote import Remote

REMOTE_FILE = "remote.json"
GONE_FILE = "remote.gone.json"  # remote.json, set aside by `ferry forget`; the next copy puts it back
APPEND = ("metrics.jsonl", "notes.jsonl", "output.log")
WHOLE = ("run.json", "files.json", "config.json", "env.json", "summary.json", "end.json")
POLL_SECONDS = 2.0
LIVENESS_EVERY = 5  # polls between checks that the run's supervisor is still alive


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).astimezone().isoformat(timespec="seconds")


def write_state(run_dir: Path, state: dict) -> None:
    write_json(run_dir / REMOTE_FILE, state)
    (run_dir / GONE_FILE).unlink(missing_ok=True)


def read_state(run_dir: Path) -> dict:
    state = read_json(run_dir / REMOTE_FILE) or read_json(run_dir / GONE_FILE)
    if state is None:
        raise FerryError(f"{run_dir.name} did not run on another machine")
    return state


def pull(sftp: paramiko.SFTPClient, state: dict, run_dir: Path, box_clock_offset: float, echo=None) -> bool:
    """Copies what is new. Returns True once the run has ended and is fully home."""
    remote_dir = state["dir"]
    entries = {e.filename: e for e in sftp.listdir_attr(remote_dir) if stat.S_ISREG(e.st_mode or 0)}
    ended = "end.json" in entries
    offsets = state.setdefault("offsets", {})
    seen = state.setdefault("seen", {})

    for name in APPEND:
        entry = entries.get(name)
        if entry is None:
            continue
        local = run_dir / name
        offset = offsets.get(name, local.stat().st_size if local.exists() else 0)
        if entry.st_size > offset:
            with sftp.open(f"{remote_dir}/{name}", "rb") as f:
                f.seek(offset)
                # Ask for every piece at once: one request per round trip read a big log at 0.35 MB/s
                # from a machine 71 ms away, this at 9.9 MB/s (measured 8 Nov 2026).
                f.prefetch(entry.st_size)
                data = f.read(entry.st_size - offset)
            if not ended:
                data = data[: data.rfind(b"\n") + 1]  # complete lines only; the rest comes next time
            if data:
                with open(local, "ab") as out:
                    out.write(data)
                offset += len(data)
                if echo is not None and name == "output.log":
                    echo(data)
        offsets[name] = offset

    for name in WHOLE:
        entry = entries.get(name)
        if entry is None:
            continue
        mark = f"{entry.st_size}:{entry.st_mtime}"
        if seen.get(name) == mark and (run_dir / name).exists():
            continue
        with sftp.open(f"{remote_dir}/{name}", "rb") as f:
            data = f.read()
        tmp = run_dir / f".{name}.ferry"
        tmp.write_bytes(data)
        os.replace(tmp, run_dir / name)
        seen[name] = mark

    beat = entries.get("heartbeat")
    if beat is not None:
        age = (time.time() + box_clock_offset) - beat.st_mtime
        state["heartbeat_age"] = round(age, 1)
        if age < LOST_AFTER_SECONDS:
            (run_dir / "heartbeat").touch()
    state["synced_at"] = _iso(time.time())
    write_state(run_dir, state)
    return ended


def verify(remote: Remote, state: dict, run_dir: Path) -> list[str]:
    """Checks every file at home against the machine's copy. Returns the names that differ."""
    differ = []
    for name, digest in remote.hashes(state["dir"]).items():
        local = run_dir / name
        if not local.exists() or hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            differ.append(name)
    return differ


def being_copied(run_dir: Path) -> bool:
    """Whether a ferry is copying this run home right now."""
    try:
        with _CopyLock(run_dir):
            return False
    except FerryError:
        return True


class _CopyLock:
    """Only one ferry copies a run at a time; the OS lets go of the lock if that ferry dies."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / ".ferry.lock"
        self.file = None

    def __enter__(self):
        self.file = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise FerryError(f"another ferry is already copying {self.path.parent.name}") from None
        return self

    def __exit__(self, *exc):
        self.file.close()  # closing releases the lock on both platforms
        try:
            self.path.unlink()
        except OSError:
            pass  # another ferry holds it now


MISSING_FILE = (b"FileNotFoundError", b"No such file or directory")


def _report_files(remote: Remote, state: dict, run_dir: Path, label: str, status: str) -> None:
    """Says what the run left on the machine, and, when it crashed looking for a file,
    that only the code goes there on its own."""
    short = run_dir.name[-4:]
    if status == "crashed":
        try:
            with open(run_dir / "output.log", "rb") as f:
                tail = f.read()[-4000:]
        except OSError:
            tail = b""
        if any(mark in tail for mark in MISSING_FILE):
            print(f"ferry: the run could not find a file. Only the code goes to {label} on its own; send data with "
                  f"--data PATH, and it stays there for later runs.", file=sys.stderr)
    try:
        outputs = remote.outputs(data.work_dir(state), state["dir"])
    except (FerryError, OSError, paramiko.SSHException):
        return  # an older run folder, or the machine went away: nothing to say
    if outputs:
        try:
            where = data.here(run_dir.parent.parent.parent)  # <project>/.hunch/runs/<id>
        except ValueError:
            where = ""
        print(f"ferry: the run wrote {', '.join(data.summary(outputs, where)[:4])} on {label}. "
              f"Bring them home with: ferry pull {short} PATH", file=sys.stderr)


CAPPED = 124  # what follow returns when a rented machine's time is up before the run ended


def follow(remote: Remote, run_dir: Path, echo_output: bool = True, until: float | None = None) -> int:
    """Copies the run home until it ends, or until `until` (a rented machine's time cap).
    Ctrl+C stops copying, never the run."""
    with _CopyLock(run_dir):
        return _follow(remote, run_dir, echo_output, until)


def _follow(remote: Remote, run_dir: Path, echo_output: bool, until: float | None = None) -> int:
    state = read_state(run_dir)
    label = state.get("machine", remote.box.label)
    offset = remote.now() - time.time()
    echo = (lambda data: (sys.stdout.write(data.decode("utf-8", "replace")), sys.stdout.flush())) if echo_output else None
    failures = 0
    polls = 0
    checking = bool(state.get("venv") and state.get("pidfile"))
    try:
        while True:
            try:
                ended = pull(remote.box.sftp(), state, run_dir, offset, echo)
                failures = 0
                polls += 1
                if not ended and checking and polls % LIVENESS_EVERY == 0:
                    # A supervisor that died without writing end.json must not look like a running run.
                    try:
                        verdict = remote.reap(state["venv"], state["dir"], state["pidfile"])
                    except FerryError as e:
                        checking = False  # a helper problem, not a lost connection: say so once, keep copying
                        print(f"\nferry: cannot check whether the run is alive ({e})", file=sys.stderr)
                        verdict = None
                    if verdict == "reaped":
                        print(f"\nferry: the run stopped on {label} without recording an end; marked it crashed.", file=sys.stderr)
                        continue
            except FileNotFoundError:  # the connection is fine, the run's folder is not there
                short = run_dir.name[-4:]
                raise FerryError(f"run {short} is no longer on {label}. If it is gone for good, "
                                 f"`ferry forget {short}` marks it lost") from None
            except (OSError, EOFError, paramiko.SSHException, FerryError) as e:
                failures += 1
                wait = min(60, 2 ** failures)
                print(f"\nferry: lost the connection to {label} ({e}); trying again in {wait}s. The run keeps going.", file=sys.stderr)
                time.sleep(wait)
                try:
                    remote.box.connect()
                    offset = remote.now() - time.time()
                except FerryError as again:
                    print(f"ferry: {again}", file=sys.stderr)
                continue
            if ended:
                break
            if until is not None and time.time() >= until:
                print(f"\nferry: the time you rented {label} for is up, before the run ended.", file=sys.stderr)
                return CAPPED
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print(f"\nferry: stopped copying. The run is still going on {label}; `ferry attach {run_dir.name[-4:]}` picks it up again.",
              file=sys.stderr)
        return 130

    differ = verify(remote, state, run_dir)
    for name in differ:  # a file rewritten on the machine after it was copied: take it whole
        with remote.box.sftp().open(f"{state['dir']}/{name}", "rb") as f:
            f.prefetch()
            (run_dir / name).write_bytes(f.read())
    if differ and verify(remote, state, run_dir):
        raise FerryError(f"the copy of {run_dir.name} does not match {label}'s; run `ferry attach` to try again")
    state["complete"] = True
    write_state(run_dir, state)
    if state.get("provider"):
        rent.mark_home(remote.box, run_dir.name)
    end = read_json(run_dir / "end.json", {})
    status = end.get("status", "finished")
    print(f"ferry: run {run_dir.name} {status} on {label}; everything is home.", file=sys.stderr)
    _report_files(remote, state, run_dir, label, status)
    code = end.get("exit_code")
    return 0 if status == "finished" else (code if isinstance(code, int) and code > 0 else 1)
