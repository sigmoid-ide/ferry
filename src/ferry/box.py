"""Ferry's helper on the machine. It is copied to ~/.ferry/box.py and run with the
machine's python3, except `supervise`, which runs in the run's own environment so
it can use Hunch. Standard library only, Python 3.8 and newer.

Usage: python3 box.py <command> [arguments]
"""

import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
import zlib

SHA = re.compile(r"^[0-9a-f]{40}$")
ENV_MARK = "FERRY-ENV: "


def _object_path(objects, sha):
    return os.path.join(objects, sha[:2], sha[2:])


def _write_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def cmd_mkdirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def cmd_missing(objects):
    """Reads object ids on stdin and prints the ones this machine does not have."""
    for sha in sys.stdin.read().split():
        if SHA.match(sha) and not os.path.exists(_object_path(objects, sha)):
            print(sha)


def cmd_unpack(objects):
    """Reads a tar of objects on stdin. Each entry is checked against its name."""
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tar:
        count = 0
        for member in tar:
            if not member.isfile():
                continue
            sha = member.name.replace("/", "")
            if not SHA.match(sha) or member.name != sha[:2] + "/" + sha[2:]:
                raise SystemExit("unexpected entry in the upload: %r" % member.name)
            data = tar.extractfile(member).read()
            if hashlib.sha1(zlib.decompress(data)).hexdigest() != sha:
                raise SystemExit("object %s arrived damaged" % sha)
            _write_atomic(_object_path(objects, sha), data)
            count += 1
    print(count)


def cmd_checkout(objects, files_json, workdir):
    """Writes the run's code, exactly as recorded, into an empty folder."""
    with open(files_json, encoding="utf-8") as f:
        files = json.load(f)["files"]
    os.makedirs(workdir, exist_ok=True)
    for rel, entry in files.items():
        parts = rel.split("/")
        if rel.startswith("/") or "\\" in rel or ".." in parts or not all(parts):
            raise SystemExit("refusing an unsafe path in the snapshot: %r" % rel)
        with open(_object_path(objects, entry["sha"]), "rb") as f:
            raw = zlib.decompress(f.read())
        dest = os.path.join(workdir, *parts)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(raw[raw.index(b"\0") + 1:])
    print(len(files))


def cmd_bootstrap_uv(root, url, sha256):
    """Installs uv into ~/.ferry/bin, checked against the checksum Ferry pins."""
    dest = os.path.join(root, "bin", "uv")
    if os.access(dest, os.X_OK):
        print(dest)
        return
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit("the uv download does not match its pinned checksum; not installing it")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        member = next(m for m in tar if m.isfile() and m.name.rsplit("/", 1)[-1] == "uv")
        binary = tar.extractfile(member).read()
    _write_atomic(dest, binary)
    os.chmod(dest, 0o755)
    print(dest)


def cmd_env(root, key, python):
    """Builds (once) the environment named `key` from requirements on stdin, and
    prints its path. Torch is picked to match this machine's driver."""
    requirements = sys.stdin.read()
    venv = os.path.join(root, "envs", key)
    if os.path.exists(os.path.join(venv, ".ready")):
        print(ENV_MARK + venv)
        return
    uv = os.path.join(root, "bin", "uv")
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = os.path.join(root, "cache", "uv")
    env["UV_PYTHON_INSTALL_DIR"] = os.path.join(root, "python")
    shutil.rmtree(venv, ignore_errors=True)
    steps = [
        [uv, "venv", "--quiet", "--python", python, venv],
        [uv, "pip", "install", "--python", os.path.join(venv, "bin", "python"), "--torch-backend", "auto",
         "-r", os.path.join(venv, "requirements.txt")],
    ]
    os.makedirs(venv, exist_ok=True)
    for i, step in enumerate(steps):
        if i == 1:
            with open(os.path.join(venv, "requirements.txt"), "w", encoding="utf-8") as f:
                f.write(requirements)
        code = subprocess.call(step, env=env, stdin=subprocess.DEVNULL)
        if code != 0:
            raise SystemExit("building the environment failed (%s exited with %d)" % (os.path.basename(step[0]), code))
    open(os.path.join(venv, ".ready"), "w").close()
    print(ENV_MARK + venv)


STARTED_MARK = ".ferry-started"
START_WAIT_SECONDS = 60


def _tail(path, lines=12):
    try:
        with open(path, "rb") as f:
            return b"".join(f.readlines()[-lines:]).decode("utf-8", "replace")
    except OSError:
        return ""


def cmd_start(run_dir, workdir, venv, shared, pidfile, *command):
    """Starts the supervisor in its own session, so it outlives the SSH connection,
    and waits until it has really started the command, or reports why it did not."""
    if command[:1] == ("--",):
        command = command[1:]
    os.makedirs(os.path.dirname(pidfile), exist_ok=True)
    os.makedirs(shared, exist_ok=True)
    os.makedirs(workdir, exist_ok=True)  # a subfolder the code has no files in
    log_path = os.path.join(run_dir, "output.log")
    log = open(log_path, "ab")
    proc = subprocess.Popen(
        [os.path.join(venv, "bin", "python"), os.path.abspath(__file__), "supervise", run_dir, workdir, venv, shared, "--", *command],
        cwd=workdir, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True,
    )
    with open(pidfile, "w") as f:
        f.write(str(proc.pid))
    deadline = time.monotonic() + START_WAIT_SECONDS
    while time.monotonic() < deadline:
        if os.path.exists(os.path.join(run_dir, STARTED_MARK)):
            print(proc.pid)
            return
        if proc.poll() is not None:
            sys.stderr.write(_tail(log_path))
            raise SystemExit("the run stopped while starting (exit %d)" % proc.returncode)
        time.sleep(0.2)
    raise SystemExit("the run did not start within %d seconds" % START_WAIT_SECONDS)


def cmd_supervise(run_dir, workdir, venv, shared, *command):
    """Runs the command like `hunch run` does: output to output.log, a heartbeat, and
    end.json when it ends, even when something here fails. Runs inside the run's
    environment, which has Hunch."""
    from pathlib import Path

    from hunch.capture import capture_env, capture_host
    from hunch.store import Heartbeat, finish_run, read_json, write_json

    if command[:1] == ("--",):
        command = command[1:]
    run = Path(run_dir)
    started = time.monotonic()
    heartbeat = None
    try:
        write_json(run / "env.json", capture_env(None))
        meta = read_json(run / "run.json", {})
        meta["host"] = capture_host()
        write_json(run / "run.json", meta)

        env = dict(os.environ)
        env["PATH"] = os.path.join(venv, "bin") + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = venv
        env["HUNCH_RUN_DIR"] = run_dir
        env["FERRY_SHARED"] = shared
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        heartbeat = Heartbeat(run / "heartbeat")
        heartbeat.start()
        log = open(run / "output.log", "ab", buffering=0)
        proc = subprocess.Popen(list(command), cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        (run / STARTED_MARK).touch()
    except BaseException as e:
        sys.stderr.write("ferry: the run could not start: %r\n" % (e,))
        if heartbeat is not None:
            heartbeat.stop()
        (run / STARTED_MARK).touch()
        finish_run(run, status="crashed", exit_code=None, duration=time.monotonic() - started)
        raise SystemExit(1)

    stopped = []

    def forward(signum, frame):
        stopped.append(signum)
        proc.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    code = None
    try:
        code = proc.wait()
    finally:
        duration = time.monotonic() - started
        heartbeat.stop()
        if code == 0 and not stopped:
            status = "finished"
        elif stopped or code is None or code < 0:
            status = "killed"
        else:
            status = "crashed"
        finish_run(run, status=status, exit_code=code, duration=duration)


def cmd_reap(run_dir, pidfile):
    """Ends a run whose supervisor is gone without writing end.json, so it never
    looks like it is still running. Runs inside the run's environment."""
    from pathlib import Path

    from hunch.store import finish_run

    run = Path(run_dir)
    if (run / "end.json").exists():
        print("ended")
        return
    try:
        with open(pidfile) as f:
            os.kill(int(f.read().strip()), 0)
        print("alive")
        return
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    with open(run / "output.log", "ab") as f:
        f.write(b"ferry: the run's supervisor on this machine stopped without recording an end\n")
    finish_run(run, status="crashed", exit_code=None, duration=0.0)
    print("reaped")


def cmd_stop(pidfile):
    try:
        with open(pidfile) as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
    except (FileNotFoundError, ProcessLookupError, ValueError):
        print("not running")
        return
    print("stopping")


def cmd_alive(pidfile):
    try:
        with open(pidfile) as f:
            os.kill(int(f.read().strip()), 0)
        print("yes")
    except (FileNotFoundError, ProcessLookupError, ValueError, PermissionError):
        print("no")


def cmd_hash(run_dir):
    """Checksums of a run's files, so the copy at home can be checked."""
    out = {}
    for name in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path) and name not in ("heartbeat", STARTED_MARK):
            with open(path, "rb") as f:
                out[name] = hashlib.sha256(f.read()).hexdigest()
    print(json.dumps(out))


def cmd_now():
    print("%.3f" % time.time())


COMMANDS = {
    "mkdirs": cmd_mkdirs,
    "missing": cmd_missing,
    "unpack": cmd_unpack,
    "checkout": cmd_checkout,
    "bootstrap-uv": cmd_bootstrap_uv,
    "env": cmd_env,
    "start": cmd_start,
    "supervise": cmd_supervise,
    "stop": cmd_stop,
    "alive": cmd_alive,
    "reap": cmd_reap,
    "hash": cmd_hash,
    "now": cmd_now,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit("usage: box.py {%s} ..." % ",".join(COMMANDS))
    COMMANDS[sys.argv[1]](*sys.argv[2:])
