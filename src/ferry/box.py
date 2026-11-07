"""Ferry's helper on the machine. It is copied to ~/.ferry/box.py and run with the
machine's python3, except `supervise`, which runs in the run's own environment so
it can use Hunch. Standard library only, Python 3.8 and newer.

Usage: python3 box.py <command> [arguments]
"""

import glob
import hashlib
import io
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zlib

SHA = re.compile(r"^[0-9a-f]{40}$")
ENV_MARK = "FERRY-ENV: "
WSL_GPU_TOOLS = "/usr/lib/wsl/lib"  # nvidia-smi on WSL; an SSH session's PATH lacks it


def _with_gpu_tools(path):
    """PATH with WSL's GPU tools, so uv picks a CUDA torch and nvidia-smi works there too."""
    if os.path.isdir(WSL_GPU_TOOLS) and WSL_GPU_TOOLS not in path.split(os.pathsep):
        return path + os.pathsep + WSL_GPU_TOOLS
    return path


def _object_path(objects, sha):
    return os.path.join(objects, sha[:2], sha[2:])


def _write_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _safe_parts(rel):
    """A project-relative path as its parts, refusing anything that could leave the folder."""
    parts = rel.split("/")
    if rel.startswith("/") or "\\" in rel or ".." in parts or not all(parts):
        raise SystemExit("refusing an unsafe path: %r" % rel)
    return parts


PART_SUFFIX = ".ferry-part"  # a data file still arriving; never linked into a run


def _link_shared(shared, workdir):
    """Links what the project keeps on this machine into a run's folder, at the same paths.
    Where the code has a file of the same name, the code wins."""
    if not os.path.isdir(shared):
        return
    for name in os.listdir(shared):
        if name.endswith(PART_SUFFIX):
            continue
        source, dest = os.path.join(shared, name), os.path.join(workdir, name)
        if not os.path.lexists(dest):
            os.symlink(source, dest)
        elif os.path.isdir(source) and not os.path.islink(source) and os.path.isdir(dest) and not os.path.islink(dest):
            _link_shared(source, dest)


def _alive(pidfile):
    """Whether the supervisor named in `pidfile` still runs. A reused pid, or a finished
    process nobody has reaped yet, does not count."""
    try:
        with open(pidfile) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None
    try:
        with open("/proc/%d/stat" % pid) as f:
            if f.read().rsplit(")", 1)[1].split()[0] == "Z":
                return None
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            if os.path.basename(pidfile).encode() not in f.read():
                return None
    except (OSError, IndexError):
        pass  # no /proc here: trust the signal check
    return pid


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


def cmd_checkout(objects, files_json, workdir, shared=None):
    """Writes the run's code, exactly as recorded, into an empty folder, then links in
    what the project keeps on this machine (its data)."""
    with open(files_json, encoding="utf-8") as f:
        files = json.load(f)["files"]
    os.makedirs(workdir, exist_ok=True)
    for rel, entry in files.items():
        parts = _safe_parts(rel)
        with open(_object_path(objects, entry["sha"]), "rb") as f:
            raw = zlib.decompress(f.read())
        dest = os.path.join(workdir, *parts)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(raw[raw.index(b"\0") + 1:])
    if shared:
        _link_shared(shared, workdir)
    print(len(files))


def cmd_want(shared):
    """Reads [path, size, mtime] for data files on stdin and prints the paths this
    machine lacks or has in another version, as rsync's quick check does."""
    want = []
    for rel, size, mtime in json.load(sys.stdin):
        try:
            st = os.stat(os.path.join(shared, *_safe_parts(rel)))
        except FileNotFoundError:
            want.append(rel)
            continue
        if st.st_size != size or int(st.st_mtime) != int(mtime):
            want.append(rel)
    print(json.dumps(want))


def cmd_outputs(workdir, files_json):
    """Prints [path, size] for each file a run wrote in its folder: everything but its
    code and the linked data."""
    if not os.path.isdir(workdir):
        raise SystemExit("the run's folder is no longer on this machine (ferry clean --outputs removes it)")
    with open(files_json, encoding="utf-8") as f:
        code = set(json.load(f)["files"])
    out = []
    for folder, dirs, names in os.walk(workdir):
        dirs[:] = sorted(d for d in dirs if not os.path.islink(os.path.join(folder, d)) and d != "__pycache__")
        for name in sorted(names):
            path = os.path.join(folder, name)
            rel = os.path.relpath(path, workdir).replace(os.sep, "/")
            if rel in code or os.path.islink(path):
                continue
            out.append([rel, os.path.getsize(path)])
    print(json.dumps(out))


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
    import fcntl

    requirements = sys.stdin.read()
    os.makedirs(os.path.join(root, "envs"), exist_ok=True)
    venv = os.path.join(root, "envs", key)
    with open(venv + ".lock", "w") as lock:  # two runs starting at once build it once
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("another run is building this environment; waiting for it", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        _build_env(root, venv, python, requirements)
        open(os.path.join(venv, ".used"), "w").close()  # when a run last asked for it, for `ferry clean`
    print(ENV_MARK + venv)


def _build_env(root, venv, python, requirements):
    if os.path.exists(os.path.join(venv, ".ready")):
        return
    uv = os.path.join(root, "bin", "uv")
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = os.path.join(root, "cache", "uv")
    env["UV_PYTHON_INSTALL_DIR"] = os.path.join(root, "python")
    env["PATH"] = _with_gpu_tools(env.get("PATH", ""))
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
        sys.stdout.flush()
        code = subprocess.call(step, env=env, stdin=subprocess.DEVNULL)
        if code != 0:
            raise SystemExit("building the environment failed (%s exited with %d)" % (os.path.basename(step[0]), code))
    open(os.path.join(venv, ".ready"), "w").close()


STARTED_MARK = ".ferry-started"
START_WAIT_SECONDS = 60
STOP_GRACE_SECONDS = 30  # after asking a run to stop, before killing it


def _signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
    except (ProcessLookupError, PermissionError):
        pass


def _tail(path, lines=12):
    try:
        with open(path, "rb") as f:
            return b"".join(f.readlines()[-lines:]).decode("utf-8", "replace")
    except OSError:
        return ""


def cmd_start(run_dir, workdir, venv, shared, pidfile, *command):
    """Starts the supervisor in its own session, so it outlives the SSH connection,
    and waits until it has really started the command, or reports why it did not.
    Variables for the run (tokens, the GPUs to use) arrive as JSON on stdin and are
    handed over in memory, never written to a file."""
    if command[:1] == ("--",):
        command = command[1:]
    env = dict(os.environ)
    env.update(json.loads(sys.stdin.read() or "{}"))
    os.makedirs(os.path.dirname(pidfile), exist_ok=True)
    os.makedirs(shared, exist_ok=True)
    os.makedirs(workdir, exist_ok=True)  # a subfolder the code has no files in
    log_path = os.path.join(run_dir, "output.log")
    log = open(log_path, "ab")
    proc = subprocess.Popen(
        [os.path.join(venv, "bin", "python"), os.path.abspath(__file__), "supervise", run_dir, workdir, venv, shared, "--", *command],
        cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True,
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
    os.environ["PATH"] = _with_gpu_tools(os.environ.get("PATH", ""))  # for the GPU in env.json, and for the run
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
        # Its own process group, so stopping the run reaches everything it started (data loader workers, torchrun ranks).
        proc = subprocess.Popen(list(command), cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        (run / STARTED_MARK).touch()
    except BaseException as e:
        sys.stderr.write("ferry: the run could not start: %r\n" % (e,))
        if heartbeat is not None:
            heartbeat.stop()
        (run / STARTED_MARK).touch()
        finish_run(run, status="crashed", exit_code=None, duration=time.monotonic() - started)
        raise SystemExit(1)

    stopped = []

    def kill_if_still_running():
        try:
            proc.wait(STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _signal_group(proc.pid, signal.SIGKILL)

    def forward(signum, frame):
        if stopped:
            return
        stopped.append(signum)
        _signal_group(proc.pid, signal.SIGTERM)
        threading.Thread(target=kill_if_still_running, daemon=True).start()

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    code = None
    try:
        code = proc.wait()
    finally:
        duration = time.monotonic() - started
        if stopped:
            _signal_group(proc.pid, signal.SIGKILL)  # whatever the run started and left behind
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
    if _alive(pidfile):
        print("alive")
        return
    with open(run / "output.log", "ab") as f:
        f.write(b"ferry: the run's supervisor on this machine stopped without recording an end\n")
    finish_run(run, status="crashed", exit_code=None, duration=0.0)
    print("reaped")


def cmd_stop(pidfile):
    """Asks the run to stop and waits until it has, or until it is killed."""
    pid = _alive(pidfile)
    if pid is None:
        print("not running")
        return
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_GRACE_SECONDS + 10
    while time.monotonic() < deadline:
        if not _alive(pidfile):
            print("stopped")
            return
        time.sleep(0.5)
    print("still stopping")


def cmd_alive(pidfile):
    print("yes" if _alive(pidfile) else "no")


def cmd_states():
    """Reads [[run_dir, pidfile], ...] on stdin and prints how each run is, as JSON:
    "running", "ended", "gone" (its supervisor stopped without an end) or "missing"."""
    out = []
    for run_dir, pidfile in json.load(sys.stdin):
        if not os.path.isdir(run_dir):
            out.append("missing")
        elif os.path.exists(os.path.join(run_dir, "end.json")):
            out.append("ended")
        else:
            out.append("running" if _alive(pidfile) else "gone")
    print(json.dumps(out))


CLEAN_AFTER_DAYS = 14  # `ferry clean` removes environments no run has asked for in this long


def _envs_in_use(root):
    """The environments of the runs going on this machine now."""
    used = set()
    for pidfile in glob.glob(os.path.join(root, "projects", "*", "pids", "*")):
        pid = _alive(pidfile)
        if pid is None:
            continue
        try:
            with open("/proc/%d/cmdline" % pid, "rb") as f:
                args = f.read().split(b"\0")
            used.add(os.fsdecode(args[args.index(b"supervise") + 3]))  # supervise run_dir workdir venv
        except (OSError, ValueError, IndexError):
            pass
    return used


def _last_used(venv):
    for name in (".used", ".ready"):
        try:
            return os.path.getmtime(os.path.join(venv, name))
        except OSError:
            pass
    return os.path.getmtime(venv)  # a build that never finished


def _idle_envs(root):
    """Environments no run uses now and none has asked for in CLEAN_AFTER_DAYS, and how many others there are."""
    folder = os.path.join(root, "envs")
    in_use = _envs_in_use(root)
    old = time.time() - CLEAN_AFTER_DAYS * 86400
    idle, kept = [], 0
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        venv = os.path.join(folder, name)
        if not os.path.isdir(venv):
            continue
        if venv in in_use or _last_used(venv) > old:
            kept += 1
        else:
            idle.append(name)
    return idle, kept


def _freed_by(folders):
    """The disk space removing these folders gives back: their files that nothing outside
    them links to. uv links packages from its cache into environments, so an environment
    alone frees little until the cache goes too."""
    seen = {}
    for top in folders:
        for folder, _, names in os.walk(top):
            for name in names:
                try:
                    st = os.lstat(os.path.join(folder, name))
                except OSError:
                    continue
                if stat.S_ISREG(st.st_mode):
                    key = (st.st_dev, st.st_ino)
                    found, links, size = seen.get(key, (0, st.st_nlink, getattr(st, "st_blocks", 0) * 512 or st.st_size))
                    seen[key] = (found + 1, links, size)
    return sum(size for found, links, size in seen.values() if found >= links)


def _ended_runs(root, project):
    """The ids of the project's runs that ended here and still have their folder."""
    base = os.path.join(root, "projects", project)
    work = os.path.join(base, "work")
    return [run_id for run_id in (sorted(os.listdir(work)) if os.path.isdir(work) else [])
            if os.path.exists(os.path.join(base, ".hunch", "runs", run_id, "end.json"))
            and not _alive(os.path.join(base, "pids", run_id))]


def cmd_clean_plan(root, project=""):
    """What `ferry clean` can remove, as JSON: idle environments and what removing them
    with uv's cache frees, and the project's ended runs with the size of their folders."""
    idle, kept = _idle_envs(root)
    folders = [os.path.join(root, "envs", name) for name in idle] + [os.path.join(root, "cache", "uv")]
    runs = []
    if project:
        work = os.path.join(root, "projects", project, "work")
        runs = [{"id": run_id, "size": _freed_by([os.path.join(work, run_id)])} for run_id in _ended_runs(root, project)]
    print(json.dumps({"envs": idle, "kept": kept, "freed": _freed_by(folders) if idle else 0, "runs": runs,
                      "free": shutil.disk_usage(root).free}))


def cmd_clean(root, project=""):
    """Removes what stdin names ({"envs": [...], "runs": [...]}), checking each again: an
    environment a run asked for since, or a run that is going, stays. Clears uv's cache
    once an environment went, and prints what it removed and freed."""
    import fcntl

    asked = json.load(sys.stdin)
    before = shutil.disk_usage(root).free
    idle = set(_idle_envs(root)[0])
    envs = 0
    for name in asked.get("envs", []):
        if name not in idle:
            continue
        venv = os.path.join(root, "envs", name)
        with open(venv + ".lock", "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                continue  # a run is having it built right now
            shutil.rmtree(venv)
        envs += 1
    cache = None
    if envs:
        env = dict(os.environ)
        env["UV_CACHE_DIR"] = os.path.join(root, "cache", "uv")
        try:
            cache = subprocess.run([os.path.join(root, "bin", "uv"), "cache", "clean", "--quiet"], env=env,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=600).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            cache = False
    runs = 0
    if project:
        ended = set(_ended_runs(root, project))
        for run_id in asked.get("runs", []):
            if run_id in ended:
                shutil.rmtree(os.path.join(root, "projects", project, "work", run_id))
                runs += 1
    print(json.dumps({"envs": envs, "runs": runs, "cache": cache, "freed": shutil.disk_usage(root).free - before}))


def cmd_hash(run_dir):
    """Checksums of a run's files, so the copy at home can be checked."""
    out = {}
    for name in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path) and name not in ("heartbeat", STARTED_MARK):
            digest = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
            out[name] = digest.hexdigest()
    print(json.dumps(out))


def cmd_now():
    print("%.3f" % time.time())


def _nvidia_smi(query, fields):
    env = dict(os.environ)
    env["PATH"] = _with_gpu_tools(env.get("PATH", ""))
    try:
        proc = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"], env=env, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    rows = (line.split(",", fields - 1) for line in proc.stdout.decode("utf-8", "replace").splitlines())
    return [[part.strip() for part in row] for row in rows if len(row) == fields]


def cmd_gpus():
    """Prints the NVIDIA GPUs as JSON: index, name, memory and how much of it other
    programs already compute with (MiB). An empty list without nvidia-smi."""
    gpus, by_uuid = [], {}
    for index, uuid, memory, name in _nvidia_smi("--query-gpu=index,uuid,memory.total,name", 4):
        if index.isdigit() and memory.isdigit():
            gpu = {"index": int(index), "name": name, "memory": int(memory), "busy": 0}
            gpus.append(gpu)
            by_uuid[uuid] = gpu
    for uuid, used in _nvidia_smi("--query-compute-apps=gpu_uuid,used_memory", 2):
        if uuid in by_uuid and used.isdigit():
            by_uuid[uuid]["busy"] += int(used)
    print(json.dumps(gpus))


COMMANDS = {
    "mkdirs": cmd_mkdirs,
    "missing": cmd_missing,
    "unpack": cmd_unpack,
    "checkout": cmd_checkout,
    "want": cmd_want,
    "outputs": cmd_outputs,
    "bootstrap-uv": cmd_bootstrap_uv,
    "env": cmd_env,
    "start": cmd_start,
    "supervise": cmd_supervise,
    "stop": cmd_stop,
    "alive": cmd_alive,
    "states": cmd_states,
    "reap": cmd_reap,
    "hash": cmd_hash,
    "now": cmd_now,
    "gpus": cmd_gpus,
    "clean-plan": cmd_clean_plan,
    "clean": cmd_clean,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit("usage: box.py {%s} ..." % ",".join(COMMANDS))
    COMMANDS[sys.argv[1]](*sys.argv[2:])
