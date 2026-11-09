"""Ferry's watchdog on a rented machine: onstart starts it at every boot. Standard library only.

At the deadline it destroys the machine, with the key Vast gives each instance
(CONTAINER_API_KEY), which can act on this instance only; the account's key never comes
here. When a run has ended and no laptop has brought it home for IDLE seconds, it stops
the machine instead: GPU billing ends, the disk and the run stay, and Ferry starts the
machine again to bring the run home. After such a start it allows GRACE seconds for that,
even past the deadline, and destroys the machine as soon as the run is home.

    python3 watch.py STATE_DIR PROJECTS_DIR IDLE_SECONDS

STATE_DIR holds `deadline` (Unix time), `home/<run id>` for each run a laptop brought
home, and `idle-stopped` after an idle stop.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://console.vast.ai/api/v0/instances/"
GRACE = 30 * 60
EVERY = 20


def from_main_process(name: str) -> str:
    """A variable of the container's main process: SSH sessions and onstart may not have it."""
    try:
        with open("/proc/1/environ", "rb") as f:
            items = f.read().split(b"\0")
    except OSError:
        return ""
    for item in items:
        key, _, value = item.partition(b"=")
        if key.decode("utf-8", "replace") == name:
            return value.decode("utf-8", "replace")
    return ""


def variable(name: str) -> str:
    return os.environ.get(name) or from_main_process(name)


def runs(projects: str, home: str) -> tuple[bool, bool, float]:
    """Whether a run is still going, whether an ended run has not been brought home, and
    when the last such run ended."""
    going = waiting = False
    last = 0.0
    for run in glob.glob(os.path.join(projects, "*", ".hunch", "runs", "*")):
        end = os.path.join(run, "end.json")
        if not os.path.exists(end):
            going = True
        elif not os.path.exists(os.path.join(home, os.path.basename(run))):
            waiting = True
            last = max(last, os.path.getmtime(end))
    return going, waiting, last


def decide(now: float, deadline: float, booted: float, idle: float, going: bool, waiting: bool, last: float,
           woke_from_idle: bool) -> str | None:
    """'destroy', 'stop' or None. Past the deadline, only a run that still waits to come
    home, on a machine just started again for it, holds the destroy off, for GRACE at most."""
    bringing_home = woke_from_idle and waiting and now < booted + GRACE
    if now >= deadline and not bringing_home:
        return "destroy"
    if waiting and not going and now - max(last, booted) >= idle:
        return "stop"
    return None


def call(method: str, body: dict, key: str, instance: str) -> int:
    request = urllib.request.Request(API + instance + "/", data=json.dumps(body).encode(), method=method, headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "ferry-watch"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def say(text: str) -> None:
    line = "FERRY-WATCH %d %s" % (time.time(), text)
    print(line, flush=True)
    try:  # into the instance's log too, which the laptop can read over HTTPS
        with open("/proc/1/fd/1", "w") as out:
            out.write(line + "\n")
    except OSError:
        pass


def main() -> None:
    state, projects, idle = sys.argv[1], sys.argv[2], float(sys.argv[3])
    key, instance = variable("CONTAINER_API_KEY"), variable("CONTAINER_ID")
    booted = time.time()
    marker = os.path.join(state, "idle-stopped")
    woke_from_idle = os.path.exists(marker)
    say("watching instance %s; key %s; woke from an idle stop: %s" % (instance, "found" if key else "MISSING", woke_from_idle))
    while True:
        try:
            with open(os.path.join(state, "deadline")) as f:
                deadline = float(f.read().strip())
            going, waiting, last = runs(projects, os.path.join(state, "home"))
            action = decide(time.time(), deadline, booted, idle, going, waiting, last, woke_from_idle)
        except (OSError, ValueError) as e:
            say("cannot read the state: %s" % e)
            action = None
        if action == "stop":
            open(marker, "w").close()
        if action:
            code = call("DELETE", {}, key, instance) if action == "destroy" else call("PUT", {"state": "stopped"}, key, instance)
            say("%s answered %s" % (action, code))
            time.sleep(60 if code == 200 else 15)  # on 200 the machine goes away; otherwise try again
            continue
        time.sleep(EVERY)


if __name__ == "__main__":
    main()
