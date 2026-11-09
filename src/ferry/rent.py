"""Renting a GPU with your own Vast.ai account: pick an offer, rent it, put the time cap on
it, and give it back.

A rented machine is a machine like any other once it is up: `vast-<instance>` in
machines.json, reached over its direct SSH port as root, with its own SSH key and its own
pinned host keys (in rented/<name>/, deleted at release). Its onstart script installs that
key, prints the host keys to the instance log, which Ferry reads over HTTPS before the
first connect, and starts watch.py, which holds the time cap on the machine itself.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import re
import shlex
import shutil
import socket
import sys
import time
from importlib import resources

import paramiko
from hunch.store import HunchError, Store, read_json, write_json

from .config import FerryError, Machine, can_ask, get_machine, home, remove_machine, rented_dir, save_machine
from .remote import Remote
from .ssh import Box, connect_first
from .vast import DISK_GB, Offer, Vast, VastError, stored_key

IMAGE = "vastai/base-image:@vastai-automatic-tag"
LABEL = "ferry"
STATE = "/root/.ferry/watch"
PROJECTS = "/root/.ferry/projects"
IDLE_SECONDS = 15 * 60  # an ended run nobody brought home: the machine stops itself after this
ONSTART_LIMIT = 4048  # characters Vast takes in onstart
BOOT_WAIT = 15 * 60  # an image that has to be downloaded first took about 5 minutes (8 Nov 2026)
LOG_WAIT = 3 * 60
PORT_WAIT = 15  # direct ports either answered within a second or never (8 Nov 2026)
LOGIN_WAIT = 90
TRIES = 3
NEARBY = 1.2  # when an offer fails, the next may cost up to this much more per hour
SKIP_DAYS = 7
MIN_SECONDS = 10 * 60
MARGIN = 3600  # an offer whose host contract ends sooner after the cap than this is left out
DURATION = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+)m)?")

ONSTART = """W=__STATE__
mkdir -p /root/.ssh "$W/home" && chmod 700 /root/.ssh
grep -qxF '__PUBKEY__' /root/.ssh/authorized_keys 2>/dev/null || echo '__PUBKEY__' >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
[ -f "$W/deadline" ] || echo $(( $(date +%s) + __SECONDS__ )) > "$W/deadline"
echo '__WATCH__' | base64 -d | gunzip > "$W/watch.py"
p=$(cat "$W/pid" 2>/dev/null); [ -n "$p" ] && grep -q watch.py "/proc/$p/cmdline" 2>/dev/null && kill "$p"
nohup setsid python3 "$W/watch.py" "$W" __PROJECTS__ __IDLE__ >> "$W/log" 2>&1 < /dev/null &
echo $! > "$W/pid"
( for i in $(seq 1 90); do [ -f /etc/ssh/ssh_host_ed25519_key.pub ] && break; sleep 2; done
  for f in /etc/ssh/ssh_host_ed25519_key.pub /etc/ssh/ssh_host_ecdsa_key.pub; do [ -f "$f" ] && echo "FERRY-HOSTKEY $(cat "$f")"; done ) &
"""


def say(text: str) -> None:
    print(f"ferry: {text}", file=sys.stderr)


def parse_duration(text: str) -> int:
    """Seconds in 2h, 90m, 1h30m or 1.5h."""
    match = DURATION.fullmatch(text.strip().lower())
    if not text.strip() or match is None or not any(match.groups()):
        raise FerryError(f"--for {text!r}: give a time such as 2h, 90m or 1h30m")
    seconds = round(float(match[1] or 0) * 3600 + int(match[2] or 0) * 60)
    if seconds < MIN_SECONDS:
        raise FerryError(f"--for {text}: at least 10m, since a machine can take 5 minutes to start")
    return seconds


def duration(seconds: float) -> str:
    minutes = max(0, round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def money(dollars: float) -> str:
    return "under a cent" if dollars < 0.01 else f"${dollars:.2f}"


def client() -> Vast:
    key = stored_key()
    if not key:
        raise FerryError("log in to Vast.ai first: ferry login vast (or set VAST_API_KEY)")
    return Vast(key)


def onstart(public_key: str, seconds: int) -> str:
    watch = resources.files("ferry").joinpath("watch.py").read_bytes()
    packed = base64.b64encode(gzip.compress(watch, mtime=0)).decode()
    script = (ONSTART.replace("__STATE__", STATE).replace("__PROJECTS__", PROJECTS).replace("__PUBKEY__", public_key)
              .replace("__SECONDS__", str(int(seconds))).replace("__IDLE__", str(IDLE_SECONDS)).replace("__WATCH__", packed))
    if len(script) > ONSTART_LIMIT:
        raise FerryError(f"the machine's start script is {len(script)} characters, over Vast's {ONSTART_LIMIT}")
    return script


def new_key(comment: str) -> tuple[str, str]:
    """A fresh ed25519 key for one machine: (private key in OpenSSH form, public key line)."""
    from cryptography.hazmat.primitives import serialization  # paramiko's own dependency
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH,
                                serialization.NoEncryption()).decode()
    public = private.public_key().public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH).decode()
    return pem, f"{public} {comment}"


# Hosts whose direct port never answered, so they are not picked again for a while.

def _skip_path():
    return home() / "vast-skip.json"


def skipped() -> dict[str, float]:
    try:
        entries = json.loads(_skip_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    cutoff = time.time() - SKIP_DAYS * 86400
    return {machine: at for machine, at in entries.items() if isinstance(at, (int, float)) and at > cutoff}


def skip(machine: int) -> None:
    entries = skipped()
    entries[str(machine)] = time.time()
    path = _skip_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")


def is_offer(choice: str) -> bool:
    """An offer id, such as 52708846, rather than a GPU such as 4090: Vast's offer ids run to
    seven or eight digits, GPU model numbers to four."""
    return choice.isdigit() and len(choice) >= 6


def candidates(v: Vast, choice: str, seconds: int, disk: int) -> list[Offer]:
    """The offer to rent first, then up to two near it in price to fall back on: the same
    GPUs, from hosts not skipped, whose contracts outlast the cap."""
    avoid = skipped()
    latest = time.time() + seconds + MARGIN

    def usable(o: Offer) -> bool:
        return str(o.machine) not in avoid and (o.ends is None or o.ends >= latest)

    if is_offer(choice):
        found = v.search(offer=int(choice), disk=disk, any_host=True)
        if not found:
            raise FerryError(f"offer {choice} is not free any more, or no longer meets Ferry's bar. "
                             f"`ferry gpus` shows what is")
        first = found[0]
        if first.ends is not None and first.ends < latest:
            raise FerryError(f"offer {choice}'s host is rented out only until {time.strftime('%d %b %H:%M', time.localtime(first.ends))}, "
                             f"before your time is up. Pick another offer, or a shorter --for")
        others = v.search(first.gpu, disk=disk, any_host=first.tier == "unverified")
    else:
        others = v.search(choice, disk=disk)
        found = [o for o in others if usable(o)]
        if not found:
            raise FerryError(f"no {choice} on Vast.ai right now that Ferry can use. `ferry gpus` shows what there is")
        first = found[0]
    near = [o for o in others if o.id != first.id and o.gpu == first.gpu and o.count == first.count
            and o.machine != first.machine and o.price <= first.price * NEARBY and usable(o)]
    return [first, *near[:TRIES - 1]]


def describe(offers: list[Offer], seconds: int) -> str:
    """What renting costs, in words, with the most it can cost."""
    first = offers[0]
    top = max(o.price for o in offers)
    gpu = f"{first.count}x {first.gpu}" if first.count > 1 else first.gpu
    rate = f"${first.price:.3f} an hour" + (f" (up to ${top:.3f} if Ferry has to take the next offer)" if top > first.price else "")
    most = top * seconds / 3600
    host = "an unverified host, which, like any host, can see your code and data" if first.tier == "unverified" else f"a {first.tier} host"
    text = (f"{gpu} on Vast.ai, offer {first.id}: {rate} with {first.disk} GB disk, {host} in {first.place}.\n"
            f"For {duration(seconds)} it costs at most ${most:.2f}, plus ${first.up_cost:.4f} per GB you send it "
            f"and ${first.down_cost:.4f} per GB you bring back.")
    return text


def rent(choice: str, seconds: int, disk: int = DISK_GB, yes: bool = False, once: bool = False) -> Machine:
    """Rents an offer (an id, or a GPU name for its cheapest offer), after saying what it
    costs and asking, and returns it as a machine Ferry can run on."""
    v = client()
    offers = candidates(v, choice, seconds, disk)
    for line in describe(offers, seconds).splitlines():
        say(line)
    if not yes:
        if not can_ask():
            raise FerryError("nothing rented: run again with --yes to rent it")
        if input("Rent it? [y/N] ").strip().lower() not in ("y", "yes"):
            raise FerryError("nothing rented")
    for i, offer in enumerate(offers):
        machine = _try(v, offer, seconds, disk, once)
        if machine is not None:
            return machine
        if i + 1 < len(offers):
            say(f"trying the next offer, {offers[i + 1].id} (${offers[i + 1].price:.3f} an hour)")
    raise FerryError("none of the offers tried could be reached; nothing is rented now. Try another GPU, or again later")


class _Unusable(Exception):
    """This machine will not do; destroy it and take the next offer."""


def _try(v: Vast, offer: Offer, seconds: int, disk: int, once: bool) -> Machine | None:
    pem, public = new_key("ferry")
    try:
        instance = v.create(offer.id, disk, onstart(public, seconds), LABEL, IMAGE)
    except VastError as e:
        if e.status in (400, 404, 410):  # gone, or not rentable after all: nothing was rented
            say(f"offer {offer.id}: {e}")
            return None
        raise
    name = f"vast-{instance}"
    now = time.time()
    # Saved at once, so whatever happens next, `ferry machines` and `ferry release` know it.
    machine = Machine(name=name, host="", user="root", provider="vast", instance=instance, price=offer.price,
                      disk_rate=offer.disk_rate, tier=offer.tier, place=offer.place, rented_at=now,
                      deadline=now + BOOT_WAIT + seconds, once=once, gpu=offer.gpu)
    save_machine(machine)
    say(f"rented {name}; waiting for it to start (a minute, or about five if the host has to download the image first)")
    try:
        details = _wait_running(v, instance, name)
        keys = _host_keys(v, instance)
        host, port = _direct_address(details)
        _check_port(host, port)
        folder = rented_dir(name)
        folder.mkdir(parents=True, exist_ok=True)
        key_file = folder / "id_ed25519"
        key_file.write_text(pem, encoding="utf-8")
        if os.name != "nt":
            os.chmod(key_file, 0o600)
        pinned = paramiko.HostKeys()
        for key in keys:
            pinned.add(f"[{host}]:{port}" if port != 22 else host, key.get_name(), key)
        pinned.save(str(folder / "known_hosts"))
        machine.host, machine.port, machine.key = host, port, str(key_file)
        save_machine(machine)
        box = _login(machine)
        try:
            info = Remote(box).probe()
            code, out, _ = box.run(f"cat {STATE}/deadline; date +%s")
        finally:
            box.close()
        if code == 0 and len(out.split()) == 2:
            machine_deadline, machine_now = (float(x) for x in out.split())
            machine.deadline = machine_deadline - (machine_now - time.time())  # on this computer's clock
        machine.hostname = info.get("hostname") or None
        machine.arch = info.get("arch") or None
        machine.gpu = info.get("gpu") or offer.gpu
        machine.driver = info.get("driver") or None
        machine.python = info.get("python") or None
        save_machine(machine)
        say(f"{name} is ready: {machine.gpu}, until {time.strftime('%H:%M', time.localtime(machine.deadline))} "
            f"({duration(machine.deadline - time.time())} from now), when it destroys itself")
        return machine
    except _Unusable as e:
        say(f"{name}: {e}. Giving it back (it cost cents)")
        skip(offer.machine)
        _give_back(v, machine)
        return None
    except BaseException:
        say(f"giving {name} back")
        _give_back(v, machine)
        raise


def _wait_running(v: Vast, instance: int, name: str, waking: bool = False) -> dict:
    """Waits for the instance to run. A new one that goes exited, unknown or offline never
    will; one being started again shows exited until it does (Vast's word for stopped,
    9 Nov 2026)."""
    deadline = time.monotonic() + BOOT_WAIT
    last = None
    while time.monotonic() < deadline:
        details = v.instance(instance)
        if details is None:
            raise _Unusable("it disappeared while starting")
        status = details.get("actual_status")
        message = (details.get("status_msg") or "").strip()
        if (status, message) != last and status:
            say(f"{name}: {status}{': ' + message[:100] if message and status != 'running' else ''}")
            last = (status, message)
        if status == "running" and details.get("intended_status", "running") == "running":
            return details
        if status in ("unknown", "offline") or (status == "exited" and not waking):
            raise _Unusable(f"it went {status}{': ' + message[:100] if message else ''}")
        time.sleep(5)
    raise _Unusable(f"it did not start within {BOOT_WAIT // 60} minutes")


def _host_keys(v: Vast, instance: int) -> list[paramiko.PKey]:
    """The host keys onstart printed to the instance log, read over HTTPS, so the first
    connection is checked too."""
    deadline = time.monotonic() + LOG_WAIT
    while time.monotonic() < deadline:
        keys: list[paramiko.PKey] = []
        # The log cut an RSA key's long line short (9 Nov 2026), and paramiko read what was left as a
        # key, a wrong one. So only the short types onstart prints, and only when whole.
        for match in re.finditer(r"FERRY-HOSTKEY (ssh-ed25519|ecdsa-sha2-nistp\d+) (\S+)", v.log(instance) or ""):
            if not _whole(match[1], match[2]):
                continue
            try:
                entry = paramiko.hostkeys.HostKeyEntry.from_line(f"x {match[1]} {match[2]}")
            except (paramiko.hostkeys.InvalidHostKey, paramiko.SSHException, ValueError, TypeError):
                continue
            if entry is not None and entry.key is not None and all(k.asbytes() != entry.key.asbytes() for k in keys):
                keys.append(entry.key)
        if keys:
            return sorted(keys, key=lambda k: k.get_name() != "ssh-ed25519")  # paramiko asks for the first one's type
        time.sleep(10)
    raise _Unusable("its host keys never showed up in its log")


def _whole(kind: str, text: str) -> bool:
    """Whether a public key is all there: its SSH strings add up exactly, an ed25519 key has
    its 32 bytes and an ECDSA key its full point."""
    try:
        blob = base64.b64decode(text, validate=True)
    except ValueError:
        return False
    parts = []
    while blob:
        size = int.from_bytes(blob[:4], "big")
        if len(blob) < 4 + size:
            return False
        parts.append(blob[4:4 + size])
        blob = blob[4 + size:]
    if not parts or parts[0] != kind.encode():
        return False
    if kind == "ssh-ed25519":
        return len(parts) == 2 and len(parts[1]) == 32
    return len(parts) == 3 and parts[2][:1] == b"\x04" and len(parts[2]) in (65, 97, 133)


def _direct_address(details: dict) -> tuple[str, int]:
    try:
        return details["public_ipaddr"].strip(), int(details["ports"]["22/tcp"][0]["HostPort"])
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise _Unusable("it has no direct SSH port") from None


def _check_port(host: str, port: int) -> None:
    deadline = time.monotonic() + PORT_WAIT
    while True:
        try:
            connect_first(host, port, timeout=max(1.0, deadline - time.monotonic())).close()
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise _Unusable(f"its SSH port ({host}:{port}) did not answer within {PORT_WAIT} seconds") from None
            time.sleep(1)


def _login(machine: Machine) -> Box:
    """Connects, waiting a little for the machine's key to be in place."""
    deadline = time.monotonic() + LOGIN_WAIT
    while True:
        try:
            return Box(machine).connect()
        except FerryError as e:
            if "refused the login" not in str(e) or time.monotonic() >= deadline:
                raise
            time.sleep(5)


def _give_back(v: Vast, machine: Machine) -> None:
    """Destroys the instance and forgets the machine. Says how to do it by hand if it fails."""
    try:
        v.destroy(machine.instance)
        for _ in range(12):
            if v.instance(machine.instance) is None:
                break
            time.sleep(5)
    except (VastError, OSError) as e:
        say(f"could not destroy {machine.name} ({e}). Destroy instance {machine.instance} on vast.ai, "
            f"or run: ferry release {machine.name}. Its time cap destroys it in any case")
        return
    remove_machine(machine.name)
    shutil.rmtree(rented_dir(machine.name), ignore_errors=True)


def cost_so_far(machine: Machine, until: float | None = None) -> float:
    """What the machine has cost since it was rented: its hourly price while running, its
    disk alone while stopped. Its time cap ends the count: it is destroyed then."""
    end = min(until or time.time(), machine.deadline or float("inf"))
    hours = max(0.0, (end - (machine.rented_at or end)) / 3600)
    stopped = (machine.stopped_seconds or 0) / 3600
    return (machine.price or 0) * max(0.0, hours - stopped) + (machine.disk_rate or 0) * stopped


def release(name: str, yes: bool = False) -> float:
    """Destroys a rented machine now and forgets it. Returns about what it cost."""
    machine = get_machine(name)
    if not machine.provider:
        raise FerryError(f"{name} is not rented: Ferry only gives back machines it rented")
    v = client()
    pending = runs_not_home(name)
    if pending and not yes:
        say(f"{len(pending)} run{'s' if len(pending) != 1 else ''} on {name} {'have' if len(pending) != 1 else 'has'} "
            f"not come home yet: {', '.join(pending)}. Releasing loses what is still only there.")
        if not can_ask():
            raise FerryError(f"nothing released: `ferry attach {pending[0]}` brings it home first, or add --yes")
        if input("Release anyway? [y/N] ").strip().lower() not in ("y", "yes"):
            raise FerryError("nothing released")
    cost = cost_so_far(machine)
    v.destroy(machine.instance)
    for _ in range(12):
        if v.instance(machine.instance) is None:
            break
        time.sleep(5)
    else:
        say(f"Vast.ai still lists instance {machine.instance}; it should be gone within a minute")
    remove_machine(name)
    shutil.rmtree(rented_dir(name), ignore_errors=True)
    say(f"released {name}; it cost {'about ' if cost >= 0.01 else ''}{money(cost)}")
    return cost


def runs_not_home(name: str) -> list[str]:
    """Short ids of this project's runs on the machine that are not fully home."""
    try:
        store = Store.find()
    except HunchError:
        return []
    out = []
    for run in store.runs():
        state = read_json(run.dir / "remote.json")
        if isinstance(state, dict) and state.get("machine") == name and not state.get("complete"):
            out.append(run.id[-4:])
    return out


def write_cost(run_dir, machine: Machine, running: float, stopped: float) -> None:
    """cost.json in the run's folder: what the run cost on its rented machine, as an estimate
    (bandwidth is not counted) until the provider's bill says otherwise."""
    estimate = (machine.price or 0) * running / 3600 + (machine.disk_rate or 0) * stopped / 3600
    write_json(run_dir / "cost.json", {
        "provider": machine.provider,
        "machine": machine.name,
        "gpu": machine.gpu,
        "rate": machine.price,
        "disk_rate": round(machine.disk_rate or 0, 6),
        "running_seconds": round(max(0.0, running)),
        "stopped_seconds": round(max(0.0, stopped)),
        "estimate": round(estimate, 4),
        "currency": "USD",
        "billed": None,
    })


class Stopped(FerryError):
    """A rented machine that stopped itself, asked about without starting it."""


def wake(machine: Machine, start: bool = True) -> float | None:
    """Before connecting to a rented machine: says when it is gone, and, with `start`, starts
    it again when it stopped itself, waiting until it runs. Returns when it asked for the start."""
    v = client()
    details = v.instance(machine.instance)
    if details is None:
        raise FerryError(f"{machine.name} is gone: Vast.ai destroyed it, probably at its time cap. "
                         f"`ferry release {machine.name}` forgets it")
    woke = None
    if details.get("intended_status") == "stopped" or details.get("actual_status") in ("stopped", "exited"):
        if not start:
            raise Stopped(f"{machine.name} stopped itself after its run ended, so only its disk costs now; "
                          f"`ferry attach` starts it again and brings the run home")
        say(f"{machine.name} stopped itself after its run ended; starting it again to bring the run home")
        woke = time.time()
        v.set_state(machine.instance, "running")
    if details.get("actual_status") != "running" or woke:
        try:
            _wait_running(v, machine.instance, machine.name, waking=True)
        except _Unusable as e:
            raise FerryError(f"{machine.name} did not start again: {e}. `ferry release {machine.name}` gives it back") from None
    return woke


def note_stop(machine: Machine, box: Box, woke: float) -> None:
    """After starting a machine that stopped itself: counts the time it stood, from its
    idle-stop mark until the start, so its cost counts only its disk for that time."""
    code, out, _ = box.run(f"stat -c %Y {STATE}/idle-stopped && date +%s && rm -f {STATE}/idle-stopped")
    if code != 0 or len(out.split()) < 2:
        return
    stopped_at, machine_now = (float(x) for x in out.split()[:2])
    stopped_at += time.time() - machine_now  # on this computer's clock
    machine.stopped_seconds = (machine.stopped_seconds or 0) + max(0.0, woke - stopped_at)
    save_machine(machine)


def mark_home(box: Box, run_id: str) -> None:
    """Tells the machine's watchdog this run is home, so it does not stop the machine for it."""
    box.run(f"mkdir -p {STATE}/home && touch {STATE}/home/{shlex.quote(run_id)}")
