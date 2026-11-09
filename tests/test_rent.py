"""Renting on Vast.ai: offers, the machine's start script and watchdog, and giving it back."""

import base64
import gzip
import io
import json
import os
import re
import time
import urllib.error

import pytest
from hunch.store import Store, read_json, write_json

from ferry import launch, mirror, rent, vast, watch
from ferry.config import FerryError, Machine, load_machines, rented_dir, save_machine


@pytest.fixture(autouse=True)
def ferry_home(tmp_path, monkeypatch):
    monkeypatch.setenv("FERRY_HOME", str(tmp_path / "ferry"))
    monkeypatch.delenv(vast.KEY_ENV, raising=False)


# The watchdog on the machine

def test_the_watchdog_destroys_at_the_deadline_and_stops_for_a_run_nobody_brought_home():
    hour = 3600
    decide = watch.decide
    # now, deadline, booted, idle, going, waiting, last end, woke from an idle stop
    assert decide(10 * hour, 10 * hour, 0, 900, True, False, 0, False) == "destroy"
    assert decide(5 * hour, 10 * hour, 0, 900, True, False, 0, False) is None  # training
    assert decide(5 * hour, 10 * hour, 0, 900, False, True, 5 * hour - 600, False) is None  # ended 10 minutes ago
    assert decide(5 * hour, 10 * hour, 0, 900, False, True, 5 * hour - 1000, False) == "stop"
    assert decide(5 * hour, 10 * hour, 0, 900, True, True, 0, False) is None  # another run is still going
    assert decide(5 * hour, 10 * hour, 0, 900, False, False, 0, False) is None  # everything is home
    # Started again after an idle stop, past its deadline: half an hour to bring the run home,
    assert decide(12 * hour, 10 * hour, 12 * hour - 60, 900, False, True, 0, True) is None
    assert decide(12 * hour + watch.GRACE, 10 * hour, 12 * hour, 900, False, True, 0, True) == "destroy"
    # ... and not a moment longer once it is home.
    assert decide(12 * hour, 10 * hour, 12 * hour - 60, 900, False, False, 0, True) == "destroy"
    # The idle clock starts at that boot, not at the old end.
    assert decide(12 * hour + 100, 13 * hour, 12 * hour, 900, False, True, 0, True) is None


def test_the_watchdog_tells_running_ended_and_brought_home_runs_apart(tmp_path):
    runs = tmp_path / "projects" / "p-1234" / ".hunch" / "runs"
    home = tmp_path / "watch" / "home"
    home.mkdir(parents=True)
    (runs / "a").mkdir(parents=True)
    assert watch.runs(str(tmp_path / "projects"), str(home)) == (True, False, 0.0)
    (runs / "a" / "end.json").write_text("{}")
    going, waiting, last = watch.runs(str(tmp_path / "projects"), str(home))
    assert (going, waiting) == (False, True) and last == pytest.approx(os.path.getmtime(runs / "a" / "end.json"))
    (home / "a").touch()
    assert watch.runs(str(tmp_path / "projects"), str(home)) == (False, False, 0.0)


def test_the_start_script_fits_and_carries_the_key_the_cap_and_the_watchdog():
    public = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOgk7Uq0zZ9eJ1x8pH3oY2y3mZ8bq7GJ0QhQkzLwqv9k ferry"
    script = rent.onstart(public, 7200)
    assert len(script) <= rent.ONSTART_LIMIT
    assert script.count(public) == 2  # added once, and only when it is not there yet
    assert "+ 7200 ))" in script
    packed = re.search(r"echo '([A-Za-z0-9+/=]+)' \| base64 -d \| gunzip", script)[1]
    from importlib import resources
    assert gzip.decompress(base64.b64decode(packed)) == resources.files("ferry").joinpath("watch.py").read_bytes()


def test_host_keys_come_from_the_log_and_never_from_a_line_cut_short():
    host_key = rent.new_key("root@host")[1].rsplit(" ", 1)[0]
    # An RSA line the log cut short: still base64, so it would read as a (wrong) key.
    log = (f"booting\nFERRY-HOSTKEY ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCnSC0mzibUJ7+AZk5/j0HfOJeGRNBco6gGSxrYqjvX\n"
           f"FERRY-HOSTKEY ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOgk7Uq0\n"
           f"FERRY-HOSTKEY {host_key}\n")

    class Log:
        def log(self, instance):
            return log

    keys = rent._host_keys(Log(), 1)
    assert [k.get_name() for k in keys] == ["ssh-ed25519"]
    assert f"{keys[0].get_name()} {keys[0].get_base64()}" == host_key


def test_a_fresh_key_for_each_machine():
    pem, public = rent.new_key("ferry")
    assert pem.startswith("-----BEGIN OPENSSH PRIVATE KEY-----")
    assert public.startswith("ssh-ed25519 ") and public.endswith(" ferry")
    assert rent.new_key("ferry")[1] != public


# Offers

def test_gpu_names_typed_any_way_match_vasts():
    assert "RTX 4090" in vast.gpu_names("rtx_4090")
    assert "RTX 4090" in vast.gpu_names("4090")
    assert {"A100 SXM4", "A100 PCIE"} <= set(vast.gpu_names("a100"))
    assert "Titan RTX" in vast.gpu_names("titan rtx")
    assert "GTX TITAN X" in vast.gpu_names("gtx titan x")
    assert "RTX 3060 Ti" in vast.gpu_names("RTX 3060 TI")
    assert vast.gpu_names("  ") == []
    assert rent.is_offer("52708846") and not rent.is_offer("4090")


def test_durations():
    assert rent.parse_duration("2h") == 7200
    assert rent.parse_duration("90m") == rent.parse_duration("1h30m") == rent.parse_duration("1.5h") == 5400
    for bad in ("", "2", "2 hours", "5m", "h"):
        with pytest.raises(FerryError):
            rent.parse_duration(bad)
    assert rent.duration(5400) == "1h30m" and rent.duration(300) == "5m"


RAW = {"id": 52708846, "machine_id": 59091, "gpu_name": "GTX 1080", "num_gpus": 1, "gpu_ram": 8192,
       "dph_total": 0.0644, "storage_cost": 0.2, "inet_down": 278.0, "inet_up": 250.0, "inet_down_cost": 0.0065,
       "inet_up_cost": 0.0039, "reliability": 0.9948, "reliability2": 0.9950, "verification": "verified",
       "geolocation": "South Korea, KR", "cuda_max_good": 13.0, "end_date": 1796022739.0}


def test_an_offer_as_ferry_shows_it():
    offer = vast.Offer.of(RAW, 40)
    assert (offer.id, offer.machine, offer.gpu, offer.memory, offer.tier, offer.country) == (52708846, 59091, "GTX 1080", 8.0, "verified", "KR")
    assert offer.reliability == 0.9950
    assert (offer.up_cost, offer.down_cost) == (0.0065, 0.0039)  # what the host downloads is what Ferry sends
    assert offer.disk_rate == pytest.approx(0.2 * 40 / 720)
    assert vast.Offer.of(dict(RAW, verification="unverified"), 40).tier == "unverified"
    assert vast.Offer.of(dict(RAW, datacenter=True), 40).tier == "datacenter"


def offer(id, price, machine=None, gpu="RTX 4090", ends=None, tier="verified"):
    raw = dict(RAW, id=id, machine_id=machine or id, gpu_name=gpu, dph_total=price, end_date=ends or time.time() + 30 * 86400,
               verification=tier)
    return vast.Offer.of(raw, 40)


class FakeVast:
    def __init__(self, offers):
        self.offers = offers
        self.searches = []

    def search(self, gpu=None, max_price=None, any_host=False, disk=40, offer=None):
        self.searches.append((gpu, offer))
        if offer is not None:
            return [o for o in self.offers if o.id == offer]
        return sorted(self.offers, key=lambda o: o.price)


def test_a_gpu_name_rents_its_cheapest_offer_with_two_near_ones_to_fall_back_on():
    soon = time.time() + 3600
    v = FakeVast([offer(1, 0.30, ends=soon), offer(2, 0.40), offer(3, 0.41, machine=2), offer(4, 0.45), offer(5, 0.47),
                  offer(6, 0.90)])
    picked = rent.candidates(v, "4090", 7200, 40)
    # 1's host contract ends before the cap; 3 is on the same host as 2; 6 costs too much more.
    assert [o.id for o in picked] == [2, 4, 5]


def test_a_host_whose_port_never_answered_is_not_picked_again():
    rent.skip(2)
    v = FakeVast([offer(2, 0.40), offer(4, 0.45)])
    assert [o.id for o in rent.candidates(v, "4090", 7200, 40)] == [4]


def test_an_offer_id_rents_that_offer_or_says_why_not():
    v = FakeVast([offer(5270007, 0.50), offer(5270008, 0.52)])
    assert [o.id for o in rent.candidates(v, "5270007", 7200, 40)] == [5270007, 5270008]
    with pytest.raises(FerryError, match="not free any more"):
        rent.candidates(v, "5270009", 7200, 40)
    v = FakeVast([offer(5270007, 0.50, ends=time.time() + 3600)])
    with pytest.raises(FerryError, match="before your time is up"):
        rent.candidates(v, "5270007", 7200, 40)


def test_the_confirmation_says_the_most_it_can_cost():
    text = rent.describe([offer(2, 0.40), offer(4, 0.45)], 7200)
    assert "$0.400 an hour (up to $0.450 if Ferry has to take the next offer)" in text
    assert "at most $0.90" in text
    text = rent.describe([offer(2, 0.40, tier="unverified")], 3600)
    assert "unverified host, which, like any host, can see your code and data" in text


# The API

class Answer(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_rent_request_is_never_sent_twice_but_a_search_is_retried(monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append(request.get_method())
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 502, "bad gateway", {}, io.BytesIO(b""))
        return Answer(json.dumps({"offers": []}).encode())

    monkeypatch.setattr(vast.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(vast.time, "sleep", lambda s: None)
    assert vast.Vast("k").search() == []
    assert calls == ["POST", "POST"]
    calls.clear()
    with pytest.raises(vast.VastError, match="502"):
        vast.Vast("k").create(1, 40, "echo", "ferry", "image")
    assert calls == ["PUT"]


def test_a_refused_key_says_to_log_in_and_the_key_never_shows(monkeypatch):
    def urlopen(request, timeout):
        assert request.get_header("Authorization") == "Bearer secret-key"
        raise urllib.error.HTTPError(request.full_url, 401, "no", {}, io.BytesIO(b'{"error": "auth"}'))

    monkeypatch.setattr(vast.urllib.request, "urlopen", urlopen)
    client = vast.Vast("secret-key")
    assert "secret" not in repr(client)
    with pytest.raises(vast.VastError, match="ferry login vast") as e:
        client.balance()
    assert "secret" not in str(e.value)


def test_the_key_comes_from_the_environment_first_then_the_keychain(monkeypatch):
    kept = {}

    class Keyring:
        @staticmethod
        def get_password(service, user):
            return kept.get((service, user))

        @staticmethod
        def set_password(service, user, key):
            kept[(service, user)] = key

        @staticmethod
        def delete_password(service, user):
            if (service, user) not in kept:
                raise Errors.PasswordDeleteError()
            del kept[(service, user)]

    class Errors:
        class KeyringError(Exception):
            pass

        class PasswordDeleteError(KeyringError):
            pass

    monkeypatch.setattr(vast, "_keyring", lambda: (Keyring, Errors))
    assert vast.stored_key() is None
    vast.store_key("from-keychain")
    assert vast.stored_key() == "from-keychain"
    monkeypatch.setenv(vast.KEY_ENV, "from-env")
    assert vast.stored_key() == "from-env"
    assert vast.forget_key() is True and vast.forget_key() is False


# After the run

def rented(name="vast-42", once=True, **fields):
    machine = Machine(name=name, host="1.2.3.4", user="root", port=40022, provider="vast", instance=42, price=0.4,
                      disk_rate=0.02, tier="verified", place="Quebec, CA", rented_at=time.time() - 1800,
                      deadline=time.time() + 3600, once=once, gpu="RTX 4090", **fields)
    save_machine(machine)
    return machine


def a_run(tmp_path, monkeypatch, name="vast-42"):
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path)
    run_dir = store.dir / "runs" / "20261109-1200-abcdef"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "run.json", {"id": run_dir.name, "started_at": "2026-11-09T12:00:00+00:00"})
    mirror.write_state(run_dir, {"machine": name, "dir": "/root/x", "provider": "vast"})
    return run_dir


def test_a_machine_rented_for_one_run_goes_back_once_the_run_is_home(tmp_path, monkeypatch):
    rented()
    run_dir = a_run(tmp_path, monkeypatch)
    write_json(run_dir / "end.json", {"status": "finished", "duration": 1500})
    released = []
    monkeypatch.setattr(rent, "release", lambda name, yes=False: released.append(name))
    assert launch._after("vast-42", run_dir, 0) == 0
    assert released == ["vast-42"]
    cost = read_json(run_dir / "cost.json")
    assert cost["provider"] == "vast" and cost["rate"] == 0.4 and cost["billed"] is None
    assert cost["running_seconds"] == pytest.approx(1800, abs=5)  # the whole rental, start-up included
    assert cost["estimate"] == pytest.approx(0.2, abs=0.01)


def test_a_kept_machine_stays_and_the_run_costs_its_own_time(tmp_path, monkeypatch):
    rented(once=False)
    run_dir = a_run(tmp_path, monkeypatch)
    write_json(run_dir / "end.json", {"status": "finished", "duration": 900})
    monkeypatch.setattr(rent, "release", lambda *a, **k: pytest.fail("released a machine rented with ferry rent"))
    assert launch._after("vast-42", run_dir, 0) == 0
    assert read_json(run_dir / "cost.json")["running_seconds"] == 900


def test_ctrl_c_keeps_the_machine_and_the_time_cap_ends_the_run(tmp_path, monkeypatch, capsys):
    rented()
    run_dir = a_run(tmp_path, monkeypatch)
    monkeypatch.setattr(rent, "release", lambda *a, **k: pytest.fail("released on Ctrl+C"))
    assert launch._after("vast-42", run_dir, 130) == 130
    assert "`ferry attach cdef` brings the run home and gives vast-42 back" in capsys.readouterr().err
    released = []
    monkeypatch.setattr(rent, "release", lambda name, yes=False: released.append(name))
    assert launch._after("vast-42", run_dir, mirror.CAPPED) == 1
    assert released == ["vast-42"]
    assert read_json(run_dir / "end.json")["status"] == "killed"
    assert "time rented" in (run_dir / "notes.jsonl").read_text()


def test_release_asks_when_a_run_has_not_come_home(tmp_path, monkeypatch):
    rented()
    a_run(tmp_path, monkeypatch)
    monkeypatch.setenv(vast.KEY_ENV, "k")
    monkeypatch.setattr(rent, "can_ask", lambda: False)
    with pytest.raises(FerryError, match="`ferry attach cdef` brings it home first"):
        rent.release("vast-42")
    destroyed = []
    monkeypatch.setattr(vast.Vast, "destroy", lambda self, i: destroyed.append(i))
    monkeypatch.setattr(vast.Vast, "instance", lambda self, i: None)
    rented_dir("vast-42").mkdir(parents=True)
    assert rent.release("vast-42", yes=True) == pytest.approx(0.2, abs=0.01)
    assert destroyed == [42] and "vast-42" not in load_machines() and not rented_dir("vast-42").exists()


def test_an_unreachable_machine_goes_back_and_the_next_offer_is_tried(monkeypatch):
    monkeypatch.setenv(vast.KEY_ENV, "k")
    offers = [offer(2, 0.40), offer(4, 0.45)]
    monkeypatch.setattr(rent, "candidates", lambda v, choice, seconds, disk: offers)
    created, destroyed = [], []
    monkeypatch.setattr(vast.Vast, "create", lambda self, o, disk, script, label, image: created.append(o) or 100 + o)
    monkeypatch.setattr(vast.Vast, "destroy", lambda self, i: destroyed.append(i))
    monkeypatch.setattr(vast.Vast, "instance", lambda self, i: None)

    def wait(v, instance, name):
        if instance == 102:
            raise rent._Unusable("its SSH port (1.2.3.4:40022) did not answer within 15 seconds")
        raise KeyboardInterrupt  # the second: stop here, after it was saved

    monkeypatch.setattr(rent, "_wait_running", wait)
    with pytest.raises(KeyboardInterrupt):
        rent.rent("4090", 7200, yes=True)
    assert created == [2, 4] and destroyed == [102, 104]  # both given back, even on Ctrl+C
    assert "2" in rent.skipped() and "4" not in rent.skipped()
    assert load_machines() == {}
