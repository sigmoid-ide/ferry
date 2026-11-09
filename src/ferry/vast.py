"""Vast.ai, straight over its REST API with the standard library.

The account key comes from VAST_API_KEY or the OS keychain (`ferry login vast`). Ferry
never writes it to a file, prints it, or sends it to a machine. Offer lists are shown,
never stored: Vast's terms forbid passing its data on.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field

from . import __version__
from .config import FerryError, home

API = "https://console.vast.ai/api/v0"
KEY_ENV = "VAST_API_KEY"
KEYRING_SERVICE = "ferry"
KEYRING_USER = "vast.ai"
DISK_GB = 40
MIN_RELIABILITY = 0.98
MIN_DOWNLOAD = 100  # Mb/s, so packages and data arrive in minutes
MIN_CUDA = 12.0  # current PyTorch builds need a driver for CUDA 12 or newer
RETRIES = 4


class VastError(FerryError):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


def _keyring():
    import keyring  # only when needed: some systems have no keychain, and VAST_API_KEY does without
    import keyring.errors

    return keyring, keyring.errors


def stored_key() -> str | None:
    """The key from VAST_API_KEY, else from the OS keychain."""
    if os.environ.get(KEY_ENV, "").strip():
        return os.environ[KEY_ENV].strip()
    keyring, errors = _keyring()
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
    except errors.KeyringError:
        return None


def store_key(key: str) -> None:
    keyring, errors = _keyring()
    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, key)
    except errors.KeyringError as e:
        raise FerryError(f"this system has no keychain Ferry can use ({e}). Set {KEY_ENV} instead") from None


def forget_key() -> bool:
    """Removes the key from the keychain. False when there was none."""
    keyring, errors = _keyring()
    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
        return True
    except errors.PasswordDeleteError:
        return False
    except errors.KeyringError as e:
        raise FerryError(f"could not reach the keychain: {e}") from None


def tier(offer: dict) -> str:
    if offer.get("datacenter"):
        return "datacenter"
    return "verified" if offer.get("verification") == "verified" else "unverified"


@dataclass
class Offer:
    """A machine Vast.ai rents out now, as `ferry gpus` shows it."""

    id: int
    machine: int  # the host machine, which can have several offers
    gpu: str
    count: int
    memory: float  # GB per GPU
    price: float  # $ per hour, the disk included
    reliability: float
    tier: str
    place: str  # "City, CC" as Vast gives it; it does not tell the network path
    download: float  # Mb/s
    upload: float
    cuda: float
    up_cost: float  # $ per GB sent to the machine
    down_cost: float  # $ per GB copied back
    disk: int  # GB included in the price
    disk_rate: float  # $ per hour of that disk alone, which a stopped machine still pays
    ends: float | None  # the host's contract end, Unix time

    @classmethod
    def of(cls, raw: dict, disk: int) -> Offer:
        return cls(
            id=int(raw["id"]),
            machine=int(raw.get("machine_id") or 0),
            gpu=str(raw.get("gpu_name") or "?"),
            count=int(raw.get("num_gpus") or 1),
            memory=round((raw.get("gpu_ram") or 0) / 1024, 1),
            price=float(raw.get("dph_total") or 0),
            reliability=float(max(raw.get("reliability2") or 0, raw.get("reliability") or 0)),
            tier=tier(raw),
            place=str(raw.get("geolocation") or "?"),
            download=float(raw.get("inet_down") or 0),
            upload=float(raw.get("inet_up") or 0),
            cuda=float(raw.get("cuda_max_good") or 0),
            # inet_*_cost is per GB (1024 per TB, checked against *_per_tb on 8 Nov 2026). Vast names them
            # from the host's side: what the host downloads is what Ferry sends.
            up_cost=float(raw.get("inet_down_cost") or 0),
            down_cost=float(raw.get("inet_up_cost") or 0),
            disk=disk,
            disk_rate=float(raw.get("storage_cost") or 0) * disk / 720,  # storage_cost is $ per GB per 30 days
            ends=float(raw["end_date"]) if raw.get("end_date") else None,
        )

    @property
    def country(self) -> str:
        return self.place.rsplit(",", 1)[-1].strip()

    def to_json(self) -> dict:
        return asdict(self)


NAMES_FILE = "vast-gpu-names.json"  # Vast's GPU names (not offers), kept a day like Vast's own CLI does
NAMES_SECONDS = 24 * 3600


def _squash(text: str) -> str:
    return re.sub(r"[\s_-]+", "", text.lower())


def matching_gpus(names: list[str], text: str) -> list[str]:
    """Vast's GPU names that `text` means: the one it names exactly (a10, l4, rtx_4090), else
    every name that holds all its words, in any case (laptop, 4090, 3060 ti, a100). Vast's
    own filter matches whole names only."""
    exact = [name for name in names if _squash(name) == _squash(text)]
    if exact:
        return exact
    words = [_squash(word) for word in text.replace("_", " ").split()]
    return [name for name in names if words and all(word in _squash(name) for word in words)]


REGIONS = {
    "europe": "AT BE BG CH CY CZ DE DK EE ES FI FR GB GR HR HU IE IS IT LT LU LV MT NL NO PL PT RO RS SE SI SK UA",
    "north-america": "US CA MX",
    "south-america": "AR BR CL CO EC PE UY",
    "asia": "CN HK ID IN JP KR KZ MY PH SG TH TW VN",
    "oceania": "AU NZ",
    "middle-east": "AE IL QA SA TR",
    "africa": "EG KE MA NG ZA",
}
_REGION_WORDS = {"eu": "europe", "na": "north-america", "america": "north-america", "sa": "south-america",
                 "apac": "asia", "oc": "oceania", "me": "middle-east", "uk": "GB", **{r: r for r in REGIONS}}
_COUNTRIES = set(" ".join(REGIONS.values()).split())
_PRICE = re.compile(r"<=?\$?(\d+(?:\.\d+)?)|\$(\d+(?:\.\d+)?)")
_MEMORY = re.compile(r"(\d+)\s*gb?")
_COUNT = re.compile(r"(\d+)x")


@dataclass
class Query:
    """A search for offers, from words such as `4090 europe <0.5 24gb 2x`: the GPU, where,
    the most per hour, the least memory per GPU, how many GPUs, and datacenters only."""

    gpu: str | None = None
    max_price: float | None = None
    memory: int | None = None  # GB per GPU
    count: int | None = None
    where: list[str] = field(default_factory=list)  # two-letter country codes
    datacenter: bool = False

    def describe(self) -> str:
        parts = [f"{self.count}x {self.gpu or 'GPUs'}" if self.count else (self.gpu or "any GPU")]
        if self.memory:
            parts.append(f"{self.memory} GB or more")
        if self.where:
            parts.append("in " + _places(self.where))
        if self.max_price is not None:
            parts.append(f"at most ${self.max_price:.2f} an hour")
        if self.datacenter:
            parts.append("in datacenters")
        return ", ".join(parts)


def _places(codes: list[str]) -> str:
    for region, members in REGIONS.items():
        if sorted(members.split()) == sorted(codes):
            return region.replace("-", " ").title()
    return ", ".join(codes)


def parse_query(words: list[str]) -> Query:
    """Reads search words: `<0.5` or `$0.5` is the most per hour, `24gb` the least memory per
    GPU, `2x` the number of GPUs, a region (europe, north-america, asia, oceania, ...) or
    two-letter country codes where, `datacenter` (or `dc`) datacenter hosts only, and the
    other words the GPU, such as `rtx 4090` or `a100`."""
    query = Query()
    gpu: list[str] = []
    for word in " ".join(words).replace(",", " ").split():
        lower = word.lower()
        if match := _PRICE.fullmatch(lower):
            query.max_price = float(match[1] or match[2])
        elif match := _MEMORY.fullmatch(lower):
            query.memory = int(match[1])
        elif match := _COUNT.fullmatch(lower):
            query.count = int(match[1])
        elif lower in ("datacenter", "dc"):
            query.datacenter = True
        elif lower in _REGION_WORDS:
            place = _REGION_WORDS[lower]
            query.where += REGIONS[place].split() if place in REGIONS else [place]
        elif word.upper() in _COUNTRIES:  # de, us; not ti, as in 3060 ti
            query.where.append(word.upper())
        else:
            gpu.append(word)
    query.gpu = " ".join(gpu) or None
    query.where = sorted(set(query.where))
    return query


class Vast:
    def __init__(self, key: str | None = None):
        self._key = key

    def __repr__(self) -> str:
        return "Vast()"  # never the key

    def call(self, method: str, path: str, body=None, retry: bool = True, auth: bool = True):
        """One API call, as JSON. Retries when Vast says it is busy; other retries only for
        reads, so a rent request that may have arrived is never sent twice."""
        data = None if body is None else json.dumps(body).encode()
        for attempt in range(RETRIES):
            request = urllib.request.Request(API + path, data=data, method=method)
            request.add_header("Content-Type", "application/json")
            request.add_header("Accept", "application/json")
            request.add_header("User-Agent", f"ferry/{__version__}")
            if auth:
                if not self._key:
                    raise VastError("no Vast.ai key yet. Run: ferry login vast")
                request.add_header("Authorization", f"Bearer {self._key}")
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return _json(response.read())
            except urllib.error.HTTPError as e:
                answer = _json(e.read())
                busy = e.code == 429 and not (isinstance(answer, dict) and answer.get("error") == "search_quota_exceeded")
                if attempt + 1 < RETRIES and (busy or (retry and e.code >= 500)):
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise _error(e.code, answer) from None
            except OSError as e:
                if retry and attempt + 1 < RETRIES:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise VastError(f"could not reach Vast.ai: {getattr(e, 'reason', e)}") from None
        raise VastError("Vast.ai kept answering that it is busy; try again in a minute")

    # Offers

    def search(self, gpu: str | None = None, max_price: float | None = None, any_host: bool = False,
               disk: int = DISK_GB, offer: int | None = None, query: Query | None = None) -> list[Offer]:
        """Offers to rent now, cheapest first: a direct SSH port, enough disk, a driver
        current PyTorch runs on, and a verified host unless `any_host`. With `offer`, that
        one offer, if it is still free and meets the same bar. `query` narrows it further;
        `gpu` and `max_price` win over its own."""
        query = query or Query()
        gpu = gpu or query.gpu
        max_price = max_price if max_price is not None else query.max_price
        filters: dict = {
            "rentable": {"eq": True}, "rented": {"eq": False}, "direct_port_count": {"gte": 1},
            "reliability": {"gte": MIN_RELIABILITY}, "inet_down": {"gte": MIN_DOWNLOAD}, "cuda_max_good": {"gte": MIN_CUDA},
            "order": [["dph_total", "asc"]], "type": "on-demand", "allocated_storage": disk, "limit": 64,
        }
        if not any_host:
            filters["verified"] = {"eq": True}
        if gpu:
            names = matching_gpus(self.gpu_list(), gpu)
            if not names:
                return []  # no GPU is called that
            filters["gpu_name"] = {"in": names}
        if max_price is not None:
            filters["dph_total"] = {"lte": max_price}
        if offer is not None:
            filters["ask_contract_id"] = {"eq": offer}  # an offer's id; Vast ignores a filter on "id"
        if query.memory:
            filters["gpu_ram"] = {"gte": round(query.memory * 1024 * 0.92)}  # 24 GB cards say 23028 to 24564 MB
        if query.count:
            filters["num_gpus"] = {"eq": query.count}
        if query.where:
            filters["geolocation"] = {"in": query.where}  # two-letter codes; Vast matches the "City, CC" ending
        if query.datacenter:
            filters["datacenter"] = {"eq": True}
        answer = self.call("POST", "/bundles/", filters, auth=bool(self._key))
        offers = answer.get("offers") if isinstance(answer, dict) else None
        if not isinstance(offers, list):
            raise VastError("Vast.ai's offer list could not be read")
        found = [Offer.of(raw, disk) for raw in offers if isinstance(raw, dict) and raw.get("id")]
        # Vast's own price filter let a $0.625 offer through a $0.60 limit (9 Nov 2026), so check again.
        return [o for o in found if max_price is None or o.price <= max_price]

    def gpu_list(self) -> list[str]:
        """Every GPU name Vast uses, such as "RTX 3060 laptop", kept a day in Ferry's folder."""
        path = home() / NAMES_FILE
        try:
            if time.time() - path.stat().st_mtime < NAMES_SECONDS:
                names = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(names, list) and names:
                    return names
        except (OSError, ValueError):
            pass
        answer = self.call("GET", "/gpu_names/unique/", auth=False)
        names = answer.get("gpu_names") if isinstance(answer, dict) else None
        if not isinstance(names, list) or not names:
            raise VastError("Vast.ai's list of GPU names could not be read")
        names = [str(name) for name in names]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(names), encoding="utf-8")
        except OSError:
            pass  # asked again next time
        return names

    # The account

    def balance(self) -> float:
        me = self.call("GET", "/users/current/")
        if not isinstance(me, dict):
            raise VastError("Vast.ai's account answer could not be read")
        return float(me.get("credit") or 0)

    # Instances

    def create(self, offer: int, disk: int, onstart: str, label: str, image: str) -> int:
        """Rents the offer. Sent once only, never retried after it may have arrived."""
        body = {"client_id": "me", "image": image, "disk": disk, "label": label, "onstart": onstart,
                "runtype": "ssh_direc ssh_proxy", "env": {}}
        answer = self.call("PUT", f"/asks/{offer}/", body, retry=False)
        if not isinstance(answer, dict) or not answer.get("new_contract"):
            raise VastError(f"Vast.ai did not rent offer {offer}: {str(answer)[:200]}")
        return int(answer["new_contract"])

    def instance(self, instance: int) -> dict | None:
        """The instance, or None once it is gone ({"instances": null}; the account's list is
        not reliable for this)."""
        answer = self.call("GET", f"/instances/{instance}/?owner=me")
        found = answer.get("instances") if isinstance(answer, dict) else None
        return found if isinstance(found, dict) else None

    def set_state(self, instance: int, state: str) -> None:
        self.call("PUT", f"/instances/{instance}/", {"state": state})

    def destroy(self, instance: int) -> None:
        try:
            self.call("DELETE", f"/instances/{instance}/", {})
        except VastError as e:
            if e.status != 404:  # already gone
                raise

    def log(self, instance: int, tail: int = 400) -> str | None:
        """The instance's log (what onstart printed), fetched over HTTPS from where Vast puts it."""
        answer = self.call("PUT", f"/instances/request_logs/{instance}/", {"tail": str(tail)})
        url = answer.get("result_url") if isinstance(answer, dict) else None
        if not url or not url.startswith("https://"):
            return None
        for _ in range(20):  # Vast writes the file a moment after answering
            time.sleep(1.5)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": f"ferry/{__version__}"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    return response.read().decode("utf-8", "replace")
            except OSError:
                continue
        return None


def _json(raw: bytes):
    text = raw.decode("utf-8", "replace")
    try:
        return json.loads(text) if text.strip() else None
    except ValueError:
        return text[:300]


def _error(status: int, answer) -> VastError:
    detail = ""
    if isinstance(answer, dict):
        detail = str(answer.get("msg") or answer.get("error") or "")
    elif isinstance(answer, str):
        detail = answer.strip()
    if status in (401, 403):
        return VastError("Vast.ai did not accept your API key. Run: ferry login vast", status)
    if status == 429 and "search_quota" in detail:
        return VastError("Vast.ai's daily search limit is used up; it resets at midnight UTC", status)
    if status in (404, 410) and "no_such_ask" in detail:
        return VastError("that offer is gone: someone else rented it. `ferry gpus` shows what is free now", status)
    return VastError(f"Vast.ai answered {status}{': ' + detail[:200] if detail else ''}", status)
