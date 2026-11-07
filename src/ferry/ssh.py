"""SSH to a machine, in pure Python, so Windows needs no OpenSSH or rsync setup.

Host keys are pinned in Ferry's own known_hosts: a new key is shown once and
trusted only when the user says so, and a changed key stops everything. Jump hosts
(ProxyJump in ~/.ssh/config) are pinned the same way.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import paramiko

from .config import FerryError, Machine, can_ask, known_hosts_path, parse_target

MAX_JUMPS = 8  # more hops than this is a ProxyJump that leads back to itself

# Ferry says what went wrong in its own words; paramiko's log lines would only repeat it, raw.
logging.getLogger("paramiko").addHandler(logging.NullHandler())


def fingerprint(key: paramiko.PKey) -> str:
    """The key's fingerprint as OpenSSH prints it."""
    return "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")


class _UnknownKey(Exception):
    def __init__(self, key: paramiko.PKey):
        self.key = key


class _Remember(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        raise _UnknownKey(key)


class Box:
    """One SSH connection to a machine, through its jump hosts when it has any."""

    def __init__(self, machine: Machine, via: str | None = None, parent: Box | None = None):
        """`via` is the ProxyJump chain to use instead of the one in ~/.ssh/config; `parent`
        is the box this one is a jump host for."""
        self.machine = machine
        self.top: Box = parent.top if parent else self
        self.depth: int = parent.depth + 1 if parent else 0
        self.label = machine.name if parent is None else f"{self.top.label}'s jump host {machine.host}"
        settings = {}
        ssh_config = Path.home() / ".ssh" / "config"
        if ssh_config.is_file():
            settings = paramiko.SSHConfig.from_path(str(ssh_config)).lookup(machine.host)
        self.hostname = settings.get("hostname", machine.host)
        self.port = machine.port or int(settings.get("port", 22))
        self.user = machine.user or settings.get("user") or getpass.getuser()
        keys = [machine.key] if machine.key else list(settings.get("identityfile", []))
        self.keys = [str(Path(k).expanduser()) for k in keys if Path(k).expanduser().is_file()]
        jump = settings.get("proxyjump") if via is None else via
        self.jump = None if not jump or jump.lower() == "none" else jump
        self.proxy_command = settings.get("proxycommand")  # None when unset or "none"
        self._hop: Box | None = None
        self._client: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None
        self._passphrase: str | None = None  # typed once per ferry command, kept in memory only

    def _host_key_name(self) -> str:
        return self.hostname if self.port == 22 else f"[{self.hostname}]:{self.port}"

    def connect(self, trust: Callable[[str, str, str], bool] | None = None) -> Box:
        """Connects. `trust(label, fingerprint, key type)` decides about a host key Ferry
        has not seen, the machine's or a jump host's; without it, an unknown key is refused."""
        self.close()
        path = known_hosts_path()
        client = paramiko.SSHClient()
        if path.is_file():
            client.load_host_keys(str(path))
        client.set_missing_host_key_policy(_Remember() if trust else paramiko.RejectPolicy())
        try:
            self._open(client, trust)
        except _UnknownKey as unknown:
            client.close()
            if not trust(self.label, fingerprint(unknown.key), unknown.key.get_name()):
                raise FerryError(f"did not trust {self.label}'s host key") from None
            path.parent.mkdir(parents=True, exist_ok=True)
            keys = paramiko.HostKeys(str(path)) if path.is_file() else paramiko.HostKeys()
            keys.add(self._host_key_name(), unknown.key.get_name(), unknown.key)
            keys.save(str(path))
            return self.connect()
        except paramiko.BadHostKeyException:
            client.close()
            raise FerryError(
                f"{self.label}'s host key has changed since you added it. If you reinstalled the machine, "
                f"remove its line from {path} and run ferry add again; otherwise something may be in the way"
            ) from None
        except paramiko.PasswordRequiredException:
            client.close()
            if self._passphrase is not None:
                raise FerryError("that passphrase did not open your SSH key") from None
            if not can_ask():
                raise FerryError("your SSH key is protected by a passphrase: add it to ssh-agent (ssh-add), "
                                 "or run ferry in a terminal to type it") from None
            self._passphrase = getpass.getpass("passphrase for your SSH key: ")
            return self.connect(trust)
        except paramiko.AuthenticationException:
            client.close()
            raise FerryError(f"SSH refused the login to {self.user}@{self.hostname}: check your key or ssh-agent") from None
        except paramiko.SSHException as e:
            client.close()
            if "not found in known_hosts" in str(e):
                raise FerryError(f"{self.label}'s host key is not pinned yet. "
                                 f"Run: ferry add {self.top.label} {self.top.machine.host}") from None
            raise FerryError(f"could not connect to {self.label}: {e}") from None
        except OSError as e:
            client.close()
            raise FerryError(f"could not reach {self.label} ({self.hostname}:{self.port}): {e}") from None
        self._client = client
        return self

    def _open(self, client: paramiko.SSHClient, trust) -> None:
        client.connect(
            self.hostname,
            port=self.port,
            username=self.user,
            key_filename=self.keys or None,
            passphrase=self._passphrase,
            timeout=15,
            banner_timeout=15,
            auth_timeout=15,
            allow_agent=True,
            look_for_keys=True,
            sock=self._tunnel(trust),
        )
        client.get_transport().set_keepalive(30)

    def _tunnel(self, trust):
        """The way to the machine when it is not reached directly: through the last jump
        host (which reaches its own jump hosts the same way), or a ProxyCommand."""
        if self.jump:
            if self.depth >= MAX_JUMPS:
                raise FerryError(f"the ProxyJump for {self.top.label} in ~/.ssh/config leads back to itself")
            *before, last = [hop.strip() for hop in self.jump.split(",")]
            host, user, port = parse_target(last)
            self._hop = Box(Machine(name=last, host=host, user=user, port=port), via=",".join(before) or None, parent=self)
            self._hop._passphrase = self._passphrase
            self._hop.connect(trust)
            self._passphrase = self._passphrase or self._hop._passphrase  # the same key is often used for both
            try:
                return self._hop._client.get_transport().open_channel("direct-tcpip", (self.hostname, self.port), ("127.0.0.1", 0))
            except paramiko.ChannelException as e:
                raise FerryError(f"{self._hop.label} could not reach {self.hostname}:{self.port} ({e.text})") from None
        if self.proxy_command:
            if os.name == "nt":  # paramiko waits on the command's output with select(), which Windows has only for sockets
                raise FerryError(f"{self.label} is reached through a ProxyCommand in ~/.ssh/config, which Ferry cannot "
                                 f"run on Windows yet. A ProxyJump line works")
            return paramiko.ProxyCommand(self.proxy_command)
        return None

    def run(self, command: str, stdin: bytes | None = None, on_line: Callable[[str], None] | None = None,
            timeout: float = 300) -> tuple[int, bytes, bytes]:
        """Runs a shell command on the machine. With `on_line`, output (stdout and stderr
        together) is passed on line by line as it arrives. Gives up after `timeout`
        seconds without output, so nothing waits forever."""
        if self._client is None:
            raise FerryError(f"not connected to {self.label}")
        chan = self._client.get_transport().open_session()
        chan.settimeout(timeout)
        if on_line is not None:
            chan.set_combine_stderr(True)
        chan.exec_command(command)
        if stdin:
            chan.sendall(stdin)
        chan.shutdown_write()
        out, err, pending = bytearray(), bytearray(), b""
        last_activity = time.monotonic()
        while True:
            if time.monotonic() - last_activity > timeout:
                chan.close()
                raise FerryError(f"{self.label} did not answer within {int(timeout)} seconds")
            if chan.recv_ready():
                data = chan.recv(65536)
                last_activity = time.monotonic()
                out += data
                if on_line is not None:
                    pending += data
                    *lines, pending = pending.split(b"\n")
                    for line in lines:
                        on_line(line.decode("utf-8", "replace").rstrip("\r"))
            elif chan.recv_stderr_ready():
                err += chan.recv_stderr(65536)
                last_activity = time.monotonic()
            elif chan.exit_status_ready():
                break
            else:
                time.sleep(0.02)
        if on_line is not None and pending:
            on_line(pending.decode("utf-8", "replace").rstrip("\r"))
        code = chan.recv_exit_status()
        chan.close()
        return code, bytes(out), bytes(err)

    def sftp(self) -> paramiko.SFTPClient:
        if self._client is None:
            raise FerryError(f"not connected to {self.label}")
        if self._sftp is None:
            self._sftp = self._client.open_sftp()
        return self._sftp

    def close(self) -> None:
        if self._sftp is not None:
            self._sftp.close()
            self._sftp = None
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._hop is not None:
            self._hop.close()
            self._hop = None
