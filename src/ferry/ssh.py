"""SSH to a machine, in pure Python, so Windows needs no OpenSSH or rsync setup.

Host keys are pinned in Ferry's own known_hosts: a new key is shown once and
trusted only when the user says so, and a changed key stops everything.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import time
from collections.abc import Callable
from pathlib import Path

import paramiko

from .config import FerryError, Machine, known_hosts_path


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
    """One SSH connection to a machine."""

    def __init__(self, machine: Machine):
        self.machine = machine
        self.label = machine.name
        settings = {}
        ssh_config = Path.home() / ".ssh" / "config"
        if ssh_config.is_file():
            settings = paramiko.SSHConfig.from_path(str(ssh_config)).lookup(machine.host)
        self.hostname = settings.get("hostname", machine.host)
        self.port = machine.port or int(settings.get("port", 22))
        self.user = machine.user or settings.get("user") or getpass.getuser()
        keys = [machine.key] if machine.key else list(settings.get("identityfile", []))
        self.keys = [str(Path(k).expanduser()) for k in keys if Path(k).expanduser().is_file()]
        self._client: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None

    def _host_key_name(self) -> str:
        return self.hostname if self.port == 22 else f"[{self.hostname}]:{self.port}"

    def connect(self, trust: Callable[[str, str], bool] | None = None) -> Box:
        """Connects. `trust(fingerprint, key type)` decides about a host key Ferry has
        not seen; without it, an unknown key is refused."""
        self.close()
        path = known_hosts_path()
        client = paramiko.SSHClient()
        if path.is_file():
            client.load_host_keys(str(path))
        client.set_missing_host_key_policy(_Remember() if trust else paramiko.RejectPolicy())
        try:
            self._open(client)
        except _UnknownKey as unknown:
            client.close()
            if not trust(fingerprint(unknown.key), unknown.key.get_name()):
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
        except paramiko.AuthenticationException:
            client.close()
            raise FerryError(f"SSH refused the login to {self.user}@{self.hostname}: check your key or ssh-agent") from None
        except paramiko.SSHException as e:
            client.close()
            if "not found in known_hosts" in str(e):
                raise FerryError(f"{self.label}'s host key is not pinned yet. Run: ferry add {self.label} {self.machine.host}") from None
            raise FerryError(f"could not connect to {self.label}: {e}") from None
        except OSError as e:
            client.close()
            raise FerryError(f"could not reach {self.label} ({self.hostname}:{self.port}): {e}") from None
        self._client = client
        return self

    def _open(self, client: paramiko.SSHClient) -> None:
        client.connect(
            self.hostname,
            port=self.port,
            username=self.user,
            key_filename=self.keys or None,
            timeout=15,
            banner_timeout=15,
            auth_timeout=15,
            allow_agent=True,
            look_for_keys=True,
        )
        client.get_transport().set_keepalive(30)

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
