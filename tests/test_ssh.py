"""Jump hosts come from ~/.ssh/config as OpenSSH reads them, hop by hop."""

import os

import pytest

from ferry.config import FerryError, Machine
from ferry.ssh import Box


@pytest.fixture
def ssh_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() on Windows
    monkeypatch.setenv("FERRY_HOME", str(tmp_path / "ferry"))
    (tmp_path / ".ssh").mkdir()

    def write(text):
        (tmp_path / ".ssh" / "config").write_text(text)

    return write


def test_a_proxyjump_chain_is_followed_hop_by_hop(ssh_config):
    ssh_config("Host gpu\n  HostName 10.0.0.5\n  User me\n  ProxyJump first, admin@second:2200\n"
               "Host second\n  HostName second.lab\n")
    box = Box(Machine(name="gpu", host="gpu"))
    assert (box.hostname, box.user, box.jump) == ("10.0.0.5", "me", "first, admin@second:2200")
    last = Box(Machine(name="second", host="second", user="admin", port=2200), via="first", parent=box)
    assert (last.hostname, last.port, last.user, last.jump) == ("second.lab", 2200, "admin", "first")
    assert last.label == "gpu's jump host second"
    assert Box(Machine(name="first", host="first"), via=None, parent=last).jump is None


def test_proxyjump_none_means_a_direct_connection(ssh_config):
    ssh_config("Host gpu\n  ProxyJump none\n")
    assert Box(Machine(name="gpu", host="gpu")).jump is None


def test_a_proxyjump_that_leads_back_to_itself_is_refused(ssh_config):
    ssh_config("Host *\n  ProxyJump hop\n")
    with pytest.raises(FerryError, match="leads back to itself"):
        Box(Machine(name="gpu", host="gpu")).connect()


@pytest.mark.skipif(os.name != "nt", reason="ProxyCommand works elsewhere")
def test_a_proxycommand_on_windows_says_to_use_proxyjump(ssh_config):
    ssh_config("Host gpu\n  ProxyCommand ssh -W %h:%p bastion\n")
    with pytest.raises(FerryError, match="A ProxyJump line works"):
        Box(Machine(name="gpu", host="gpu")).connect()
