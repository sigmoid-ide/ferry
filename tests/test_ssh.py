"""Reaching a machine: jump hosts from ~/.ssh/config, hop by hop, and the first of its addresses that answers."""

import os
import socket
import time

import pytest

from ferry.config import FerryError, Machine
from ferry.ssh import Box, connect_first


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


def listening():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    return server


def closed_port():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def at(host, port):
    return (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (host, port))


def test_an_address_that_refuses_or_hangs_does_not_stop_the_next():
    server = listening()
    port = server.getsockname()[1]
    try:
        sock = connect_first("box", port, addresses=[at("127.0.0.1", closed_port()), at("127.0.0.1", port)])
        assert sock.getpeername()[1] == port
        sock.close()
        # 192.0.2.1 is for documentation only: it never answers (or is unreachable at once).
        started = time.monotonic()
        sock = connect_first("box", port, timeout=5, addresses=[at("192.0.2.1", 22), at("127.0.0.1", port)])
        assert sock.getpeername()[1] == port
        assert time.monotonic() - started < 2  # a quarter second, not the whole timeout
        sock.close()
    finally:
        server.close()


@pytest.mark.skipif(os.name != "nt", reason="elsewhere the system grows the send buffer itself")
def test_on_windows_the_connection_gets_a_send_buffer_big_enough_for_far_machines():
    server = listening()
    port = server.getsockname()[1]
    try:
        sock = connect_first("box", port, addresses=[at("127.0.0.1", port)])
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF) >= 4 << 20
        sock.close()
    finally:
        server.close()


def test_when_no_address_answers_each_says_why():
    with pytest.raises(OSError, match="127.0.0.1"):
        connect_first("box", 22, timeout=2, addresses=[at("127.0.0.1", closed_port())])
