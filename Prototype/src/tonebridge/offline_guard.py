"""Offline evidence: block and COUNT network use inside this process (test/eval/CLI only; the inference path never imports this).

``OfflineGuard`` patches the Python socket layer (connect / connect_ex / sendto / getaddrinfo / gethostbyname / create_connection): every attempt to reach
a non-unix address is counted in ``attempts`` and raises ``OfflineViolation`` (a ConnectionError). A sampler thread additionally polls the OS table
(``psutil.Process.net_connections('inet')``) so a native library that opened a socket behind Python's back would still show up in ``os_conns_seen``.
It also sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE (inherited by child processes).
Limits (stated, not hidden): the guard cannot see sockets that a native library opens without going through Python AND closes between two samples
(20 ms); loopback is counted like any other address.
"""
from __future__ import annotations

import os
import socket
import threading
import time

import psutil

OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


class OfflineViolation(ConnectionError):
    pass


class OfflineGuard:
    def __init__(self, sample_s: float = 0.02) -> None:
        self.attempts: list[str] = []
        self.os_conns_seen: set[str] = set()
        self._sample_s = sample_s
        self._orig: dict[str, object] = {}
        self._env_prev: dict[str, str | None] = {}
        self._stop = threading.Event()
        self._thr: threading.Thread | None = None

    # ---- patches ---------------------------------------------------------------------------------------------------------------------
    def _deny(self, what: str, target) -> None:
        self.attempts.append(f"{what}:{target!r}")
        raise OfflineViolation(f"network blocked (offline guard): {what} {target!r}")

    def __enter__(self) -> "OfflineGuard":
        for k, v in OFFLINE_ENV.items():
            self._env_prev[k] = os.environ.get(k)
            os.environ[k] = v
        guard = self
        S = socket.socket
        for name in ("connect", "connect_ex", "sendto"):
            self._orig[name] = getattr(S, name)

        def mk(name):
            orig = self._orig[name]

            def wrapped(sock, *a, **kw):
                if sock.family == getattr(socket, "AF_UNIX", -1):
                    return orig(sock, *a, **kw)
                guard._deny(name, a[-1] if a else None)
            return wrapped

        for name in ("connect", "connect_ex", "sendto"):
            setattr(S, name, mk(name))
        for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "create_connection"):
            self._orig[name] = getattr(socket, name)
            setattr(socket, name, (lambda n: lambda *a, **kw: guard._deny(n, a[:2]))(name))
        self._stop.clear()
        self._thr = threading.Thread(target=self._sample, daemon=True)
        self._thr.start()
        return self

    def _sample(self) -> None:
        p = psutil.Process(os.getpid())
        while not self._stop.is_set():
            self._poll(p)
            time.sleep(self._sample_s)

    def _poll(self, p: psutil.Process) -> None:
        try:
            for c in p.net_connections(kind="inet"):
                self.os_conns_seen.add(f"{c.laddr}->{c.raddr} {c.status}")
        except (psutil.Error, AttributeError):
            pass

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=1)
        self._poll(psutil.Process(os.getpid()))
        for name in ("connect", "connect_ex", "sendto"):
            setattr(socket.socket, name, self._orig[name])
        for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "create_connection"):
            setattr(socket, name, self._orig[name])
        for k, v in self._env_prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def report(self) -> dict:
        return {"net_attempts_python": len(self.attempts), "net_conns_os_seen": len(self.os_conns_seen), "offline_env": dict(OFFLINE_ENV),
                "attempt_detail": self.attempts[:5], "os_detail": sorted(self.os_conns_seen)[:5]}
