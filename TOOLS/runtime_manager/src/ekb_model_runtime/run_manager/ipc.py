"""Authenticated JSON over loopback TCP; no pickle, shell or bearer on wire."""
from __future__ import annotations
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import socket

from .common import RunManagerError, canonical

MAX_MESSAGE = 2 * 1024 * 1024


def secure_root(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
        advapi.SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
        advapi.SetFileSecurityW.restype = wintypes.BOOL
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        descriptor = ctypes.c_void_p()
        # SYSTEM and the object owner only; inherited by this user's task files.
        if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW("D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;OW)", 1, ctypes.byref(descriptor), None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not advapi.SetFileSecurityW(str(root), 4 | 0x80000000, descriptor):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.LocalFree(descriptor)
    else:
        root.chmod(0o700)


def auth_key(root, *, create=False):
    path = Path(root) / "auth.key"
    if create and not path.exists():
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(secrets.token_bytes(32))
    value = path.read_bytes()
    if len(value) != 32:
        raise RunManagerError("INVALID_AUTH_KEY")
    return value


def mac(key, challenge, value):
    return hmac.new(key, (challenge + "\n" + canonical(value)).encode("utf-8"), hashlib.sha256).hexdigest()


def send(stream, value):
    data = canonical(value).encode("utf-8") + b"\n"
    if len(data) > MAX_MESSAGE:
        raise RunManagerError("IPC_MESSAGE_TOO_LARGE")
    stream.write(data)
    stream.flush()


def receive(stream):
    line = stream.readline(MAX_MESSAGE + 1)
    if not line or len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
        raise RunManagerError("INVALID_IPC_FRAME")
    value = json.loads(line.decode("utf-8"))
    if not isinstance(value, dict):
        raise RunManagerError("INVALID_IPC_MESSAGE")
    return value


def exchange(root, endpoint, operation, arguments, timeout=10):
    key = auth_key(root)
    if endpoint.get("host") != "127.0.0.1":
        raise RunManagerError("NONLOCAL_ENDPOINT_REJECTED")
    with socket.create_connection(("127.0.0.1", endpoint["port"]), timeout=timeout) as connection:
        connection.settimeout(timeout)
        with connection.makefile("rwb") as stream:
            greeting = receive(stream)
            challenge = greeting["challenge"]
            if not isinstance(challenge, str) or len(challenge) != 64:
                raise RunManagerError("INVALID_AUTH_CHALLENGE")
            request = {"operation": operation, "arguments": arguments}
            send(stream, {"request": request, "mac": mac(key, challenge, request)})
            response = receive(stream)
            value = response["response"]
            if not hmac.compare_digest(response.get("mac", ""), mac(key, challenge, value)):
                raise RunManagerError("UNAUTHENTICATED_RESPONSE")
            if not value.get("ok"):
                error = value.get("error", {})
                raise RunManagerError(error.get("code", "MANAGER_ERROR"), error.get("message"), error.get("details"))
            return value["value"]
