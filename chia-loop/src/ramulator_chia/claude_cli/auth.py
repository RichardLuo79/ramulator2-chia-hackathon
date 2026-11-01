"""Auth-only access for the unmodified Claude CLI, never an OAuth reimplementation.

Native-login mode binds one selected credential file read-only. Setup-token
mode injects one private file's opaque token into the native CLI environment
after installing the process boundary; no token is serialized in its policy.
No host Claude directory is copied or exposed. This adapter does not rotate
subscription tokens, turn on extra usage, or fall back to API billing.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
import stat
import time

KINDS = ("native_login", "setup_token")
from ramulator_chia.layout import ROOT as REPO


def read_setup_token(path):
    """Return a private opaque token in memory only; never infer its expiry."""
    path = pathlib.Path(path).resolve(strict=True)
    if path.is_relative_to(REPO):
        raise RuntimeError("setup token must be stored outside the repository and evaluation artifacts")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or not 1 <= info.st_size <= 4096):
            raise RuntimeError("setup token must be an owner-only regular private file")
        raw = os.read(fd, 4097)
    finally:
        os.close(fd)
    try:
        token = raw.decode("ascii").strip()
    except UnicodeError:
        raise RuntimeError("invalid private setup-token file") from None
    if not re.fullmatch(r"sk-ant-oat01-[A-Za-z0-9_-]{40,2048}", token):
        raise RuntimeError("setup-token file must contain one subscription OAuth token only")
    return token


def check(path, *, now=None, credential_kind="native_login"):
    if credential_kind not in KINDS:
        raise ValueError("unsupported explicit Claude credential kind")
    if credential_kind == "setup_token":
        read_setup_token(path)
        return {"auth_mode": "claude_subscription", "credential_kind": credential_kind,
                "subscription_type": None, "expires_at": None, "generation_calls": 0,
                "expiry_validation": "opaque token; provider authoritative, not inferred from file age"}
    path = pathlib.Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise RuntimeError("Claude auth must be an operator-owned private credential file")
    # Never return credential values, account IDs, or raw parsing errors.
    try:
        data = json.loads(path.read_text())
        oauth = data["claudeAiOauth"]
        if (not isinstance(oauth.get("accessToken"), str) or not oauth["accessToken"]
                or type(oauth.get("expiresAt")) not in (int, float) or not math.isfinite(oauth["expiresAt"])):
            raise ValueError()
        expiry = oauth["expiresAt"] / 1000
        if expiry <= (time.time() if now is None else now) + 2000:
            raise RuntimeError("Claude login expires too soon; renew through the official CLI before dispatch")
        if oauth.get("subscriptionType") not in {"max", "pro", "team", "enterprise"}:
            raise RuntimeError("recognized Claude subscription login required; no API fallback")
    except (KeyError, ValueError, TypeError):
        raise RuntimeError("invalid native Claude subscription credential record") from None
    return {"auth_mode": "claude_subscription", "subscription_type": oauth["subscriptionType"],
            "expires_at": expiry, "generation_calls": 0}


def bind(work, path, *, credential_kind="native_login"):
    check(path, credential_kind=credential_kind)
    if credential_kind == "setup_token":
        # The process boundary reads this one file after sandbox installation.
        # Do not copy the token into the private workspace or JSON policy.
        return pathlib.Path(path).resolve(strict=True)
    config = pathlib.Path(work) / "home/.claude"
    config.mkdir(parents=True, exist_ok=True)
    (config / ".credentials.json").symlink_to(pathlib.Path(path).resolve(strict=True))
    return pathlib.Path(path).resolve(strict=True)
