"""Auth-only access for the unmodified Claude CLI, never an OAuth reimplementation.

The CLI reads one explicitly selected credential file through a temporary,
read-only binding. No host Claude directory is copied or exposed. Expiring
credentials stop before dispatch: this adapter does not rotate subscription
tokens, turn on extra usage, or fall back to API billing.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import time


def check(path, *, now=None):
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


def bind(work, path):
    check(path)
    config = pathlib.Path(work) / "home/.claude"
    config.mkdir(parents=True, exist_ok=True)
    (config / ".credentials.json").symlink_to(pathlib.Path(path).resolve(strict=True))
    return pathlib.Path(path).resolve(strict=True)
