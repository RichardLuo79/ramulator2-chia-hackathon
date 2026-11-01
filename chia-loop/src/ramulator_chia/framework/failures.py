"""Small, credential-safe error records and conservative transport classification."""

import re
import subprocess
import traceback

import httpx


def safe_message(text):
    text = re.sub(r"(?i)Bearer\s+[^\s\"'\\,;]+", "Bearer [redacted]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", text)
    text = re.sub(
        r"(?i)((?:X-CHIA-Token|api[_-]?key|access_token|refresh_token)[\"'\\\s:=]+)[^\s\"'\\,;}]+",
        r"\1[redacted]", text,
    )
    return text


def exception_record(error, seen=None):
    """Keep group leaves and stack locations, without locals, source lines or argv."""
    seen = set() if seen is None else seen
    if id(error) in seen:
        return {"type": type(error).__name__, "message": "already recorded"}
    seen.add(id(error))
    message = (f"native process interrupted after {error.timeout} seconds"
               if isinstance(error, subprocess.TimeoutExpired) else safe_message(str(error)))
    record = {"type": type(error).__name__, "message": message}
    record["traceback"] = [
        {"file": frame.filename, "line": frame.lineno, "function": frame.name}
        for frame in traceback.extract_tb(error.__traceback__)
    ]
    children = getattr(error, "exceptions", ())
    if children:
        record["exceptions"] = [exception_record(child, seen) for child in children]
    if error.__cause__ is not None:
        record["cause"] = exception_record(error.__cause__, seen)
    elif error.__context__ is not None and not error.__suppress_context__:
        record["context"] = exception_record(error.__context__, seen)
    return record


def transient_error(error):
    """Every group leaf must be a known transient failure; unknowns never retry."""
    children = getattr(error, "exceptions", ())
    if children:
        return all(transient_error(child) for child in children)
    return (
        isinstance(error, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError))
        or getattr(error, "error_type", None) in {"server_error", "rate_limit", "max_output_tokens"}
    )
