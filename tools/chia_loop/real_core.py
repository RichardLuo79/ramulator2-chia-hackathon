"""Pure protocol, source-boundary, and crash-safe spending rules for real trials."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import pathlib
import re
import time

from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.run_records import model_pricing
from tools.chia_loop import prompt_cache as K
from tools.chia_loop.pareto import objectives, dominates, pareto

MUTABLE = "src/ramulator/controller/impl/atomic_controller.cpp"
MODELS = {"pro": "gemini-3.1-pro-preview", "flash": "gemini-3.8-flash"}
# Both selected Vertex models support at most 65,536 generated tokens.
# Omitting the field uses a model default; it does not mean unlimited.
MAX_OUTPUT = 65_536
MAX_CONTEXT_BYTES = 8_000_000  # transport/runaway guard, not the model context limit
MAX_INPUT_TOKENS = 900_000  # leaves room for 65,536 output tokens in a 1M context
MAX_ITERATIONS = 5
CAP_USD = 50.0
PRICING = {
    "source": "https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing",
    "checked_utc": "2026-09-05",
    "units": "USD per million tokens; global standard; output includes thinking",
    "pro": {"input": 2.0, "output": 12.0, "long_input": 4.0, "long_output": 18.0},
    "flash": {"input": 0.75, "output": 3.75, "long_input": 0.75, "long_output": 3.75},
    "flash_introductory_pricing_until": "2026-12-31",
    "conservative_cap_rates": {
        "pro": {"input": 4.8, "output": 21.6},
        "flash": {"input": 1.8, "output": 9.0},
    },
}


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def parse_provider_response(raw):
    """Treat provider truncation as such, before trying to parse proposal JSON."""
    candidates = raw.get("candidates") or []
    if not candidates:
        return {"status": "failed", "failure_kind": "missing_candidate", "reason": "provider returned no candidate"}
    candidate = candidates[0]
    finish = candidate.get("finish_reason")
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text") or "" for p in parts if not p.get("thought"))
    if finish == "MAX_TOKENS":
        return {"status": "failed", "failure_kind": "generation_truncated", "finish_reason": finish,
            "reason": "MAX_TOKENS at the configured generation ceiling; incomplete patches are not repaired or applied",
            "usage": raw.get("usage_metadata"), "response_excerpt": text[-4000:]}
    if finish != "STOP":
        return {"status": "failed", "failure_kind": "provider_finish", "finish_reason": finish,
                "reason": f"provider did not finish normally: {finish}"}
    try:
        proposal = json.loads(text)
        if not isinstance(proposal, dict):
            raise ValueError("proposal must be a JSON object")
        return proposal
    except (ValueError, TypeError) as exc:
        return {"status": "failed", "failure_kind": "invalid_json",
                "reason": f"invalid proposal JSON: {exc}", "response_excerpt": text[-4000:]}


class BudgetExhausted(RuntimeError):
    pass


class Ledger:
    """Reserve pessimistically BEFORE dispatch; unresolved calls keep reservations.

    Reservations use counted tokens plus 10%/4K framing slack, or UTF-8 bytes
    if no count is supplied, and twice the output cap including thinking.
    Successful usage replaces the reservation with a conservative tariff bound.
    Actual standard-rate estimates are reported separately, never used to raise
    the authorized per-run ceiling. No SDK hidden retries are allowed.
    """
    def __init__(self, path: pathlib.Path, arm: str, *, run_id=None, cap_usd=None, iteration_guard=False):
        self.path, self.arm = pathlib.Path(path), arm
        self.run_id = run_id
        if type(iteration_guard) is not bool or (iteration_guard and (cap_usd is not None or not run_id)):
            raise ValueError("iteration-only accounting requires a named run and no USD cap")
        self.iteration_guard = iteration_guard
        if cap_usd is not None and (isinstance(cap_usd, bool) or not isinstance(cap_usd, (int, float))
                                   or not math.isfinite(cap_usd) or cap_usd <= 0):
            raise ValueError("budget cap must be finite and positive")
        self.cap_usd = cap_usd
        if arm not in MODELS:
            raise ValueError("unknown model backend")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _check_owner(self, data):
        if self.iteration_guard != (data.get("guard_mode") == "iterations"):
            raise ValueError("cannot change an initialized ledger guard mode")
        if self.iteration_guard and data.get("cap_usd") is not None:
            raise ValueError("iteration-only ledger cannot carry a dollar ceiling")
        if data.get("backend", data.get("arm")) != self.arm:
            raise ValueError("ledger belongs to a different model backend")
        if data.get("model", MODELS[self.arm]) != MODELS[self.arm]:
            raise ValueError("ledger belongs to a different model version")
        if self.run_id is not None and data.get("run_id") != self.run_id:
            raise ValueError("ledger belongs to a different run")
        if self.cap_usd is not None and data["cap_usd"] != self.cap_usd:
            raise ValueError("cannot change an initialized run budget cap")

    def _transaction(self, update):
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {
                "schema_version": 2, "record_type": "run_budget", "run_id": self.run_id,
                "backend": self.arm, "model": MODELS[self.arm],
                "cap_usd": None if self.iteration_guard else self.cap_usd if self.cap_usd is not None else CAP_USD,
                **({"guard_mode": "iterations"} if self.iteration_guard else {}),
                "pricing": model_pricing(PRICING, self.arm), "calls": []}
            self._check_owner(data)
            result = update(data)
            atomic_write_json(self.path, data)
            return result

    def initialize_carryover(self, carryover):
        """Keep earlier infrastructure-attempt charges inside the same authorization."""
        charge = carryover.get("cap_charge_usd", 0)
        estimate = carryover.get("estimated_standard_usd", 0)
        ceiling = math.inf if self.iteration_guard else self.cap_usd if self.cap_usd is not None else CAP_USD
        if (not math.isfinite(charge) or not math.isfinite(estimate)
                or not 0 <= estimate <= charge <= ceiling):
            raise ValueError("invalid budget carryover")
        def update(data):
            if data["calls"] or data.get("carryover"):
                raise ValueError("cannot replace an initialized run budget")
            data["carryover"] = carryover
        self._transaction(update)

    def initialize(self):
        """Create an owned empty ledger, including for runs that make no calls."""
        self._transaction(lambda data: None)

    def reserve(self, payload: str, iteration: int, turn: int, input_tokens=None, *,
                purpose="proposal", backend=None, operation_key=None):
        # A run still has one proposing model. Its isolated reviewer is a
        # separately identified service, charged to this SAME authorization.
        backend = self.arm if backend is None else backend
        if backend not in MODELS or purpose not in {"proposal", "review"}:
            raise ValueError("invalid generation role/backend")
        if purpose == "proposal" and backend != self.arm:
            raise ValueError("cannot substitute a different proposing model")
        size = len(payload.encode("utf-8"))
        if size > MAX_CONTEXT_BYTES:
            raise BudgetExhausted("finite context-byte limit reached")
        if input_tokens is not None and (isinstance(input_tokens, bool) or
                not isinstance(input_tokens, int) or not 0 <= input_tokens <= MAX_INPUT_TOKENS):
            raise BudgetExhausted("input token limit reached or invalid token count")
        bound = size + 4096 if input_tokens is None else int(input_tokens * 1.1) + 4096
        rates = PRICING["conservative_cap_rates"][backend]
        reserve = (bound * rates["input"] + 2 * MAX_OUTPUT * rates["output"]) / 1e6
        def update(data):
            committed = data.get("carryover", {}).get("cap_charge_usd", 0) + sum(c["cap_charge_usd"] for c in data["calls"])
            if data["cap_usd"] is not None and committed + reserve > data["cap_usd"]:
                raise BudgetExhausted(f"per-run ${data['cap_usd']:g} ceiling would be exceeded by next call")
            call_id = len(data["calls"])
            data["calls"].append({"id": call_id, "iteration": iteration, "turn": turn,
                "purpose": purpose, "backend": backend, "model": MODELS[backend],
                "operation_key": operation_key, "pricing": model_pricing(PRICING, backend),
                "reserved_at": time.time(), "state": "reserved", "payload_sha256": sha(payload),
                "counted_input_tokens": input_tokens,
                "reserved_usd": reserve, "cap_charge_usd": reserve,
                "estimated_standard_usd": None})
            # Both new layout arms use the same accounting convention. A
            # legacy-layout ablation may still receive provider cache hits.
            if K.load(self.path.parent) is not None:
                data["calls"][-1]["cache_accounting"] = "reported_reads_discount_v1"
            return call_id
        return self._transaction(update)

    def settle(self, call_id: int, usage: dict | None, error: str | None = None, http_status=None):
        def update(data):
            row = data["calls"][call_id]
            if row["state"] == "usage_recorded":
                if usage == row.get("usage") and error is None and http_status is None:
                    return  # Idempotent replay preserves the original receipt time.
                raise RuntimeError("cannot replace a settled provider usage receipt")
            if row["state"] == "unbilled_http_error":
                if http_status == row.get("http_status") and usage is None:
                    return
                raise RuntimeError("cannot replace a recorded HTTP error with a different outcome")
            row.update({"usage": usage, "error": error, "settled_at": time.time()})
            if isinstance(http_status, int) and 400 <= http_status <= 599:
                # Google's published policy bills only HTTP 200 responses.
                # A transport timeout without an HTTP response is still unknown.
                row.update({"state": "unbilled_http_error", "http_status": http_status,
                    "cap_charge_usd": 0.0, "estimated_standard_usd": 0.0})
                return
            if not usage or usage.get("prompt_token_count") is None:
                row["state"] = "usage_unknown_reservation_retained"
                return
            prompt = usage["prompt_token_count"]
            # Gemini candidates_token_count excludes thoughts_token_count.
            output = (usage.get("candidates_token_count") or 0) + (usage.get("thoughts_token_count") or 0)
            output = max(output, (usage.get("total_token_count") or 0) - prompt)
            if any(not isinstance(n, int) or n < 0 for n in (prompt, output)):
                raise RuntimeError("invalid provider token usage")
            backend = row.get("backend", self.arm)
            rate = PRICING[backend]
            long = prompt > 200_000
            row["estimated_standard_usd"] = (prompt * rate["long_input" if long else "input"]
                + output * rate["long_output" if long else "output"]) / 1e6
            if row.get("cache_accounting") == "reported_reads_discount_v1":
                cached = usage.get("cached_content_token_count")
                if cached is not None and (type(cached) is not int or not 0 <= cached <= prompt):
                    raise RuntimeError("invalid cached input token count")
                row["cached_input_tokens"] = cached
                row["cache_cost_assumptions"] = [] if cached is not None else ["missing_cache_counter_priced_as_uncached_upper_estimate"]
                row["estimated_standard_usd"] -= (cached or 0) * .9 * rate["long_input" if long else "input"] / 1e6
                row["cache_price_source"] = "https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview"
            # The stopping guard intentionally remains conservative: a cache hit
            # does not silently expand an existing campaign's authorization.
            guard = PRICING["conservative_cap_rates"][backend]
            row["cap_charge_usd"] = (prompt * guard["input"] + output * guard["output"]) / 1e6
            row["billed_output_including_thinking"] = output
            row["state"] = "usage_recorded"
            if row["cap_charge_usd"] > row["reserved_usd"]:
                raise RuntimeError("provider usage exceeded conservative reservation")
        self._transaction(update)

    def totals(self):
        if not self.path.exists():
            return {"api_attempts": 0, "cap_charge_usd": 0, "estimated_standard_usd": 0}
        data = json.loads(self.path.read_text())
        self._check_owner(data)
        calls, carry = data["calls"], data.get("carryover", {})
        return {"api_attempts": len(calls),
            "carryover_api_attempts": carry.get("api_attempts", 0),
            "carryover_cap_charge_usd": carry.get("cap_charge_usd", 0),
            "carryover_estimated_standard_usd": carry.get("estimated_standard_usd", 0),
            "cap_charge_usd": carry.get("cap_charge_usd", 0) + sum(c["cap_charge_usd"] for c in calls),
            "estimated_standard_usd": carry.get("estimated_standard_usd", 0) + sum(c["estimated_standard_usd"] or 0 for c in calls),
            "unknown_usage_calls": carry.get("unknown_usage_calls", 0) + sum(c["estimated_standard_usd"] is None for c in calls),
            "prompt_tokens": sum((c.get("usage") or {}).get("prompt_token_count") or 0 for c in calls),
            "thinking_tokens": sum((c.get("usage") or {}).get("thoughts_token_count") or 0 for c in calls),
            "output_tokens_including_thinking": sum(c.get("billed_output_including_thinking", 0) for c in calls)}

    def calls(self):
        """Read the durable journal; the runner/operation locks serialize dispatch."""
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text())
        self._check_owner(data)
        return data["calls"]


def regions(source: str):
    result = {}
    backbone = source
    for name in ("INCLUDES", "CODE"):
        tag = "CHIA_MODEL_INCLUDES" if name == "INCLUDES" else "CHIA_MODEL"
        pattern = rf"(// {tag}_BEGIN\n)(.*?)([ \t]*// {tag}_END)"
        matches = list(re.finditer(pattern, backbone, re.S))
        if len(matches) != 1:
            raise ValueError(f"exactly one {tag} region required")
        result[name] = matches[0].group(2)
        backbone = re.sub(pattern, r"\1<MODEL REGION>\3", backbone, flags=re.S)
    return backbone, result


def assemble_regions(parent: str, proposed: dict) -> str:
    """Assemble exact model-owned bodies; never ask an agent to copy the scaffold."""
    if not isinstance(proposed, dict) or set(proposed) != {"includes", "code"}:
        raise ValueError("regions must contain exactly string fields includes and code")
    if any(not isinstance(v, str) for v in proposed.values()):
        raise ValueError("region bodies must be strings")
    if any("CHIA_MODEL" in v for v in proposed.values()):
        raise ValueError("return region bodies only, without CHIA_MODEL boundary markers")
    candidate = parent
    for key, tag in (("includes", "CHIA_MODEL_INCLUDES"), ("code", "CHIA_MODEL")):
        body = proposed[key]
        if body and not body.endswith("\n"):
            body += "\n"
        pattern = rf"(// {tag}_BEGIN\n)(.*?)([ \t]*// {tag}_END)"
        candidate, count = re.subn(pattern, lambda m: m[1] + body + m[3], candidate, flags=re.S)
        if count != 1:
            raise ValueError("invalid parent region boundaries")
    if candidate == parent:
        raise ValueError("proposal makes no source change")
    return candidate


def validate_source(seed: str, candidate: str):
    if len(candidate.encode()) > 250_000:
        raise ValueError("candidate exceeds 250 KB source limit")
    trusted, _ = regions(seed)
    backbone, body = regions(candidate)
    if trusted != backbone:
        raise ValueError("changed protected lifecycle/instrumentation outside model regions")
    allowed_headers = {"array", "cmath", "deque", "map", "numeric", "tuple", "utility",
                       "unordered_map", "set", "unordered_set", "optional", "cstddef",
                       "algorithm", "cstdint", "limits", "queue", "string", "vector",
                       "bit", "bitset", "functional", "numbers", "span", "type_traits", "stdexcept"}
    includes = re.sub(r"//[^\n]*|/\*.*?\*/", "", body["INCLUDES"], flags=re.S)
    for line in includes.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*#include <([\w]+)>\s*", line)
        if not match or match[1] not in allowed_headers:
            raise ValueError(f"unsupported model include: {line}")
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", body["CODE"], flags=re.S)
    if "#" in code or "\\\n" in code:
        raise ValueError("preprocessor/escaped lines prohibited in model code")
    forbidden = (r"\b(?:asm|__asm__|reinterpret_cast|const_cast|dlopen|dlsym|syscall|system|"
        r"popen|fopen|freopen|open|ifstream|fstream|filesystem|socket|fork|execve|getenv|"
        r"m_parent|m_pending|m_trace_file|m_trace_path|m_stats|m_config|frontend_id|"
        r"RAMULATOR_PARSE_PARAM|m_model_parameters|m_model_parameter_entries|"
        r"m_used_model_parameters|m_model_initializing|parse_model_parameters|"
        r"frontend_sub_id|source_id|admission_ordinal|callback|Factory|RAMULATOR_CREATE_CHILD|"
        r"get_preq_command|issue_command|check_ready|check_rowbuffer_hit|"
        r"s_num_read_reqs|s_num_write_reqs|s_read_latency)\b")
    match = re.search(forbidden, code)
    if match:
        raise ValueError("model uses forbidden I/O, metadata, scheduler, or lifecycle access: " + match[0])
    if re.search(r"m_device\s*\.\s*(?!m_spec\b)\w+", code):
        raise ValueError("model may access DRAM specification, not command-device operations")
    return {"source_sha256": sha(candidate), "backbone_sha256": sha(backbone),
            "static_boundary_pass": True, "requires_semantic_review": True}


def apply_unified(parent: str, patch: str) -> str:
    """Apply exact hunks in memory; no shell, git history, fuzz, or extra paths."""
    lines = patch.splitlines(keepends=True)
    start = next((i for i, s in enumerate(lines) if s.startswith("--- ")), None)
    if start is None or start + 1 >= len(lines):
        raise ValueError("missing unified diff file headers")
    def target(s):
        return s[4:].strip().split("\t")[0].removeprefix("a/").removeprefix("b/")
    if target(lines[start]) != MUTABLE or not lines[start + 1].startswith("+++ ") or target(lines[start + 1]) != MUTABLE:
        raise ValueError("patch must target only the Atomic controller")
    source = parent.splitlines(keepends=True)
    output, cursor, pos = [], 0, start + 2
    while pos < len(lines):
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[pos])
        if not match:
            if lines[pos].strip():
                raise ValueError(f"invalid hunk header: {lines[pos][:120]}")
            pos += 1
            continue
        pos += 1
        old, new = [], []
        while pos < len(lines) and not lines[pos].startswith("@@ "):
            line = lines[pos]
            if line.startswith(" "): old.append(line[1:]); new.append(line[1:])
            elif line.startswith("-") and not line.startswith("--- "): old.append(line[1:])
            elif line.startswith("+") and not line.startswith("+++ "): new.append(line[1:])
            elif line.startswith("\\ No newline"):
                raise ValueError("non-newline-terminated patch not supported")
            else:
                raise ValueError("patch contains malformed or additional file data")
            pos += 1
        # Context itself is authoritative; tolerate only inaccurate line/count
        # annotations, never a fuzzy context match or an ambiguous insertion.
        expected = int(match[1]) - 1
        locations = [i for i in range(cursor, len(source) - len(old) + 1) if source[i:i+len(old)] == old]
        if not old or (expected not in locations and len(locations) != 1):
            raise ValueError("hunk context not found uniquely in parent")
        at = expected if expected in locations else locations[0]
        output.extend(source[cursor:at]); output.extend(new); cursor = at + len(old)
    output.extend(source[cursor:])
    result = "".join(output)
    if result == parent:
        raise ValueError("proposal makes no source change")
    return result


def render(template: str, fields: dict):
    result = re.sub(r"{{(\w+)}}", lambda m: str(fields[m[1]]), template)
    if re.search(r"{{\w+}}", result):
        raise ValueError("unresolved prompt field")
    return result
