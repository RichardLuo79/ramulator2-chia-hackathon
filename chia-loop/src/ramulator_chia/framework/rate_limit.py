"""Bounded HTTP-429 backoff; never retry a conversation or a tool effect."""

from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import hashlib
import math
import random
import time
from uuid import uuid4

import httpx


@dataclass(frozen=True)
class RateLimitPolicy:
    attempts: int = 5
    initial_delay_seconds: float = 30
    maximum_delay_seconds: float = 300
    jitter_fraction: float = 0.2

    def __post_init__(self):
        if type(self.attempts) is not int or not 1 <= self.attempts <= 10:
            raise ValueError("rate-limit attempts must be between 1 and 10")
        for value in (self.initial_delay_seconds, self.maximum_delay_seconds, self.jitter_fraction):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("backoff settings must be finite numbers")
        if not 0 < self.initial_delay_seconds <= self.maximum_delay_seconds:
            raise ValueError("backoff delays must be positive and ordered")
        if not 0 <= self.jitter_fraction <= 1:
            raise ValueError("backoff jitter must be between 0 and 1")


class RateLimitTransport(httpx.BaseTransport):
    """Replay only an explicit 429 rejection, retaining identical request bytes.

    Other responses and transport exceptions pass through unchanged. SDK retries
    remain disabled. Each retry gets a durable operational receipt before sleep;
    no credentials, response bodies, or prompt text enter those receipts.
    """

    def __init__(self, inner, policy, record, *, sleep=time.sleep, clock=time.time,
                 jitter=random.uniform):
        self.inner, self.policy, self.record = inner, policy, record
        self.sleep, self.clock, self.jitter = sleep, clock, jitter

    def handle_request(self, request):
        body = request.read()
        identity = hashlib.sha256(body).hexdigest()
        request_id = uuid4().hex
        for attempt in range(1, self.policy.attempts + 1):
            response = self.inner.handle_request(request)
            if response.status_code != 429 or attempt == self.policy.attempts:
                return response
            delay = min(self.policy.maximum_delay_seconds,
                        self.policy.initial_delay_seconds * 2 ** (attempt - 1))
            delay = min(self.policy.maximum_delay_seconds,
                        delay * self.jitter(1 - self.policy.jitter_fraction,
                                            1 + self.policy.jitter_fraction))
            hint = response.headers.get("Retry-After")
            if hint:
                try:
                    minimum = float(hint)
                except ValueError:
                    try:
                        minimum = parsedate_to_datetime(hint).timestamp() - self.clock()
                    except (ValueError, TypeError, OverflowError):
                        minimum = 0
                if math.isfinite(minimum):
                    if minimum > self.policy.maximum_delay_seconds:
                        # Do not violate a server's longer cooldown or wait
                        # unboundedly. Return the rejection to normal recovery.
                        return response
                    delay = max(delay, minimum)
            response.close()
            deadline = self.clock() + delay
            self.record({
                "event": "vertex_rate_limit_backoff", "request_id": request_id,
                "request_sha256": identity, "operation": request.url.path.rsplit(":", 1)[-1],
                "http_status": 429, "after_attempt": attempt,
                "maximum_attempts": self.policy.attempts, "delay_seconds": delay,
                "retry_not_before": deadline,
            })
            while (remaining := deadline - self.clock()) > 0:
                self.sleep(min(remaining, 55))

    def close(self):
        self.inner.close()
