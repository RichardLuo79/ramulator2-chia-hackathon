"""Conservative public-log redaction; this does not certify redistribution rights.

Native state and authentication stores are never inputs to this exporter. Each
derived file is linked to its original hash by the operator's export inventory.
"""
from collections import Counter
import re

POLICY = 'public-evidence-redaction-v1'
PRIVATE_KEYS = frozenset({
    'access_token', 'refresh_token', 'api_key', 'authorization', 'cookie',
    'client_secret', 'credentials', 'session_state_paths', 'private_state',
    'account_id', 'email', 'project_id', 'subscription_id', 'organization_id',
})
PATTERNS = (
    (r'(?i)\bBearer\s+[A-Za-z0-9_.+/=-]+', '[redacted-bearer]'),
    (r'\b(?:sk-ant-[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]{20,}|ya29\.[A-Za-z0-9_.-]+)', '[redacted-token]'),
    (r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b', '[redacted-token]'),
    (r'-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?-----END [^-]*PRIVATE KEY-----', '[redacted-private-key]'),
    (r'https?://www\.googleapis\.com/compute/[^\s\"\'<>`,;)\]}]+', '[private-compute-resource]'),
    (r'/home/(?!runner/|user/)[^\s\"\'<>`,;:)\]}]+', '[private-path]'),
    (r'/tmp/(?:chia|ramulator|pytest)[^\s\"\'<>`,;:)\]}]*', '[private-temporary-path]'),
    (r'gs://[^\s\"\'<>`,;)\]}]+', '[private-archive-location]'),
    (r'\ba3-[A-Za-z0-9_-]+\b', '[private-project]'),
    (r'\b[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b', '[private-account]'),
    (r'\b(?:34|35)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[private-ip]'),
    (r'\bchia-(?:staged|upstream|dram-speed|lat-tp|opus55|gemini|astra|deepseek)[A-Za-z0-9_-]*-2026[0-9A-Za-z_.-]*', '[private-host-label]'),
)


def redact_text(text, counts=None):
    counts = counts if counts is not None else Counter()
    for pattern, replacement in PATTERNS:
        text, n = re.subn(pattern, replacement, text)
        counts[replacement] += n
    return text


def redact(value, counts=None):
    counts = counts if counts is not None else Counter()
    if isinstance(value, str):
        return redact_text(value, counts)
    if isinstance(value, list):
        return [redact(x, counts) for x in value]
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if key.lower() in PRIVATE_KEYS:
                counts['private-field'] += 1
                output[key] = '[redacted-private-field]'
            else:
                output[redact_text(key, counts)] = redact(item, counts)
        return output
    return value
