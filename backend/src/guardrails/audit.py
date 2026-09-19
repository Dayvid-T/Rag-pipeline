"""
Structured audit log for guardrail decisions. Every block or filter is
logged as one JSON line with the rule that fired, so a decision can be
reviewed after the fact instead of just disappearing into a 200 response.

Logs to stdout via the standard logging module rather than a file - that's
what a container's log driver (and App Runner -> CloudWatch) already
captures, so there's no volume to manage.
"""

import json
import logging
import time
from typing import Any, Dict

logger = logging.getLogger("guardrails.audit")
logger.setLevel(logging.INFO)

# Attached directly to this logger rather than relying on root-logger
# config, which turned out to behave differently under `uvicorn --reload`
# (its subprocess reconfigures logging in a way that swallowed INFO
# records reaching root even though a standalone `uvicorn` process was
# fine) - an audit trail shouldn't depend on getting that order right.
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


def log_event(event: str, **fields: Any) -> Dict[str, Any]:
    """Log a structured guardrail event and return the record (handy for tests)."""
    record = {"event": event, "ts": time.time(), **fields}
    logger.info(json.dumps(record, default=str))
    return record
