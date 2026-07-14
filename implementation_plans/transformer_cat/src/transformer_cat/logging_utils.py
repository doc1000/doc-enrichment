"""
Tracing / logging initialisation.

If LANGCHAIN_TRACING_V2 + LANGCHAIN_API_KEY are both present in the environment,
LangChain's native LangSmith tracing activates automatically (no extra code needed).
Otherwise a standard Python logging config is applied so local runs are not silent.

Call init_tracing() once at startup (also called from __init__.py).
The function is idempotent.
"""
from __future__ import annotations

import logging
import os

_initialised = False
logger = logging.getLogger("transformer_cat")


def init_tracing() -> None:
    """Idempotent setup: LangSmith passthrough or local logging."""
    global _initialised
    if _initialised:
        return
    _initialised = True

    has_tracing = bool(
        os.getenv("LANGCHAIN_TRACING_V2") and os.getenv("LANGCHAIN_API_KEY")
    )

    if has_tracing:
        # LangChain picks up these env vars automatically — just log a confirmation.
        logger.info(
            "LangSmith tracing active — project: %s",
            os.getenv("LANGCHAIN_PROJECT", "(default)"),
        )
    else:
        # Apply a simple console handler so local runs surface INFO+ logs.
        if not logging.root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )
        logger.info("LangSmith not configured — using local Python logging.")
