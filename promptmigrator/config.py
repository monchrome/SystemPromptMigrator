"""Service configuration, read from the environment at import time."""

import os

# Upper bound on the uploaded prompt file (decoded as UTF-8 text).
MAX_PROMPT_BYTES = int(os.getenv("PM_MAX_PROMPT_BYTES", str(1024 * 1024)))

# How many candidate rewrites the proposer generates per migration.
DEFAULT_NUM_CANDIDATES = int(os.getenv("PM_NUM_CANDIDATES", "3"))
MAX_NUM_CANDIDATES = 5

# max_tokens for every model call in the pipeline.
MAX_OUTPUT_TOKENS = int(os.getenv("PM_MAX_OUTPUT_TOKENS", "16000"))
