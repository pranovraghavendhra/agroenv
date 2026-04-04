# Lazy imports — do not eagerly import env.py (requires pydantic at server start)
# The full server is initialized by FastAPI startup, not import time.
