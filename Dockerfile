# AgroEnv Dockerfile
# Optimised for HuggingFace Spaces (2 vCPU, 8GB RAM)
# Multi-stage build keeps final image lean

FROM python:3.11-slim AS base

# System-level hardening
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install only what we need (no build tools needed — pure Python)
COPY server/requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy server source
COPY server/ ./server/

# Copy data files
COPY server/data/ ./server/data/

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Non-root user for security
RUN useradd -m -u 1000 agroenv
USER agroenv

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Start server
CMD ["python", "-m", "uvicorn", "server.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
