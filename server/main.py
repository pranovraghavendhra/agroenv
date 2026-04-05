from __future__ import annotations
"""
AgroEnv FastAPI Server
=======================
OpenEnv-compliant HTTP API for the AgroEnv Precision Agriculture environment.

Endpoints:
  POST /reset        — Start a new episode
  POST /step         — Take an action, advance one day
  GET  /state        — Get current episode state (lightweight)
  POST /grade        — Grade current/completed episode
  GET  /tasks        — List available tasks
  GET  /health       — Health check
  GET  /docs         — Auto-generated API docs (FastAPI default)

Session management:
  Each HTTP session maintains one AgroEnv instance.
  Multiple concurrent sessions are supported via per-session state.
  Sessions are identified by a cookie-based session_id.
"""

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import time
import logging
from typing import Optional

from .env import AgroEnv
from .models import (
    AgroAction, ResetRequest, ResetResponse, StepResult,
    StateResponse, ErrorResponse,
)
from .config import API_VERSION, ENV_NAME, ENV_DISPLAY_NAME, TASK_DEFAULTS, MAX_STEPS
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("agroenv")

# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=ENV_DISPLAY_NAME,
    description=(
        "An OpenEnv-compliant Precision Agriculture environment simulating "
        "real Indian farming decisions using FAO-56 crop models, ICAR IPM standards, "
        "and AGMARKNET-calibrated market prices."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Session Store ───────────────────────────────────────────────────────────
# In-memory session store: session_id → AgroEnv instance
# For production HF Spaces: this is fine for single-worker deployment.
_sessions: dict[str, dict] = {}
SESSION_COOKIE = "agroenv_session"
SESSION_TTL_SECONDS = 3600  # 1 hour


def _get_or_create_session(request: Request, response: Response) -> tuple[str, AgroEnv]:
    """Get existing session or create a new one."""
    session_id = request.cookies.get(SESSION_COOKIE)

    # Clean up stale sessions
    now = time.time()
    stale = [k for k, v in _sessions.items() if now - v["last_active"] > SESSION_TTL_SECONDS]
    for k in stale:
        del _sessions[k]
        logger.info(f"Cleaned stale session {k}")

    if session_id and session_id in _sessions:
        _sessions[session_id]["last_active"] = now
        return session_id, _sessions[session_id]["env"]

    # New session
    session_id = str(uuid.uuid4())
    env = AgroEnv()
    _sessions[session_id] = {"env": env, "last_active": now}
    response.set_cookie(SESSION_COOKIE, session_id, max_age=SESSION_TTL_SECONDS, httponly=True)
    logger.info(f"New session: {session_id}")
    return session_id, env


def _get_session(request: Request) -> Optional[AgroEnv]:
    """Get existing session env or None."""
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id and session_id in _sessions:
        _sessions[session_id]["last_active"] = time.time()
        return _sessions[session_id]["env"]
    return None

@app.get("/", response_class=HTMLResponse)
async def frontend():
    with open("/app/index.html", "r", encoding="utf-8") as f:
        return f.read()


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — returns 200 if server is alive."""
    return {
        "status": "ok",
        "env": ENV_NAME,
        "version": API_VERSION,
        "active_sessions": len(_sessions),
    }


@app.get("/tasks")
async def list_tasks():
    """List all available tasks with descriptions and defaults."""
    return {
        "tasks": [
            {
                "name": "irrigation_scheduling",
                "difficulty": "easy",
                "max_steps": MAX_STEPS["irrigation_scheduling"],
                "pass_threshold": 0.65,
                "crop": TASK_DEFAULTS["irrigation_scheduling"]["crop"],
                "region": TASK_DEFAULTS["irrigation_scheduling"]["region"],
                "description": (
                    "Manage rice irrigation for 14 days during tillering stage. "
                    "Maintain optimal soil moisture using FAO-56 ET₀ guidance."
                ),
            },
            {
                "name": "pest_management",
                "difficulty": "medium",
                "max_steps": MAX_STEPS["pest_management"],
                "pass_threshold": 0.60,
                "crop": TASK_DEFAULTS["pest_management"]["crop"],
                "region": TASK_DEFAULTS["pest_management"]["region"],
                "description": (
                    "Manage cotton pest pressure (whitefly + bollworm) for 30 days "
                    "using ICAR IPM thresholds. Balance control with resistance management."
                ),
            },
            {
                "name": "season_optimizer",
                "difficulty": "hard",
                "max_steps": MAX_STEPS["season_optimizer"],
                "pass_threshold": 0.55,
                "crop": TASK_DEFAULTS["season_optimizer"]["crop"],
                "region": TASK_DEFAULTS["season_optimizer"]["region"],
                "description": (
                    "Manage complete 110-day tomato season in Andhra Pradesh. "
                    "Optimize irrigation + pests + harvest timing to maximize net revenue."
                ),
            },
        ]
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(request: Request, response: Response, body: ResetRequest = None):
    """
    Start a new episode.
    Call this before the first step() of any episode.
    If body is empty, uses default task (irrigation_scheduling).
    """
    if body is None:
        body = ResetRequest()

    session_id, env = _get_or_create_session(request, response)

    try:
        result = env.reset(body)
        logger.info(
            f"Session {session_id} | RESET | task={body.task} "
            f"crop={result.observation.crop.crop_name} | episode={result.episode_id}"
        )
        return result
    except Exception as e:
        logger.error(f"Session {session_id} | RESET ERROR | {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResult)
async def step(request: Request, action: AgroAction):
    """
    Take one action and advance the simulation by one day.
    Must call /reset first to initialize an episode.
    """
    env = _get_session(request)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call POST /reset first.",
        )

    session_id = request.cookies.get(SESSION_COOKIE, "unknown")

    try:
        result = env.step(action)
        logger.info(
            f"Session {session_id} | STEP {env._step_count} "
            f"| reward={result.reward:.3f} done={result.done}"
        )
        return result
    except Exception as e:
        logger.error(f"Session {session_id} | STEP ERROR | {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=StateResponse)
async def state(request: Request):
    """
    Get lightweight current episode state.
    Does NOT advance the simulation.
    """
    env = _get_session(request)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call POST /reset first.",
        )
    return env.state()


@app.post("/grade")
async def grade(request: Request):
    """
    Run the grader on the current episode.
    Can be called mid-episode for partial scores, or after done=True for final score.
    Returns score in [0.0, 1.0] and detailed breakdown.
    """
    env = _get_session(request)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call POST /reset first.",
        )

    session_id = request.cookies.get(SESSION_COOKIE, "unknown")

    try:
        score, breakdown = env.grade()
        logger.info(f"Session {session_id} | GRADE | score={score:.3f}")
        return {
            "score": round(score, 4),
            "passed": breakdown.get("passed", False),
            "breakdown": breakdown,
            "episode_id": env._episode_id,
            "steps_completed": env._step_count,
            "done": env._done,
        }
    except Exception as e:
        logger.error(f"Session {session_id} | GRADE ERROR | {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False, workers=1)
