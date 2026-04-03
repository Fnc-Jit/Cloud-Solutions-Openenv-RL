"""FastAPI application for the CloudFinOps Environment.

Uses a persistent singleton CloudFinOpsEnvironment so that state is preserved
across /reset and /step calls.  OpenEnv's create_app() is intentionally
bypassed here because it creates a fresh environment instance for every request
(stateless factory pattern), which means the state set by /reset is thrown away
before /step runs — resulting in every action returning -2.0 (server not found)
and done=True on step 1.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup (needed for Docker where server/ is not installed as a package)
# ---------------------------------------------------------------------------
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

try:
    from ..models import CloudFinOpsAction, CloudFinOpsObservation
    from .cloudfinops_env_environment import CloudFinOpsEnvironment
except (ImportError, SystemError):
    from models import CloudFinOpsAction, CloudFinOpsObservation
    from server.cloudfinops_env_environment import CloudFinOpsEnvironment

# ---------------------------------------------------------------------------
# Logging — suppress noisy poll endpoints
# ---------------------------------------------------------------------------
class _PollFilter(logging.Filter):
    _SUPPRESSED = ("/state", "/history", "/dashboard", "/health")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in self._SUPPRESSED)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cloudfinops")
logging.getLogger("uvicorn.access").addFilter(_PollFilter())

# ---------------------------------------------------------------------------
# Singleton environment — shared across all HTTP requests
# ---------------------------------------------------------------------------
_env = CloudFinOpsEnvironment()

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequestBody(BaseModel):
    action: Dict[str, Any]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CloudFinOps-Env",
    version="1.0.0",
    description="Cloud Cost-Optimization RL Environment — OpenEnv compatible.",
)


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root ping endpoint for platform health probes."""
    return {
        "status": "ok",
        "env": "cloudfinops-env",
        "endpoints": [
            "/health",
            "/reset",
            "/step",
            "/state",
            "/schema",
            "/dashboard",
            "/history",
        ],
    }


@app.get("/web")
async def web_probe() -> Dict[str, Any]:
    """Compatibility endpoint for external probes that hit /web."""
    return {"status": "ok", "hint": "Use /dashboard for the live UI"}


@app.on_event("startup")
async def on_startup() -> None:
    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ☁️  CloudFinOps-Env  v1.0.0                                ║
    ║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━                               ║
    ║                                                              ║
    ║   Endpoints:                                                 ║
    ║     GET  /health     → Health check                          ║
    ║     POST /reset      → Reset environment for a task          ║
    ║     POST /step       → Advance engine by 1 tick              ║
    ║     GET  /state      → Current observation (read-only)       ║
    ║     GET  /schema     → Action/Observation schemas            ║
    ║     GET  /dashboard  → Live visualization UI                 ║
    ║     GET  /history    → Agent action history (JSON)           ║
    ║                                                              ║
    ║   Tasks: easy, medium, hard, green                           ║
    ║                                                              ║
    ║   ┌──────────────────────────────────────────────────────┐   ║
    ║   │  📊 Live Dashboard: http://localhost:8000/dashboard   │   ║
    ║   └──────────────────────────────────────────────────────┘   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    log.info("Server started — using persistent singleton environment")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "env": "cloudfinops-env"}


# ---------------------------------------------------------------------------
# /schema
# ---------------------------------------------------------------------------
@app.get("/schema")
async def schema() -> Dict[str, Any]:
    return {
        "action": CloudFinOpsAction.model_json_schema(),
        "observation": CloudFinOpsObservation.model_json_schema(),
    }


# ---------------------------------------------------------------------------
# /reset  — POST {"task_id": "easy"}
# ---------------------------------------------------------------------------
@app.post("/reset")
async def reset(req: ResetRequest) -> Dict[str, Any]:
    valid = {"easy", "medium", "hard", "green"}
    if req.task_id not in valid:
        raise HTTPException(status_code=422, detail=f"task_id must be one of {sorted(valid)}")

    obs = _env.reset(task_id=req.task_id)
    log.info("Reset → task=%s", req.task_id)
    return {
        "observation": obs.model_dump(),
        "reward": None,
        "done": False,
    }


# ---------------------------------------------------------------------------
# /step  — POST {"action": {"command": "...", "target_id": "...", "reply": ""}}
# ---------------------------------------------------------------------------
@app.post("/step")
async def step(req: StepRequestBody) -> Dict[str, Any]:
    try:
        action = CloudFinOpsAction(**req.action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action: {exc}")

    obs = _env.step(action)
    log.info(
        "Step %d | cmd=%s target=%s | reward=%+.2f done=%s",
        obs.time_step,
        action.command,
        action.target_id,
        obs.reward,
        obs.done,
    )

    result: Dict[str, Any] = {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "info": obs.metadata or {},
    }
    return result


# ---------------------------------------------------------------------------
# /state — GET (read-only, no side effects)
# ---------------------------------------------------------------------------
@app.get("/state")
async def state() -> Dict[str, Any]:
    """Return flat observation — dashboard JS reads obs.servers etc. directly."""
    obs = _env.engine.get_state()
    return obs.model_dump()


# ---------------------------------------------------------------------------
# /dashboard — serve the glassmorphism HTML UI
# ---------------------------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


# ---------------------------------------------------------------------------
# /history — action history for the current episode
# ---------------------------------------------------------------------------
@app.get("/history")
async def history() -> List[Dict[str, Any]]:
    return list(_env.action_history)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
