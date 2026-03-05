"""
HydroTwin OS — Plane 3: Operator Dashboard API

FastAPI application providing real-time access to the Nexus agent's
state, actions, Pareto frontier, and operator controls.

Endpoints:
    GET  /api/agent/status    — Current agent state, last action, live metrics
    GET  /api/agent/pareto    — Pareto frontier data for visualization
    POST /api/agent/weights   — Update operator reward weights (α, β, γ, δ)
    GET  /api/metrics/history — Historical WUE, PUE, carbon time series
    WS   /api/agent/stream    — WebSocket for real-time action streaming
    GET  /api/health          — Health check
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ─────────────────────── Request/Response Models ───────────────────────

class WeightsUpdate(BaseModel):
    alpha: float
    beta: float
    gamma: float
    delta: float


class AgentStatus(BaseModel):
    status: str  # "running", "idle", "training"
    last_action: dict[str, Any] | None = None
    current_metrics: dict[str, float] = {}
    active_weights: dict[str, float] = {}
    scenario: str = ""
    uptime_seconds: float = 0.0
    total_actions: int = 0


class ParetoPoint(BaseModel):
    wue: float
    pue: float
    carbon_intensity: float
    reward: float
    weights: dict[str, float]


# ─────────────────────── Application State ───────────────────────

class DashboardState:
    """Shared mutable state for the dashboard API."""

    def __init__(self):
        self.status: str = "idle"
        self.last_action: dict[str, Any] | None = None
        self.current_metrics: dict[str, float] = {}
        self.active_weights: dict[str, float] = {"alpha": 0.4, "beta": 0.2, "gamma": 0.3, "delta": 0.1}
        self.history: list[dict[str, Any]] = []
        self.pareto_frontier: list[dict[str, float]] = []
        self.total_actions: int = 0
        self.start_time: datetime = datetime.utcnow()
        self._websockets: list[WebSocket] = []

    def update(self, action: dict[str, Any], metrics: dict[str, float], weights: dict[str, float]):
        """Record a new action and its metrics."""
        self.last_action = action
        self.current_metrics = metrics
        self.active_weights = weights
        self.total_actions += 1
        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "metrics": metrics,
            "weights": weights,
        })
        # Keep last 10000 records
        if len(self.history) > 10000:
            self.history = self.history[-10000:]

    async def broadcast(self, data: dict):
        """Broadcast data to all connected WebSocket clients."""
        disconnected = []
        for ws in self._websockets:
            try:
                await ws.send_json(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self._websockets.remove(ws)


# ─────────────────────── FastAPI Application ───────────────────────

def create_dashboard_app(
    state: DashboardState | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI dashboard application."""

    app = FastAPI(
        title="HydroTwin OS — Carbon Nexus Dashboard",
        description="Real-time monitoring and control of the Water-Energy-Carbon Nexus Agent",
        version="0.1.0",
    )

    # CORS
    origins = cors_origins or ["http://localhost:3000", "http://localhost:8501"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared state
    _state = state or DashboardState()
    app.state.dashboard = _state

    # ── Routes ──

    @app.get("/api/health")
    async def health():
        return {"status": "healthy", "service": "hydrotwin-plane3", "timestamp": datetime.utcnow().isoformat()}

    @app.get("/api/agent/status", response_model=AgentStatus)
    async def get_agent_status():
        s = app.state.dashboard
        elapsed = (datetime.utcnow() - s.start_time).total_seconds()
        return AgentStatus(
            status=s.status,
            last_action=s.last_action,
            current_metrics=s.current_metrics,
            active_weights=s.active_weights,
            scenario="",
            uptime_seconds=elapsed,
            total_actions=s.total_actions,
        )

    @app.get("/api/agent/pareto")
    async def get_pareto_frontier():
        """Return Pareto frontier points for visualization."""
        s = app.state.dashboard
        if not s.history:
            return {"frontier": [], "total_points": 0}

        # Extract Pareto-relevant data from history
        points = []
        for record in s.history[-1000:]:  # last 1000 points
            m = record.get("metrics", {})
            points.append({
                "wue": m.get("wue", 0),
                "pue": m.get("pue", 1),
                "carbon_intensity": m.get("carbon_intensity", 0),
                "thermal_satisfaction": m.get("thermal_satisfaction", 0),
                "timestamp": record.get("timestamp"),
            })

        return {"frontier": points, "total_points": len(points)}

    @app.post("/api/agent/weights")
    async def update_weights(update: WeightsUpdate):
        """Update the operator's reward weight preferences."""
        s = app.state.dashboard
        s.active_weights = update.model_dump()
        logger.info(f"Operator updated weights: {s.active_weights}")
        return {"status": "updated", "weights": s.active_weights}

    @app.get("/api/metrics/history")
    async def get_metrics_history(limit: int = 500):
        """Return historical metrics time series."""
        s = app.state.dashboard
        history = s.history[-limit:]
        return {
            "data": [
                {
                    "timestamp": r.get("timestamp"),
                    "wue": r.get("metrics", {}).get("wue"),
                    "pue": r.get("metrics", {}).get("pue"),
                    "carbon_intensity": r.get("metrics", {}).get("carbon_intensity"),
                    "thermal_satisfaction": r.get("metrics", {}).get("thermal_satisfaction"),
                    "inlet_temp_c": r.get("metrics", {}).get("inlet_temp_c"),
                }
                for r in history
            ],
            "total_records": len(history),
        }

    @app.websocket("/api/agent/stream")
    async def websocket_stream(websocket: WebSocket):
        """WebSocket for real-time action streaming."""
        await websocket.accept()
        s = app.state.dashboard
        s._websockets.append(websocket)
        logger.info("WebSocket client connected.")

        try:
            while True:
                # Keep connection alive, listen for pings
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            if websocket in s._websockets:
                s._websockets.remove(websocket)
            logger.info("WebSocket client disconnected.")

    return app


# Convenience: create default app instance
app = create_dashboard_app()
