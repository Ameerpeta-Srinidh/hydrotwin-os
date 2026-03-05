"""
HydroTwin OS — Plane 5: Dashboard API Server

FastAPI backend serving real-time data from all 4 planes
to the public transparency dashboard.
"""

from __future__ import annotations

import random
import math
import time
from datetime import datetime, timedelta
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import threading

from prometheus_client import make_asgi_app, Summary, Counter, Gauge
from hydrotwin.inference.rl_service import RLInferenceService
from hydrotwin.events.kafka_consumer import NexusKafkaConsumer
from hydrotwin.events.kafka_producer import NexusKafkaProducer
from hydrotwin.events.schemas import RLAction

app = FastAPI(title="HydroTwin OS — Dashboard API", version="1.0.0")

# ─── Observability (Prometheus) ───
app.mount("/metrics", make_asgi_app())
rl_latency = Summary('rl_inference_latency_seconds', 'Time spent in RL inference')
kafka_lag = Gauge('kafka_consumer_lag_ms', 'Lag between message generation and inference')
temp_violations = Counter('temp_violations_total', 'Count of inlet temperature violations')
carbon_savings = Counter('carbon_savings_gco2_total', 'Estimated carbon grams saved')

# ─── Inference Module ───
rl_service = RLInferenceService()
kafka_producer = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


# ─── Simulation State ───
class LiveSimulation:
    """Generates realistic, time-varying data center metrics."""

    def __init__(self):
        self.start_time = time.time()
        self.step = 0
        self.alerts: list[dict] = []
        self.audit_log: list[dict] = []
        self.rl_decisions: list[dict] = []

    def tick(self) -> dict:
        self.step += 1
        t = time.time() - self.start_time
        hour = (datetime.now().hour + t / 60) % 24

        # Diurnal patterns
        ambient = 28 + 8 * math.sin((hour - 14) * math.pi / 12) + random.gauss(0, 0.5)
        it_load = 5000 + 2000 * math.sin((hour - 10) * math.pi / 12) + random.gauss(0, 100)
        it_load = max(3000, min(9000, it_load))

        # Cooling response
        cooling_mix = 0.3 + 0.4 * max(0, (ambient - 25) / 20)
        cooling_kw = it_load * 0.25 * (1 + cooling_mix * 0.5)
        inlet = 22 + (it_load - 5000) / 3000 + random.gauss(0, 0.3)
        inlet = max(15, min(35, inlet))

        # Metrics
        pue = 1.0 + cooling_kw / max(it_load, 1)
        wue = 1.8 * (1 - cooling_mix) * max(0.2, (ambient - 15) / 25)
        carbon_intensity = 150 + 100 * math.sin((hour - 6) * math.pi / 12)
        water_liters = wue * it_load / 60

        return {
            "timestamp": datetime.now().isoformat(),
            "step": self.step,
            "physics": {
                "pue": round(pue, 3),
                "wue": round(max(0, wue), 3),
                "inlet_temp_c": round(inlet, 1),
                "ambient_temp_c": round(ambient, 1),
                "it_load_kw": round(it_load, 0),
                "cooling_kw": round(cooling_kw, 0),
                "hotspots": max(0, int((inlet - 27) * 2)) if inlet > 27 else 0,
                "num_racks": 64,
                "num_crahs": 8,
            },
            "carbon": {
                "cooling_mix": round(cooling_mix, 3),
                "carbon_intensity_gco2": round(carbon_intensity, 1),
                "carbon_kg_hour": round(carbon_intensity * (it_load + cooling_kw) / 1000, 1),
                "water_liters_hour": round(water_liters * 60, 0),
                "thermal_satisfaction": round(max(0, 1 - abs(inlet - 23) / 10), 3),
                "reward": round(-0.4 * wue - 0.2 * pue - 0.3 * carbon_intensity / 500 + 0.1 * max(0, 1 - abs(inlet - 23) / 10), 3),
            },
            "compliance": self._compliance(pue, wue, inlet),
        }

    def _compliance(self, pue, wue, inlet):
        rules = [
            {"rule": "EPA PUE Limit", "limit": 1.5, "actual": pue, "unit": "ratio",
             "status": "pass" if pue < 1.35 else ("warn" if pue < 1.5 else "fail")},
            {"rule": "EU WUE Target", "limit": 1.8, "actual": round(wue, 2), "unit": "L/kWh",
             "status": "pass" if wue < 1.44 else ("warn" if wue < 1.8 else "fail")},
            {"rule": "ASHRAE A1 Temp", "limit": "18-27°C", "actual": round(inlet, 1), "unit": "°C",
             "status": "pass" if 19 < inlet < 26 else ("warn" if 18 < inlet < 27 else "fail")},
            {"rule": "Daily Water Cap", "limit": 500000, "actual": round(wue * 5000 * 24, 0), "unit": "L/day",
             "status": "pass"},
        ]
        passed = sum(1 for r in rules if r["status"] == "pass")
        warned = sum(1 for r in rules if r["status"] == "warn")
        score = (passed + 0.5 * warned) / len(rules)
        return {"score": round(score, 3), "rules": rules}

    def generate_alert(self) -> dict | None:
        if random.random() < 0.08:
            types = [
                ("hotspot", "warning", "Rack-R3C5"),
                ("leak", "critical", "Pipe-CW-12"),
                ("vibration_fault", "warning", "CRAH-04"),
                ("flow_deviation", "info", "Loop-B"),
            ]
            t, s, l = random.choice(types)
            alert = {
                "id": f"ALT-{len(self.alerts)+1:04d}",
                "timestamp": datetime.now().isoformat(),
                "type": t, "severity": s, "location": l,
                "confidence": round(random.uniform(0.7, 0.98), 2),
            }
            self.alerts.append(alert)
            return alert
        return None

    def generate_rl_explanation(self, metrics: dict) -> dict:
        cooling_mix = metrics["carbon"]["cooling_mix"]
        ambient = metrics["physics"]["ambient_temp_c"]
        carbon = metrics["carbon"]["carbon_intensity_gco2"]

        reasons = []
        if ambient > 35:
            reasons.append(f"High ambient ({ambient:.1f}°C) — increasing chiller fraction")
        elif ambient < 20:
            reasons.append(f"Cool ambient ({ambient:.1f}°C) — enabling free cooling via economizer")
        if carbon > 300:
            reasons.append(f"Dirty grid ({carbon:.0f} gCO₂/kWh) — shifting to evaporative cooling")
        elif carbon < 150:
            reasons.append(f"Clean grid ({carbon:.0f} gCO₂/kWh) — mechanical cooling acceptable")
        if not reasons:
            reasons.append("Balanced operating conditions — maintaining optimal mix")

        mode = "mechanical chiller" if cooling_mix > 0.6 else ("evaporative" if cooling_mix < 0.3 else "balanced")
        return {
            "action": f"Set cooling to {mode} (mix={cooling_mix:.0%})",
            "reasoning": reasons,
            "reward": metrics["carbon"]["reward"],
            "confidence": round(random.uniform(0.75, 0.95), 2),
        }


# ─── Live State (Populated by Kafka) ───
CURRENT_STATE = {
    "timestamp": datetime.now().isoformat(),
    "step": 0,
    "physics": {"pue": 1.0, "wue": 0.0, "inlet_temp_c": 22.0, "ambient_temp_c": 25.0, "it_load_kw": 5000, "cooling_kw": 1000, "hotspots": 0},
    "carbon": {"cooling_mix": 0.5, "carbon_intensity_gco2": 200.0, "reward": 0.0},
    "compliance": {"score": 1.0, "rules": []},
    "latest_alert": None,
    "rl_decision": None
}

RECENT_ALERTS = []


@app.get("/api/metrics")
def get_metrics():
    """Serves real-time metrics fetched from the Event Mesh"""
    return CURRENT_STATE


@app.get("/api/alerts")
def get_alerts():
    return {"alerts": RECENT_ALERTS[-20:]}


@app.get("/api/audit")
def get_audit():
    return {"entries": []}


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Dashboard loading...</h1>")


from pydantic import BaseModel
class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    from hydrotwin.rag.agent import RAGAgent
    agent = RAGAgent()
    response = agent.answer_query(request.query, CURRENT_STATE, RECENT_ALERTS)
    return {"response": response}


@app.post("/api/v1/predict")
@rl_latency.time()
def predict_action(state: dict):
    """Direct synchronous REST endpoint for the RL engine."""
    if state.get("inlet_temp_c", 22.0) > 27.0:
        temp_violations.inc()
    return rl_service.predict(state)


def rl_kafka_loop():
    """Background task: Sensor -> Kafka -> RL -> Action -> Kafka"""
    time.sleep(10) # Wait for Kafka broker
    try:
        consumer = NexusKafkaConsumer(group_id="hydrotwin-rl-engine")
        consumer.subscribe(["hydrotwin.telemetry", "anomaly.alerts"])
        global kafka_producer
        kafka_producer = NexusKafkaProducer(mock_mode=False)
        
        def handle_message(topic, value):
            global CURRENT_STATE
            global RECENT_ALERTS
            
            # --- Anomaly Alerts ---
            if topic == "anomaly.alerts":
                alert_dict = {
                    "id": value.get("alert_id", "ALT-0000"),
                    "timestamp": value.get("timestamp"),
                    "type": value.get("anomaly_type"),
                    "severity": value.get("severity"),
                    "location": value.get("location"),
                    "confidence": value.get("confidence", 0.0),
                    "message": value.get("details", {}).get("message", "Anomaly Detected")
                }
                CURRENT_STATE["latest_alert"] = alert_dict
                RECENT_ALERTS.append(alert_dict)
                if len(RECENT_ALERTS) > 20:
                    RECENT_ALERTS.pop(0)
                return

            # --- Telemetry (Physics/RL) ---
            with rl_latency.time():
                action_data = rl_service.predict(value)
            
            inlet = value.get("inlet_temp_c", 22.0)
            if inlet > 27.0:
                temp_violations.inc()
                
            # Populate Global Context for Dashboard
            CURRENT_STATE["physics"].update({
                "pue": value.get("pue", 1.0),
                "wue": value.get("wue", 0.0),
                "inlet_temp_c": inlet,
                "ambient_temp_c": value.get("ambient_temp_c", 25.0),
                "it_load_kw": value.get("it_load_kw", 5000),
            })
            CURRENT_STATE["carbon"].update({
                "cooling_mix": action_data["cooling_mix"],
                "carbon_intensity_gco2": value.get("carbon_intensity", 200.0),
            })
            CURRENT_STATE["rl_decision"] = {
                "action": f"Set cooling mix to {action_data['cooling_mix']:.0%}",
                "reasoning": ["Real-time SAC Policy output via Kafka Event Mesh"],
                "confidence": 0.95
            }
                
            # Form action payload
            action_event = RLAction(
                cooling_mode_mix=action_data["cooling_mix"],
                supply_air_temp_setpoint=action_data["supply_air_temp_sp"],
                fan_speed_pct=action_data["fan_speed_pct"],
                economizer_damper=action_data["economizer_damper"],
                inlet_temp_c=inlet,
                ambient_temp_c=value.get("ambient_temp_c", 25.0),
                it_load_kw=value.get("it_load_kw", 5000.0),
                grid_carbon_intensity=value.get("carbon_intensity", 200.0),
                water_stress_index=value.get("water_stress_index", 1.0),
                wue=value.get("wue", 0.0),
                pue=value.get("pue", 1.0),
                carbon_intensity=value.get("carbon_intensity", 200.0),
                thermal_satisfaction=1.0, # Approximate or fetch from telemetry
                reward=value.get("reward", 0.0),
                reward_weights={"alpha": 0.5, "beta": 0.5, "gamma": 0.0, "delta": 0.0},
            )
            # Send to Mesh
            kafka_producer.publish_action(action_event)
            kafka_producer._producer.flush()
            
        consumer.consume_loop(handle_message)
    except Exception as e:
        print(f"Failed to start RL Kafka loop: {e}")

@app.on_event("startup")
def startup_event():
    threading.Thread(target=rl_kafka_loop, daemon=True).start()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
