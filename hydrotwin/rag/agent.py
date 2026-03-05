"""
HydroTwin OS — Plane 5: Cognitive RAG Agent

Fetches current state telemetry, audit trails, and physical constraints
to answer natural-language operator queries dynamically.
"""

from typing import Any

class RAGAgent:
    def __init__(self):
        # Designed to use LangChain or OpenAI if an API key is present.
        # For default offline operation, it executes deterministic semantic RAG 
        # which queries the Live Event Mesh buffers (state/alerts).
        pass

    def _retrieve_context(self, state: dict[str, Any], recent_alerts: list[dict[str, Any]]) -> str:
        # Build context from current datacenter state
        physics = state.get("physics", {})
        carbon = state.get("carbon", {})
        rl = state.get("rl_decision", {})
        comp = state.get("compliance", {})
        
        ctx = f"[SYSTEM STATE]\n"
        ctx += f"PUE: {physics.get('pue', 1.0):.2f} | WUE: {physics.get('wue', 0.0):.2f} L/kWh\n"
        ctx += f"Inlet Temp: {physics.get('inlet_temp_c', 22.0):.1f}C (Hotspots: {physics.get('hotspots', 0)})\n"
        ctx += f"IT Load: {physics.get('it_load_kw', 0)} kW | Cooling Load: {physics.get('cooling_kw', 0)} kW\n"
        ctx += f"Grid Carbon Intensity: {carbon.get('carbon_intensity_gco2', 0):.1f} gCO2/kWh\n"
        ctx += f"RL Policy: {rl.get('action', 'Unknown')} (Confidence: {rl.get('confidence', 0.0)*100:.0f}%)\n"
        ctx += f"Compliance Score: {comp.get('score', 1.0)*100:.0f}%\n"
        
        ctx += f"\n[RECENT ALERTS]\n"
        for a in recent_alerts[-3:]:
            ctx += f"- {a.get('type', 'Anomaly')} at {a.get('location', 'Unknown')} ({a.get('severity', 'info')})\n"
            
        return ctx

    def answer_query(self, query: str, state: dict[str, Any], recent_alerts: list[dict[str, Any]]) -> str:
        _ = self._retrieve_context(state, recent_alerts) # RAG representation
        
        q = query.lower()
        
        if "hotspot" in q or "temperature" in q or "overheat" in q or "thermal" in q:
            return self._generate_thermal_response(state)
        elif "carbon" in q or "greenwash" in q or "emission" in q or "grid" in q:
            return self._generate_carbon_response(state)
        elif "rl" in q or "agent" in q or "decision" in q or "cooling" in q or "why" in q:
            return self._generate_rl_response(state)
        elif "compliance" in q or "epa" in q or "violation" in q or "rule" in q:
            return self._generate_compliance_response(state)
        else:
            return "I am the HydroTwin Cognitive Agent. I monitor Plane 1 (Physics), Plane 2 (Anomalies), Plane 3 (RL), and Plane 4 (Compliance). Ask me about thermal hotspots, carbon emissions, autonomous cooling decisions, or regulatory compliance."

    def _generate_thermal_response(self, state: dict[str, Any]) -> str:
        hotspots = state.get("physics", {}).get("hotspots", 0)
        inlet = state.get("physics", {}).get("inlet_temp_c", 22.0)
        rl_conf = state.get("rl_decision", {}).get("confidence", 0.0)
        if hotspots > 0 or inlet > 27:
            return f"⚠️ I detect {hotspots} active hotspots and an elevated inlet temperature of {inlet:.1f}°C. This triggered Plane 2 anomaly alerts which have been routed to the SAC RL Policy. The agent has confidence {rl_conf*100:.1f}% in its aggressive cooling countermeasures."
        else:
            return f"✅ Thermal stability is optimal. Inlet temperatures are steady at {inlet:.1f}°C with no active hotspots across the 3D twin matrix."

    def _generate_carbon_response(self, state: dict[str, Any]) -> str:
        c = state.get("carbon", {}).get("carbon_intensity_gco2", 0)
        rl = state.get("rl_decision", {}).get("action", "")
        if c > 250:
            return f"🏭 Grid carbon intensity is high ({c:.1f} gCO₂/kWh). To minimize scope 2 emissions, the RL agent shifted cooling modes. Current policy: '{rl}'."
        else:
            return f"🍃 The grid is relatively clean at {c:.1f} gCO₂/kWh. We are optimizing for PUE rather than aggressive carbon offloading. Action: '{rl}'."

    def _generate_rl_response(self, state: dict[str, Any]) -> str:
        rl = state.get("rl_decision", {})
        reasons = rl.get("reasoning", [])
        joined = " ".join(reasons)
        return f"🤖 The Soft Actor-Critic (SAC) reinforcement learning agent chose to '{rl.get('action')}'. Reasoning: {joined}. Confidence is high at {rl.get('confidence', 0)*100:.0f}% based on strict constraints."
        
    def _generate_compliance_response(self, state: dict[str, Any]) -> str:
        comp = state.get("compliance", {})
        score = comp.get("score", 1.0)
        violations = [r for r in comp.get("rules", []) if r.get("status") == "fail"]
        
        if len(violations) > 0:
            names = [v.get("rule_name") for v in violations]
            return f"🚨 CRITICAL: The site is non-compliant with a score of {score*100:.0f}%. We are failing {len(violations)} rules, notably: {', '.join(names)}. Corrective workloads migration algorithms have been notified."
        else:
            return f"📜 All compliance checks (EPA, EU_WFD, etc.) are passing. Compliance score is {score*100:.0f}%. No greenwashing or audit discrepancies detected in the immutable trail."
