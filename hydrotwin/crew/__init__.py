"""
HydroTwin OS — Plane 4: CrewAI Multi-Agent Coordination Layer

Orchestrates specialised AI agents for end-to-end decision support:
  • NexusExecutor   — translates SAC RL actions into infrastructure commands
  • AquiferForecaster — predicts groundwater and thermal demand trajectories
  • ComplianceGuardian — audits decisions against regulatory constraints
"""

from hydrotwin.crew.crew import HydroTwinCrew

__all__ = ["HydroTwinCrew"]
