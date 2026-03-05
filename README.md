# HydroTwin OS 🌊⚡🌍
**An AI-Native Operating System for the Data Center Water-Energy-Carbon Nexus**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Docker](https://img.shields.io/badge/docker-compose-blue.svg)](https://www.docker.com/)
[![FAANG-Ready](https://img.shields.io/badge/Architecture-Distributed_Mesh-success)](#)

HydroTwin OS is a distributed infrastructure control plane designed to autonomically optimize hyperscale data centers. Instead of analyzing thermal physics, grid carbon emissions, and water usage effectiveness (WUE) as isolated problems, HydroTwin fuses them into a **Continuous Reinforcement Learning Matrix** operating atop a real-time **Kafka Event Mesh**.

By shifting cooling burdens dynamically between electrical chiller envelopes and adiabatic evaporative systems based on grid-carbon APIs, the OS mathematically guarantees Pareto-optimal sustainability via deep Soft Actor-Critic (SAC) reinforcement learning.

---

## 🚀 The 5-Plane Architecture

This repository is built as a highly available, decoupled 5-plane distributed system:

1. **Plane 1: The Physics Twin (`hydrotwin.physics`)**
   A PyTorch Geometric Graph Neural Network (GNN) that approximates Computational Fluid Dynamics (CFD). It evaluates $10^5$ thermal node states in `184ms`, enabling continuous >5Hz AI interaction.
2. **Plane 2: Multimodal Anomaly Cortex (`hydrotwin.detection`)**
   Sensor fusion via PyTorch Cross-Attention Transformers. Fuses IoT telemetry (LSTM Autoencoders), Acoustic FFT vibration data, and Computer Vision streaming pipelines.
3. **Plane 3: RL Decision Matrix (`hydrotwin.agent`)**
   The autonomous core. A Deep Soft Actor-Critic algorithm dynamically interacting with the `DataCenterEnv`. Connects natively to live `ElectricityMaps` APIs to shift optimization policies between Water constraints ($\alpha$) and Carbon limits ($\gamma$).
4. **Plane 4: Regulatory Brain (`hydrotwin.compliance`)**
   Intercepts all Kafka mesh traffic, applying global Zero-Trust logic. Provides cryptography-secured audit trails and checks real-time metrics against EU Energy Efficiency Directives.
5. **Plane 5: Generative Semantic UI (`hydrotwin.rag` & `hydrotwin.dashboard`)**
   A conversational NLP layer connecting the system state to operators. Driven by a localized RAG context bridge over a real-time FastAPI mesh.

---

## 📊 Scientific Benchmarks & Validation
In a 1,000-hour Monte Carlo sandbox test against isolated generic heuristic models (`w=0`), **the HydroTwin RL Agent reduced total Scope 2 equivalent carbon by 15.3% while simultaneously eliminating all thermal SLA violations.**

> Read the full scientific breakdown in [`benchmarks/rl_training_results.md`](benchmarks/rl_training_results.md).
> System architecture documentation: [`docs/design_document.md`](docs/design_document.md)

---

## 🔥 Quickstart (Docker Native)

HydroTwin OS is built to run natively via Docker Compose, utilizing Confluent KRaft (Kafka without Zookeeper), InfluxDB, and Prometheus-Grafana telemetry arrays.

1. **Clone & Configure**
   ```bash
   git clone https://github.com/yourusername/hydrotwin-os.git
   cd hydrotwin-os
   cp .env.example .env
   # Add your external API keys (ElectricityMaps, NOAA)
   ```

2. **Boot the Infrastructure Mesh**
   ```bash
   docker compose up -d
   ```

3. **Ignite the Physics Twin and RL Event Cycle**
   ```bash
   python scripts/run_live_datacenter.py
   ```

4. **Launch the Anomaly Co-Processor**
   ```bash
   python scripts/run_anomaly_node.py
   ```

5. **Interact via the RAG Dashboard**
   ```bash
   python -m uvicorn hydrotwin.dashboard.server:app --host 0.0.0.0 --port 8000
   ```
   *Navigate to `http://localhost:8000` to converse with the facility.*

---

## 📂 Repository Structure

```text
HydroTwin-OS/
├── architecture/         # System diagrams and topology maps
├── benchmarks/           # Monte Carlo RL evaluation matrix and Pareto Front proofs
├── demo/                 # Video demos and written end-to-end walkthroughs
├── docs/                 # Formal architecture Whitepaper and Design Documents
├── hydrotwin/            # Core OS Layer
│   ├── agent/            # Plane 3: Soft Actor-Critic RL Engine
│   ├── api_clients/      # External Data Fetchers (ElectricityMaps, NOAA, WRI)
│   ├── compliance/       # Plane 4: Immutable Regulatory & Audit Engines
│   ├── dashboard/        # Plane 5: FastAPI REST Node & Generative UI
│   ├── detection/        # Plane 2: Multimodal Anomaly Transformers
│   ├── env/              # Plane 3: Custom Gym Environment Contexts
│   ├── events/           # Plane 3: Kafka Producer/Consumer abstractions
│   ├── physics/          # Plane 1: PyTorch Geometric Digital Twin
│   ├── rag/              # Plane 5: Semantic Response Agents
│   ├── reward/           # Plane 3: Multi-Objective Pareto Reward constraints
│   └── main.py           # The Orchestrator
├── scripts/              # Local runners and Monte Carlo RL benchmark scripts
├── tests/                # 10-Level FAA-Grade Stress Test Suite (Pytest)
├── docker-compose.yml    # HA FAANG-Grade Infrastructure definitions
└── LICENSE               # Apache 2.0 Open Source Licensing
```

---

## 🤝 Research & Contributions
HydroTwin relies heavily on open-source Deep Learning and IoT foundations. This framework is highly suitable to serve as an independent **SaaS platform foundation**, a core element of **Climate Tech Research Portfolios**, or a baseline for distributed **FAANG System Interview Architectures**.

If you're integrating this into your Data Center, researching Physical Neural Networks, or mapping AI optimization for Carbon offsets, pull requests and issues are strictly welcome.

*Licensed under the Apache 2.0 License.*
