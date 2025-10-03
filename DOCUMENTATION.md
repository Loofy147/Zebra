# Zebra Project: Comprehensive Documentation

This document contains the complete project documentation, synthesizing the roadmap, best practices, and key learnings from the implementation journey. It serves as a full project summary and a guide for future development.

---

## 1. Project Overview & Vision

Zebra is a prototype implementation of a self-improving, multi-ecosystem AI architecture. Its purpose is to autonomously observe, analyze, and evolve its own software components to achieve high-level strategic goals. It is built on a foundation of mathematical guarantees for stability, rational decision-making, and structural integrity.

The core principle is a gradual, multi-phase evolution where each stage builds trust and capability upon the last: **start by observing, then suggesting, then acting under supervision, and only then achieving full autonomy.**

---

## 2. The Phased Implementation Roadmap (Current Status)

Zebra's development follows a four-phase maturity model. The system has been significantly upgraded with a professional deep learning stack, accelerating its progress.

*   #### **Phase 0: The Substrate - Universal Observability**
    *   **Goal:** Give the system perfect senses and a perfect memory.
    *   **Status:** **Completed ✅**
    *   **Artifacts:** A fully observable `sample-service` and a robust OpenTelemetry pipeline.

*   #### **Phase 1: The Professional Observer - From Data to Actionable Insight**
    *   **Goal:** Build the system's "brain" with state-of-the-art deep learning. The system can now analyze, suggest, and generate production-quality code optimizations.
    *   **Status:** **Completed & Enhanced ✅**
    *   **Artifacts:** A `zebra-observer` service powered by a deep learning pipeline, including modules for anomaly detection, causal inference, reinforcement learning, and code generation.

*   #### **Phase 2: The Apprentice - Supervised Actuation**
    *   **Goal:** Connect the brain to the hands, with a strict human approval gate for every action.
    *   **Status:** **In Progress 🟡**
    *   **Artifacts:** The system now automatically generates Pull Requests (PRs) with suggested code changes, which serves as the "human approval gate." A persistent `Supabase` backend stores all actions and outcomes for review.

*   #### **Phase 3: The Practitioner - Bounded Autonomy**
    *   **Goal:** Remove the human safety gate for specific, well-understood classes of changes.
    *   **Status:** Not Started ⚪

---

## 3. System Architecture & Core Components

The Zebra system has been upgraded to a professional-grade, deep learning-native architecture. The `zebra-observer` service now orchestrates a sophisticated pipeline to transform telemetry data into intelligent, autonomous actions.

### 3.1. Enhanced Pipeline Overview
The end-to-end pipeline is designed for robust, continuous self-improvement:
```
Telemetry → Anomaly Detection → Causal Analysis → RL Recommendation → Code Understanding → PR Generation → Storage & Learning
```

### 3.2. Neural Network Anomaly Detection
- **Purpose:** Advanced time-series anomaly detection using an ensemble of LSTM and Transformer autoencoders.
- **Key Features:** Learns normal behavior to detect deviations, uses adaptive thresholding to minimize false positives, and identifies which metrics contribute most to an anomaly.

### 3.3. Deep Learning Causal Inference Engine
- **Purpose:** Replaces simple rule-based logic with a neural network-based doubly robust estimation model to identify the root causes of anomalies.
- **Key Features:** Provides calibrated confidence scores for causal claims and robustly estimates the causal effects of system events.

### 3.4. Reinforcement Learning (RL) Agent
- **Purpose:** Learns the optimal intervention policy through experience, using a Deep Q-Network (DQN).
- **Key Features:** Balances exploration and exploitation to decide on the best action (e.g., optimize algorithm, scale resources) based on the current system state, with a reward function tuned to prioritize stability and performance.

### 3.5. Advanced Code Understanding & Generation
- **Purpose:** NLP-based code analysis and generation using pre-trained transformers (CodeBERT and CodeT5).
- **Key Features:** Generates semantic embeddings of code to understand its function, identifies optimization opportunities, and automatically refactors code to implement improvements.

### 3.6. Continuous Learning & Supabase Integration
- **Purpose:** Enables the system to learn and improve from operational experience.
- **Key Features:** All interventions and their outcomes are stored in a persistent Supabase database. This experience is used to automatically retrain the RL agent and other models on a continuous schedule, allowing the system to improve over time.

---

## 4. Best Practices & Key Technical Learnings

#### **Deep Learning Development**
*   **Continuous Learning:** The system's performance is directly tied to the quality and quantity of its experience. Ensure the continuous learning pipeline is active and that models are retraining on schedule.
*   **Benchmarking:** Regularly run the performance benchmarks (`src/zebra_orchestrator/benchmarking.py`) to monitor model latency and prevent regressions.
*   **Model Versioning:** (Future) Integrate a model registry to track versions and facilitate rollbacks if a new model underperforms.

#### **Python Instrumentation & Service Design**
*   **Manual In-Code Instrumentation:** Continue to use manual OpenTelemetry setup within the application code for reliability.
*   **Use a Standard WSGI Server:** Always run services with Gunicorn, not the Flask development server.
*   **Robustness:** Services must handle empty or malformed telemetry payloads gracefully.

#### **Docker & Docker Compose**
*   **Use `--force-recreate`:** When changing configurations, use this flag to avoid stale container states.
*   **Install from `requirements.txt`:** Ensure all dependencies are managed centrally in `requirements.txt`.

---

## 5. Testing & Verification

The system includes a comprehensive suite of tests and benchmarks to ensure correctness and performance.

*   **Run Unit & Integration Tests:**
    ```bash
    pytest test_zebra_system.py
    ```
*   **Run Performance Benchmarks:**
    ```bash
    python3 -c "from src.zebra_orchestrator.benchmarking import benchmark_suite; benchmark_suite.run_all_benchmarks()"
    ```

---

## 6. Meta-System: Self-Documentation
A core principle of Zebra is self-reflection. This document should be considered part of the system's "source code." Future iterations of Zebra should be able to parse this documentation to understand their own architecture, goals, and best practices, enabling them to make more informed decisions.