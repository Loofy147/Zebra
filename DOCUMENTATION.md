# Zebra Project: Comprehensive Documentation

This document contains the complete project documentation, synthesizing the roadmap, best practices, and key learnings from the implementation journey. It serves as a full project summary and a guide for future development.

---

## 1. Project Overview & Vision

Zebra is a prototype implementation of a self-improving, multi-ecosystem AI architecture. Its purpose is to autonomously observe, analyze, and evolve its own software components to achieve high-level strategic goals. It is built on a foundation of mathematical guarantees for stability, rational decision-making, and structural integrity.

The core principle is a gradual, multi-phase evolution where each stage builds trust and capability upon the last: **start by observing, then suggesting, then acting under supervision, and only then achieving full autonomy.**

## 2. The Phased Implementation Roadmap (Current Status)

Zebra's development follows a four-phase maturity model.

*   #### **Phase 0: The Substrate - Universal Observability & A Formal System Model**
    *   **Goal:** Give the system perfect senses and a perfect memory.
    *   **Status:** **Completed ✅**
    *   **Artifacts:**
        *   A fully observable `sample-service` that emits traces.
        *   A Docker Compose environment with an OpenTelemetry Collector and Jaeger for visualization.
        *   A robust pattern for manual, in-code Python instrumentation.

*   #### **Phase 1: The Observer - From Data to Actionable Insight**
    *   **Goal:** Build the system's "brain" but keep it disconnected from the "hands." The system will analyze and suggest, but take no action.
    *   **Status:** **Completed ✅**
    *   **Artifacts:**
        *   A `zebra-observer` service that ingests telemetry from the collector.
        *   A simulated **Causal Inference Engine (CIE)** that performs root cause analysis on anomalies.
        *   A **"Shadow" LLM** that generates code optimization proposals based on the CIE's findings.
        *   A complete **"Observe -> Analyze -> Suggest"** pipeline has been established and verified.

*   #### **Phase 2: The Apprentice - Supervised Actuation**
    *   **Goal:** Connect the brain to the hands, with a strict human approval gate for every action.
    *   **Status:** Not Started ⚪

*   #### **Phase 3: The Practitioner - Bounded Autonomy**
    *   **Goal:** Remove the human safety gate for specific, well-understood classes of changes.
    *   **Status:** Not Started ⚪

## 3. Best Practices & Key Technical Learnings

This project involved a significant amount of debugging to arrive at a correct and robust solution. The following are the key technical best practices and learnings that should guide all future development on this project.

#### **Python Instrumentation:**
*   **Use Manual In-Code Instrumentation:** The most robust and reliable method for instrumenting Python web services in this project is to configure OpenTelemetry manually within the application code.
*   **Initialize at the Module Level:** All OpenTelemetry configuration (e.g., creating the `TracerProvider`) must be done at the top level of the application module to ensure it's executed when a WSGI server (Gunicorn) imports the application object.
*   **Use a Standard WSGI Server:** Always run the application with a standard WSGI server like **Gunicorn**. Flask's built-in development server is not suitable for reliable instrumentation.

#### **OpenTelemetry Collector Configuration:**
*   **Forking Pipelines:** The collector can send the same data to multiple destinations simultaneously (e.g., to Jaeger and a custom observer) by defining a pipeline with multiple exporters.
*   **OTLP/HTTP Exporter Configuration:**
    *   The `endpoint` for an `otlphttp` exporter must contain the full path to the receiving endpoint (e.g., `http://service-name:port/v1/traces`).
    *   To send telemetry as JSON, the exporter must be explicitly configured with a `headers` block: `headers: { "Content-Type": "application/json" }`.

#### **Docker & Docker Compose:**
*   **Use `--force-recreate`:** During development, if configuration changes are not being applied, use the `--force-recreate` flag with `docker compose up` to ensure all containers are destroyed and recreated from scratch, bypassing any stale state.
*   **Install Dependencies from `requirements.txt`:** Each service's `Dockerfile` should copy the project's `requirements.txt` file and run `pip install -r requirements.txt` to ensure all dependencies are correctly installed in a reproducible way.

#### **Robust Service Design:**
*   **Handle Empty Payloads:** Services that ingest telemetry from the OTel collector must be robust against receiving empty request bodies. The receiving endpoint should check if the request data is empty before parsing it, ideally with a `try...except JSONDecodeError` block.

## 4. Meta-System: Self-Documentation
A core principle of Zebra is self-reflection. This document should be considered part of the system's "source code." Future iterations of Zebra should be able to parse this documentation to understand their own architecture, goals, and best practices, enabling them to make more informed decisions.