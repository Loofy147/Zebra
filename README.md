# Zebra 🦓: A Self-Governing AI System

Zebra is a prototype implementation of a self-improving, multi-ecosystem AI architecture. Its purpose is to autonomously observe, analyze, and evolve its own software components to achieve high-level strategic goals. It is built on a foundation of mathematical guarantees for stability, rational decision-making, and structural integrity.

---

## Comprehensive Documentation

This repository contains a fully functional prototype demonstrating the "Observe -> Analyze -> Suggest" pipeline. For a detailed explanation of the project's vision, phased roadmap, technical architecture, and key learnings, please see the complete project documentation:

➡️ **[Zebra Project: Comprehensive Documentation](./DOCUMENTATION.md)**

---

## Quick Start

This project uses Docker and Docker Compose to run the multi-service environment.

1.  **Prerequisites:** Ensure you have Docker installed and running.
2.  **Build & Run:** From the project root, run the following command. `sudo` may be required depending on your Docker setup.
    ```bash
    docker compose up --build -d
    ```
3.  **Generate Telemetry:** Send requests to the sample service.
    ```bash
    curl http://localhost:8080/rolldice
    ```
4.  **Observe the Pipeline:**
    *   **View Traces:** Open the Jaeger UI at `http://localhost:16686`.
    *   **View Analysis:** Check the logs of the observer service to see the output of the Causal Inference Engine and the Shadow LLM.
        ```bash
        docker compose logs -f zebra-observer
        ```