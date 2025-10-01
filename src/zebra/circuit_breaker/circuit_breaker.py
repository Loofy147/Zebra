# src/zebra/circuit_breaker/circuit_breaker.py
from __future__ import annotations
from typing import Dict, Any, Optional, Callable
import threading
import logging
import time

logger = logging.getLogger("zebra.circuit")

class BreakerBase:
    def __init__(self, name: str, reason: str = ""):
        self.name = name
        self.reason = reason

    def should_break(self, metrics: Dict[str, Any]) -> bool:
        raise NotImplementedError

class PerformanceBreaker(BreakerBase):
    def __init__(self, threshold: float = -0.1):
        super().__init__("performance_degradation", f"performance drop < {threshold}")
        self.threshold = threshold

    def should_break(self, metrics: Dict[str, Any]) -> bool:
        perf_delta = float(metrics.get("performance_delta", 0.0))
        return perf_delta < self.threshold

class ErrorRateBreaker(BreakerBase):
    def __init__(self, threshold: float = 0.05):
        super().__init__("error_rate", f"error_rate > {threshold}")
        self.threshold = threshold

    def should_break(self, metrics: Dict[str, Any]) -> bool:
        err = float(metrics.get("error_rate", 0.0))
        return err > self.threshold

class ResourceBreaker(BreakerBase):
    def __init__(self, cpu_thresh: float = 0.9, mem_thresh: float = 0.9):
        super().__init__("resource_exhaustion", "resource utilization exceeded")
        self.cpu_thresh = cpu_thresh
        self.mem_thresh = mem_thresh

    def should_break(self, metrics: Dict[str, Any]) -> bool:
        cpu = float(metrics.get("cpu_usage", 0.0))
        mem = float(metrics.get("memory_usage", 0.0))
        return cpu > self.cpu_thresh or mem > self.mem_thresh

class CascadingFailureBreaker(BreakerBase):
    def __init__(self, dependent_error_threshold: float = 0.1):
        super().__init__("cascading_failures", "dependent service errors high")
        self.dependent_error_threshold = dependent_error_threshold

    def should_break(self, metrics: Dict[str, Any]) -> bool:
        dep_err = float(metrics.get("dependent_error_rate", 0.0))
        return dep_err > self.dependent_error_threshold

class CircuitBreaker:
    """
    CircuitBreaker central manager.
    state: 'closed', 'open', 'half-open'
    Integrate with governance/audit/observability.
    """

    def __init__(self, alert_fn: Optional[Callable[[str, str], None]] = None):
        self.breakers = {
            'performance_degradation': PerformanceBreaker(-0.1),
            'error_rate': ErrorRateBreaker(0.05),
            'resource_exhaustion': ResourceBreaker(0.90, 0.95),
            'cascading_failures': CascadingFailureBreaker(0.10),
        }
        self.state = 'closed'
        self.current_reason = None
        self.alert_fn = alert_fn

    def monitor_and_break(self, metrics: Dict[str, Any]) -> bool:
        for name, breaker in self.breakers.items():
            try:
                if breaker.should_break(metrics):
                    self.open_circuit(name, breaker.reason)
                    return True
            except Exception:
                logger.exception("breaker %s failed during should_break", name)
        return False

    def open_circuit(self, breaker_name: str, reason: str):
        if self.state == 'open':
            logger.info("Circuit already open: %s", self.current_reason)
            return
        self.state = 'open'
        self.current_reason = f"{breaker_name}: {reason}"
        logger.critical("🚨 Circuit Breaker triggered by %s: %s", breaker_name, reason)
        # Stop optimizations (hook)
        self.stop_all_optimizations()
        # rollback
        self.rollback_to_safe_state()
        # alert
        self.send_alert(breaker_name, reason)
        # schedule half-open test after configurable delay (30m PoC)
        threading.Timer(30 * 60, self.half_open_test).start()

    def stop_all_optimizations(self):
        # Integrator should implement real logic to disable autoscaling/improvements.
        logger.warning("Stopping all automated optimizations (PoC)")

    def rollback_to_safe_state(self):
        # Integrator should implement domain-specific rollback (revert model, revert infra flags)
        logger.warning("Rolling back to safe state (PoC)")

    def send_alert(self, breaker_name: str, reason: str):
        msg = f"Circuit breaker: {breaker_name} reason={reason}"
        logger.warning("Alert: %s", msg)
        if callable(self.alert_fn):
            try:
                self.alert_fn(breaker_name, reason)
            except Exception:
                logger.exception("alert_fn failed")

    def half_open_test(self):
        logger.info("Half-open test starting")
        self.state = 'half-open'
        # perform a cautious test: integrator should configure the actual test function
        # here we simulate a test by sleeping and setting closed
        time.sleep(5)  # PoC short test
        # In production: run smoke tests / Canary for a small fraction of traffic
        healthy = True  # result of those tests
        if healthy:
            self.state = 'closed'
            self.current_reason = None
            logger.info("Circuit breaker closed after successful half-open test")
        else:
            self.state = 'open'
            logger.info("Circuit breaker remains open after half-open test")