# src/zebra/decision/safe_policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional
import numpy as np
import logging
import time

# Optional observability integration
try:
    from src.zebra_telemetry.opentelemetry_setup import ZebraObservability
except Exception:
    ZebraObservability = None  # graceful degradation

logger = logging.getLogger("zebra.decision.safe_policy")

@dataclass
class SimulationResult:
    success: bool
    performance: float  # higher is better; can be reward or business metric
    metadata: Dict[str, Any]

class SimulationEnvInterface:
    """
    واجهة محاكاة بسيطة يتوقّع منها:
    - simulate(policy) -> SimulationResult
    Implement this interface in your env.
    """
    def simulate(self, policy: Any) -> SimulationResult:
        raise NotImplementedError

class SafePolicyLearning:
    """
    تنفيذ عملي لسياسة آمنة:
    - evaluate_policy_safety: تشغيل محاكاة monte-carlo + قياسات
    - gradual_rollout: نشر تدريجي مع مراقبة
    - formal_verification: محاولة تحقق رسمي (Z3) إن توفر
    """

    def __init__(self, observability: Optional[ZebraObservability] = None, safety_threshold: float = 0.95):
        self.safety_threshold = float(safety_threshold)
        self.rollback_history: List[Dict[str, Any]] = []
        self.obs = observability

    def evaluate_policy_safety(self, policy: Any, simulation_env: SimulationEnvInterface, n_sim: int = 1000) -> Dict[str, Any]:
        # Monte-Carlo simulation
        results: List[SimulationResult] = []
        start = time.time()
        for i in range(n_sim):
            r = simulation_env.simulate(policy)
            results.append(r)
        duration = time.time() - start

        perf = np.array([r.performance for r in results], dtype=float)
        success_rate = float(np.mean([1.0 if r.success else 0.0 for r in results]))
        worst_case = float(np.min(perf)) if perf.size > 0 else float("nan")

        # Stability: simple stationarity check on performance series (ADF if available)
        stability = self.test_stability(perf)

        is_safe = (
            (success_rate >= self.safety_threshold)
            and (worst_case > -0.1)
            and stability.get("is_stationary", False)
        )

        # Observability metrics
        if self.obs:
            try:
                # emit simple metrics/events
                self.obs.record_request(path="decision.evaluate_policy", method="simulate", status_code=200, latency_ms=duration*1000)
                # (more advanced: create custom metrics for success_rate/worst_case)
            except Exception:
                logger.exception("observability emit failed")

        return {
            "is_safe": bool(is_safe),
            "success_rate": success_rate,
            "worst_case": worst_case,
            "stability": stability,
            "duration_s": duration,
            "recommendation": "deploy" if is_safe else "reject",
        }

    def test_stability(self, performance_series: np.ndarray) -> Dict[str, Any]:
        # Lightweight stability: if statsmodels available run ADF; else use variance heuristic
        try:
            from statsmodels.tsa.stattools import adfuller
            series = np.asarray(performance_series)
            if series.size < 30:
                return {"is_stationary": False, "reason": "too_short"}
            res = adfuller(series, autolag="AIC", regression="c")
            p_value = float(res[1])
            return {"is_stationary": p_value < 0.05, "p_value": p_value}
        except Exception:
            # fallback heuristic: low coefficient of variation => stable
            arr = np.asarray(performance_series)
            if arr.size == 0:
                return {"is_stationary": False, "reason": "empty"}
            cv = float(np.std(arr) / (np.abs(np.mean(arr)) + 1e-9))
            return {"is_stationary": cv < 0.5, "cv": cv}

    def gradual_rollout(self, new_policy: Any, apply_policy_fn: Callable[[Any, float], List[SimulationResult]],
                        current_policy: Any, stages: Optional[List[float]] = None, monitor_window_s: int = 60) -> bool:
        """
        apply_policy_fn(policy, traffic_fraction) -> list of runtime results (or metrics)
        - stages: list of fractions [0.01, 0.05, ...]
        - monitor_window_s: how long to observe each stage (time-based or event-based in real infra)
        """
        if stages is None:
            stages = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

        for frac in stages:
            logger.info("Rollout stage: %.2f%%", frac * 100)
            results = apply_policy_fn(new_policy, frac)
            # We expect apply_policy_fn return list-like with performance attribute
            if not results:
                logger.warning("No results from apply_policy_fn at stage %.2f", frac)
                self.rollback(current_policy)
                return False

            perf_arr = np.array([r.performance for r in results], dtype=float)
            success_rate = float(np.mean([1.0 if r.success else 0.0 for r in results]))
            # simple safety heuristic: require success_rate >= safety_threshold and mean perf not dropping heavily
            if success_rate < self.safety_threshold or np.mean(perf_arr) < -0.1:
                logger.warning("Rollback triggered at stage %.2f (success_rate=%.3f, mean_perf=%.3f)", frac, success_rate, np.mean(perf_arr))
                self.rollback(current_policy)
                return False

            logger.info("Stage %.2f succeeded (success_rate=%.3f)", frac, success_rate)
            # optionally wait/monitor; real infra should collect metrics via observability
            time.sleep(0.5)  # quick pause for PoC; replace by real monitoring

        logger.info("Rollout completed successfully")
        return True

    def is_safe_to_continue(self, metrics: Dict[str, Any]) -> bool:
        # placeholder policy, to be extended with domain metrics
        return metrics.get("success_rate", 0.0) >= self.safety_threshold

    def rollback(self, previous_policy: Any):
        ts = time.time()
        self.rollback_history.append({"timestamp": ts, "policy": previous_policy})
        logger.info("Rollback executed at %s", time.ctime(ts))

    def formal_verification(self, assertions: List[Any]) -> Dict[str, Any]:
        """
        assertions: list of callables or z3 constraints (flexible)
        - If z3 is available and user provides z3.ExprRef constraints, we try to solve them.
        - If assertions are callables, execute them (should return True/False).
        """
        # Try z3 path
        try:
            import z3  # type: ignore
            # If user passed z3 constraints, we expect them in assertions
            sz = z3.Solver()
            for a in assertions:
                if isinstance(a, z3.ExprRef):
                    sz.add(a)
                else:
                    # skip non-z3 in this path
                    pass
            sat = sz.check()
            if sat == z3.sat:
                return {"verified": True, "method": "z3", "model": str(sz.model())}
            else:
                return {"verified": False, "method": "z3", "reason": str(sat)}
        except Exception:
            # fallback: treat assertions as callables
            results = []
            for a in assertions:
                try:
                    results.append(bool(a()))
                except Exception as e:
                    results.append(False)
            return {"verified": all(results), "method": "callable", "results": results}