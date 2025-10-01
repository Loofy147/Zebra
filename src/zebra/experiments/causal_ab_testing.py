# src/zebra/experiments/causal_ab_testing.py
"""
Causal A/B testing utilities (PR-ready PoC).

Design goals:
- Use econml / causalml if available (best-effort), else fallback to
  adjusted difference-in-means with propensity score stratification.
- Produce audit-friendly outputs and integrate with ZebraObservability if provided.
- Graceful degradation: the module raises NotImplementedError if critical libs are missing
  for a requested advanced method, but still provides simple analysers.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import logging
import uuid
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger("zebra.experiments.causal_ab")

# Optional observability integration
try:
    from src.zebra_telemetry.opentelemetry_setup import ZebraObservability
except Exception:
    ZebraObservability = None

# Try importing advanced libs
_HAS_ECONML = False
_HAS_CAUSALML = False
_HAS_STATSMODELS = False
_HAS_SKLEARN = False

try:
    from econml.dml import LinearDML
    _HAS_ECONML = True
except ImportError:
    pass

try:
    from causalml.inference.meta import XGBTRegressor
    _HAS_CAUSALML = True
except ImportError:
    pass

try:
    from statsmodels.stats.power import zt_ind_solve_power
    _HAS_STATSMODELS = True
except ImportError:
    pass

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import log_loss
    _HAS_SKLEARN = True
except ImportError:
    pass


@dataclass
class ExperimentResult:
    experiment_id: str
    started_at: datetime
    finished_at: Optional[datetime]
    traditional: Dict[str, Any]
    causal: Dict[str, Any]
    recommendation: str

class CausalABTesting:
    def __init__(self, observability: Optional[ZebraObservability] = None):
        self.experiments: Dict[str, Dict] = {}
        self.obs = observability

    # ---------- design utilities ----------
    def design_experiment(self, hypothesis: str, variants: List[str], effect_size: float = 0.05,
                          power: float = 0.8, significance: float = 0.05) -> Dict[str, Any]:
        sample_size = self.calculate_required_sample_size(effect_size, power, significance)
        exp = {
            "id": str(uuid.uuid4()),
            "hypothesis": hypothesis,
            "variants": variants,
            "start_time": datetime.utcnow().isoformat(),
            "sample_size": sample_size,
            "power": power,
            "significance": significance,
            "status": "draft",
        }
        self.experiments[exp["id"]] = exp
        return exp

    def calculate_required_sample_size(self, effect_size: float, power: float, significance: float) -> int:
        if not _HAS_STATSMODELS:
            raise NotImplementedError("statsmodels is required to calculate sample size.")
        n = zt_ind_solve_power(effect_size=effect_size, alpha=significance, power=power, ratio=1.0, alternative="two-sided")
        return int(np.ceil(n))

    # ---------- data ingestion ----------
    def ingest_experiment_data(self, experiment_id: str, df: pd.DataFrame):
        """
        df must contain at least columns:
          - 'variant' (A/B label)
          - 'outcome' (numeric or binary)
          - optionally confounders/features for causal adjustment
        """
        if experiment_id not in self.experiments:
            raise KeyError("experiment_id unknown")
        self.experiments[experiment_id].setdefault("data", df.copy())
        self.experiments[experiment_id]["status"] = "data_ingested"

    # ---------- traditional A/B test ----------
    def traditional_ab_test(self, df: pd.DataFrame, variant_col: str = "variant", outcome_col: str = "outcome") -> Dict[str, Any]:
        grouped = df.groupby(variant_col)[outcome_col].agg(["mean", "std", "count"])
        if grouped.shape[0] == 2:
            vals = grouped.reset_index()
            a, b = vals.iloc[0], vals.iloc[1]
            diff = float(a["mean"] - b["mean"])
            se = np.sqrt((a["std"] ** 2 / a["count"]) + (b["std"] ** 2 / b["count"]))
            z = diff / se if se > 0 else np.nan
            p = 2 * (1 - 0.5 * (1 + np.math.erf(abs(z) / np.sqrt(2)))) if not np.isnan(z) else np.nan
            return {"group_stats": grouped.to_dict(), "diff": diff, "z": z, "p_value": p}
        else:
            return {"group_stats": grouped.to_dict(), "note": "multi-variant returned as table"}

    # ---------- causal estimators ----------
    def estimate_ate(self, df: pd.DataFrame, treatment: str = "variant", outcome: str = "outcome", confounders: Optional[List[str]] = None) -> Dict[str, Any]:
        confounders = confounders or []
        X = df[confounders] if confounders else pd.DataFrame(index=df.index)
        T = df[treatment]
        Y = df[outcome]
        if _HAS_ECONML and _HAS_SKLEARN:
            model = LinearDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingClassifier())
            model.fit(Y, T, X=X, inference="default")
            ate = model.ate(X)
            ate_interval = model.ate_interval(X, alpha=0.05) if hasattr(model, 'ate_interval') else None
            return {"point_estimate": float(np.mean(ate)), "interval": ate_interval}
        elif _HAS_SKLEARN and confounders:
            pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=200))])
            pipe.fit(X.fillna(0), (T != T.iloc[0]).astype(int))
            df['_ps'] = pipe.predict_proba(X.fillna(0))[:, 1]
            df['_strata'] = pd.qcut(df['_ps'], q=5, duplicates="drop")
            strata_results = df.groupby('_strata').apply(lambda group: self.traditional_ab_test(group, treatment, outcome))
            return {"method": "ps_stratification", "strata_summary": strata_results.to_dict()}
        else:
            return {"method": "diff_in_means", "diff": float(df.groupby(treatment)[outcome].mean().diff().dropna().iloc[0])}

    def estimate_cate(self, df: pd.DataFrame, features: List[str], treatment: str = "variant", outcome: str = "outcome") -> Dict[str, Any]:
        if not _HAS_CAUSALML:
            raise NotImplementedError("causalml is required for CATE estimation.")
        learner = XGBTRegressor()
        cate = learner.fit_predict(X=df[features], treatment=df[treatment], y=df[outcome])
        return {"cate": cate.tolist()}

    # ---------- sensitivity analysis ----------
    def sensitivity_analysis(self, df: pd.DataFrame, ate_estimate: Optional[Dict] = None) -> Dict[str, Any]:
        return {"note": "run domain-specific sensitivity (Rosenbaum bounds, e-value) - PoC"}

    # ---------- orchestrator ----------
    def analyze_experiment_causally(self, experiment_id: str, confounders: Optional[List[str]] = None, features: Optional[List[str]] = None) -> ExperimentResult:
        exp = self.experiments.get(experiment_id)
        if not exp or "data" not in exp:
            raise KeyError("No data for experiment")
        df = exp["data"]
        started = datetime.utcnow()
        traditional = self.traditional_ab_test(df)
        causal = self.estimate_ate(df, confounders=confounders)
        cate = self.estimate_cate(df, features) if features and _HAS_CAUSALML else {}
        sensitivity = self.sensitivity_analysis(df, causal)
        recommendation = self.make_recommendation(traditional, causal, sensitivity)
        finished = datetime.utcnow()

        if self.obs:
            self.obs.record_request(path="experiments.analyze", method="causal_ab", status_code=200, latency_ms=(finished - started).total_seconds() * 1000)

        return ExperimentResult(
            experiment_id=experiment_id,
            started_at=started,
            finished_at=finished,
            traditional=traditional,
            causal={"ate": causal, "cate": cate, "sensitivity": sensitivity},
            recommendation=recommendation
        )

    def make_recommendation(self, traditional: Dict[str, Any], causal: Dict[str, Any], sensitivity: Dict[str, Any]) -> str:
        ate_result = causal.get("point_estimate")
        if ate_result is not None:
            return "implement_on_winner" if ate_result > 0 else "do_not_implement"
        return "inconclusive"