from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

class MultivariateStabilityAnalyzer:
    """
    MultivariateStabilityAnalyzer:
    - اختبار Johansen للتكامل المشترك
    - فحص استقرار VAR (جذور المعادلة)
    """

    def johansen_cointegration_test(self, matrix: np.ndarray, det_order: int = 1, k_ar_diff: int = 1) -> Dict[str, Any]:
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
        except Exception:
            raise NotImplementedError("statsmodels required for Johansen cointegration (pip install statsmodels)")

        res = coint_johansen(matrix, det_order=det_order, k_ar_diff=k_ar_diff)
        # res.lr1 = trace statistics, res.lr2 = max eigenvalue statistics, res.cvt = critical values
        return {
            "trace_statistic": np.asarray(res.lr1).tolist(),
            "max_eigen_statistic": np.asarray(res.lr2).tolist(),
            "critical_values": np.asarray(res.cvt).tolist(),
            "eigenvectors": None,  # res.evec can be added if needed
            "cointegration_rank": self.determine_rank(res),
        }

    def determine_rank(self, johansen_result) -> int:
        """
        Determine cointegration rank with a simple rule:
        compare trace statistics to critical values (5%) for each r.
        """
        # res.lr1: trace stats; res.cvt: critical values matrix [:, idx], idx 1=5% in many versions
        trace_stats = np.asarray(johansen_result.lr1)
        crit_table = np.asarray(johansen_result.cvt)
        # try index 1 for 5% (older statsmodels store 90/95/99) — fallback safe
        try:
            crit_5 = crit_table[:, 1]
        except Exception:
            crit_5 = crit_table[:, 0]
        rank = 0
        for t, c in zip(trace_stats[::-1], crit_5[::-1]):
            if t > c:
                rank += 1
            else:
                break
        return int(rank)

    def vector_autoregression_stability(self, data: np.ndarray, maxlags: int = 5) -> Dict[str, Any]:
        """
        يناسب بيانات على شكل (T, N) حيث N = عدد السلاسل.
        يرجع جذور نموذج VAR ويحدد إذا كان كل الجذور داخل الوحدة (مستقر).
        """
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
        except Exception:
            raise NotImplementedError("statsmodels required for VAR (pip install statsmodels)")

        # data expected as 2D: (T, N)
        model = VAR(data)
        fitted = model.fit(maxlags=maxlags, ic="aic")
        roots = np.asarray(fitted.roots)
        is_stable = bool(np.all(np.abs(roots) < 1.0))
        return {
            "is_stable": is_stable,
            "roots": roots.tolist(),
            "max_root": float(np.max(np.abs(roots))),
        }