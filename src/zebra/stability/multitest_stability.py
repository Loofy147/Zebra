from __future__ import annotations
import warnings
from typing import Any, Dict, Tuple, List, Optional
import numpy as np

# Third-party imports kept local to functions to allow graceful degradation if missing
# (so tests can run even if some optional packages are not installed)

class MultiTestStabilityAnalyzer:
    """
    MultiTestStabilityAnalyzer:
    - يجمع عدة اختبارات استقرار أحادية المتغير (ADF, KPSS, PP, Zivot-Andrews, Variance Ratio)
    - يوفر نتيجة إجماعية (majority vote) + ثقة مبسطة (نسبة الاختبارات التي تدعم الإستقرار)
    - واجهة مرنة للتوسيع أو استبدال الاختبارات
    """

    def __init__(self, significance_level: float = 0.05):
        self.alpha = float(significance_level)
        # mapping from test name -> function(self, series) -> dict result
        self.tests = {
            "adf": self.augmented_dickey_fuller,
            "kpss": self.kwiatkowski_phillips_schmidt_shin,
            "pp": self.phillips_perron,
            "zivot": self.zivot_andrews,
            "variance_ratio": self.variance_ratio_test,
        }

    # ----------------------------- orchestration -----------------------------
    def comprehensive_stability_test(self, series: np.ndarray) -> Dict[str, Any]:
        """
        تطبِق كل الاختبارات وتعيد نتيجة إجماعية.
        النتيجة تحتوي: is_stationary (bool), confidence (0..1), individual_results
        """
        series = np.asarray(series).astype(float)
        results: Dict[str, Any] = {}

        for name, fn in self.tests.items():
            try:
                results[name] = fn(series)
            except NotImplementedError as e:
                results[name] = {"error": str(e), "is_stationary": None}
            except Exception as e:
                results[name] = {"error": f"runtime: {e}", "is_stationary": None}

        # Aggregate: count tests that returned boolean decision
        decisions = [r.get("is_stationary") for r in results.values() if isinstance(r, dict)]
        votes = [d for d in decisions if d is True]  # True = stationary
        valid_votes = [d for d in decisions if d is not None]
        confidence = float(len(votes) / len(valid_votes)) if valid_votes else 0.0
        is_stationary = confidence >= 0.5  # majority vote (can be parameterized)

        return {
            "is_stationary": is_stationary,
            "confidence": confidence,
            "individual_results": results,
            "consensus_method": "majority_vote",
        }

    # ----------------------------- individual tests -----------------------------
    def augmented_dickey_fuller(self, series: np.ndarray) -> Dict[str, Any]:
        """
        ADF test (uses statsmodels). Returns dict with statistic, p_value, used_lags.
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except Exception:
            raise NotImplementedError("statsmodels required for ADF (pip install statsmodels)")

        # suppress warnings from adfuller about regression failure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = adfuller(series, autolag="AIC", regression="ct")
        # adfuller returns: statistic, pvalue, usedlag, nobs, critical_values, icbest
        return {
            "statistic": float(res[0]),
            "p_value": float(res[1]),
            "is_stationary": float(res[1]) < self.alpha,
            "critical_values": res[4],
            "used_lags": int(res[2]),
        }

    def kwiatkowski_phillips_schmidt_shin(self, series: np.ndarray) -> Dict[str, Any]:
        """
        KPSS test for stationarity (null = stationary). We invert decision so True => stationary.
        """
        try:
            from statsmodels.tsa.stattools import kpss
        except Exception:
            raise NotImplementedError("statsmodels required for KPSS (pip install statsmodels)")

        # KPSS may warn on regression choices; set nlags='auto' supported in recent versions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p_value, lags, crit = kpss(series, regression="ct", nlags="auto")
        # For KPSS null hypothesis: stationary. So p_value < alpha => reject stationarity.
        is_stationary = float(p_value) >= self.alpha
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "is_stationary": is_stationary,
            "used_lags": int(lags),
            "critical_values": crit,
        }

    def phillips_perron(self, series: np.ndarray) -> Dict[str, Any]:
        """
        Phillips-Perron test. statsmodels has phillips_perron in tsa.stattools in some versions.
        """
        try:
            from statsmodels.tsa.stattools import phillips_perron
        except Exception:
            # Some older/newer versions might not expose it; raise explicit message
            raise NotImplementedError("Phillips-Perron not available: require statsmodels with phillips_perron")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = phillips_perron(series)
        # phillips_perron returns (stat, pvalue, lags, crit)
        return {
            "statistic": float(res[0]),
            "p_value": float(res[1]),
            "is_stationary": float(res[1]) < self.alpha,
            "used_lags": int(res[2]),
            "critical_values": res[3],
        }

    def zivot_andrews(self, series: np.ndarray) -> Dict[str, Any]:
        """
        Zivot-Andrews for structural break detection.
        """
        try:
            from statsmodels.tsa.stattools import zivot_andrews
        except Exception:
            raise NotImplementedError("Zivot-Andrews not available: require statsmodels.orig")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = zivot_andrews(series, trim=0.15)
        # returns (stat, pvalue, breakpoint)
        return {
            "statistic": float(res[0]),
            "p_value": float(res[1]),
            "is_stationary": float(res[1]) < self.alpha,
            "breakpoint": int(res[2]) if res[2] is not None else None,
        }

    def variance_ratio_test(self, series: np.ndarray, lag: int = 2) -> Dict[str, Any]:
        """
        Variance ratio test (Lo-MacKinlay or similar). Implement a simple version using numpy.
        """
        # Simple implementation for small lags; not as feature-complete as specialized libs
        # Null: random walk (non-stationary). We expect variance ratio ~1 under RW.
        n = len(series)
        if n < lag + 3:
            return {"error": "series too short for variance ratio", "is_stationary": None}
        diffs = np.diff(series)
        var1 = np.var(diffs, ddof=1)
        # aggregated k-step differences
        k = lag
        # k-step returns
        agg = series[k:] - series[:-k]
        vark = np.var(agg / np.sqrt(k), ddof=1)
        vr = vark / var1 if var1 > 0 else np.nan
        # naive pvalue: treat vr ~ 1 under H0 — we compute distance
        p_value = float(np.exp(-abs(vr - 1)))  # heuristic; not statistically rigorous
        is_stationary = p_value < self.alpha
        return {"vr": float(vr), "p_value": p_value, "is_stationary": is_stationary}

    # ----------------------------- utility -----------------------------
    def adaptive_significance_level(self, series: np.ndarray, base_alpha: float = 0.05) -> float:
        """
        حساب مستوى معنوية تكيفي بناءً على خصائص السلسلة:
        - يزيد الصرامة إذا التباين عالي أو الضوضاء مرتفعة
        - يقلل الصرامة إذا البيانات شديدة الانحراف/الثِقل
        """
        try:
            from scipy import stats
        except Exception:
            # إذا لم يتوفر scipy، أعد alpha الأساسي
            return float(base_alpha)

        variance = float(np.var(series))
        skewness = float(stats.skew(series))
        kurtosis = float(stats.kurtosis(series))
        adjusted_alpha = float(base_alpha)

        if variance > 100:  # threshold heuristic — يمكن ضبطه عبر config
            adjusted_alpha = base_alpha * 0.8
        elif abs(skewness) > 2:
            adjusted_alpha = base_alpha * 1.2
        return max(min(adjusted_alpha, 0.5), 0.0001)