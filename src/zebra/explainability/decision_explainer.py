# src/zebra/explainability/decision_explainer.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import logging

# Third-party optional libs (graceful fallback)
try:
    import shap
except Exception:
    shap = None

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

try:
    import networkx as nx
except Exception:
    nx = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

logger = logging.getLogger("zebra.explainability")

@dataclass
class Decision:
    id: str
    action: str
    target_variable: str
    input_variables: Dict[str, Any]
    current_value: Dict[str, Any]
    model: Any
    confidence: float
    expected_outcome: Dict[str, Any]
    risks: Optional[List[str]] = None
    timestamp: str = datetime.utcnow().isoformat()

class DecisionExplainer:
    """
    DecisionExplainer: يوفّر تفسيرات سببية وميزة-مهمة وcounterfactuals وملخص NL.
    - يعمل مع أو بدون shap/lime (fallback heuristics).
    - ينتج هيكل JSON قابل للأرشفة.
    """

    def __init__(self, observability: Optional[Any] = None):
        self.obs = observability

    # ---------- high-level API ----------
    def explain_causal_decision(self, decision: Decision, causal_graph: Any, data: Any) -> Dict[str, Any]:
        """
        Returns a structured explanation dict:
        {
          decision: {...},
          causal_chain: [...],
          feature_importance: {...},
          counterfactuals: [...],
          confidence: float,
          natural_language: str,
          visualization: optional serializable fig
        }
        """
        out = {
            "decision": vars(decision),
            "timestamp": datetime.utcnow().isoformat()
        }

        # causal chain extraction
        try:
            out["causal_chain"] = self.extract_causal_chain(causal_graph, decision)
        except Exception as e:
            out["causal_chain"] = {"error": f"causal chain extraction failed: {e}"}
            logger.exception("causal_chain error")

        # feature importance
        try:
            out["feature_importance"] = self.calculate_feature_importance(decision, data)
        except Exception as e:
            out["feature_importance"] = {"error": f"feature importance failed: {e}"}
            logger.exception("feature_importance error")

        # counterfactuals
        try:
            out["counterfactuals"] = self.generate_counterfactual_explanations(decision, causal_graph, data)
        except Exception as e:
            out["counterfactuals"] = {"error": f"counterfactual generation failed: {e}"}
            logger.exception("counterfactuals error")

        # natural language
        try:
            out["natural_language"] = self.generate_nl_explanation(decision, out.get("feature_importance", {}), out.get("causal_chain", {}))
        except Exception as e:
            out["natural_language"] = f"NL generation failed: {e}"
            logger.exception("nl error")

        # optional visualization (serializable)
        try:
            fig = self.generate_visual_explanation(decision, causal_graph)
            out["visualization"] = self._serialize_figure(fig) if fig is not None else None
        except Exception as e:
            logger.exception("visualization error")
            out["visualization"] = None

        # emit observability event if available
        if self.obs:
            try:
                self.obs.record_request(path="explain.decision", method="explain", status_code=200, latency_ms=0)
            except Exception:
                logger.exception("observability emit failed")

        return out

    # ---------- causal chain helpers ----------
    def extract_causal_chain(self, causal_graph: Any, decision: Decision) -> List[Dict[str, Any]]:
        """
        Find paths from root nodes to target_variable, score by aggregated edge weights.
        Requires causal_graph to have nodes() and edges(data=True) or be networkx-like.
        """
        if nx is None:
            raise RuntimeError("networkx is required for causal chain extraction (pip install networkx)")

        G = causal_graph if isinstance(causal_graph, nx.DiGraph) else nx.DiGraph(causal_graph)
        target = decision.target_variable
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        paths_info = []
        for r in roots:
            try:
                path = nx.shortest_path(G, source=r, target=target)
            except Exception:
                continue
            # compute simple path strength = product of (1 - p_value) or sum weights if present
            strengths = []
            for u, v in zip(path[:-1], path[1:]):
                ed = G.get_edge_data(u, v) or {}
                weight = ed.get("weight")
                pval = ed.get("p_value")
                if weight is not None:
                    strengths.append(float(weight))
                elif pval is not None:
                    strengths.append(1.0 - float(pval))
                else:
                    strengths.append(0.5)
            path_strength = float(sum(strengths) / max(1, len(strengths)))
            paths_info.append({"path": path, "strength": path_strength, "edges_meta": [G.get_edge_data(u, v) for u, v in zip(path[:-1], path[1:])]})

        paths_info_sorted = sorted(paths_info, key=lambda x: x["strength"], reverse=True)
        return paths_info_sorted

    # ---------- feature importance ----------
    def calculate_feature_importance(self, decision: Decision, data: Any) -> Dict[str, Any]:
        """
        Prefer SHAP -> LIME -> fallback permutation importance.
        Expects decision.model to be scikit-learn-like fitted model or callable predictor.
        """
        model = decision.model
        X = getattr(data, "X", None)
        if X is None:
            # if data is a DataFrame-like
            X = data

        # SHAP
        if shap is not None:
            try:
                # TreeExplainer if model has tree attributes, else KernelExplainer fallback
                if hasattr(model, "predict_proba") and hasattr(model, "feature_importances_"):
                    expl = shap.TreeExplainer(model)
                else:
                    expl = shap.KernelExplainer(lambda x: model.predict(x), shap.sample(X, min(50, len(X))))
                vals = expl.shap_values(X)
                # produce mean absolute shap per feature
                import numpy as np
                if isinstance(vals, list):
                    # multiclass -> take mean absolute of margin classes
                    arr = np.abs(np.vstack([v for v in vals])).mean(axis=0)
                else:
                    arr = np.abs(vals).mean(axis=0)
                features = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(arr.shape[0])]
                return {"method": "shap", "importances": dict(zip(features, arr.tolist()))}
            except Exception:
                logger.exception("SHAP failed, falling back")

        # LIME
        if LimeTabularExplainer is not None:
            try:
                exp = LimeTabularExplainer(X.values, mode="regression" if hasattr(model, "predict") else "classification", feature_names=list(X.columns))
                i = 0
                e = exp.explain_instance(X.values[i], model.predict if hasattr(model, "predict") else model, num_features=min(10, X.shape[1]))
                return {"method": "lime", "explanation": e.as_list()}
            except Exception:
                logger.exception("LIME failed, falling back")

        # fallback: permutation importance (naive)
        try:
            from sklearn.inspection import permutation_importance
            r = permutation_importance(model, X, getattr(data, "y", None), n_repeats=5, random_state=0)
            features = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(len(r.importances_mean))]
            return {"method": "permutation", "importances": dict(zip(features, r.importances_mean.tolist()))}
        except Exception:
            logger.exception("permutation importance failed")
            return {"method": "none", "importances": {}}

    # ---------- counterfactuals ----------
    def generate_counterfactual_explanations(self, decision: Decision, causal_graph: Any, data: Any, max_vars: int = 3) -> List[Dict[str, Any]]:
        """
        PoC: search minimal single-feature perturbations that flip model prediction.
        Not a rigorous CF generator — suitable as a first-order explanation.
        """
        model = decision.model
        x0 = decision.current_value.copy()
        cfs = []
        features = list(decision.input_variables.keys())[:max_vars]
        for feat in features:
            orig = x0.get(feat)
            # generate simple perturbation grid around orig
            try:
                import numpy as np
                deltas = [(-0.1), (-0.05), 0.05, 0.1]
                for d in deltas:
                    new = orig + d if isinstance(orig, (int, float)) else (d if orig is None else orig)
                    x_test = x0.copy()
                    x_test[feat] = new
                    # model.predict expects array-like
                    try:
                        pred_orig = model.predict([list(x0.values())])[0]
                        pred_new = model.predict([list(x_test.values())])[0]
                    except Exception:
                        # fallback: use model as callable returning score
                        pred_orig = model(x0) if callable(model) else None
                        pred_new = model(x_test) if callable(model) else None
                    if pred_new != pred_orig:
                        cfs.append({
                            "variable": feat,
                            "current_value": orig,
                            "counterfactual_value": new,
                            "delta": d,
                            "alternative_decision": str(pred_new)
                        })
                        break
            except Exception:
                continue
        return cfs

    # ---------- NL explanation ----------
    def generate_nl_explanation(self, decision: Decision, feature_importance: Dict[str, Any], causal_chain: Any) -> str:
        top = []
        try:
            imps = feature_importance.get("importances") or {}
            sorted_feats = sorted(imps.items(), key=lambda kv: kv[1], reverse=True)[:3]
            for name, score in sorted_feats:
                top.append({"name": name, "causal_effect": float(score)})
        except Exception:
            pass
        pieces = []
        pieces.append(f"قرر النظام '{decision.action}' على {decision.target_variable} بثقة {decision.confidence:.2%}.")
        if top:
            pieces.append("العوامل الأهم:")
            for i, f in enumerate(top, 1):
                pieces.append(f"{i}. {f['name']} (تأثير تقريبي {f['causal_effect']:.2%})")
        if decision.expected_outcome:
            desc = decision.expected_outcome.get("description", "تحسن متوقع")
            tf = decision.expected_outcome.get("timeframe", "قريبًا")
            pieces.append(f"التأثير المتوقع: {desc} خلال {tf}.")
        if decision.risks:
            pieces.append("المخاطر: " + ", ".join(decision.risks))
        # limit to a few sentences
        return " ".join(pieces[:6])

    # ---------- visualization ----------
    def generate_visual_explanation(self, decision: Decision, causal_graph: Any):
        if go is None or nx is None:
            return None
        G = causal_graph if isinstance(causal_graph, nx.DiGraph) else nx.DiGraph(causal_graph)
        pos = nx.spring_layout(G)
        edge_traces = []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = float(G.get_edge_data(u, v).get("weight", 0.5))
            edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                                          line=dict(width=max(1, w * 5), color="#888"), hoverinfo="none"))
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
        node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=node_text,
                                marker=dict(size=[(G.degree(n)+1)*8 for n in G.nodes()], color="lightblue", line=dict(width=1)))
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(title=f"Decision causal graph: {decision.id}", showlegend=False)
        return fig

    def _serialize_figure(self, fig):
        """Return JSON serializable representation of plotly fig"""
        try:
            return fig.to_dict()
        except Exception:
            return None