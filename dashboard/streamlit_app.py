# dashboard/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime
from typing import Any

# This is a mock import. In a real scenario, the zebra package would be installed.
# For demonstration purposes, we will mock the necessary classes.
# from zebra.explainability.decision_explainer import Decision, DecisionExplainer

@st.cache_data
def get_explainer():
    # Mocking DecisionExplainer for demo purposes
    class MockDecisionExplainer:
        def explain_causal_decision(self, decision, causal_graph, data):
            # A mock visualization using plotly's graph_objects
            fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode='markers', name='Mock Data')],
                            layout=go.Layout(title=go.layout.Title(text="Causal Graph (Mock)")))
            return {
                "natural_language": "The system decided to 'Increase cache size' because CPU usage was high, which causally influences latency. The most important factor was 'cpu_usage'. A counterfactual scenario suggests that if 'cpu_usage' were lower, the decision would be different.",
                "causal_chain": [{"path": ["cpu", "latency"], "strength": 0.8}],
                "counterfactuals": [{"variable": "cpu", "counterfactual_value": 0.5, "alternative_decision": "no_action"}],
                "visualization": fig.to_dict()
            }
    return MockDecisionExplainer()

@st.cache_data
def get_recent_decisions():
    # demo decision
    return [{
        "id": "d-123",
        "title": "Increase cache size",
        "status": "معلق",
        "confidence": 0.87,
        "type": "تحسين",
        "timestamp": datetime.utcnow().isoformat(),
        "expected_impact": {"performance_change": 0.05, "cost_change": -0.01, "risk_level": "medium"},
        "actual_results": {"performance_change": 0.0, "cost_change": 0.0, "risk_level": "unknown"},
        "explanation": None
    }]

@st.cache_data
def get_current_causal_graph():
    # return a simple dict for the mock
    return {"nodes": ["cpu", "mem", "latency"], "edges": [("cpu","latency"),("mem","latency")]}

# Mock Decision class
class Decision:
    def __init__(self, id, action, target_variable, input_variables, current_value, model, confidence, expected_outcome, risks=None, timestamp=None):
        self.id = id
        self.action = action
        self.target_variable = target_variable
        self.input_variables = input_variables
        self.current_value = current_value
        self.model = model
        self.confidence = confidence
        self.expected_outcome = expected_outcome
        self.risks = risks or []
        self.timestamp = timestamp or datetime.utcnow().isoformat()

explainer = get_explainer()

st.set_page_config(page_title="Zebra Control Panel", page_icon="🦓", layout="wide")
st.title("🦓 Zebra Control Panel — Explainability")

decisions = get_recent_decisions()
for d in decisions:
    with st.expander(f"{d['title']} — الثقة: {d['confidence']:.1%} — {d['status']}"):
        cols = st.columns(3)
        cols[0].metric("الثقة", f"{d['confidence']:.1%}")
        cols[1].metric("نوع القرار", d["type"])
        cols[2].metric("الحالة", d["status"])
        if st.button("شرح القرار", key=f"explain_{d['id']}"):
            # build Decision dataclass
            decision_obj = Decision(
                id=d["id"],
                action=d["title"],
                target_variable="latency",
                input_variables={"cpu": 0.9, "mem": 0.7},
                current_value={"cpu": 0.9, "mem": 0.7},
                model=lambda x: 1 if sum(x.values())>1.0 else 0,  # demo model
                confidence=d["confidence"],
                expected_outcome=d["expected_impact"],
                risks=["cost increase"]
            )
            cg = get_current_causal_graph()
            # use some sample data (DataFrame)
            sample_df = pd.DataFrame([{"cpu":0.9,"mem":0.7,"latency":120}])
            explanation = explainer.explain_causal_decision(decision_obj, cg, sample_df)
            st.subheader("تفسير مختصر (NL)")
            st.write(explanation["natural_language"])
            st.subheader("سلسلة السببية")
            st.json(explanation.get("causal_chain", {}))
            st.subheader("Counterfactuals")
            st.json(explanation.get("counterfactuals", []))
            if explanation.get("visualization"):
                st.subheader("تصوّر الشبكة")
                try:
                    fig = go.Figure(explanation["visualization"])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render visualization: {e}")