# src/zebra_telemetry/opentelemetry_setup.py
# PR-ready PoC — ضبط Tracing + Metrics + safe baggage usage + programmatic instrumentation
from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Callable, Dict, Any, Optional

from opentelemetry import trace, metrics, baggage
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from opentelemetry.sdk.metrics import MeterProvider, PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import CallbackOptions, Observation

# Optional: instrument frameworks programmatically (FastAPI/requests/sqlalchemy)
# from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
# from opentelemetry.instrumentation.requests import RequestsInstrumentor
# from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

logger = logging.getLogger("zebra.telemetry")
logger.setLevel(logging.INFO)


class ZebraObservability:
    """
    إعداد OpenTelemetry (traces + metrics) بطريقة عملية وآمنة.
    - يستخدم OTLP exporter (endpoint يُضبط عبر env vars أو args)
    - يوفر واجهات لتسجيل مقاييس سببية بشكل صحيح (بمقاييس async للـgauge)
    - لا يضع الرسم السببي الكامل في baggage (يستخدم graph_id بدلاً منه)
    """

    def __init__(
        self,
        service_name: str = "zebra",
        otlp_endpoint: Optional[str] = None,
        metrics_export_interval_s: int = 5,
    ):
        # resource metadata
        resource = Resource.create({"service.name": service_name})
        self._setup_tracing(resource, otlp_endpoint)
        self._setup_metrics(resource, otlp_endpoint, metrics_export_interval_s)

        # placeholders for created instruments
        self.meter = metrics.get_meter(__name__)
        self.request_counter = self.meter.create_counter(
            "zebra.requests.total", description="عدد الطلبات الكلي"
        )
        self.latency_histogram = self.meter.create_histogram(
            "zebra.latency.ms", description="زمن الاستجابة (ملليثانية)", unit="ms"
        )

        # causal-specific instruments
        self.causal_discovery_duration = self.meter.create_histogram(
            "zebra.causal.discovery.duration",
            description="مدة اكتشاف العلاقات السببية (ms)",
            unit="ms",
        )
        self.causal_edges_discovered = self.meter.create_counter(
            "zebra.causal.edges.discovered", description="عدد الحواف السببية المكتشفة"
        )

        # stability: use ObservableGauge (async) for p-value / latest score
        self._latest_stability = {"p_value": None}

        def stability_callback(options: CallbackOptions):
            # callback must yield Observation(s)
            p = self._latest_stability.get("p_value")
            if p is None:
                return []
            yield Observation(p, {"zebra.metric": "stability.p_value"})

        # register ObservableGauge
        try:
            self.stability_observable = self.meter.create_observable_gauge(
                "zebra.stability.p_value",
                [stability_callback],
                description="قيمة p لاختبار الاستقرار",
            )
        except Exception:
            # بعض إصدارات SDK قد تستخدم naming مختلف؛ log and continue
            logger.exception("ObservableGauge registration failed (check SDK version)")

    def _setup_tracing(self, resource: Resource, otlp_endpoint: Optional[str]):
        # create tracer provider with resource
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # OTLP exporter - prefer env var OTEL_EXPORTER_OTLP_ENDPOINT if otlp_endpoint not passed
        otlp_endpoint = otlp_endpoint or None
        span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        span_processor = BatchSpanProcessor(
            span_exporter, max_queue_size=2048, max_export_batch_size=512, schedule_delay_millis=5000
        )
        tracer_provider.add_span_processor(span_processor)
        logger.info("Tracing configured (OTLP endpoint=%s)", otlp_endpoint)

    def _setup_metrics(self, resource: Resource, otlp_endpoint: Optional[str], interval_s: int):
        # OTLP Metric exporter requires a MetricReader; use PeriodicExportingMetricReader
        metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
        metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=interval_s * 1000)
        meter_provider = MeterProvider(metric_readers=[metric_reader], resource=resource)
        metrics.set_meter_provider(meter_provider)
        logger.info("Metrics configured (OTLP endpoint=%s, interval=%ds)", otlp_endpoint, interval_s)

    # ---------- helper APIs ----------
    def record_request(self, path: str, method: str, status_code: int, latency_ms: float):
        self.request_counter.add(1, {"http.path": path, "http.method": method, "http.status_code": status_code})
        self.latency_histogram.record(latency_ms, {"http.path": path, "http.method": method})

    def observe_causal_stability(self, p_value: float):
        """أحدث قيمة اختبار الاستقرار (تُنقل إلى ObservableGauge عبر callback)."""
        self._latest_stability["p_value"] = float(p_value)

    # ========= safe causal context propagation =========
    def propagate_causal_context(self, graph_id: str, next_service_call: Callable[[], Any]):
        """
        ضع فقط مرجع الرسم السببي (graph_id) في baggage/headers — لا ترسل الرسم نفسه.
        امثلة: baggage.set_baggage("zebra.causal.graph_id", graph_id)
        ثم احفظ الرسم الكامل في Redis/DB مع هذا graph_id.
        """
        ctx = baggage.set_baggage("zebra.causal.graph_id", graph_id)
        ctx = baggage.set_baggage("zebra.causal.timestamp", datetime.utcnow().isoformat(), context=ctx)
        # call next service (which can fetch full graph by id)
        return next_service_call()