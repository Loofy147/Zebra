# Zebra Implementation Checklist

ملف تنفيذ ومراجعة عملي لنظام Zebra — قائمة فحص شاملة قابلة للاستخدام في PR وعمليات النشر.

> ملاحظة: انسخ هذا الملف في docs/implementation-checklist.md أو افتح كـPR.




---

0. ملخّص المراجعة السريعة

بنية المشروع جيدة: يوجد src/, tests/, وثائق وملفات إعداد OTel/Prometheus.

مطلوب تنفيذ CI/Matrix واختبارات أخلاقية/حوكمة ومراقبة أسرار وPII.

هناك مكونات PoC متعدّدة (observability, causal engine, experiments, explainability, continuous learning, governance, circuit-breakers, integration). تحتاج إلى ربط ومراجعة اعتماديات قبل التشغيل في الإنتاج.



---

1. متطلبات أساسية (التحقق قبل أي نشر)

[ ] Secrets: لا توجد أسرار (API keys, DB passwords) في repo. شغّل git-secrets/truffleHog لفحص التاريخ.

[ ] Environment: وثّق متغيرات البيئة المطلوبة في .env.example (DB URLs, Redis URL, OTEL endpoint, PagerDuty key, SMTP creds).

[ ] License & Codeowners: أضف LICENSE, CODEOWNERS, وCONTRIBUTING.md.

[ ] Pre-commit: فعِّل pre-commit (black/isort/ruff/git hooks).



---

2. CI / Build / Releases

[ ] GitHub Actions: job أساسي: lint (ruff/flake8), tests (pytest), build Docker image.

[ ] CI Matrix: jobs منفصلة لـ(1) lightweight PoC deps، (2) heavy ML deps (econml, causalml, shap) — اختبرها فقط عند الطلب.

[ ] Image scanning: Trivy / GitHub Container Scanning في pipeline.

[ ] Artifact publishing: push images لـ GHCR/ECR عند نجاح الاختبارات.

[ ] CD: Canary/Blue-Green deploy scripts (k8s manifests + helm charts أو kustomize).



---

3. Tests & Quality

[ ] Unit tests: تغطية لـ core modules (decision, bandit, experiments, explainability, continuous learning, caching).

[ ] Integration tests: Docker Compose based end-to-end smoke tests (Timescale + Redis + app) على job منفصل.

[ ] Load tests: k6/locust scenarios للـqueue processing وcausal discovery.

[ ] Policy tests: تحقق من سلوك PolicyChecker وGovernanceSystem على حالات افتراضية (safety/ethics/legal).



---

4. Observability & Monitoring

[ ] OTel Collector: تكوين مركزي (receive/export to Jaeger/OTLP/Prometheus).

[ ] Semantic convention: تأكد من إعداد ZebraSemanticConventions وملء السمات في spans.

[ ] Metrics: أنشئ وصنّف المقاييس الأساسية (decision.success_rate, causal.discovery.duration, explain.latency, continuous.buffer_size).

[ ] Prometheus rules & Alerts: حمل/فشل/rollback/circuit-breaker/queue-length/drift alerts مهيئة مع Alertmanager->PagerDuty.

[ ] Dashboards: Grafana dashboards for: Overview, Causal Graph health, Experiments, Decisions, Continuous learning drift.



---

5. Infrastructure & Scaling

[ ] Kubernetes: manifests (Deployment, Service, HPA, PDB, NetworkPolicy). استخدم KEDA لقياس الطوابير.

[ ] DB: TimescaleDB production config (no host port map), PgBouncer, backups (pgbackrest -> S3), monitoring.

[ ] Redis: caching & idempotency — use Redis cluster or managed service.

[ ] Autoscaling & Cluster Autoscaler: تأكد من node pools وquotas.



---

6. Model / ML Ops

[ ] Model registry: MLflow / DVC integration (artifact store + model_version + training_data_hash).

[ ] Retrain pipeline: retrain_from_scratch() ⇒ trigger offline training job (CI/GHA/Argo) + validation step (holdout) + canary deploy.

[ ] Experiment artifacts: store experiment_id, data_hash, analysis artifacts, model versions, recommendations.



---

7. Causal Engine & Experiments

[ ] Causal discovery: ensemble approach (PC/FCI/NOTEARS/ LiNGAM) with voting and edge confidence.

[ ] Counterfactual tests: bootstrap CI for counterfactual estimates and test-simulations for top-k CFs.

[ ] A/B governance: ADRs require ATE + sensitivity + human sign-off for actionable recommendations.



---

8. Decisioning & Safety

[ ] SafePolicyLearning: Monte-Carlo sim harness + rollback history + formal verification hooks (Z3).

[ ] Gradual rollout: canary stages + metric gates + automatic rollback on violations.

[ ] Circuit Breaker: integrate with Prometheus metrics and runbook for alerts & half-open tests.



---

9. Explainability & Privacy

[ ] DecisionExplainer: Cache heavy explainers (SHAP), sample-based explain generation, and exposure control (internal vs external views).

[ ] PII masking: enforce PII detection (regexes + blacklists) before saving or sending any explanation.

[ ] Retention: explain artifacts retention (90 days or per policy) and audit logs append-only.



---

10. Integration & Notifications

[ ] NotificationSystem: HMAC signing for webhooks, idempotency (Redis), rate-limits, retry/backoff.

[ ] Providers: Slack, Email, PagerDuty configured via env and secrets manager.

[ ] Webhook receivers: implement signature verification & idempotency on receiving side.



---

11. Security & Governance

[ ] Policy engine: integrate OPA/Cerbos for dynamic policy checks (PolicyChecker adapter).

[ ] AuditLog: move from sqlite PoC → append-only store (S3 + KMS or managed DB with immutability).

[ ] Human oversight: ticketing integration (Jira/ServiceNow/Slack) for high-priority reviews.

[ ] Access control: RBAC for endpoints, model registry, and secrets.



---

12. Operational Runbooks & ADRs

[ ] Runbooks: DriftDetected, CircuitBreakerTriggered, ExperimentImplement, RetrainComplete.

[ ] ADRs: store ADRs (0001..0010) under docs/adr/ (experiments, governance, explainability, continuous learning, infra).

[ ] Post-mortem template and schedule for RCA after incidents.



---

13. Deployment Checklist (pre-release)

[ ] All unit+integration tests pass in CI.

[ ] No secrets in repo; secrets present in CI/CD env only.

[ ] Canary deployment plan + metric gates defined.

[ ] Alerts & runbooks published and on-call notified.

[ ] Backups and restore tested (DB WAL restore test).



---

14. Priority action items (first 7 days)

1. [ ] Add CI + matrix (lint/tests/build) — يحجز #1 حسب الأهمية.


2. [ ] Enable secret scanning + .env.example.


3. [ ] Integrate NotificationSystem PoC (webhook signing + idempotency) and configure Redis for idempotency.


4. [ ] Add Prometheus alerts for CB / drift / rollback and a Grafana board for quick observability.


5. [ ] PII masking middleware: في مسارات شرح/notifications قبل أي إرسال.


6. [ ] Model registry + retrain job skeleton (triggered by retrain_from_scratch).


7. [ ] Run an end-to-end smoke test (docker-compose) and document results.




---

15. How  you can help (اختر ما تريد تنفيذًا الآن)

[ ] أجهز GitHub Actions CI (lint + tests + docker build) — أضعه كـPR-ready.

[ ] أجهز PRs منفصلة للـNotificationSystem, Explainability, Continuous Learning أو أي مكوّن تختاره.

[ ] أكتب Runbooks وADRs النهائية بصيغة قابلة للنشر.

[ ] أعدّ ملفات manifests لـK8s (Helm / Kustomize) وPR للـinfrastructure.



---

انتهى الملف.