# src/zebra/integration/notification_system.py
from __future__ import annotations
import json
import hmac
import hashlib
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
# optional redis for idempotency / distributed rate-limit
try:
    import redis
except Exception:
    redis = None

logger = logging.getLogger("zebra.integration.notification")

# -------------------------
# Helpers: idempotency + signing
# -------------------------
def compute_message_id(message: Dict[str, Any]) -> str:
    """
    Stable id for message: hash of title + text + sorted fields (best-effort).
    Integrator may override by passing 'message_id' in message metadata.
    """
    s = (message.get("title", "") + "|" + message.get("text", ""))
    fields = message.get("fields", [])
    try:
        s += "|" + json.dumps(fields, sort_keys=True)
    except Exception:
        s += "|fields"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def compute_hmac_signature(secret: str, payload: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


# -------------------------
# Rate limiter (token bucket, in-memory fallback; Redis available)
# -------------------------
@dataclass
class TokenBucketRateLimiter:
    rate_per_sec: float = 1.0  # tokens/sec
    burst: int = 5
    namespace: str = "zebra_notif"
    redis_client: Optional[Any] = None  # redis.Redis()
    _local_state: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def _now(self) -> float:
        return time.time()

    def allow(self, key: str) -> bool:
        """
        key: channel or global key.
        If redis_client provided, use Redis for distributed token bucket (approximate).
        """
        if self.redis_client is not None and redis is not None:
            # simple redis-based token bucket using a Lua-less approach (best-effort)
            try:
                rkey = f"{self.namespace}:bucket:{key}"
                data = self.redis_client.hgetall(rkey)
                now = self._now()
                if not data:
                    # initialize
                    self.redis_client.hmset(rkey, {"tokens": self.burst, "ts": now})
                    self.redis_client.expire(rkey, 3600)
                    tokens = float(self.burst)
                    ts = now
                else:
                    tokens = float(data.get(b"tokens", data.get("tokens", self.burst)))
                    ts = float(data.get(b"ts", data.get("ts", now)))
                # replenish
                delta = now - ts
                tokens = min(self.burst, tokens + delta * self.rate_per_sec)
                if tokens >= 1.0:
                    tokens -= 1.0
                    self.redis_client.hmset(rkey, {"tokens": tokens, "ts": now})
                    return True
                else:
                    # update timestamp
                    self.redis_client.hmset(rkey, {"tokens": tokens, "ts": now})
                    return False
            except Exception:
                logger.exception("redis rate limiter failed, falling back to local")
                # fallthrough to local
        # local in-memory token bucket
        st = self._local_state.setdefault(key, {"tokens": float(self.burst), "ts": self._now()})
        now = self._now()
        elapsed = now - st["ts"]
        st["tokens"] = min(self.burst, st["tokens"] + elapsed * self.rate_per_sec)
        st["ts"] = now
        if st["tokens"] >= 1.0:
            st["tokens"] -= 1.0
            return True
        return False


# -------------------------
# Base notifier and implementations
# -------------------------
class NotifierBase:
    def __init__(self, observability: Optional[Any] = None):
        self.obs = observability

    def send(self, message: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        raise NotImplementedError


def tenacity_retry_decorator(logger_obj: logging.Logger):
    # retry on requests exceptions and generic Exception as fallback
    return retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger_obj, logging.WARNING),
    )


class SlackNotifier(NotifierBase):
    def __init__(self, webhook_url: str, username: Optional[str] = "zebra", observability: Optional[Any] = None):
        super().__init__(observability)
        self.webhook_url = webhook_url
        self.username = username

    @tenacity_retry_decorator(logger)
    def send(self, message: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        payload = {
            "username": self.username,
            "text": f"*{message.get('title','')}*\n{message.get('text','')}",
            "attachments": [
                {
                    "fields": [
                        {"title": f.get("title", ""), "value": f.get("value", ""), "short": True}
                        for f in message.get("fields", [])
                    ]
                }
            ],
        }
        resp = requests.post(self.webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        # observability hook
        if self.obs:
            try:
                self.obs.record_request(path="notify.slack", method="post", status_code=resp.status_code, latency_ms=0)
            except Exception:
                logger.exception("obs slack emit failed")
        return True


import smtplib
from email.message import EmailMessage

class EmailNotifier(NotifierBase):
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
        from_addr: str = "zebra@localhost",
        observability: Optional[Any] = None,
    ):
        super().__init__(observability)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.from_addr = from_addr

    @tenacity_retry_decorator(logger)
    def send(self, message: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        recipients = metadata.get("recipients") if metadata else None
        if not recipients:
            raise ValueError("EmailNotifier requires recipients in metadata")
        msg = EmailMessage()
        msg["Subject"] = message.get("title", "Zebra Notification")
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(recipients)
        body = message.get("text", "")
        # include fields
        if message.get("fields"):
            body += "\n\n" + "\n".join([f"{f.get('title')}: {f.get('value')}" for f in message.get("fields", [])])
        msg.set_content(body)
        # send
        s = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10)
        try:
            if self.use_tls:
                s.starttls()
            if self.username and self.password:
                s.login(self.username, self.password)
            s.send_message(msg)
        finally:
            s.quit()
        if self.obs:
            try:
                self.obs.record_request(path="notify.email", method="smtp", status_code=200, latency_ms=0)
            except Exception:
                logger.exception("obs email emit failed")
        return True


class PagerDutyNotifier(NotifierBase):
    def __init__(self, routing_key: str, observability: Optional[Any] = None):
        super().__init__(observability)
        self.routing_key = routing_key
        self.url = "https://events.pagerduty.com/v2/enqueue"

    @tenacity_retry_decorator(logger)
    def send(self, message: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        payload = {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": message.get("title", ""),
                "source": metadata.get("source", "zebra") if metadata else "zebra",
                "severity": metadata.get("severity", "info") if metadata else "info",
                "custom_details": {"text": message.get("text", ""), "fields": message.get("fields", [])},
            },
        }
        resp = requests.post(self.url, json=payload, timeout=10)
        resp.raise_for_status()
        if self.obs:
            try:
                self.obs.record_request(path="notify.pagerduty", method="post", status_code=resp.status_code, latency_ms=0)
            except Exception:
                logger.exception("obs pagerduty emit failed")
        return True


class WebhookNotifier(NotifierBase):
    def __init__(self, endpoint: str, secret: Optional[str] = None, observability: Optional[Any] = None):
        """
        endpoint: full URL
        secret: optional HMAC secret to sign payloads (recommended)
        """
        super().__init__(observability)
        self.endpoint = endpoint
        self.secret = secret

    @tenacity_retry_decorator(logger)
    def send(self, message: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        payload_bytes = json.dumps(message).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.secret:
            headers["X-Zebra-Signature"] = compute_hmac_signature(self.secret, payload_bytes)
        timeout = (3, 10)  # (connect, read)
        resp = requests.post(self.endpoint, data=payload_bytes, headers=headers, timeout=timeout)
        resp.raise_for_status()
        if self.obs:
            try:
                self.obs.record_request(path="notify.webhook", method="post", status_code=resp.status_code, latency_ms=0)
            except Exception:
                logger.exception("obs webhook emit failed")
        return True


# -------------------------
# Main NotificationSystem
# -------------------------
class NotificationSystem:
    """
    Central dispatcher.
    Usage:
      ns = NotificationSystem(config=..., observability=...)
      ns.register_channel("slack", SlackNotifier(...))
      ns.notify(decision, channels=['slack','webhook'])
    """
    def __init__(self, rate_limiter: Optional[TokenBucketRateLimiter] = None, redis_client: Optional[Any] = None, observability: Optional[Any] = None):
        self.channels: Dict[str, NotifierBase] = {}
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter()
        self.redis = redis_client
        self.obs = observability
        # idempotency TTL seconds (prevent duplicate sends)
        self.idempotency_ttl = 60 * 60  # 1 hour

    def register_channel(self, name: str, notifier: NotifierBase):
        self.channels[name] = notifier

    def determine_channels(self, decision: Dict[str, Any]) -> List[str]:
        # Basic routing logic: can be extended with rules/config
        if decision.get("risk_level") == "high":
            return ["pagerduty", "slack", "email"]
        return decision.get("channels", ["slack"])

    def _is_idempotent_sent(self, message_id: str) -> bool:
        if self.redis is not None:
            try:
                key = f"zebra:notif:sent:{message_id}"
                return self.redis.exists(key)
            except Exception:
                logger.exception("redis idempotency check failed")
        else:
            # best-effort in-memory set (not persistent across processes)
            if not hasattr(self, "_local_sent"):
                self._local_sent = set()
            return message_id in self._local_sent

    def _mark_idempotent_sent(self, message_id: str):
        if self.redis is not None:
            try:
                key = f"zebra:notif:sent:{message_id}"
                self.redis.set(key, "1", ex=self.idempotency_ttl)
                return
            except Exception:
                logger.exception("redis idempotency mark failed")
        if not hasattr(self, "_local_sent"):
            self._local_sent = set()
        self._local_sent.add(message_id)

    def notify(self, decision: Dict[str, Any], channels: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        dispatch notifications to channels with:
         - idempotency (message_id)
         - rate limiting per-channel
         - retries handled in notifier implementations
         - observability events returned in result
        """
        result = {"sent": {}, "skipped": {}, "errors": {}}
        msg = self.format_decision_message(decision)
        # allow override message_id by decision metadata
        message_id = decision.get("message_id") or compute_message_id(msg)
        # idempotency: skip if already sent
        if self._is_idempotent_sent(message_id):
            result["skipped"]["reason"] = "idempotent_already_sent"
            return result

        if channels is None:
            channels = self.determine_channels(decision)

        for ch in channels:
            notifier = self.channels.get(ch)
            if notifier is None:
                result["errors"][ch] = "no_such_channel"
                continue
            # rate limit check
            if not self.rate_limiter.allow(ch):
                result["skipped"][ch] = "rate_limited"
                continue
            try:
                notifier.send(msg, metadata=metadata or {})
                result["sent"][ch] = "ok"
            except Exception as e:
                logger.exception("notify %s failed", ch)
                result["errors"][ch] = str(e)
                # If critical channel failed and decision is high risk — escalate via fallback channel
                if ch == "pagerduty":
                    # best-effort: try Slack as fallback
                    fallback = self.channels.get("slack")
                    if fallback:
                        try:
                            fallback.send({"title": f"[ESCALATION] {msg.get('title')}", "text": msg.get("text")})
                            result["sent"]["slack_fallback"] = "ok"
                        except Exception:
                            result["errors"]["slack_fallback"] = "failed"
        # mark idempotency only if at least one success
        if result["sent"]:
            self._mark_idempotent_sent(message_id)
        # emit observability summary
        if self.obs:
            try:
                self.obs.record_request(path="notify.dispatch", method="notify", status_code=200, latency_ms=0)
            except Exception:
                logger.exception("obs dispatch failed")
        return result

    def format_decision_message(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        # keep format consistent with your earlier format
        msg = {
            "title": f"🦓 Zebra قرر: {decision.get('title', decision.get('id', 'decision'))}",
            "text": decision.get("explanation", {}).get("natural_language", "") if isinstance(decision.get("explanation"), dict) else decision.get("explanation", ""),
            "fields": [
                {"title": "الثقة", "value": f"{decision.get('confidence', 0.0):.1%}"},
                {"title": "التأثير المتوقع", "value": json.dumps(decision.get("expected_impact", {}))},
                {"title": "المخاطر", "value": decision.get("risk_level", "unknown")}
            ],
        }
        # include actions if present
        if decision.get("actions"):
            msg["actions"] = decision["actions"]
        return msg