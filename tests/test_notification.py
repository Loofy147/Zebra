# tests/test_notification.py
import json
import sys
import os
import pytest
from types import SimpleNamespace

# Add src to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.zebra.integration.notification_system import (
    NotificationSystem,
    SlackNotifier,
    WebhookNotifier,
    compute_hmac_signature,
)

# monkeypatch requests.post and smtplib in relevant tests

def test_slack_notifier(monkeypatch):
    calls = {}
    def fake_post(url, json=None, timeout=None):
        calls['url'] = url
        class R:
            status_code = 200
            def raise_for_status(self): pass
        return R()
    monkeypatch.setattr("requests.post", fake_post)
    s = SlackNotifier(webhook_url="https://hooks.slack/test")
    ok = s.send({"title":"t","text":"hello","fields":[]})
    assert ok is True
    assert calls['url'] == "https://hooks.slack/test"

def test_webhook_signature_and_notify(monkeypatch):
    calls = {}
    def fake_post(url, data=None, headers=None, timeout=None):
        calls['url'] = url
        calls['data'] = data
        calls['headers'] = headers
        class R:
            status_code = 200
            def raise_for_status(self): pass
        return R()
    monkeypatch.setattr("requests.post", fake_post)
    secret = "topsecret"
    w = WebhookNotifier(endpoint="https://example.com/hook", secret=secret)
    msg = {"title":"t","text":"payload"}
    ok = w.send(msg)
    assert ok is True
    assert calls['url'] == "https://example.com/hook"
    # verify signature header format
    sig = calls['headers'].get("X-Zebra-Signature")
    assert sig.startswith("sha256=")
    # verify computed signature matches header
    computed = compute_hmac_signature(secret, json.dumps(msg).encode("utf-8"))
    assert sig == computed

def test_notification_system_idempotency(monkeypatch, tmp_path):
    # stub SlackNotifier to record calls
    class Dummy( SlackNotifier ):
        def __init__(self):
            # bypass parent init
            super().__init__(webhook_url="https://noop")
        def send(self, message, metadata=None):
            # pretend success
            return True
    ns = NotificationSystem(redis_client=None)  # use in-memory idempotency
    ns.register_channel("slack", Dummy())
    decision = {"id":"d1","title":"T","explanation":"ok","confidence":0.9,"expected_impact":{},"risk_level":"low"}
    res1 = ns.notify(decision, channels=["slack"])
    assert "slack" in res1["sent"]
    res2 = ns.notify(decision, channels=["slack"])
    assert res2.get("skipped", {}).get("reason") == "idempotent_already_sent"