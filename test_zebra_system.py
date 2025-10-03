#!/usr/bin/env python3
"""
Comprehensive test suite for the Zebra Self-Governing AI System.
Tests all major components and their integration.
"""

import sys
import logging
import pytest
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def test_imports():
    """Test that all modules can be imported."""
    logging.info("Testing module imports...")
    try:
        from src.zebra_orchestrator import causal_engine
        from src.zebra_orchestrator import shadow_llm
        from src.zebra_orchestrator import pr_bot
        from src.zebra_orchestrator import observer
        logging.info("  ✓ Classic modules imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import classic modules: {e}")

    try:
        from src.zebra_orchestrator import deep_learning_causal_engine
        from src.zebra_orchestrator import anomaly_detector
        from src.zebra_orchestrator import reinforcement_learning
        from src.zebra_orchestrator import code_understanding
        from src.zebra_orchestrator import supabase_storage
        from src.zebra_orchestrator import continuous_learning
        from src.zebra_orchestrator import benchmarking
        logging.info("  ✓ Deep learning modules imported successfully")
    except Exception as e:
        pytest.fail(f"Failed to import deep learning modules: {e}")


def test_causal_inference():
    """Test causal inference engine."""
    logging.info("Testing causal inference engine...")
    try:
        from src.zebra_orchestrator.causal_engine import CIE
        result = CIE.analyze_anomaly("high_roll_rate")
        assert 'root_cause' in result
        assert 'confidence' in result
        logging.info(f"  ✓ Classic CIE analysis: {result['root_cause']}")
    except Exception as e:
        pytest.fail(f"Classic CIE test failed: {e}")

    try:
        from src.zebra_orchestrator.deep_learning_causal_engine import deep_cie_engine
        test_data = {
            'latency_p50': 50.0, 'latency_p95': 100.0, 'latency_p99': 150.0,
            'request_rate': 100.0, 'error_rate': 0.01, 'cpu_usage': 0.3, 'memory_usage': 0.5
        }
        result = deep_cie_engine.analyze_anomaly_deep(test_data, "test_anomaly")
        assert 'root_cause' in result
        assert 'confidence' in result
        logging.info(f"  ✓ Deep CIE analysis: {result['root_cause']} (confidence: {result['confidence']:.3f})")
    except Exception as e:
        pytest.fail(f"Deep CIE test failed: {e}")


def test_anomaly_detection():
    """Test anomaly detection system."""
    logging.info("Testing anomaly detection...")
    try:
        from src.zebra_orchestrator.anomaly_detector import anomaly_detector
        test_data = {
            'latency_p50': 50.0, 'latency_p95': 100.0, 'latency_p99': 150.0,
            'request_rate': 100.0, 'error_rate': 0.01, 'cpu_usage': 0.3, 'memory_usage': 0.5
        }
        for _ in range(15):
            anomaly_detector.detect_anomaly(test_data)
        result = anomaly_detector.detect_anomaly(test_data)
        assert 'is_anomaly' in result
        assert 'anomaly_score' in result
        logging.info(f"  ✓ Anomaly detection result: score={result['anomaly_score']:.3f}")
    except Exception as e:
        pytest.fail(f"Anomaly detection test failed: {e}")


def test_reinforcement_learning():
    """Test RL agent."""
    logging.info("Testing reinforcement learning agent...")
    try:
        from src.zebra_orchestrator.reinforcement_learning import rl_agent, InterventionEnvironment
        test_state = {
            'latency_p50': 50.0, 'latency_p95': 100.0, 'latency_p99': 150.0,
            'request_rate': 100.0, 'error_rate': 0.01, 'cpu_usage': 0.3, 'memory_usage': 0.5
        }
        recommendation = rl_agent.recommend_intervention(test_state)
        assert 'recommended_action' in recommendation
        assert 'confidence' in recommendation
        logging.info(f"  ✓ RL recommendation: {recommendation['recommended_action']} (confidence: {recommendation['confidence']:.3f})")
        env = InterventionEnvironment()
        episode_result = rl_agent.train_episode(env)
        assert 'reward' in episode_result
        logging.info(f"  ✓ RL training episode: reward={episode_result['reward']:.2f}")
    except Exception as e:
        pytest.fail(f"RL agent test failed: {e}")


def test_code_understanding():
    """Test code analysis system."""
    logging.info("Testing code understanding...")
    try:
        from src.zebra_orchestrator.code_understanding import code_analyzer
        test_code = "def f(x): return x"
        analysis = code_analyzer.analyze_bottleneck(test_code, "performance_issue")
        assert 'quality_metrics' in analysis
        logging.info(f"  ✓ Code analysis completed")
        features = code_analyzer.extract_code_features(test_code)
        assert 'quality_score' in features
        logging.info(f"  ✓ Feature extraction: quality={features['quality_score']:.3f}")
    except Exception as e:
        pytest.fail(f"Code understanding test failed: {e}")


def test_continuous_learning():
    """Test continuous learning pipeline."""
    logging.info("Testing continuous learning pipeline...")
    try:
        from src.zebra_orchestrator.continuous_learning import continuous_learning
        experience = {
            'state': {'latency': 50.0}, 'action': {'type': 'optimize', 'index': 1},
            'reward': 10.0, 'next_state': {'latency': 40.0}
        }
        continuous_learning.experience_collector.add_experience(experience)
        stats = continuous_learning.get_learning_statistics()
        assert 'total_experiences' in stats
        logging.info(f"  ✓ Continuous learning: {stats['total_experiences']} experiences")
    except Exception as e:
        pytest.fail(f"Continuous learning test failed: {e}")


def test_storage():
    """Test Supabase storage integration."""
    logging.info("Testing storage integration...")
    try:
        from src.zebra_orchestrator.supabase_storage import supabase_storage
        logging.info("  ✓ Supabase storage client initialized")
    except Exception as e:
        pytest.fail(f"Storage test failed: {e}")


def test_integration():
    """Test end-to-end integration."""
    logging.info("Testing end-to-end integration...")
    try:
        from src.zebra_orchestrator.anomaly_detector import anomaly_detector
        from src.zebra_orchestrator.deep_learning_causal_engine import deep_cie_engine
        from src.zebra_orchestrator.reinforcement_learning import rl_agent
        test_data = {
            'latency_p50': 80.0, 'latency_p95': 150.0, 'latency_p99': 200.0,
            'request_rate': 120.0, 'error_rate': 0.02, 'cpu_usage': 0.6, 'memory_usage': 0.7
        }
        for _ in range(15):
            anomaly_detector.detect_anomaly(test_data)
        anomaly_result = anomaly_detector.detect_anomaly(test_data)
        logging.info(f"  → Anomaly detection: {anomaly_result['is_anomaly']}")
        if anomaly_result['is_anomaly']:
            causal_result = deep_cie_engine.analyze_anomaly_deep(test_data, "integration_test")
            logging.info(f"  → Causal analysis: {causal_result['root_cause']}")
            recommendation = rl_agent.recommend_intervention(test_data)
            logging.info(f"  → RL recommendation: {recommendation['recommended_action']}")
        logging.info("  ✓ End-to-end integration successful")
    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")