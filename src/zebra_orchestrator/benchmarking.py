import logging
import time
import numpy as np
import torch
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    duration_ms: float
    throughput: float
    memory_usage_mb: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict] = None


class PerformanceBenchmark:
    """
    Comprehensive benchmarking suite for the Zebra system.
    Measures performance, accuracy, and resource usage of all components.
    """

    def __init__(self):
        self.results = []
        self.start_time = None
        logging.info("Performance Benchmark suite initialized")

    def _measure_time(self, func: Callable, *args, **kwargs) -> tuple:
        """Measure execution time of a function."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000
        return result, duration

    def _measure_memory(self) -> float:
        """Measure current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def benchmark_causal_inference(self, num_iterations: int = 100) -> BenchmarkResult:
        """
        Benchmark the causal inference engine.

        Args:
            num_iterations: Number of inference calls to make

        Returns:
            BenchmarkResult with performance metrics
        """
        logging.info(f"Benchmarking causal inference ({num_iterations} iterations)...")

        try:
            from .deep_learning_causal_engine import deep_cie_engine

            test_data = {
                'latency_p50': 50.0,
                'latency_p95': 100.0,
                'latency_p99': 150.0,
                'request_rate': 100.0,
                'error_rate': 0.01,
                'cpu_usage': 0.3,
                'memory_usage': 0.5
            }

            total_duration = 0.0
            successes = 0

            for _ in range(num_iterations):
                _, duration = self._measure_time(
                    deep_cie_engine.analyze_anomaly_deep,
                    test_data,
                    "performance_test"
                )
                total_duration += duration
                successes += 1

            avg_duration = total_duration / num_iterations
            throughput = 1000.0 / avg_duration if avg_duration > 0 else 0
            memory = self._measure_memory()

            result = BenchmarkResult(
                name="causal_inference",
                duration_ms=avg_duration,
                throughput=throughput,
                memory_usage_mb=memory,
                success=True,
                metadata={'iterations': num_iterations, 'successes': successes}
            )

            logging.info(f"Causal inference: {avg_duration:.2f}ms avg, {throughput:.2f} ops/s")
            return result

        except Exception as e:
            logging.error(f"Causal inference benchmark failed: {e}")
            return BenchmarkResult(
                name="causal_inference",
                duration_ms=0,
                throughput=0,
                memory_usage_mb=0,
                success=False,
                error=str(e)
            )

    def benchmark_anomaly_detection(self, num_iterations: int = 100) -> BenchmarkResult:
        """Benchmark the anomaly detection system."""
        logging.info(f"Benchmarking anomaly detection ({num_iterations} iterations)...")

        try:
            from .anomaly_detector import anomaly_detector

            test_data = {
                'latency_p50': 50.0 + np.random.normal(0, 5),
                'latency_p95': 100.0 + np.random.normal(0, 10),
                'latency_p99': 150.0 + np.random.normal(0, 15),
                'request_rate': 100.0 + np.random.normal(0, 10),
                'error_rate': 0.01 + np.random.normal(0, 0.001),
                'cpu_usage': 0.3 + np.random.normal(0, 0.05),
                'memory_usage': 0.5 + np.random.normal(0, 0.05)
            }

            for _ in range(10):
                anomaly_detector.detect_anomaly(test_data)

            total_duration = 0.0
            anomalies_detected = 0

            for _ in range(num_iterations):
                result, duration = self._measure_time(
                    anomaly_detector.detect_anomaly,
                    test_data
                )
                total_duration += duration
                if result.get('is_anomaly', False):
                    anomalies_detected += 1

            avg_duration = total_duration / num_iterations
            throughput = 1000.0 / avg_duration if avg_duration > 0 else 0
            memory = self._measure_memory()

            benchmark_result = BenchmarkResult(
                name="anomaly_detection",
                duration_ms=avg_duration,
                throughput=throughput,
                memory_usage_mb=memory,
                success=True,
                metadata={
                    'iterations': num_iterations,
                    'anomalies_detected': anomalies_detected,
                    'detection_rate': anomalies_detected / num_iterations
                }
            )

            logging.info(f"Anomaly detection: {avg_duration:.2f}ms avg, {throughput:.2f} ops/s")
            return benchmark_result

        except Exception as e:
            logging.error(f"Anomaly detection benchmark failed: {e}")
            return BenchmarkResult(
                name="anomaly_detection",
                duration_ms=0,
                throughput=0,
                memory_usage_mb=0,
                success=False,
                error=str(e)
            )

    def benchmark_rl_agent(self, num_iterations: int = 50) -> BenchmarkResult:
        """Benchmark the reinforcement learning agent."""
        logging.info(f"Benchmarking RL agent ({num_iterations} iterations)...")

        try:
            from .reinforcement_learning import rl_agent

            test_state = {
                'latency_p50': 50.0,
                'latency_p95': 100.0,
                'latency_p99': 150.0,
                'request_rate': 100.0,
                'error_rate': 0.01,
                'cpu_usage': 0.3,
                'memory_usage': 0.5
            }

            total_duration = 0.0

            for _ in range(num_iterations):
                _, duration = self._measure_time(
                    rl_agent.recommend_intervention,
                    test_state
                )
                total_duration += duration

            avg_duration = total_duration / num_iterations
            throughput = 1000.0 / avg_duration if avg_duration > 0 else 0
            memory = self._measure_memory()

            result = BenchmarkResult(
                name="rl_agent",
                duration_ms=avg_duration,
                throughput=throughput,
                memory_usage_mb=memory,
                success=True,
                metadata={'iterations': num_iterations}
            )

            logging.info(f"RL Agent: {avg_duration:.2f}ms avg, {throughput:.2f} ops/s")
            return result

        except Exception as e:
            logging.error(f"RL agent benchmark failed: {e}")
            return BenchmarkResult(
                name="rl_agent",
                duration_ms=0,
                throughput=0,
                memory_usage_mb=0,
                success=False,
                error=str(e)
            )

    def benchmark_code_analysis(self, num_iterations: int = 20) -> BenchmarkResult:
        """Benchmark the code analysis system."""
        logging.info(f"Benchmarking code analysis ({num_iterations} iterations)...")

        try:
            from .code_understanding import code_analyzer

            test_code = """
def calculate_metrics(data):
    result = 0
    for item in data:
        result += item * 2
    return result
"""

            total_duration = 0.0

            for _ in range(num_iterations):
                _, duration = self._measure_time(
                    code_analyzer.analyze_bottleneck,
                    test_code,
                    "performance_test"
                )
                total_duration += duration

            avg_duration = total_duration / num_iterations
            throughput = 1000.0 / avg_duration if avg_duration > 0 else 0
            memory = self._measure_memory()

            result = BenchmarkResult(
                name="code_analysis",
                duration_ms=avg_duration,
                throughput=throughput,
                memory_usage_mb=memory,
                success=True,
                metadata={'iterations': num_iterations}
            )

            logging.info(f"Code Analysis: {avg_duration:.2f}ms avg")
            return result

        except Exception as e:
            logging.error(f"Code analysis benchmark failed: {e}")
            return BenchmarkResult(
                name="code_analysis",
                duration_ms=0,
                throughput=0,
                memory_usage_mb=0,
                success=False,
                error=str(e)
            )

    def benchmark_end_to_end(self, num_iterations: int = 10) -> BenchmarkResult:
        """Benchmark complete end-to-end pipeline."""
        logging.info(f"Benchmarking end-to-end pipeline ({num_iterations} iterations)...")

        try:
            from .anomaly_detector import anomaly_detector
            from .deep_learning_causal_engine import deep_cie_engine
            from .reinforcement_learning import rl_agent

            test_data = {
                'latency_p50': 50.0,
                'latency_p95': 100.0,
                'latency_p99': 150.0,
                'request_rate': 100.0,
                'error_rate': 0.01,
                'cpu_usage': 0.3,
                'memory_usage': 0.5
            }

            def run_pipeline(data):
                anomaly_result = anomaly_detector.detect_anomaly(data)
                if anomaly_result['is_anomaly']:
                    causal_result = deep_cie_engine.analyze_anomaly_deep(
                        data,
                        "e2e_test"
                    )
                    recommendation = rl_agent.recommend_intervention(data)
                    return {'anomaly': anomaly_result, 'causal': causal_result, 'recommendation': recommendation}
                return None

            total_duration = 0.0
            pipeline_executions = 0

            for _ in range(num_iterations):
                _, duration = self._measure_time(run_pipeline, test_data)
                total_duration += duration
                pipeline_executions += 1

            avg_duration = total_duration / num_iterations
            throughput = 1000.0 / avg_duration if avg_duration > 0 else 0
            memory = self._measure_memory()

            result = BenchmarkResult(
                name="end_to_end_pipeline",
                duration_ms=avg_duration,
                throughput=throughput,
                memory_usage_mb=memory,
                success=True,
                metadata={'iterations': num_iterations, 'executions': pipeline_executions}
            )

            logging.info(f"End-to-end: {avg_duration:.2f}ms avg")
            return result

        except Exception as e:
            logging.error(f"End-to-end benchmark failed: {e}")
            return BenchmarkResult(
                name="end_to_end_pipeline",
                duration_ms=0,
                throughput=0,
                memory_usage_mb=0,
                success=False,
                error=str(e)
            )

    def run_all_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks and return comprehensive results."""
        logging.info("=" * 60)
        logging.info("STARTING COMPREHENSIVE BENCHMARK SUITE")
        logging.info("=" * 60)

        self.start_time = datetime.utcnow()

        benchmarks = {
            'causal_inference': lambda: self.benchmark_causal_inference(100),
            'anomaly_detection': lambda: self.benchmark_anomaly_detection(100),
            'rl_agent': lambda: self.benchmark_rl_agent(50),
            'code_analysis': lambda: self.benchmark_code_analysis(20),
            'end_to_end': lambda: self.benchmark_end_to_end(10)
        }

        results = {}
        for name, benchmark_func in benchmarks.items():
            logging.info(f"\nRunning {name} benchmark...")
            result = benchmark_func()
            results[name] = result
            self.results.append(result)

        end_time = datetime.utcnow()
        total_duration = (end_time - self.start_time).total_seconds()

        logging.info("\n" + "=" * 60)
        logging.info("BENCHMARK RESULTS SUMMARY")
        logging.info("=" * 60)

        for name, result in results.items():
            if result.success:
                logging.info(
                    f"{name:25s}: {result.duration_ms:8.2f}ms | "
                    f"{result.throughput:8.2f} ops/s | "
                    f"{result.memory_usage_mb:6.2f} MB"
                )
            else:
                logging.error(f"{name:25s}: FAILED - {result.error}")

        logging.info(f"\nTotal benchmark duration: {total_duration:.2f}s")
        logging.info("=" * 60)

        return results

    def export_results(self, filepath: str):
        """Export benchmark results to JSON file."""
        export_data = {
            'benchmark_run_time': self.start_time.isoformat() if self.start_time else None,
            'results': [
                {
                    'name': r.name,
                    'duration_ms': r.duration_ms,
                    'throughput': r.throughput,
                    'memory_usage_mb': r.memory_usage_mb,
                    'success': r.success,
                    'error': r.error,
                    'metadata': r.metadata
                }
                for r in self.results
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logging.info(f"Benchmark results exported to {filepath}")


benchmark_suite = PerformanceBenchmark()