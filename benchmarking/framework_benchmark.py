"""
Framework Benchmarking Tools

Provides comprehensive benchmarking capabilities for comparing AI frameworks
across different metrics like response time, accuracy, throughput, and resource usage.
"""

import asyncio
import json
import logging
import os
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil


class BenchmarkType(str, Enum):
    """Types of benchmarks that can be run."""

    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"
    STRESS_TEST = "stress_test"


class BenchmarkStatus(str, Enum):
    """Status of benchmark execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BenchmarkMetric:
    """Individual benchmark metric."""

    name: str
    value: float
    unit: str
    description: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class FrameworkBenchmarkResult:
    """Result of benchmarking a single framework."""

    framework_name: str
    benchmark_type: BenchmarkType
    status: BenchmarkStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    metrics: List[BenchmarkMetric] = None
    errors: List[str] = None
    raw_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = []
        if self.errors is None:
            self.errors = []
        if self.raw_data is None:
            self.raw_data = {}

    def add_metric(self, name: str, value: float, unit: str, description: str):
        """Add a metric to the result."""
        metric = BenchmarkMetric(name, value, unit, description)
        self.metrics.append(metric)

    def get_metric(self, name: str) -> Optional[BenchmarkMetric]:
        """Get a specific metric by name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "framework_name": self.framework_name,
            "benchmark_type": self.benchmark_type.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time_seconds": ((self.end_time or datetime.now()) - self.start_time).total_seconds(),
            "metrics": [asdict(m) for m in self.metrics],
            "errors": self.errors,
            "raw_data": self.raw_data,
        }


@dataclass
class BenchmarkComparison:
    """Comparison results between multiple frameworks."""

    benchmark_id: str
    benchmark_type: BenchmarkType
    frameworks: List[str]
    results: Dict[str, FrameworkBenchmarkResult]
    comparison_metrics: Dict[str, Any]
    winner: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_type": self.benchmark_type.value,
            "frameworks": self.frameworks,
            "results": {name: result.to_dict() for name, result in self.results.items()},
            "comparison_metrics": self.comparison_metrics,
            "winner": self.winner,
            "timestamp": self.timestamp.isoformat(),
        }


class FrameworkBenchmarker:
    """Main benchmarking class for AI frameworks."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_benchmarks: Dict[str, BenchmarkComparison] = {}

        # Standard test datasets
        self.test_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain neural networks in simple terms",
            "What are the benefits of deep learning?",
            "How is AI used in healthcare?",
            "What is natural language processing?",
            "Describe computer vision applications",
            "How do recommendation systems work?",
            "What is reinforcement learning?",
            "Explain the difference between AI and ML",
        ]

        self.test_documents = [
            {
                "filename": "ai_basics.txt",
                "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence.",
                "metadata": {"category": "education", "length": "short"},
            },
            {
                "filename": "ml_guide.txt",
                "content": "Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
                "metadata": {"category": "technical", "length": "medium"},
            },
            {
                "filename": "deep_learning.txt",
                "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs.",
                "metadata": {"category": "advanced", "length": "long"},
            },
        ]

    async def run_performance_benchmark(
        self, frameworks: Dict[str, Any], test_iterations: int = 10
    ) -> BenchmarkComparison:
        """Run performance benchmark comparing response times."""
        benchmark_id = str(uuid.uuid4())

        self.logger.info(f"Starting performance benchmark {benchmark_id} with {len(frameworks)} frameworks")

        benchmark = BenchmarkComparison(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.PERFORMANCE,
            frameworks=list(frameworks.keys()),
            results={},
            comparison_metrics={},
        )

        self.active_benchmarks[benchmark_id] = benchmark

        # Run benchmark for each framework
        for framework_name, framework_obj in frameworks.items():
            result = await self._benchmark_framework_performance(framework_name, framework_obj, test_iterations)
            benchmark.results[framework_name] = result

        # Calculate comparison metrics
        benchmark.comparison_metrics = self._calculate_performance_comparison(benchmark.results)
        benchmark.winner = self._determine_performance_winner(benchmark.results)

        self.logger.info(f"Performance benchmark {benchmark_id} completed. Winner: {benchmark.winner}")

        return benchmark

    async def _benchmark_framework_performance(
        self, framework_name: str, framework_obj: Any, iterations: int
    ) -> FrameworkBenchmarkResult:
        """Benchmark performance of a single framework."""
        result = FrameworkBenchmarkResult(
            framework_name=framework_name,
            benchmark_type=BenchmarkType.PERFORMANCE,
            status=BenchmarkStatus.RUNNING,
            start_time=datetime.now(),
        )

        try:
            response_times = []
            successful_queries = 0
            failed_queries = 0

            # Test RAG queries
            for i in range(iterations):
                query = self.test_queries[i % len(self.test_queries)]

                try:
                    start_time = time.time()
                    response = await framework_obj.execute_rag_query(query=query)
                    end_time = time.time()

                    response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                    response_times.append(response_time)

                    if response.get("success", False):
                        successful_queries += 1
                    else:
                        failed_queries += 1

                except Exception as e:
                    failed_queries += 1
                    result.errors.append(f"Query {i}: {str(e)}")

            # Calculate metrics
            if response_times:
                result.add_metric("avg_response_time", statistics.mean(response_times), "ms", "Average response time")
                result.add_metric("min_response_time", min(response_times), "ms", "Minimum response time")
                result.add_metric("max_response_time", max(response_times), "ms", "Maximum response time")
                result.add_metric(
                    "median_response_time", statistics.median(response_times), "ms", "Median response time"
                )
                if len(response_times) > 1:
                    result.add_metric(
                        "std_response_time",
                        statistics.stdev(response_times),
                        "ms",
                        "Standard deviation of response time",
                    )

            result.add_metric("success_rate", (successful_queries / iterations) * 100, "%", "Success rate")
            result.add_metric(
                "throughput",
                successful_queries / ((result.end_time or datetime.now()) - result.start_time).total_seconds(),
                "queries/sec",
                "Throughput",
            )

            result.raw_data = {
                "response_times": response_times,
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "total_iterations": iterations,
            }

            result.status = BenchmarkStatus.COMPLETED
            result.end_time = datetime.now()

        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.errors.append(f"Benchmark failed: {str(e)}")
            result.end_time = datetime.now()

        return result

    async def run_throughput_benchmark(
        self, frameworks: Dict[str, Any], duration_seconds: int = 60, concurrent_requests: int = 10
    ) -> BenchmarkComparison:
        """Run throughput benchmark testing concurrent request handling."""
        benchmark_id = str(uuid.uuid4())

        self.logger.info(
            f"Starting throughput benchmark {benchmark_id} for {duration_seconds}s with {concurrent_requests} concurrent requests"
        )

        benchmark = BenchmarkComparison(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.THROUGHPUT,
            frameworks=list(frameworks.keys()),
            results={},
            comparison_metrics={},
        )

        self.active_benchmarks[benchmark_id] = benchmark

        # Run benchmark for each framework
        for framework_name, framework_obj in frameworks.items():
            result = await self._benchmark_framework_throughput(
                framework_name, framework_obj, duration_seconds, concurrent_requests
            )
            benchmark.results[framework_name] = result

        # Calculate comparison metrics
        benchmark.comparison_metrics = self._calculate_throughput_comparison(benchmark.results)
        benchmark.winner = self._determine_throughput_winner(benchmark.results)

        self.logger.info(f"Throughput benchmark {benchmark_id} completed. Winner: {benchmark.winner}")

        return benchmark

    async def _benchmark_framework_throughput(
        self, framework_name: str, framework_obj: Any, duration_seconds: int, concurrent_requests: int
    ) -> FrameworkBenchmarkResult:
        """Benchmark throughput of a single framework."""
        result = FrameworkBenchmarkResult(
            framework_name=framework_name,
            benchmark_type=BenchmarkType.THROUGHPUT,
            status=BenchmarkStatus.RUNNING,
            start_time=datetime.now(),
        )

        try:
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(concurrent_requests)

            # Track metrics
            completed_requests = 0
            failed_requests = 0
            response_times = []

            async def make_request():
                nonlocal completed_requests, failed_requests

                async with semaphore:
                    query = self.test_queries[completed_requests % len(self.test_queries)]

                    try:
                        start_time = time.time()
                        response = await framework_obj.execute_rag_query(query=query)
                        end_time = time.time()

                        response_times.append((end_time - start_time) * 1000)

                        if response.get("success", False):
                            completed_requests += 1
                        else:
                            failed_requests += 1

                    except Exception as e:
                        failed_requests += 1
                        result.errors.append(f"Request failed: {str(e)}")

            # Run requests for specified duration
            end_time = time.time() + duration_seconds
            tasks = []

            while time.time() < end_time:
                # Launch concurrent requests
                for _ in range(concurrent_requests):
                    if time.time() >= end_time:
                        break
                    task = asyncio.create_task(make_request())
                    tasks.append(task)

                # Wait a bit before next batch
                await asyncio.sleep(0.1)

            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)

            # Calculate metrics
            total_requests = completed_requests + failed_requests
            actual_duration = (datetime.now() - result.start_time).total_seconds()

            result.add_metric(
                "requests_per_second", completed_requests / actual_duration, "req/s", "Requests per second"
            )
            result.add_metric("total_requests", total_requests, "count", "Total requests sent")
            result.add_metric("successful_requests", completed_requests, "count", "Successful requests")
            result.add_metric("failed_requests", failed_requests, "count", "Failed requests")
            result.add_metric(
                "success_rate",
                (completed_requests / total_requests) * 100 if total_requests > 0 else 0,
                "%",
                "Success rate",
            )

            if response_times:
                result.add_metric("avg_response_time", statistics.mean(response_times), "ms", "Average response time")
                result.add_metric(
                    "p95_response_time",
                    statistics.quantiles(response_times, n=20)[18],
                    "ms",
                    "95th percentile response time",
                )

            result.raw_data = {
                "completed_requests": completed_requests,
                "failed_requests": failed_requests,
                "response_times": response_times,
                "duration_seconds": actual_duration,
                "concurrent_requests": concurrent_requests,
            }

            result.status = BenchmarkStatus.COMPLETED
            result.end_time = datetime.now()

        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.errors.append(f"Throughput benchmark failed: {str(e)}")
            result.end_time = datetime.now()

        return result

    async def run_resource_usage_benchmark(
        self, frameworks: Dict[str, Any], test_duration: int = 30
    ) -> BenchmarkComparison:
        """Run resource usage benchmark monitoring CPU and memory."""
        benchmark_id = str(uuid.uuid4())

        self.logger.info(f"Starting resource usage benchmark {benchmark_id} for {test_duration}s")

        benchmark = BenchmarkComparison(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.RESOURCE_USAGE,
            frameworks=list(frameworks.keys()),
            results={},
            comparison_metrics={},
        )

        self.active_benchmarks[benchmark_id] = benchmark

        # Run benchmark for each framework
        for framework_name, framework_obj in frameworks.items():
            result = await self._benchmark_framework_resources(framework_name, framework_obj, test_duration)
            benchmark.results[framework_name] = result

        # Calculate comparison metrics
        benchmark.comparison_metrics = self._calculate_resource_comparison(benchmark.results)
        benchmark.winner = self._determine_resource_winner(benchmark.results)

        self.logger.info(f"Resource usage benchmark {benchmark_id} completed. Winner: {benchmark.winner}")

        return benchmark

    async def _benchmark_framework_resources(
        self, framework_name: str, framework_obj: Any, duration: int
    ) -> FrameworkBenchmarkResult:
        """Benchmark resource usage of a single framework."""
        result = FrameworkBenchmarkResult(
            framework_name=framework_name,
            benchmark_type=BenchmarkType.RESOURCE_USAGE,
            status=BenchmarkStatus.RUNNING,
            start_time=datetime.now(),
        )

        try:
            # Get initial resource usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            cpu_samples = []
            memory_samples = []

            # Monitor resources while running queries
            end_time = time.time() + duration
            query_count = 0

            while time.time() < end_time:
                # Make a query
                query = self.test_queries[query_count % len(self.test_queries)]

                try:
                    await framework_obj.execute_rag_query(query=query)
                    query_count += 1
                except Exception as e:
                    result.errors.append(f"Query {query_count} failed: {str(e)}")

                # Sample resource usage
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024

                cpu_samples.append(cpu_percent)
                memory_samples.append(memory_mb)

                await asyncio.sleep(1)  # Sample every second

            # Calculate metrics
            if cpu_samples:
                result.add_metric("avg_cpu_usage", statistics.mean(cpu_samples), "%", "Average CPU usage")
                result.add_metric("max_cpu_usage", max(cpu_samples), "%", "Maximum CPU usage")

            if memory_samples:
                result.add_metric("avg_memory_usage", statistics.mean(memory_samples), "MB", "Average memory usage")
                result.add_metric("max_memory_usage", max(memory_samples), "MB", "Maximum memory usage")
                result.add_metric(
                    "memory_increase", max(memory_samples) - initial_memory, "MB", "Memory increase during test"
                )

            result.add_metric("queries_processed", query_count, "count", "Queries processed during test")

            result.raw_data = {
                "cpu_samples": cpu_samples,
                "memory_samples": memory_samples,
                "initial_memory": initial_memory,
                "queries_processed": query_count,
                "duration_seconds": duration,
            }

            result.status = BenchmarkStatus.COMPLETED
            result.end_time = datetime.now()

        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.errors.append(f"Resource benchmark failed: {str(e)}")
            result.end_time = datetime.now()

        return result

    def _calculate_performance_comparison(self, results: Dict[str, FrameworkBenchmarkResult]) -> Dict[str, Any]:
        """Calculate performance comparison metrics."""
        comparison = {"avg_response_times": {}, "success_rates": {}, "throughput": {}}

        for name, result in results.items():
            if result.status == BenchmarkStatus.COMPLETED:
                avg_time = result.get_metric("avg_response_time")
                success_rate = result.get_metric("success_rate")
                throughput = result.get_metric("throughput")

                if avg_time:
                    comparison["avg_response_times"][name] = avg_time.value
                if success_rate:
                    comparison["success_rates"][name] = success_rate.value
                if throughput:
                    comparison["throughput"][name] = throughput.value

        return comparison

    def _determine_performance_winner(self, results: Dict[str, FrameworkBenchmarkResult]) -> Optional[str]:
        """Determine performance benchmark winner based on avg response time and success rate."""
        best_score = -1
        winner = None

        for name, result in results.items():
            if result.status == BenchmarkStatus.COMPLETED:
                avg_time = result.get_metric("avg_response_time")
                success_rate = result.get_metric("success_rate")

                if avg_time and success_rate and success_rate.value > 80:  # Must have >80% success rate
                    # Lower response time is better, so invert it for scoring
                    score = (success_rate.value / 100) * (1000 / avg_time.value)

                    if score > best_score:
                        best_score = score
                        winner = name

        return winner

    def _calculate_throughput_comparison(self, results: Dict[str, FrameworkBenchmarkResult]) -> Dict[str, Any]:
        """Calculate throughput comparison metrics."""
        comparison = {"requests_per_second": {}, "success_rates": {}, "avg_response_times": {}}

        for name, result in results.items():
            if result.status == BenchmarkStatus.COMPLETED:
                rps = result.get_metric("requests_per_second")
                success_rate = result.get_metric("success_rate")
                avg_time = result.get_metric("avg_response_time")

                if rps:
                    comparison["requests_per_second"][name] = rps.value
                if success_rate:
                    comparison["success_rates"][name] = success_rate.value
                if avg_time:
                    comparison["avg_response_times"][name] = avg_time.value

        return comparison

    def _determine_throughput_winner(self, results: Dict[str, FrameworkBenchmarkResult]) -> Optional[str]:
        """Determine throughput benchmark winner."""
        best_rps = 0
        winner = None

        for name, result in results.items():
            if result.status == BenchmarkStatus.COMPLETED:
                rps = result.get_metric("requests_per_second")
                success_rate = result.get_metric("success_rate")

                if rps and success_rate and success_rate.value > 80:
                    if rps.value > best_rps:
                        best_rps = rps.value
                        winner = name

        return winner

    def _calculate_resource_comparison(self, results: Dict[str, FrameworkBenchmarkResult]) -> Dict[str, Any]:
        """Calculate resource usage comparison metrics."""
        comparison = {"avg_cpu_usage": {}, "avg_memory_usage": {}, "memory_increase": {}}

        for name, result in results.items():
            if result.status == BenchmarkStatus.COMPLETED:
                cpu = result.get_metric("avg_cpu_usage")
                memory = result.get_metric("avg_memory_usage")
                memory_inc = result.get_metric("memory_increase")

                if cpu:
                    comparison["avg_cpu_usage"][name] = cpu.value
                if memory:
                    comparison["avg_memory_usage"][name] = memory.value
                if memory_inc:
                    comparison["memory_increase"][name] = memory_inc.value

        return comparison

    def _determine_resource_winner(self, results: Dict[str, FrameworkBenchmarkResult]) -> Optional[str]:
        """Determine resource usage benchmark winner (lowest resource usage)."""
        best_score = float('inf')
        winner = None

        for name, result in results.items():
            if result.status == BenchmarkStatus.COMPLETED:
                cpu = result.get_metric("avg_cpu_usage")
                memory = result.get_metric("avg_memory_usage")

                if cpu and memory:
                    # Lower is better for resource usage
                    score = cpu.value + (memory.value / 100)  # Normalize memory

                    if score < best_score:
                        best_score = score
                        winner = name

        return winner

    def get_benchmark_results(self, benchmark_id: str) -> Optional[BenchmarkComparison]:
        """Get benchmark results by ID."""
        return self.active_benchmarks.get(benchmark_id)

    def list_benchmarks(self) -> List[Dict[str, Any]]:
        """List all benchmark results."""
        return [benchmark.to_dict() for benchmark in self.active_benchmarks.values()]

    def export_benchmark_results(self, benchmark_id: str = None) -> str:
        """Export benchmark results to JSON."""
        if benchmark_id:
            benchmark = self.active_benchmarks.get(benchmark_id)
            if benchmark:
                return json.dumps(benchmark.to_dict(), indent=2, default=str)
            else:
                raise ValueError(f"Benchmark {benchmark_id} not found")
        else:
            # Export all benchmarks
            all_benchmarks = [b.to_dict() for b in self.active_benchmarks.values()]
            return json.dumps(all_benchmarks, indent=2, default=str)


# Global benchmarker instance
framework_benchmarker = FrameworkBenchmarker()
