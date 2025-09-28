import time
import psutil
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from threading import Lock

from app.models.responses import MetricData, MetricsResponse
from app.services.logger_service import get_logger_service


logger_service = get_logger_service()


@dataclass
class RequestMetric:
    """Individual request metric data."""
    timestamp: datetime
    method: str
    endpoint: str
    status_code: int
    duration_ms: float
    user_id: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class SystemMetric:
    """System performance metric."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float


class MetricsService:
    """Service for collecting and managing application metrics."""

    def __init__(self, max_metrics_history: int = 10000):
        self.max_metrics_history = max_metrics_history
        self.start_time = datetime.utcnow()

        # Thread-safe storage
        self._lock = Lock()

        # Request metrics storage
        self.request_metrics: deque = deque(maxlen=max_metrics_history)

        # System metrics storage
        self.system_metrics: deque = deque(maxlen=1000)  # Keep last 1000 system metrics

        # Real-time counters
        self.request_count: Dict[str, int] = defaultdict(int)
        self.error_count: Dict[str, int] = defaultdict(int)
        self.endpoint_counts: Dict[str, int] = defaultdict(int)
        self.status_code_counts: Dict[int, int] = defaultdict(int)

        # Performance tracking
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Will start system metrics collection when event loop is available
        self._background_task = None

    def start_background_tasks(self):
        """Start background task for system metrics collection."""
        if self._background_task is None:
            try:
                self._background_task = asyncio.create_task(self._collect_system_metrics_loop())
            except RuntimeError:
                # No event loop running, will try again later
                pass

    async def _collect_system_metrics_loop(self):
        """Background loop to collect system metrics."""
        while True:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(60)  # Collect every minute
            except Exception as e:
                logger_service.log_error(e, {"operation": "system_metrics_collection"})
                await asyncio.sleep(60)

    async def _collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            metric = SystemMetric(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent
            )

            with self._lock:
                self.system_metrics.append(metric)

        except Exception as e:
            logger_service.log_error(e, {"operation": "collect_system_metrics"})

    def record_request(self, method: str, endpoint: str, status_code: int,
                      duration_ms: float, user_id: Optional[str] = None,
                      error_type: Optional[str] = None):
        """Record a request metric."""
        try:
            metric = RequestMetric(
                timestamp=datetime.utcnow(),
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration_ms=duration_ms,
                user_id=user_id,
                error_type=error_type
            )

            with self._lock:
                # Store detailed metric
                self.request_metrics.append(metric)

                # Update counters
                self.request_count[f"{method} {endpoint}"] += 1
                self.endpoint_counts[endpoint] += 1
                self.status_code_counts[status_code] += 1

                # Track errors
                if status_code >= 400:
                    error_key = f"{status_code}_{error_type}" if error_type else str(status_code)
                    self.error_count[error_key] += 1

                # Track response times
                self.response_times[endpoint].append(duration_ms)

            # Log performance warning for slow requests
            if duration_ms > 5000:  # 5 seconds
                logger_service.log_performance(
                    operation=f"{method} {endpoint}",
                    duration_ms=duration_ms,
                    status_code=status_code,
                    user_id=user_id
                )

        except Exception as e:
            logger_service.log_error(e, {"operation": "record_request_metric"})

    def get_uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        return (datetime.utcnow() - self.start_time).total_seconds()

    def get_request_rate(self, time_window_minutes: int = 5) -> float:
        """Get requests per minute for the specified time window."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)

            with self._lock:
                recent_requests = [
                    m for m in self.request_metrics
                    if m.timestamp >= cutoff_time
                ]

            return len(recent_requests) / time_window_minutes if recent_requests else 0.0

        except Exception as e:
            logger_service.log_error(e, {"operation": "get_request_rate"})
            return 0.0

    def get_error_rate(self, time_window_minutes: int = 5) -> float:
        """Get error rate percentage for the specified time window."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)

            with self._lock:
                recent_requests = [
                    m for m in self.request_metrics
                    if m.timestamp >= cutoff_time
                ]

            if not recent_requests:
                return 0.0

            error_requests = [m for m in recent_requests if m.status_code >= 400]
            return (len(error_requests) / len(recent_requests)) * 100

        except Exception as e:
            logger_service.log_error(e, {"operation": "get_error_rate"})
            return 0.0

    def get_average_response_time(self, endpoint: Optional[str] = None,
                                 time_window_minutes: int = 5) -> float:
        """Get average response time for endpoint or all endpoints."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)

            with self._lock:
                recent_requests = [
                    m for m in self.request_metrics
                    if m.timestamp >= cutoff_time and (endpoint is None or m.endpoint == endpoint)
                ]

            if not recent_requests:
                return 0.0

            total_time = sum(m.duration_ms for m in recent_requests)
            return total_time / len(recent_requests)

        except Exception as e:
            logger_service.log_error(e, {"operation": "get_average_response_time"})
            return 0.0

    def get_top_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top endpoints by request count."""
        try:
            with self._lock:
                sorted_endpoints = sorted(
                    self.endpoint_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

            return [
                {"endpoint": endpoint, "count": count}
                for endpoint, count in sorted_endpoints[:limit]
            ]

        except Exception as e:
            logger_service.log_error(e, {"operation": "get_top_endpoints"})
            return []

    def get_status_code_distribution(self) -> Dict[str, int]:
        """Get distribution of HTTP status codes."""
        with self._lock:
            return dict(self.status_code_counts)

    def get_current_system_metrics(self) -> Optional[SystemMetric]:
        """Get the most recent system metrics."""
        try:
            with self._lock:
                return self.system_metrics[-1] if self.system_metrics else None
        except Exception as e:
            logger_service.log_error(e, {"operation": "get_current_system_metrics"})
            return None

    def get_metrics_summary(self, time_window_minutes: int = 5) -> MetricsResponse:
        """Get comprehensive metrics summary."""
        try:
            current_time = datetime.utcnow()

            # Basic metrics
            metrics_data = [
                MetricData(
                    name="uptime_seconds",
                    value=self.get_uptime_seconds(),
                    unit="seconds",
                    timestamp=current_time
                ),
                MetricData(
                    name="total_requests",
                    value=len(self.request_metrics),
                    unit="count",
                    timestamp=current_time
                ),
                MetricData(
                    name="request_rate",
                    value=round(self.get_request_rate(time_window_minutes), 2),
                    unit="requests/minute",
                    timestamp=current_time
                ),
                MetricData(
                    name="error_rate",
                    value=round(self.get_error_rate(time_window_minutes), 2),
                    unit="percentage",
                    timestamp=current_time
                ),
                MetricData(
                    name="average_response_time",
                    value=round(self.get_average_response_time(time_window_minutes=time_window_minutes), 2),
                    unit="milliseconds",
                    timestamp=current_time
                )
            ]

            # System metrics
            system_metric = self.get_current_system_metrics()
            if system_metric:
                metrics_data.extend([
                    MetricData(
                        name="cpu_usage",
                        value=round(system_metric.cpu_percent, 2),
                        unit="percentage",
                        timestamp=system_metric.timestamp
                    ),
                    MetricData(
                        name="memory_usage",
                        value=round(system_metric.memory_percent, 2),
                        unit="percentage",
                        timestamp=system_metric.timestamp
                    ),
                    MetricData(
                        name="memory_used",
                        value=round(system_metric.memory_used_mb, 2),
                        unit="MB",
                        timestamp=system_metric.timestamp
                    )
                ])

            # Status code distribution
            status_codes = self.get_status_code_distribution()
            for code, count in status_codes.items():
                metrics_data.append(
                    MetricData(
                        name=f"status_code_{code}",
                        value=count,
                        unit="count",
                        timestamp=current_time
                    )
                )

            return MetricsResponse(
                metrics=metrics_data,
                time_range={
                    "start": current_time - timedelta(minutes=time_window_minutes),
                    "end": current_time
                },
                summary={
                    "total_requests": len(self.request_metrics),
                    "error_rate": round(self.get_error_rate(time_window_minutes), 2),
                    "uptime_hours": round(self.get_uptime_seconds() / 3600, 2),
                    "top_endpoints": self.get_top_endpoints(5)
                }
            )

        except Exception as e:
            logger_service.log_error(e, {"operation": "get_metrics_summary"})
            # Return empty metrics on error
            return MetricsResponse(
                metrics=[],
                time_range={"start": current_time, "end": current_time},
                summary={}
            )

    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self.request_metrics.clear()
            self.system_metrics.clear()
            self.request_count.clear()
            self.error_count.clear()
            self.endpoint_counts.clear()
            self.status_code_counts.clear()
            self.response_times.clear()
            self.start_time = datetime.utcnow()


# Global metrics service instance
metrics_service = MetricsService()


def get_metrics_service() -> MetricsService:
    """Get the global metrics service instance."""
    return metrics_service