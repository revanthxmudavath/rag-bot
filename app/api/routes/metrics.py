from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import time
import psutil
import os
from datetime import datetime, timezone
from app.services.metrics_service import get_metrics_service
from app.services.rag_service import get_rag_service

router = APIRouter()
metrics_service = get_metrics_service()
rag_service = get_rag_service()


@router.get("/", response_model=Dict[str, Any])
async def get_system_metrics():
    """
    Get comprehensive system and application metrics.

    Returns:
        System metrics including performance, RAG service stats, and health
    """
    try:
        start_time = time.time()

        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Get process info
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info()

        # Get RAG service stats
        try:
            rag_stats = await rag_service.get_service_stats()
            rag_health = await rag_service.health_check()
        except Exception as e:
            rag_stats = {"error": str(e)}
            rag_health = {"overall": False, "error": str(e)}

        # Get application metrics from metrics service
        app_metrics = metrics_service.get_metrics()

        processing_time = (time.time() - start_time) * 1000

        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(processing_time, 2),
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent": round((disk.used / disk.total) * 100, 1)
                },
                "process": {
                    "memory_mb": round(process_memory.rss / (1024**2), 2),
                    "threads": process.num_threads(),
                    "cpu_percent": process.cpu_percent(interval=None)
                }
            },
            "application": app_metrics,
            "rag_service": {
                "stats": rag_stats,
                "health": rag_health
            }
        }

        return metrics

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


@router.get("/health", response_model=Dict[str, Any])
async def get_health_metrics():
    """
    Get simplified health metrics for monitoring.

    Returns:
        Basic health status and key metrics
    """
    try:
        start_time = time.time()

        # Basic system checks
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        # RAG service health
        try:
            rag_health = await rag_service.health_check()
            rag_healthy = rag_health.get("overall", False)
        except:
            rag_healthy = False

        # Application metrics
        app_metrics = metrics_service.get_metrics()

        # Determine overall health
        healthy = (
            cpu_percent < 90 and
            memory.percent < 90 and
            rag_healthy
        )

        processing_time = (time.time() - start_time) * 1000

        return {
            "healthy": healthy,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(processing_time, 2),
            "checks": {
                "cpu_ok": cpu_percent < 90,
                "memory_ok": memory.percent < 90,
                "rag_service_ok": rag_healthy
            },
            "quick_stats": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "total_requests": app_metrics.get("requests", {}).get("total", 0),
                "error_rate": app_metrics.get("requests", {}).get("error_rate", 0)
            }
        }

    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@router.get("/requests", response_model=Dict[str, Any])
async def get_request_metrics():
    """
    Get detailed request metrics.

    Returns:
        Request statistics and performance data
    """
    try:
        app_metrics = metrics_service.get_metrics()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requests": app_metrics.get("requests", {}),
            "endpoints": app_metrics.get("endpoints", {}),
            "performance": app_metrics.get("performance", {})
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve request metrics: {str(e)}"
        )


@router.post("/reset")
async def reset_metrics():
    """
    Reset application metrics counters.

    Returns:
        Confirmation of reset
    """
    try:
        metrics_service.reset_metrics()
        return {
            "message": "Metrics reset successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset metrics: {str(e)}"
        )