"""
NubHQ Health Check Routes
Comprehensive system health monitoring
"""
from fastapi import APIRouter
from datetime import datetime
import os
import sys
import sqlite3
from pathlib import Path
import psutil
import asyncio
from typing import Dict, Any

router = APIRouter(prefix="/api/health", tags=["health"])

# Config
DB_PATH = os.environ.get("NUBHQ_DB", "./data/nubhq.db")
MEDIA_ROOT = os.environ.get("NUBHQ_MEDIA_ROOT", "./data/media")

START_TIME = datetime.utcnow()


def get_uptime() -> str:
    """Get system uptime as human-readable string"""
    delta = datetime.utcnow() - START_TIME
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if delta.days > 0:
        return f"{delta.days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    else:
        return f"{minutes}m {seconds}s"


def check_database() -> Dict[str, Any]:
    """Check database connectivity and stats"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get row counts for key tables
        counts = {}
        for table in ["content_items", "media_assets", "templates", "jobs", "activity_log"]:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cursor.fetchone()[0]
        
        # Get database size
        db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
        
        conn.close()
        
        return {
            "status": "healthy",
            "path": DB_PATH,
            "size_mb": round(db_size / (1024 * 1024), 2),
            "tables": len(tables),
            "row_counts": counts,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


def check_storage() -> Dict[str, Any]:
    """Check storage availability"""
    try:
        media_path = Path(MEDIA_ROOT)
        
        if not media_path.exists():
            return {
                "status": "unhealthy",
                "error": f"Media root does not exist: {MEDIA_ROOT}",
            }
        
        # Get disk usage
        usage = psutil.disk_usage(str(media_path))
        
        # Count files in key directories
        subdirs = ["files", "thumbs", "uploads", "clips", "exports"]
        file_counts = {}
        for subdir in subdirs:
            subpath = media_path / subdir
            if subpath.exists():
                file_counts[subdir] = len(list(subpath.iterdir()))
        
        # Determine health
        free_percent = usage.free / usage.total * 100
        status = "healthy" if free_percent > 10 else "warning" if free_percent > 5 else "critical"
        
        return {
            "status": status,
            "path": MEDIA_ROOT,
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "free_percent": round(free_percent, 1),
            "file_counts": file_counts,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


def check_ffmpeg() -> Dict[str, Any]:
    """Check FFmpeg availability"""
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            return {
                "status": "healthy",
                "version": version_line,
            }
        return {
            "status": "unhealthy",
            "error": "FFmpeg returned non-zero exit code",
        }
    except FileNotFoundError:
        return {
            "status": "unhealthy",
            "error": "FFmpeg not found in PATH",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


def check_system() -> Dict[str, Any]:
    """Check system resources"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        return {
            "status": "healthy" if memory.percent < 90 else "warning",
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "python_version": sys.version.split()[0],
        }
    except Exception as e:
        return {
            "status": "unknown",
            "error": str(e),
        }


# ============================================================
# ROUTES
# ============================================================

@router.get("/")
@router.get("/live")
async def health_live():
    """
    Liveness probe - is the service running?
    Returns 200 if the service is alive.
    """
    return {
        "ok": True,
        "status": "alive",
        "uptime": get_uptime(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/ready")
async def health_ready():
    """
    Readiness probe - is the service ready to accept traffic?
    Checks database and storage connectivity.
    """
    db = check_database()
    storage = check_storage()
    
    all_healthy = db["status"] == "healthy" and storage["status"] in ["healthy", "warning"]
    
    return {
        "ok": all_healthy,
        "status": "ready" if all_healthy else "not_ready",
        "checks": {
            "database": db["status"],
            "storage": storage["status"],
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/full")
async def health_full():
    """
    Full health check - detailed status of all components.
    Use for monitoring dashboards.
    """
    db = check_database()
    storage = check_storage()
    ffmpeg = check_ffmpeg()
    system = check_system()
    
    # Calculate overall status
    statuses = [db["status"], storage["status"], ffmpeg["status"], system["status"]]
    if "unhealthy" in statuses:
        overall = "unhealthy"
    elif "warning" in statuses or "critical" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"
    
    return {
        "ok": overall == "healthy",
        "status": overall,
        "uptime": get_uptime(),
        "started_at": START_TIME.isoformat() + "Z",
        "checks": {
            "database": db,
            "storage": storage,
            "ffmpeg": ffmpeg,
            "system": system,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/metrics")
async def health_metrics():
    """
    Prometheus-style metrics endpoint.
    """
    db = check_database()
    storage = check_storage()
    system = check_system()
    
    metrics = []
    
    # Uptime
    uptime_seconds = (datetime.utcnow() - START_TIME).total_seconds()
    metrics.append(f"nubhq_uptime_seconds {uptime_seconds}")
    
    # System
    if system["status"] != "unknown":
        metrics.append(f"nubhq_cpu_percent {system.get('cpu_percent', 0)}")
        metrics.append(f"nubhq_memory_percent {system.get('memory_percent', 0)}")
    
    # Storage
    if storage["status"] != "unhealthy":
        metrics.append(f"nubhq_storage_free_gb {storage.get('free_gb', 0)}")
        metrics.append(f"nubhq_storage_free_percent {storage.get('free_percent', 0)}")
    
    # Database
    if db["status"] == "healthy":
        metrics.append(f"nubhq_db_size_mb {db.get('size_mb', 0)}")
        for table, count in db.get("row_counts", {}).items():
            metrics.append(f'nubhq_db_rows{{table="{table}"}} {count}')
    
    # Health status (1 = healthy, 0 = unhealthy)
    metrics.append(f"nubhq_health_database {1 if db['status'] == 'healthy' else 0}")
    metrics.append(f"nubhq_health_storage {1 if storage['status'] in ['healthy', 'warning'] else 0}")
    
    return "\n".join(metrics)
