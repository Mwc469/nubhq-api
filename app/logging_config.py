"""
NubHQ Logging Configuration
Structured logging with context for debugging and monitoring
"""
import logging
import sys
import json
import traceback
from datetime import datetime
from typing import Optional
from functools import wraps
import time
import os

# ============================================================
# LOG LEVELS
# ============================================================

LOG_LEVEL = os.environ.get("NUBHQ_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.environ.get("NUBHQ_LOG_FORMAT", "json")  # json or text

# ============================================================
# STRUCTURED LOGGING
# ============================================================

class StructuredLogger:
    """Logger that outputs structured JSON logs"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Add handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter() if LOG_FORMAT == "json" else TextFormatter())
        self.logger.addHandler(handler)
    
    def _log(self, level: str, message: str, **context):
        extra = {
            "context": context,
            "logger_name": self.name,
        }
        getattr(self.logger, level.lower())(message, extra=extra)
    
    def debug(self, message: str, **context):
        self._log("debug", message, **context)
    
    def info(self, message: str, **context):
        self._log("info", message, **context)
    
    def warning(self, message: str, **context):
        self._log("warning", message, **context)
    
    def error(self, message: str, error: Optional[Exception] = None, **context):
        if error:
            context["error_type"] = type(error).__name__
            context["error_message"] = str(error)
            context["traceback"] = traceback.format_exc()
        self._log("error", message, **context)
    
    def critical(self, message: str, error: Optional[Exception] = None, **context):
        if error:
            context["error_type"] = type(error).__name__
            context["error_message"] = str(error)
            context["traceback"] = traceback.format_exc()
        self._log("critical", message, **context)


class StructuredFormatter(logging.Formatter):
    """Formats logs as JSON"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": getattr(record, "logger_name", record.name),
            "message": record.getMessage(),
        }
        
        # Add context if present
        if hasattr(record, "context"):
            log_data.update(record.context)
        
        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Formats logs as readable text"""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        
        parts = [
            f"{color}[{timestamp}]",
            f"[{record.levelname}]",
            f"{self.RESET}{record.getMessage()}",
        ]
        
        # Add context
        if hasattr(record, "context") and record.context:
            context_str = " ".join(f"{k}={v}" for k, v in record.context.items() if k != "traceback")
            if context_str:
                parts.append(f"\033[90m({context_str}){self.RESET}")
        
        return " ".join(parts)


# ============================================================
# REQUEST LOGGING MIDDLEWARE
# ============================================================

def log_request(logger: StructuredLogger):
    """FastAPI middleware for request logging"""
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware
    
    class RequestLoggingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            request_id = f"req_{int(time.time() * 1000)}"
            start_time = time.time()
            
            # Log request
            logger.info(
                f"{request.method} {request.url.path}",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                query=str(request.query_params),
            )
            
            try:
                response = await call_next(request)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log response
                level = "info" if response.status_code < 400 else "warning" if response.status_code < 500 else "error"
                getattr(logger, level)(
                    f"{request.method} {request.url.path} -> {response.status_code}",
                    request_id=request_id,
                    status_code=response.status_code,
                    duration_ms=round(duration_ms, 2),
                )
                
                return response
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{request.method} {request.url.path} -> ERROR",
                    error=e,
                    request_id=request_id,
                    duration_ms=round(duration_ms, 2),
                )
                raise
    
    return RequestLoggingMiddleware


# ============================================================
# FUNCTION TIMING DECORATOR
# ============================================================

def timed(logger: StructuredLogger):
    """Decorator to log function execution time"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                logger.debug(
                    f"{func.__name__} completed",
                    function=func.__name__,
                    duration_ms=round(duration, 2),
                )
                return result
            except Exception as e:
                duration = (time.time() - start) * 1000
                logger.error(
                    f"{func.__name__} failed",
                    error=e,
                    function=func.__name__,
                    duration_ms=round(duration, 2),
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                logger.debug(
                    f"{func.__name__} completed",
                    function=func.__name__,
                    duration_ms=round(duration, 2),
                )
                return result
            except Exception as e:
                duration = (time.time() - start) * 1000
                logger.error(
                    f"{func.__name__} failed",
                    error=e,
                    function=func.__name__,
                    duration_ms=round(duration, 2),
                )
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# ============================================================
# LOGGER INSTANCES
# ============================================================

# Create loggers for different modules
api_logger = StructuredLogger("nubhq.api")
worker_logger = StructuredLogger("nubhq.worker")
video_logger = StructuredLogger("nubhq.video")
db_logger = StructuredLogger("nubhq.db")


def get_logger(name: str) -> StructuredLogger:
    """Get or create a logger by name"""
    return StructuredLogger(f"nubhq.{name}")
