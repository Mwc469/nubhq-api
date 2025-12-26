"""
NubHQ API Response Utilities
Standardized response format and error handling
"""
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime
import traceback

from .logging_config import api_logger

T = TypeVar('T')


# ============================================================
# RESPONSE MODELS
# ============================================================

class ApiResponse(BaseModel, Generic[T]):
    """Standard API response wrapper"""
    ok: bool = True
    data: Optional[T] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    timestamp: str = ""
    
    def __init__(self, **data):
        if "timestamp" not in data:
            data["timestamp"] = datetime.utcnow().isoformat() + "Z"
        super().__init__(**data)


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated list response"""
    ok: bool = True
    data: List[T] = []
    pagination: Dict[str, Any] = {}
    timestamp: str = ""
    
    def __init__(self, items: List[T], total: int, page: int = 1, per_page: int = 20, **kwargs):
        super().__init__(
            data=items,
            pagination={
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page,
                "has_next": page * per_page < total,
                "has_prev": page > 1,
            },
            timestamp=datetime.utcnow().isoformat() + "Z",
            **kwargs
        )


# ============================================================
# SUCCESS RESPONSES
# ============================================================

def success(data: Any = None, message: str = None, meta: Dict = None) -> Dict:
    """Create success response"""
    response = {
        "ok": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    
    if data is not None:
        response["data"] = data
    
    if message:
        response["message"] = message
    
    if meta:
        response["meta"] = meta
    
    return response


def created(data: Any, message: str = "Created successfully") -> Dict:
    """201 Created response"""
    return success(data, message)


def updated(data: Any = None, message: str = "Updated successfully") -> Dict:
    """200 Updated response"""
    return success(data, message)


def deleted(message: str = "Deleted successfully") -> Dict:
    """200 Deleted response"""
    return success(message=message)


def paginated(items: List, total: int, page: int = 1, per_page: int = 20) -> Dict:
    """Paginated list response"""
    return {
        "ok": True,
        "data": items,
        "pagination": {
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
            "has_next": page * per_page < total,
            "has_prev": page > 1,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ============================================================
# ERROR RESPONSES
# ============================================================

class ApiException(HTTPException):
    """Custom API exception with error codes"""
    
    def __init__(
        self,
        status_code: int,
        message: str,
        error_code: str = None,
        details: Dict = None,
    ):
        self.error_code = error_code or f"ERR_{status_code}"
        self.details = details
        super().__init__(status_code=status_code, detail=message)


# Common exceptions
def bad_request(message: str, code: str = "BAD_REQUEST", details: Dict = None):
    raise ApiException(400, message, code, details)

def unauthorized(message: str = "Authentication required"):
    raise ApiException(401, message, "UNAUTHORIZED")

def forbidden(message: str = "Access denied"):
    raise ApiException(403, message, "FORBIDDEN")

def not_found(resource: str = "Resource", id: str = None):
    message = f"{resource} not found" if not id else f"{resource} '{id}' not found"
    raise ApiException(404, message, "NOT_FOUND")

def conflict(message: str = "Resource conflict"):
    raise ApiException(409, message, "CONFLICT")

def validation_error(message: str, details: Dict = None):
    raise ApiException(422, message, "VALIDATION_ERROR", details)

def rate_limited(message: str = "Too many requests"):
    raise ApiException(429, message, "RATE_LIMITED")

def server_error(message: str = "Internal server error"):
    raise ApiException(500, message, "INTERNAL_ERROR")


# ============================================================
# EXCEPTION HANDLER
# ============================================================

async def api_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for API errors"""
    
    # Handle ApiException
    if isinstance(exc, ApiException):
        api_logger.warning(
            f"API Error: {exc.detail}",
            status_code=exc.status_code,
            error_code=exc.error_code,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "ok": False,
                "error": exc.detail,
                "error_code": exc.error_code,
                "details": exc.details,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )
    
    # Handle HTTPException
    if isinstance(exc, HTTPException):
        api_logger.warning(
            f"HTTP Error: {exc.detail}",
            status_code=exc.status_code,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "ok": False,
                "error": exc.detail,
                "error_code": f"HTTP_{exc.status_code}",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )
    
    # Handle unexpected errors
    api_logger.error(
        f"Unexpected error: {exc}",
        error=exc,
        path=request.url.path,
        traceback=traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": "An unexpected error occurred",
            "error_code": "INTERNAL_ERROR",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )


# ============================================================
# VALIDATION HELPERS
# ============================================================

def require(value: Any, field_name: str):
    """Require a field to be present"""
    if value is None or (isinstance(value, str) and not value.strip()):
        validation_error(f"{field_name} is required", {"field": field_name})
    return value


def require_file_exists(path: str, file_type: str = "File"):
    """Require a file to exist"""
    import os
    if not os.path.exists(path):
        not_found(file_type, path)


def require_valid_id(id: str, resource: str = "Resource"):
    """Require a valid UUID-like ID"""
    import re
    if not id or not re.match(r'^[a-zA-Z0-9_-]{8,}$', id):
        bad_request(f"Invalid {resource} ID", "INVALID_ID", {"id": id})
