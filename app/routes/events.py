"""
NubHQ Real-time Events (SSE)
Server-Sent Events for live job updates and notifications
"""
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Dict, Set
import asyncio
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid

router = APIRouter(prefix="/api/events", tags=["events"])


# ============================================================
# EVENT TYPES
# ============================================================

@dataclass
class Event:
    """Server-sent event structure"""
    type: str
    data: Dict
    id: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = f"evt_{uuid.uuid4().hex[:8]}"
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_sse(self) -> str:
        """Format as SSE message"""
        payload = {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp,
        }
        return f"id: {self.id}\nevent: {self.type}\ndata: {json.dumps(payload)}\n\n"


# ============================================================
# EVENT MANAGER (Pub/Sub)
# ============================================================

class EventManager:
    """Manages SSE connections and broadcasts"""
    
    def __init__(self):
        self._clients: Dict[str, asyncio.Queue] = {}
        self._topics: Dict[str, Set[str]] = {}  # topic -> client_ids
    
    async def connect(self, client_id: str, topics: list = None) -> asyncio.Queue:
        """Register a new client"""
        queue = asyncio.Queue()
        self._clients[client_id] = queue
        
        # Subscribe to topics
        topics = topics or ["all"]
        for topic in topics:
            if topic not in self._topics:
                self._topics[topic] = set()
            self._topics[topic].add(client_id)
        
        # Send connection confirmation
        await queue.put(Event(
            type="connected",
            data={"client_id": client_id, "topics": topics}
        ))
        
        return queue
    
    async def disconnect(self, client_id: str):
        """Remove a client"""
        if client_id in self._clients:
            del self._clients[client_id]
        
        # Remove from all topics
        for topic in self._topics.values():
            topic.discard(client_id)
    
    async def broadcast(self, event: Event, topic: str = "all"):
        """Send event to all clients subscribed to topic"""
        client_ids = self._topics.get(topic, set()) | self._topics.get("all", set())
        
        for client_id in client_ids:
            if client_id in self._clients:
                await self._clients[client_id].put(event)
    
    async def send_to_client(self, client_id: str, event: Event):
        """Send event to specific client"""
        if client_id in self._clients:
            await self._clients[client_id].put(event)
    
    @property
    def client_count(self) -> int:
        return len(self._clients)


# Global event manager
event_manager = EventManager()


# ============================================================
# HELPER FUNCTIONS (use from other routes)
# ============================================================

async def emit_job_update(job_id: str, status: str, progress: int, details: Dict = None):
    """Emit job status update event"""
    await event_manager.broadcast(
        Event(
            type="job_update",
            data={
                "job_id": job_id,
                "status": status,
                "progress": progress,
                **(details or {}),
            }
        ),
        topic="jobs"
    )


async def emit_content_update(action: str, content_id: str, data: Dict = None):
    """Emit content change event"""
    await event_manager.broadcast(
        Event(
            type="content_update",
            data={
                "action": action,  # created, updated, deleted, approved, etc.
                "content_id": content_id,
                **(data or {}),
            }
        ),
        topic="content"
    )


async def emit_notification(user_id: str, title: str, message: str, level: str = "info"):
    """Emit user notification"""
    await event_manager.send_to_client(
        user_id,
        Event(
            type="notification",
            data={
                "title": title,
                "message": message,
                "level": level,  # info, success, warning, error
            }
        )
    )


async def emit_system_event(event_type: str, data: Dict = None):
    """Emit system-wide event"""
    await event_manager.broadcast(
        Event(
            type=event_type,
            data=data or {},
        ),
        topic="system"
    )


# ============================================================
# SSE ROUTES
# ============================================================

async def event_stream(request: Request, client_id: str, topics: list) -> AsyncGenerator:
    """Generator for SSE stream"""
    queue = await event_manager.connect(client_id, topics)
    
    try:
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break
            
            try:
                # Wait for events with timeout (for keepalive)
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield event.to_sse()
            except asyncio.TimeoutError:
                # Send keepalive comment
                yield ": keepalive\n\n"
    
    finally:
        await event_manager.disconnect(client_id)


@router.get("/stream")
async def sse_stream(request: Request, topics: str = "all"):
    """
    SSE endpoint for real-time events.
    
    Query params:
    - topics: Comma-separated list of topics (all, jobs, content, system)
    
    Example:
    ```
    const source = new EventSource('/api/events/stream?topics=jobs,content');
    source.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data.type, data.data);
    };
    ```
    """
    client_id = f"client_{uuid.uuid4().hex[:8]}"
    topic_list = [t.strip() for t in topics.split(",")]
    
    return StreamingResponse(
        event_stream(request, client_id, topic_list),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.get("/status")
async def events_status():
    """Get current event system status"""
    return {
        "ok": True,
        "connected_clients": event_manager.client_count,
        "topics": list(event_manager._topics.keys()),
    }


@router.post("/test")
async def test_event(event_type: str = "test", message: str = "Hello from SSE"):
    """Send a test event (dev only)"""
    await event_manager.broadcast(
        Event(type=event_type, data={"message": message}),
        topic="all"
    )
    return {"ok": True, "message": "Event sent"}
