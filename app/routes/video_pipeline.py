"""
Video Pipeline API Routes
=========================
Endpoints for video processing, combining, and feedback.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
from sqlalchemy.orm import Session
import logging
import json
import os
import re

from ..auth import get_current_user, get_required_user
from ..database import get_db
from ..models.user import User
from ..models.job import Job
from ..limiter import limiter

# Import worker modules
try:
    from ..worker.engagement_scorer import EngagementScorer
    from ..worker.content_combiner import (
        HighlightExtractor,
        MultiAngleSync,
        TemplateCompiler,
        create_highlight_reel,
        sync_videos,
        compile_for_platform,
    )
    from ..worker.intelligent_processor import (
        PreferenceLearner,
        Config,
        VideoAnalyzer,
        ApprovalQueueIntegration,
    )
    HAS_WORKERS = True
except ImportError as e:
    logging.warning(f"Worker modules not available: {e}")
    HAS_WORKERS = False

router = APIRouter(prefix="/api/video-pipeline", tags=["video-pipeline"])


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class HighlightRequest(BaseModel):
    video_path: str
    duration: int = 60
    output_name: Optional[str] = None


class SyncRequest(BaseModel):
    video_paths: List[str]


class CompileRequest(BaseModel):
    template_id: str
    source_videos: List[str]
    intro_video: Optional[str] = None
    outro_video: Optional[str] = None
    output_name: Optional[str] = None


class EngagementRequest(BaseModel):
    video_path: str
    use_ai: bool = False


class ApprovalFeedbackRequest(BaseModel):
    video_fingerprint: str
    approved: bool
    engagement_score: float
    engagement_confidence: float
    user_edits: Optional[Dict[str, Any]] = None


class ClipResponse(BaseModel):
    source_path: str
    start_time: float
    end_time: float
    score: float
    label: str


class HighlightResponse(BaseModel):
    success: bool
    output_path: Optional[str]
    duration: float
    clips: List[ClipResponse]
    error: Optional[str] = None


class SyncResponse(BaseModel):
    reference_video: str
    offsets: Dict[str, float]
    confidence: float
    method: str


class CompileResponse(BaseModel):
    success: bool
    output_path: Optional[str]
    duration: float
    template_used: str
    clips_count: int
    error: Optional[str] = None


class EngagementResponse(BaseModel):
    technical_score: float
    performance_score: float
    ai_score: Optional[float]
    overall_score: float


class TemplateSegment(BaseModel):
    type: str  # "intro", "highlight", "outro", "cta", "hook"
    duration: int
    source: Optional[str] = None  # Path to custom asset


class CustomTemplateCreate(BaseModel):
    id: str
    name: str
    duration: int
    aspect: str  # "16:9", "9:16", "1:1"
    segments: List[TemplateSegment]


class CustomTemplateUpdate(BaseModel):
    name: Optional[str] = None
    duration: Optional[int] = None
    aspect: Optional[str] = None
    segments: Optional[List[TemplateSegment]] = None


class TemplateInfo(BaseModel):
    id: str
    name: str
    duration: int
    aspect: str
    segments: int


class PipelineStatsResponse(BaseModel):
    total_decisions: int
    preferences: Dict[str, Any]
    approval_stats: Dict[str, Any]


# ============================================================
# SECURITY HELPERS
# ============================================================

# Allowed video extensions
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'}

# Allowed directories for video processing
ALLOWED_PATHS = [
    '/Volumes/NUB_Workspace',
    '/tmp',
    os.environ.get('NUBHQ_INPUT', '/Volumes/NUB_Workspace/input'),
    os.environ.get('NUBHQ_OUTPUT', '/Volumes/NUB_Workspace/output'),
]


def validate_video_path(path_str: str) -> Path:
    """Validate a video path for security"""
    path = Path(path_str).resolve()

    # Check extension
    if path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )

    # Check path is within allowed directories
    path_str_resolved = str(path)
    allowed = any(path_str_resolved.startswith(allowed_path) for allowed_path in ALLOWED_PATHS if allowed_path)
    if not allowed:
        raise HTTPException(
            status_code=403,
            detail="Access to this path is not allowed"
        )

    return path


def validate_template_id(template_id: str) -> str:
    """Validate template ID format"""
    if not re.match(r'^[a-z0-9_-]+$', template_id):
        raise HTTPException(
            status_code=400,
            detail="Template ID must contain only lowercase letters, numbers, hyphens, and underscores"
        )
    return template_id


# ============================================================
# ENDPOINTS
# ============================================================

@router.get("/health")
async def pipeline_health():
    """Check video pipeline health"""
    return {
        "status": "ok",
        "workers_available": HAS_WORKERS,
    }


@router.get("/templates", response_model=List[TemplateInfo])
async def list_templates():
    """List available compilation templates (built-in + custom)"""
    templates = []

    # Add built-in templates from compiler
    if HAS_WORKERS:
        compiler = TemplateCompiler()
        templates.extend(compiler.list_templates())

    # Add custom templates
    for t in _custom_templates.values():
        templates.append(TemplateInfo(
            id=t["id"],
            name=t["name"],
            duration=t["duration"],
            aspect=t["aspect"],
            segments=len(t.get("segments", []))
        ))

    return templates


@router.post("/highlight", response_model=HighlightResponse)
@limiter.limit("10/minute")
async def create_highlight(
    request: Request,
    highlight_request: HighlightRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_required_user)
):
    """Create a highlight reel from a video"""
    if not HAS_WORKERS:
        raise HTTPException(status_code=503, detail="Worker modules not available")

    video_path = validate_video_path(highlight_request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {highlight_request.video_path}")

    result = create_highlight_reel(
        str(video_path),
        highlight_request.duration,
        highlight_request.output_name
    )

    return HighlightResponse(
        success=result.success,
        output_path=result.output_path,
        duration=result.duration,
        clips=[
            ClipResponse(
                source_path=c.source_path,
                start_time=c.start_time,
                end_time=c.end_time,
                score=c.score,
                label=c.label
            )
            for c in result.source_clips
        ],
        error=result.error
    )


@router.post("/sync", response_model=SyncResponse)
async def sync_multiple_videos(request: SyncRequest):
    """Synchronize multiple camera angles by audio"""
    if not HAS_WORKERS:
        raise HTTPException(status_code=503, detail="Worker modules not available")

    # Validate all videos exist
    for path in request.video_paths:
        if not Path(path).exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {path}")

    result = sync_videos(request.video_paths)

    return SyncResponse(
        reference_video=result.reference_video,
        offsets=result.synced_videos,
        confidence=result.confidence,
        method=result.method
    )


@router.post("/compile", response_model=CompileResponse)
async def compile_template(request: CompileRequest):
    """Compile videos using a template"""
    if not HAS_WORKERS:
        raise HTTPException(status_code=503, detail="Worker modules not available")

    # Validate videos exist
    for path in request.source_videos:
        if not Path(path).exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {path}")

    compiler = TemplateCompiler()

    intro = Path(request.intro_video) if request.intro_video else None
    outro = Path(request.outro_video) if request.outro_video else None

    result = compiler.compile(
        request.template_id,
        [Path(v) for v in request.source_videos],
        intro_video=intro,
        outro_video=outro,
        output_name=request.output_name
    )

    return CompileResponse(
        success=result.success,
        output_path=result.output_path,
        duration=result.duration,
        template_used=result.template_used or request.template_id,
        clips_count=len(result.source_clips),
        error=result.error
    )


@router.post("/engagement", response_model=EngagementResponse)
async def analyze_engagement(request: EngagementRequest):
    """Analyze video engagement potential"""
    if not HAS_WORKERS:
        raise HTTPException(status_code=503, detail="Worker modules not available")

    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")

    scorer = EngagementScorer(ai_enabled=request.use_ai)
    result = scorer.score(video_path)

    return EngagementResponse(
        technical_score=result.technical_score,
        performance_score=result.performance_score,
        ai_score=result.ai_score,
        overall_score=result.overall_score,
        confidence=result.confidence,
        tags=result.tags,
        best_moments=[
            {
                "start_time": m.start_time,
                "end_time": m.end_time,
                "score": m.score,
                "reason": m.reason,
                "tags": m.tags
            }
            for m in result.best_moments
        ],
        analysis_time=result.analysis_time
    )


@router.post("/feedback")
async def submit_approval_feedback(request: ApprovalFeedbackRequest):
    """Submit feedback from approval queue decision (for learning)"""
    if not HAS_WORKERS:
        raise HTTPException(status_code=503, detail="Worker modules not available")

    try:
        learner = PreferenceLearner(Config.DB_PATH)
        learner.learn_from_approval(
            video_fingerprint=request.video_fingerprint,
            approved=request.approved,
            engagement_score=request.engagement_score,
            engagement_confidence=request.engagement_confidence,
            user_edits=request.user_edits
        )

        return {"status": "ok", "message": "Feedback recorded"}

    except Exception as e:
        logging.exception("Failed to record feedback")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=PipelineStatsResponse)
async def get_pipeline_stats():
    """Get video pipeline learning statistics"""
    if not HAS_WORKERS:
        raise HTTPException(status_code=503, detail="Worker modules not available")

    try:
        learner = PreferenceLearner(Config.DB_PATH)
        stats = learner.get_stats()
        approval_stats = learner.get_approval_stats()

        return PipelineStatsResponse(
            total_decisions=stats['total_decisions'],
            preferences=stats['preferences'],
            approval_stats=approval_stats
        )

    except Exception as e:
        logging.exception("Failed to get stats")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/review")
async def get_review_queue():
    """Get videos held for manual review"""
    if not HAS_WORKERS:
        raise HTTPException(status_code=503, detail="Worker modules not available")

    review_dir = Config.REVIEW_DIR

    if not review_dir.exists():
        return {"videos": []}

    videos = []
    for f in review_dir.iterdir():
        if f.suffix.lower() in {'.mp4', '.mov', '.mkv', '.webm'}:
            videos.append({
                "path": str(f),
                "name": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime
            })

    return {"videos": sorted(videos, key=lambda x: x['modified'], reverse=True)}


@router.post("/review/{filename}/approve")
async def approve_review_video(filename: str, recipient: str = "schedule"):
    """Move a video from review to approval queue"""
    if not HAS_WORKERS:
        raise HTTPException(status_code=503, detail="Worker modules not available")

    review_path = Config.REVIEW_DIR / filename
    if not review_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found in review: {filename}")

    try:
        # Analyze the video
        analyzer = VideoAnalyzer()
        analysis = analyzer.analyze(review_path)

        # Score engagement
        scorer = EngagementScorer()
        engagement = scorer.score(str(review_path), pre_computed={
            'duration': analysis.duration,
            'fps': analysis.fps,
        })

        # Move to auto-queued directory
        dest = Config.AUTO_QUEUED_DIR / filename
        if dest.exists():
            import time
            dest = Config.AUTO_QUEUED_DIR / f"{review_path.stem}_{int(time.time())}{review_path.suffix}"
        review_path.rename(dest)

        # Push to approval queue API
        queue = ApprovalQueueIntegration()
        success, result = queue.push_to_queue(
            video_path=dest,
            analysis=analysis,
            engagement=engagement,
            decisions={"manual_approval": "true"},
            recipient=recipient
        )

        if success:
            return {
                "status": "ok",
                "message": "Moved to approval queue",
                "path": str(dest),
                "approval_id": result
            }
        else:
            # Still moved, but queue push failed
            return {
                "status": "partial",
                "message": f"Moved but queue push failed: {result}",
                "path": str(dest)
            }

    except Exception as e:
        logging.exception(f"Failed to approve video: {filename}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/review/{filename}")
async def reject_review_video(filename: str):
    """Delete a video from the review queue"""
    if not HAS_WORKERS:
        raise HTTPException(status_code=503, detail="Worker modules not available")

    review_path = Config.REVIEW_DIR / filename
    if not review_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found in review: {filename}")

    try:
        review_path.unlink()
        return {"status": "ok", "message": f"Deleted: {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# CUSTOM TEMPLATE CRUD
# ============================================================

# In-memory storage for custom templates (in production, use database)
_custom_templates: Dict[str, dict] = {}


def _load_custom_templates():
    """Load custom templates from disk"""
    global _custom_templates
    templates_dir = Path(os.environ.get('NUBHQ_OUTPUT', '/Volumes/NUB_Workspace/output')) / 'templates'
    templates_file = templates_dir / 'custom_templates.json'

    if templates_file.exists():
        try:
            with open(templates_file, 'r') as f:
                _custom_templates = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load custom templates: {e}")


def _save_custom_templates():
    """Save custom templates to disk"""
    templates_dir = Path(os.environ.get('NUBHQ_OUTPUT', '/Volumes/NUB_Workspace/output')) / 'templates'
    try:
        templates_dir.mkdir(parents=True, exist_ok=True)
        templates_file = templates_dir / 'custom_templates.json'
        with open(templates_file, 'w') as f:
            json.dump(_custom_templates, f, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save custom templates: {e}")


# Load templates on module import
_load_custom_templates()


@router.post("/templates", status_code=201)
async def create_custom_template(template: CustomTemplateCreate):
    """Create a new custom template"""
    if template.id in _custom_templates:
        raise HTTPException(status_code=400, detail=f"Template '{template.id}' already exists")

    # Validate aspect ratio
    if template.aspect not in ["16:9", "9:16", "1:1", "4:5"]:
        raise HTTPException(status_code=400, detail=f"Invalid aspect ratio: {template.aspect}")

    # Store template
    _custom_templates[template.id] = {
        "id": template.id,
        "name": template.name,
        "duration": template.duration,
        "aspect": template.aspect,
        "segments": [s.model_dump() for s in template.segments],
        "is_custom": True
    }

    _save_custom_templates()

    return {
        "status": "ok",
        "message": f"Template '{template.id}' created",
        "template": _custom_templates[template.id]
    }


@router.put("/templates/{template_id}")
async def update_custom_template(template_id: str, update: CustomTemplateUpdate):
    """Update an existing custom template"""
    if template_id not in _custom_templates:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

    template = _custom_templates[template_id]

    # Update fields
    if update.name is not None:
        template["name"] = update.name
    if update.duration is not None:
        template["duration"] = update.duration
    if update.aspect is not None:
        if update.aspect not in ["16:9", "9:16", "1:1", "4:5"]:
            raise HTTPException(status_code=400, detail=f"Invalid aspect ratio: {update.aspect}")
        template["aspect"] = update.aspect
    if update.segments is not None:
        template["segments"] = [s.model_dump() for s in update.segments]

    _save_custom_templates()

    return {
        "status": "ok",
        "message": f"Template '{template_id}' updated",
        "template": template
    }


@router.delete("/templates/{template_id}")
async def delete_custom_template(template_id: str):
    """Delete a custom template"""
    if template_id not in _custom_templates:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

    del _custom_templates[template_id]
    _save_custom_templates()

    return {
        "status": "ok",
        "message": f"Template '{template_id}' deleted"
    }


@router.get("/templates/{template_id}")
async def get_custom_template(template_id: str):
    """Get a specific template by ID"""
    # Check custom templates first
    if template_id in _custom_templates:
        return _custom_templates[template_id]

    # Check built-in templates
    if HAS_WORKERS:
        compiler = TemplateCompiler()
        for template in compiler.list_templates():
            if template["id"] == template_id:
                return template

    raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")


# ============================================================
# PHASE 3: ADVANCED FEATURES
# ============================================================

import subprocess
import time
from datetime import datetime

# Activity log storage (in-memory, would use DB in production)
_activity_log: List[dict] = []


class ThumbnailRequest(BaseModel):
    video_path: str
    count: int = 4  # Number of thumbnails to generate
    width: int = 320


class BatchProcessRequest(BaseModel):
    video_paths: List[str]
    template_id: Optional[str] = None
    generate_thumbnails: bool = True


class MultiPlatformExportRequest(BaseModel):
    video_path: str
    platforms: List[str] = ["instagram_reel", "youtube_short", "tiktok"]
    add_watermark: bool = False
    watermark_path: Optional[str] = None


class WatermarkRequest(BaseModel):
    video_path: str
    watermark_path: str
    position: str = "bottom-right"  # top-left, top-right, bottom-left, bottom-right, center
    opacity: float = 0.7
    scale: float = 0.15  # Scale relative to video width


class ActivityLogEntry(BaseModel):
    id: int
    action: str
    name: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str


def _log_activity(action: str, name: str, details: Optional[dict] = None):
    """Log an activity entry"""
    global _activity_log
    entry = {
        "id": len(_activity_log) + 1,
        "action": action,
        "name": name,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }
    _activity_log.insert(0, entry)
    # Keep only last 100 entries
    _activity_log = _activity_log[:100]
    return entry


@router.post("/thumbnails")
async def generate_thumbnails(request: ThumbnailRequest):
    """Generate thumbnail images from a video at evenly spaced intervals"""
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")

    try:
        # Get video duration using ffprobe
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip())

        # Calculate timestamps for thumbnails
        interval = duration / (request.count + 1)
        timestamps = [interval * (i + 1) for i in range(request.count)]

        # Output directory
        output_dir = Path(os.environ.get('NUBHQ_OUTPUT', '/Volumes/NUB_Workspace/output')) / 'thumbnails'
        output_dir.mkdir(parents=True, exist_ok=True)

        thumbnails = []
        for i, ts in enumerate(timestamps):
            output_path = output_dir / f"{video_path.stem}_thumb_{i+1}.jpg"

            # Generate thumbnail using ffmpeg
            cmd = [
                "ffmpeg", "-y", "-ss", str(ts),
                "-i", str(video_path),
                "-vframes", "1",
                "-vf", f"scale={request.width}:-1",
                str(output_path)
            ]
            subprocess.run(cmd, capture_output=True)

            if output_path.exists():
                thumbnails.append({
                    "path": str(output_path),
                    "timestamp": ts,
                    "index": i + 1
                })

        _log_activity("Thumbnails generated", video_path.name, {"count": len(thumbnails)})

        return {
            "status": "ok",
            "video": str(video_path),
            "thumbnails": thumbnails,
            "duration": duration
        }

    except Exception as e:
        logging.exception(f"Failed to generate thumbnails: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def batch_process(request: BatchProcessRequest, background_tasks: BackgroundTasks):
    """Process multiple videos in batch"""
    # Validate all videos exist
    for path in request.video_paths:
        if not Path(path).exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {path}")

    job_id = f"batch_{int(time.time())}"

    # Start batch processing in background
    async def process_batch():
        results = []
        for video_path in request.video_paths:
            try:
                result = {"video": video_path, "status": "completed"}

                # Generate thumbnails if requested
                if request.generate_thumbnails:
                    thumb_result = await generate_thumbnails(
                        ThumbnailRequest(video_path=video_path)
                    )
                    result["thumbnails"] = thumb_result.get("thumbnails", [])

                # Apply template if specified
                if request.template_id and HAS_WORKERS:
                    compiler = TemplateCompiler()
                    compile_result = compiler.compile(
                        request.template_id,
                        [Path(video_path)]
                    )
                    result["compiled"] = compile_result.success
                    result["output_path"] = compile_result.output_path

                results.append(result)
                _log_activity("Batch processed", Path(video_path).name)

            except Exception as e:
                results.append({
                    "video": video_path,
                    "status": "failed",
                    "error": str(e)
                })

        return results

    background_tasks.add_task(process_batch)
    _log_activity("Batch job started", f"{len(request.video_paths)} videos", {"job_id": job_id})

    return {
        "status": "ok",
        "job_id": job_id,
        "message": f"Processing {len(request.video_paths)} videos",
        "videos": request.video_paths
    }


@router.post("/export-all")
async def export_all_platforms(request: MultiPlatformExportRequest, background_tasks: BackgroundTasks):
    """Export a video to multiple platforms at once"""
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")

    if not HAS_WORKERS:
        raise HTTPException(status_code=503, detail="Worker modules not available")

    try:
        compiler = TemplateCompiler()
        results = []

        for platform in request.platforms:
            try:
                result = compiler.compile(
                    platform,
                    [video_path],
                    output_name=f"{video_path.stem}_{platform}"
                )

                output_data = {
                    "platform": platform,
                    "success": result.success,
                    "output_path": result.output_path,
                    "duration": result.duration
                }

                # Apply watermark if requested
                if request.add_watermark and result.success and request.watermark_path:
                    watermarked = await add_watermark(WatermarkRequest(
                        video_path=result.output_path,
                        watermark_path=request.watermark_path
                    ))
                    output_data["watermarked"] = watermarked.get("status") == "ok"
                    output_data["output_path"] = watermarked.get("output_path", result.output_path)

                results.append(output_data)

            except Exception as e:
                results.append({
                    "platform": platform,
                    "success": False,
                    "error": str(e)
                })

        _log_activity("Multi-platform export", video_path.name, {
            "platforms": request.platforms,
            "successful": sum(1 for r in results if r.get("success"))
        })

        return {
            "status": "ok",
            "source": str(video_path),
            "exports": results
        }

    except Exception as e:
        logging.exception(f"Failed to export: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watermark")
async def add_watermark(request: WatermarkRequest):
    """Add a watermark/logo overlay to a video"""
    video_path = Path(request.video_path)
    watermark_path = Path(request.watermark_path)

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")
    if not watermark_path.exists():
        raise HTTPException(status_code=404, detail=f"Watermark not found: {request.watermark_path}")

    try:
        output_dir = Path(os.environ.get('NUBHQ_OUTPUT', '/Volumes/NUB_Workspace/output')) / 'watermarked'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_watermarked{video_path.suffix}"

        # Position mapping for ffmpeg overlay
        positions = {
            "top-left": "10:10",
            "top-right": "main_w-overlay_w-10:10",
            "bottom-left": "10:main_h-overlay_h-10",
            "bottom-right": "main_w-overlay_w-10:main_h-overlay_h-10",
            "center": "(main_w-overlay_w)/2:(main_h-overlay_h)/2"
        }
        pos = positions.get(request.position, positions["bottom-right"])

        # Build ffmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(watermark_path),
            "-filter_complex",
            f"[1:v]scale=iw*{request.scale}:-1,format=rgba,colorchannelmixer=aa={request.opacity}[wm];"
            f"[0:v][wm]overlay={pos}",
            "-c:a", "copy",
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")

        _log_activity("Watermark added", video_path.name, {"position": request.position})

        return {
            "status": "ok",
            "input": str(video_path),
            "output_path": str(output_path),
            "watermark": str(watermark_path),
            "position": request.position
        }

    except Exception as e:
        logging.exception(f"Failed to add watermark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activity")
async def get_activity_log(limit: int = 20):
    """Get recent activity log"""
    return {
        "activities": _activity_log[:limit],
        "total": len(_activity_log)
    }


@router.delete("/activity")
async def clear_activity_log():
    """Clear activity log"""
    global _activity_log
    _activity_log = []
    return {"status": "ok", "message": "Activity log cleared"}


# ============================================================
# CAPTION GENERATION (Whisper)
# ============================================================

class CaptionRequest(BaseModel):
    video_path: str
    language: str = "en"
    format: str = "srt"  # srt, vtt, json


class CaptionSegment(BaseModel):
    start: float
    end: float
    text: str


try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@router.post("/caption")
async def generate_captions(request: CaptionRequest):
    """Generate captions/subtitles for a video using Whisper"""
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")

    if not HAS_OPENAI:
        raise HTTPException(status_code=503, detail="OpenAI module not available")

    try:
        # Extract audio from video
        audio_path = Path("/tmp") / f"{video_path.stem}_audio.mp3"
        extract_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn", "-acodec", "libmp3lame",
            "-q:a", "4",
            str(audio_path)
        ]
        subprocess.run(extract_cmd, capture_output=True)

        if not audio_path.exists():
            raise Exception("Failed to extract audio from video")

        # Transcribe with Whisper API
        client = OpenAI()

        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=request.language,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        # Clean up temp audio file
        audio_path.unlink()

        # Parse segments
        segments = []
        if hasattr(transcription, 'segments'):
            for seg in transcription.segments:
                segments.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "").strip()
                })

        # Generate output file
        output_dir = Path(os.environ.get('NUBHQ_OUTPUT', '/Volumes/NUB_Workspace/output')) / 'captions'
        output_dir.mkdir(parents=True, exist_ok=True)

        if request.format == "srt":
            output_path = output_dir / f"{video_path.stem}.srt"
            srt_content = _generate_srt(segments)
            with open(output_path, "w") as f:
                f.write(srt_content)

        elif request.format == "vtt":
            output_path = output_dir / f"{video_path.stem}.vtt"
            vtt_content = _generate_vtt(segments)
            with open(output_path, "w") as f:
                f.write(vtt_content)

        else:  # json
            output_path = output_dir / f"{video_path.stem}_captions.json"
            with open(output_path, "w") as f:
                json.dump({"segments": segments, "text": transcription.text}, f, indent=2)

        _log_activity("Captions generated", video_path.name, {
            "format": request.format,
            "segments": len(segments)
        })

        return {
            "status": "ok",
            "video": str(video_path),
            "output_path": str(output_path),
            "format": request.format,
            "segments": len(segments),
            "text": transcription.text if hasattr(transcription, 'text') else ""
        }

    except Exception as e:
        logging.exception(f"Failed to generate captions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_srt(segments: List[dict]) -> str:
    """Generate SRT subtitle format"""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_timestamp_srt(seg["start"])
        end = _format_timestamp_srt(seg["end"])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _generate_vtt(segments: List[dict]) -> str:
    """Generate WebVTT subtitle format"""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp_vtt(seg["start"])
        end = _format_timestamp_vtt(seg["end"])
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds to VTT timestamp (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


# ============================================================
# REAL-TIME PROGRESS (Server-Sent Events)
# ============================================================

from fastapi.responses import StreamingResponse
import asyncio

# Job progress storage
_job_progress: Dict[str, dict] = {}


def update_job_progress(job_id: str, progress: int, status: str, message: str = ""):
    """Update progress for a job"""
    _job_progress[job_id] = {
        "job_id": job_id,
        "progress": progress,
        "status": status,
        "message": message,
        "updated_at": datetime.now().isoformat()
    }


@router.get("/progress/{job_id}")
async def get_job_progress(job_id: str):
    """Get current progress for a job"""
    if job_id not in _job_progress:
        return {"job_id": job_id, "progress": 0, "status": "unknown", "message": "Job not found"}
    return _job_progress[job_id]


@router.get("/progress/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """Stream job progress updates via Server-Sent Events"""

    async def event_generator():
        last_progress = -1
        timeout_count = 0
        max_timeout = 300  # 5 minutes max

        while timeout_count < max_timeout:
            if job_id in _job_progress:
                current = _job_progress[job_id]
                if current["progress"] != last_progress:
                    last_progress = current["progress"]
                    yield f"data: {json.dumps(current)}\n\n"

                    # Job completed or failed
                    if current["status"] in ["completed", "failed"]:
                        break
            else:
                yield f"data: {json.dumps({'job_id': job_id, 'progress': 0, 'status': 'pending'})}\n\n"

            await asyncio.sleep(1)
            timeout_count += 1

        yield f"data: {json.dumps({'job_id': job_id, 'status': 'timeout'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/jobs")
async def list_jobs():
    """List all tracked jobs"""
    return {
        "jobs": list(_job_progress.values()),
        "total": len(_job_progress)
    }


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Remove a job from tracking"""
    if job_id in _job_progress:
        del _job_progress[job_id]
        return {"status": "ok", "message": f"Job {job_id} removed"}
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
