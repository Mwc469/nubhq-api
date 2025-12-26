"""
Video Pipeline API Routes
=========================
Endpoints for video processing, combining, and feedback.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import json
import os

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
async def create_highlight(request: HighlightRequest, background_tasks: BackgroundTasks):
    """Create a highlight reel from a video"""
    if not HAS_WORKERS:
        raise HTTPException(status_code=503, detail="Worker modules not available")

    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")

    result = create_highlight_reel(
        str(video_path),
        request.duration,
        request.output_name
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
