"""
Video Tools API

Endpoints for smart video processing features:
- Content analysis
- Smart thumbnails
- Highlight extraction
- AI tagging
- Video search
- Smart crop
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path
import logging
import os

from ..auth import get_current_user

# Import worker modules
from ..worker.content_analyzer import ContentAnalyzer
from ..worker.smart_thumbnails import SmartThumbnailGenerator
from ..worker.highlight_extractor import HighlightExtractor
from ..worker.ai_tagger import AITagger, TagDatabase
from ..worker.video_search import VideoSearch, SearchQuery
from ..worker.smart_crop import SmartCropper
from ..worker.quality_gate import QualityGate

router = APIRouter(prefix="/api/video-tools", tags=["video-tools"])

# Initialize components (paths from environment with Big Hoss defaults)
STORAGE_DIR = Path(os.environ.get("NUBHQ_STORAGE", "/Volumes/Big Hoss/NubHQ"))
OUTPUT_DIR = STORAGE_DIR / "03_Exports"
THUMBNAILS_DIR = OUTPUT_DIR / "thumbnails"
HIGHLIGHTS_DIR = OUTPUT_DIR / "highlights"
DATABASE_DIR = STORAGE_DIR / "database"

# Create directories if they don't exist (graceful handling)
for _dir in [THUMBNAILS_DIR, HIGHLIGHTS_DIR, DATABASE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODELS
# ============================================================

class AnalyzeRequest(BaseModel):
    video_path: str


class AnalyzeResponse(BaseModel):
    content_type: str
    confidence: float
    aspect_ratio: str
    suggested_profile: str
    motion_level: str
    has_face: bool
    reasons: List[str]


class ThumbnailRequest(BaseModel):
    video_path: str
    count: int = 5


class ThumbnailResponse(BaseModel):
    thumbnails: List[Dict]


class HighlightRequest(BaseModel):
    video_path: str
    max_clips: int = 3
    durations: List[int] = [15, 30, 60]


class HighlightResponse(BaseModel):
    highlights: List[Dict]


class TagRequest(BaseModel):
    video_path: str


class TagResponse(BaseModel):
    tags: List[Dict]
    description: str
    suggested_title: Optional[str]
    hashtags: List[str]
    colors: List[str]


class SearchRequest(BaseModel):
    text: Optional[str] = None
    tags: Optional[List[str]] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    resolution: Optional[str] = None
    limit: int = 50


class SearchResponse(BaseModel):
    results: List[Dict]
    total: int


class CropRequest(BaseModel):
    video_path: str
    aspect_ratio: str = "vertical"  # vertical, square, portrait
    method: str = "auto"  # auto, face_tracking, center


class CropResponse(BaseModel):
    success: bool
    output_path: Optional[str]
    method_used: str
    error: Optional[str]


class QualityCheckRequest(BaseModel):
    video_path: str


class QualityCheckResponse(BaseModel):
    passed: bool
    score: float
    can_proceed: bool
    checks: List[Dict]
    errors: List[str]
    warnings: List[str]


# ============================================================
# ENDPOINTS
# ============================================================

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_content(request: AnalyzeRequest, user=Depends(get_current_user)):
    """Analyze video content and suggest processing profile"""
    video_path = Path(request.video_path)

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        analysis = ContentAnalyzer.analyze(video_path)

        return AnalyzeResponse(
            content_type=analysis.content_type.value,
            confidence=analysis.confidence,
            aspect_ratio=analysis.aspect_ratio,
            suggested_profile=analysis.suggested_profile,
            motion_level=analysis.motion_level,
            has_face=analysis.has_face,
            reasons=analysis.reasons
        )
    except Exception as e:
        logging.exception(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/thumbnails", response_model=ThumbnailResponse)
async def generate_thumbnails(request: ThumbnailRequest, user=Depends(get_current_user)):
    """Generate smart thumbnails for a video"""
    video_path = Path(request.video_path)

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        generator = SmartThumbnailGenerator(THUMBNAILS_DIR)
        thumbnails = generator.generate(video_path, count=request.count)

        return ThumbnailResponse(
            thumbnails=[
                {
                    "path": str(t.path),
                    "timestamp": t.timestamp,
                    "score": t.score,
                    "has_face": t.has_face,
                    "reasons": t.reasons
                }
                for t in thumbnails
            ]
        )
    except Exception as e:
        logging.exception(f"Thumbnail generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/highlights", response_model=HighlightResponse)
async def extract_highlights(request: HighlightRequest, user=Depends(get_current_user)):
    """Extract highlight clips from a video"""
    video_path = Path(request.video_path)

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        extractor = HighlightExtractor(HIGHLIGHTS_DIR)
        highlights = extractor.extract_highlights(
            video_path,
            max_clips=request.max_clips,
            target_durations=request.durations
        )

        return HighlightResponse(
            highlights=[
                {
                    "path": str(h.path),
                    "start_time": h.start_time,
                    "end_time": h.end_time,
                    "duration": h.duration,
                    "score": h.score,
                    "reason": h.reason,
                    "format": h.format
                }
                for h in highlights
            ]
        )
    except Exception as e:
        logging.exception(f"Highlight extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tag", response_model=TagResponse)
async def tag_video(request: TagRequest, user=Depends(get_current_user)):
    """Auto-tag a video using AI"""
    video_path = Path(request.video_path)

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        tagger = AITagger()
        result = tagger.tag_video(video_path)

        # Save to database
        db = TagDatabase(DATABASE_DIR / "tags.db")
        db.save(result)

        return TagResponse(
            tags=[
                {
                    "name": t.name,
                    "category": t.category.value,
                    "confidence": t.confidence
                }
                for t in result.tags
            ],
            description=result.description,
            suggested_title=result.suggested_title,
            hashtags=result.suggested_hashtags,
            colors=result.dominant_colors
        )
    except Exception as e:
        logging.exception(f"Tagging failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search_videos(request: SearchRequest, user=Depends(get_current_user)):
    """Search videos by tags, content, and metadata"""
    try:
        search = VideoSearch(
            DATABASE_DIR / "search.db",
            STORAGE_DIR / "02_Library"
        )

        query = SearchQuery(
            text=request.text,
            tags=request.tags,
            min_duration=request.min_duration,
            max_duration=request.max_duration,
            resolution=request.resolution,
            limit=request.limit
        )

        results = search.search(query)

        return SearchResponse(
            results=[
                {
                    "path": str(r.path),
                    "filename": r.filename,
                    "score": r.score,
                    "tags": r.matched_tags,
                    "description": r.description,
                    "thumbnail": str(r.thumbnail) if r.thumbnail else None,
                    "duration": r.duration,
                    "resolution": r.resolution
                }
                for r in results
            ],
            total=len(results)
        )
    except Exception as e:
        logging.exception(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tags")
async def list_all_tags(user=Depends(get_current_user)):
    """Get all unique tags with counts"""
    try:
        db = TagDatabase(DATABASE_DIR / "tags.db")
        tags = db.get_all_tags()

        return {"tags": tags}
    except Exception as e:
        logging.exception(f"Failed to get tags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crop", response_model=CropResponse)
async def smart_crop(request: CropRequest, background_tasks: BackgroundTasks, user=Depends(get_current_user)):
    """Smart crop video with face tracking"""
    video_path = Path(request.video_path)

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        cropper = SmartCropper(OUTPUT_DIR / "cropped")
        result = cropper.crop(
            video_path,
            aspect_ratio=request.aspect_ratio,
            method=request.method
        )

        return CropResponse(
            success=result.success,
            output_path=str(result.output_path) if result.output_path else None,
            method_used=result.crop_method,
            error=result.error
        )
    except Exception as e:
        logging.exception(f"Crop failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality-check", response_model=QualityCheckResponse)
async def quality_check(request: QualityCheckRequest, user=Depends(get_current_user)):
    """Run quality gate checks on a video"""
    video_path = Path(request.video_path)

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        gate = QualityGate()
        report = gate.check(video_path)

        return QualityCheckResponse(
            passed=report.passed,
            score=report.overall_score,
            can_proceed=report.can_proceed,
            checks=[
                {
                    "name": c.check_name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "details": c.details
                }
                for c in report.checks
            ],
            errors=report.errors,
            warnings=report.warnings
        )
    except Exception as e:
        logging.exception(f"Quality check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_tool_stats(user=Depends(get_current_user)):
    """Get statistics for video tools"""
    try:
        search = VideoSearch(DATABASE_DIR / "search.db", STORAGE_DIR / "02_Library")
        search_stats = search.get_stats()

        db = TagDatabase(DATABASE_DIR / "tags.db")
        all_tags = db.get_all_tags()

        return {
            "indexed_videos": search_stats.get("total_videos", 0),
            "total_duration_hours": search_stats.get("total_duration_hours", 0),
            "unique_tags": len(all_tags),
            "top_tags": dict(list(all_tags.items())[:10])
        }
    except Exception as e:
        logging.exception(f"Stats failed: {e}")
        return {"indexed_videos": 0, "unique_tags": 0, "top_tags": {}}
