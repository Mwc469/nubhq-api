"""
Platform Upload Integration

Upload processed videos to:
- YouTube (via API)
- TikTok (via API or scheduling)
- Instagram (via Meta API)

Requires API credentials configured via environment variables.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Platform(Enum):
    """Supported upload platforms"""
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"


@dataclass
class UploadResult:
    """Result of an upload attempt"""
    success: bool
    platform: Platform
    video_id: Optional[str]
    url: Optional[str]
    error: Optional[str]
    scheduled_time: Optional[datetime]


@dataclass
class UploadRequest:
    """Request to upload a video"""
    video_path: Path
    platform: Platform
    title: str
    description: str
    tags: List[str]
    thumbnail_path: Optional[Path] = None
    scheduled_time: Optional[datetime] = None  # None = publish immediately
    privacy: str = "private"  # private, unlisted, public
    category: str = "Entertainment"


# ============================================================
# YOUTUBE UPLOAD
# ============================================================

class YouTubeUploader:
    """Upload videos to YouTube using the Data API v3"""

    # OAuth scopes required
    SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

    def __init__(self):
        self.client_secrets = os.environ.get('YOUTUBE_CLIENT_SECRETS')
        self.credentials_path = Path(os.environ.get('YOUTUBE_CREDENTIALS', 'youtube_credentials.json'))

    def is_configured(self) -> bool:
        """Check if YouTube upload is configured"""
        return bool(self.client_secrets) or self.credentials_path.exists()

    def upload(self, request: UploadRequest) -> UploadResult:
        """Upload a video to YouTube"""
        if not self.is_configured():
            return UploadResult(
                success=False,
                platform=Platform.YOUTUBE,
                video_id=None,
                url=None,
                error="YouTube API not configured. Set YOUTUBE_CLIENT_SECRETS env var.",
                scheduled_time=None
            )

        try:
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload

            # Get credentials
            credentials = self._get_credentials()
            if not credentials:
                return UploadResult(
                    success=False,
                    platform=Platform.YOUTUBE,
                    video_id=None,
                    url=None,
                    error="Failed to get YouTube credentials",
                    scheduled_time=None
                )

            # Build YouTube service
            youtube = build('youtube', 'v3', credentials=credentials)

            # Prepare video metadata
            body = {
                'snippet': {
                    'title': request.title[:100],  # Max 100 chars
                    'description': request.description[:5000],  # Max 5000 chars
                    'tags': request.tags[:500],  # Max 500 tags
                    'categoryId': self._get_category_id(request.category),
                },
                'status': {
                    'privacyStatus': request.privacy,
                    'selfDeclaredMadeForKids': False,
                }
            }

            # Add scheduled publish time if specified
            if request.scheduled_time:
                body['status']['publishAt'] = request.scheduled_time.isoformat() + 'Z'
                body['status']['privacyStatus'] = 'private'  # Must be private for scheduling

            # Upload video
            media = MediaFileUpload(
                str(request.video_path),
                mimetype='video/mp4',
                resumable=True,
                chunksize=1024*1024*10  # 10MB chunks
            )

            insert_request = youtube.videos().insert(
                part='snippet,status',
                body=body,
                media_body=media
            )

            response = None
            while response is None:
                status, response = insert_request.next_chunk()
                if status:
                    logging.info(f"YouTube upload progress: {int(status.progress() * 100)}%")

            video_id = response['id']
            video_url = f"https://youtube.com/watch?v={video_id}"

            # Upload thumbnail if provided
            if request.thumbnail_path and request.thumbnail_path.exists():
                self._upload_thumbnail(youtube, video_id, request.thumbnail_path)

            logging.info(f"YouTube upload complete: {video_url}")

            return UploadResult(
                success=True,
                platform=Platform.YOUTUBE,
                video_id=video_id,
                url=video_url,
                error=None,
                scheduled_time=request.scheduled_time
            )

        except Exception as e:
            logging.exception(f"YouTube upload failed: {e}")
            return UploadResult(
                success=False,
                platform=Platform.YOUTUBE,
                video_id=None,
                url=None,
                error=str(e),
                scheduled_time=None
            )

    def _get_credentials(self):
        """Get or refresh OAuth credentials"""
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request

        credentials = None

        # Load existing credentials
        if self.credentials_path.exists():
            try:
                credentials = Credentials.from_authorized_user_file(
                    str(self.credentials_path),
                    self.SCOPES
                )
            except Exception:
                pass

        # Refresh or get new credentials
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                if not self.client_secrets:
                    return None

                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secrets,
                    self.SCOPES
                )
                credentials = flow.run_local_server(port=0)

            # Save credentials
            with open(self.credentials_path, 'w') as f:
                f.write(credentials.to_json())

        return credentials

    def _upload_thumbnail(self, youtube, video_id: str, thumbnail_path: Path):
        """Upload custom thumbnail"""
        try:
            from googleapiclient.http import MediaFileUpload

            media = MediaFileUpload(str(thumbnail_path), mimetype='image/jpeg')
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=media
            ).execute()
            logging.info(f"Thumbnail uploaded for {video_id}")
        except Exception as e:
            logging.warning(f"Thumbnail upload failed: {e}")

    def _get_category_id(self, category: str) -> str:
        """Map category name to YouTube category ID"""
        categories = {
            'Film & Animation': '1',
            'Autos & Vehicles': '2',
            'Music': '10',
            'Pets & Animals': '15',
            'Sports': '17',
            'Gaming': '20',
            'People & Blogs': '22',
            'Comedy': '23',
            'Entertainment': '24',
            'News & Politics': '25',
            'Howto & Style': '26',
            'Education': '27',
            'Science & Technology': '28',
        }
        return categories.get(category, '24')  # Default to Entertainment


# ============================================================
# TIKTOK UPLOAD (Stub - requires TikTok for Developers account)
# ============================================================

class TikTokUploader:
    """Upload videos to TikTok"""

    def __init__(self):
        self.access_token = os.environ.get('TIKTOK_ACCESS_TOKEN')

    def is_configured(self) -> bool:
        return bool(self.access_token)

    def upload(self, request: UploadRequest) -> UploadResult:
        """Upload to TikTok"""
        if not self.is_configured():
            return UploadResult(
                success=False,
                platform=Platform.TIKTOK,
                video_id=None,
                url=None,
                error="TikTok API not configured. Set TIKTOK_ACCESS_TOKEN env var.",
                scheduled_time=None
            )

        # TikTok Content Posting API implementation would go here
        # Requires approved TikTok for Developers application
        return UploadResult(
            success=False,
            platform=Platform.TIKTOK,
            video_id=None,
            url=None,
            error="TikTok upload not yet implemented",
            scheduled_time=None
        )


# ============================================================
# INSTAGRAM UPLOAD (Stub - requires Meta Business account)
# ============================================================

class InstagramUploader:
    """Upload videos to Instagram"""

    def __init__(self):
        self.access_token = os.environ.get('INSTAGRAM_ACCESS_TOKEN')
        self.account_id = os.environ.get('INSTAGRAM_ACCOUNT_ID')

    def is_configured(self) -> bool:
        return bool(self.access_token and self.account_id)

    def upload(self, request: UploadRequest) -> UploadResult:
        """Upload to Instagram"""
        if not self.is_configured():
            return UploadResult(
                success=False,
                platform=Platform.INSTAGRAM,
                video_id=None,
                url=None,
                error="Instagram API not configured. Set INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_ACCOUNT_ID.",
                scheduled_time=None
            )

        # Instagram Graph API implementation would go here
        # Requires Meta Business account and approved app
        return UploadResult(
            success=False,
            platform=Platform.INSTAGRAM,
            video_id=None,
            url=None,
            error="Instagram upload not yet implemented",
            scheduled_time=None
        )


# ============================================================
# UNIFIED UPLOADER
# ============================================================

class PlatformUploader:
    """Unified interface for all platform uploads"""

    def __init__(self):
        self.uploaders = {
            Platform.YOUTUBE: YouTubeUploader(),
            Platform.TIKTOK: TikTokUploader(),
            Platform.INSTAGRAM: InstagramUploader(),
        }

    def get_configured_platforms(self) -> List[Platform]:
        """Get list of configured platforms"""
        return [p for p, u in self.uploaders.items() if u.is_configured()]

    def upload(self, request: UploadRequest) -> UploadResult:
        """Upload to specified platform"""
        uploader = self.uploaders.get(request.platform)
        if not uploader:
            return UploadResult(
                success=False,
                platform=request.platform,
                video_id=None,
                url=None,
                error=f"Unknown platform: {request.platform}",
                scheduled_time=None
            )

        return uploader.upload(request)

    def upload_to_all(self, request: UploadRequest) -> Dict[Platform, UploadResult]:
        """Upload to all configured platforms"""
        results = {}
        for platform in self.get_configured_platforms():
            req = UploadRequest(
                video_path=request.video_path,
                platform=platform,
                title=request.title,
                description=request.description,
                tags=request.tags,
                thumbnail_path=request.thumbnail_path,
                scheduled_time=request.scheduled_time,
                privacy=request.privacy,
                category=request.category
            )
            results[platform] = self.upload(req)
        return results
