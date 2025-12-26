"""
Tests for video pipeline API endpoints.
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json

from app.models.job import Job


class TestHealthEndpoint:
    """Test pipeline health endpoint (no auth required)."""

    def test_health_check(self, client):
        """Test health endpoint returns status."""
        response = client.get("/api/video-pipeline/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "workers_available" in data


class TestTemplateEndpoints:
    """Test template CRUD operations."""

    def _get_auth_headers(self, client, test_user):
        """Helper to login and get auth headers."""
        response = client.post(
            "/api/auth/login/json",
            json={"email": test_user.email, "password": "testpassword123"},
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_list_templates(self, client):
        """Test listing templates."""
        response = client.get("/api/video-pipeline/templates")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_create_custom_template(self, client, test_user):
        """Test creating a custom template."""
        headers = self._get_auth_headers(client, test_user)
        template_data = {
            "id": "test-template",
            "name": "Test Template",
            "duration": 60,
            "aspect": "16:9",
            "segments": [
                {"type": "intro", "duration": 5},
                {"type": "highlight", "duration": 50},
                {"type": "outro", "duration": 5}
            ]
        }
        response = client.post(
            "/api/video-pipeline/templates",
            json=template_data,
            headers=headers
        )
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "ok"
        assert data["template"]["id"] == "test-template"

    def test_create_duplicate_template_fails(self, client, test_user):
        """Test creating a duplicate template returns 400."""
        headers = self._get_auth_headers(client, test_user)
        template_data = {
            "id": "duplicate-test",
            "name": "Duplicate Test",
            "duration": 30,
            "aspect": "9:16",
            "segments": [{"type": "highlight", "duration": 30}]
        }
        # Create first
        client.post("/api/video-pipeline/templates", json=template_data, headers=headers)
        # Try to create duplicate
        response = client.post(
            "/api/video-pipeline/templates",
            json=template_data,
            headers=headers
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_create_template_invalid_aspect(self, client, test_user):
        """Test creating template with invalid aspect ratio fails."""
        headers = self._get_auth_headers(client, test_user)
        template_data = {
            "id": "invalid-aspect",
            "name": "Invalid Aspect",
            "duration": 30,
            "aspect": "3:2",  # Invalid
            "segments": []
        }
        response = client.post(
            "/api/video-pipeline/templates",
            json=template_data,
            headers=headers
        )
        assert response.status_code == 400
        assert "Invalid aspect ratio" in response.json()["detail"]

    def test_get_custom_template(self, client, test_user):
        """Test getting a specific template."""
        headers = self._get_auth_headers(client, test_user)
        # Create template first
        template_data = {
            "id": "get-test",
            "name": "Get Test",
            "duration": 30,
            "aspect": "1:1",
            "segments": []
        }
        client.post("/api/video-pipeline/templates", json=template_data, headers=headers)

        # Get it
        response = client.get("/api/video-pipeline/templates/get-test")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "get-test"
        assert data["name"] == "Get Test"

    def test_get_nonexistent_template(self, client):
        """Test getting a template that doesn't exist."""
        response = client.get("/api/video-pipeline/templates/nonexistent")
        assert response.status_code == 404

    def test_update_custom_template(self, client, test_user):
        """Test updating a custom template."""
        headers = self._get_auth_headers(client, test_user)
        # Create template first
        template_data = {
            "id": "update-test",
            "name": "Update Test",
            "duration": 30,
            "aspect": "16:9",
            "segments": []
        }
        client.post("/api/video-pipeline/templates", json=template_data, headers=headers)

        # Update it
        update_data = {"name": "Updated Name", "duration": 45}
        response = client.put(
            "/api/video-pipeline/templates/update-test",
            json=update_data,
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["template"]["name"] == "Updated Name"
        assert data["template"]["duration"] == 45

    def test_update_nonexistent_template(self, client, test_user):
        """Test updating a template that doesn't exist."""
        headers = self._get_auth_headers(client, test_user)
        response = client.put(
            "/api/video-pipeline/templates/nonexistent",
            json={"name": "New Name"},
            headers=headers
        )
        assert response.status_code == 404

    def test_delete_custom_template(self, client, test_user):
        """Test deleting a custom template."""
        headers = self._get_auth_headers(client, test_user)
        # Create template first
        template_data = {
            "id": "delete-test",
            "name": "Delete Test",
            "duration": 30,
            "aspect": "9:16",
            "segments": []
        }
        client.post("/api/video-pipeline/templates", json=template_data, headers=headers)

        # Delete it
        response = client.delete(
            "/api/video-pipeline/templates/delete-test",
            headers=headers
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        # Verify it's gone
        response = client.get("/api/video-pipeline/templates/delete-test")
        assert response.status_code == 404

    def test_delete_nonexistent_template(self, client, test_user):
        """Test deleting a template that doesn't exist."""
        headers = self._get_auth_headers(client, test_user)
        response = client.delete(
            "/api/video-pipeline/templates/nonexistent",
            headers=headers
        )
        assert response.status_code == 404

    def test_template_requires_auth(self, client, db):
        """Test that modifying templates requires authentication."""
        template_data = {
            "id": "auth-test",
            "name": "Auth Test",
            "duration": 30,
            "aspect": "16:9",
            "segments": []
        }
        response = client.post("/api/video-pipeline/templates", json=template_data)
        assert response.status_code == 401


class TestJobEndpoints:
    """Test job management endpoints."""

    def _get_auth_headers(self, client, test_user):
        """Helper to login and get auth headers."""
        response = client.post(
            "/api/auth/login/json",
            json={"email": test_user.email, "password": "testpassword123"},
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_list_jobs_empty(self, client, test_user):
        """Test listing jobs when none exist."""
        headers = self._get_auth_headers(client, test_user)
        response = client.get("/api/video-pipeline/jobs", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    def test_create_test_job(self, client, test_user):
        """Test creating a test job."""
        headers = self._get_auth_headers(client, test_user)
        response = client.post("/api/video-pipeline/jobs/test", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "job" in data
        assert data["job"]["type"] == "test"
        assert data["job"]["status"] == "completed"

    def test_list_jobs_after_create(self, client, test_user):
        """Test listing jobs after creating one."""
        headers = self._get_auth_headers(client, test_user)
        # Create a test job
        client.post("/api/video-pipeline/jobs/test", headers=headers)

        # List jobs
        response = client.get("/api/video-pipeline/jobs", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert len(data["jobs"]) >= 1

    def test_list_jobs_filter_by_status(self, client, test_user):
        """Test filtering jobs by status."""
        headers = self._get_auth_headers(client, test_user)
        # Create a test job (completed)
        client.post("/api/video-pipeline/jobs/test", headers=headers)

        # Filter by completed
        response = client.get(
            "/api/video-pipeline/jobs?status=completed",
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        for job in data["jobs"]:
            assert job["status"] == "completed"

    def test_list_jobs_filter_by_type(self, client, test_user):
        """Test filtering jobs by type."""
        headers = self._get_auth_headers(client, test_user)
        # Create a test job
        client.post("/api/video-pipeline/jobs/test", headers=headers)

        # Filter by type
        response = client.get(
            "/api/video-pipeline/jobs?job_type=test",
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        for job in data["jobs"]:
            assert job["type"] == "test"

    def test_get_job_progress(self, client, test_user):
        """Test getting job progress."""
        headers = self._get_auth_headers(client, test_user)
        # Create a test job
        create_response = client.post("/api/video-pipeline/jobs/test", headers=headers)
        job_id = create_response.json()["job"]["job_id"]

        # Get progress
        response = client.get(f"/api/video-pipeline/progress/{job_id}")
        assert response.status_code == 200
        data = response.json()
        # The progress endpoint returns "id" from the Job model's to_dict()
        assert data.get("id") == job_id or data.get("job_id") == job_id

    def test_get_job_progress_unknown(self, client, db):
        """Test getting progress for unknown job."""
        response = client.get("/api/video-pipeline/progress/unknown-job-id")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unknown"

    def test_delete_job(self, client, test_user):
        """Test deleting a job."""
        headers = self._get_auth_headers(client, test_user)
        # Create a test job
        create_response = client.post("/api/video-pipeline/jobs/test", headers=headers)
        job_id = create_response.json()["job"]["job_id"]

        # Delete it
        response = client.delete(
            f"/api/video-pipeline/jobs/{job_id}",
            headers=headers
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_delete_nonexistent_job(self, client, test_user):
        """Test deleting a job that doesn't exist."""
        headers = self._get_auth_headers(client, test_user)
        response = client.delete(
            "/api/video-pipeline/jobs/nonexistent",
            headers=headers
        )
        assert response.status_code == 404

    def test_job_stats(self, client, test_user):
        """Test getting job statistics."""
        headers = self._get_auth_headers(client, test_user)
        # Create some test jobs
        client.post("/api/video-pipeline/jobs/test", headers=headers)
        client.post("/api/video-pipeline/jobs/test", headers=headers)

        # Get stats
        response = client.get("/api/video-pipeline/jobs/stats", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_jobs" in data
        assert "by_status" in data
        assert "by_type" in data
        assert data["total_jobs"] >= 2

    def test_cleanup_jobs(self, client, test_user):
        """Test cleaning up old jobs."""
        headers = self._get_auth_headers(client, test_user)
        # Create a test job
        client.post("/api/video-pipeline/jobs/test", headers=headers)

        # Cleanup with days=0 to delete all
        response = client.delete(
            "/api/video-pipeline/jobs/cleanup?days=0&status=completed",
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "deleted_count" in data

    def test_retry_job_not_failed(self, client, test_user):
        """Test retrying a job that isn't in failed state."""
        headers = self._get_auth_headers(client, test_user)
        # Create a test job (which completes successfully)
        create_response = client.post("/api/video-pipeline/jobs/test", headers=headers)
        job_id = create_response.json()["job"]["job_id"]

        # Try to retry it
        response = client.post(
            f"/api/video-pipeline/jobs/{job_id}/retry",
            headers=headers
        )
        assert response.status_code == 400
        assert "not in failed state" in response.json()["detail"]

    def test_retry_nonexistent_job(self, client, test_user):
        """Test retrying a job that doesn't exist."""
        headers = self._get_auth_headers(client, test_user)
        response = client.post(
            "/api/video-pipeline/jobs/nonexistent/retry",
            headers=headers
        )
        assert response.status_code == 404

    def test_jobs_require_auth(self, client, db):
        """Test that job endpoints require authentication."""
        response = client.get("/api/video-pipeline/jobs")
        assert response.status_code == 401

        response = client.post("/api/video-pipeline/jobs/test")
        assert response.status_code == 401

        response = client.get("/api/video-pipeline/jobs/stats")
        assert response.status_code == 401


class TestActivityLogEndpoints:
    """Test activity log endpoints."""

    def _get_auth_headers(self, client, test_user):
        """Helper to login and get auth headers."""
        response = client.post(
            "/api/auth/login/json",
            json={"email": test_user.email, "password": "testpassword123"},
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_get_activity_log_empty(self, client, test_user):
        """Test getting activity log when empty."""
        headers = self._get_auth_headers(client, test_user)
        response = client.get("/api/video-pipeline/activity", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "activities" in data
        assert "total" in data

    def test_get_activity_log_with_limit(self, client, test_user):
        """Test getting activity log with limit parameter."""
        headers = self._get_auth_headers(client, test_user)
        response = client.get(
            "/api/video-pipeline/activity?limit=5",
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["activities"]) <= 5

    def test_clear_activity_log(self, client, test_user):
        """Test clearing activity log."""
        headers = self._get_auth_headers(client, test_user)
        response = client.delete("/api/video-pipeline/activity", headers=headers)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_activity_requires_auth(self, client, db):
        """Test that activity endpoints require authentication."""
        response = client.get("/api/video-pipeline/activity")
        assert response.status_code == 401

        response = client.delete("/api/video-pipeline/activity")
        assert response.status_code == 401


class TestVideoProcessingEndpoints:
    """Test video processing endpoints (mocked)."""

    def test_highlight_requires_auth(self, client, db):
        """Test highlight endpoint requires authentication."""
        response = client.post(
            "/api/video-pipeline/highlight",
            json={"video_path": "/tmp/test.mp4", "duration": 60}
        )
        assert response.status_code == 401

    def test_sync_requires_auth(self, client, db):
        """Test sync endpoint requires authentication."""
        response = client.post(
            "/api/video-pipeline/sync",
            json={"video_paths": ["/tmp/test1.mp4", "/tmp/test2.mp4"]}
        )
        assert response.status_code == 401

    def test_compile_requires_auth(self, client, db):
        """Test compile endpoint requires authentication."""
        response = client.post(
            "/api/video-pipeline/compile",
            json={"template_id": "instagram_reel", "source_videos": ["/tmp/test.mp4"]}
        )
        assert response.status_code == 401

    def test_engagement_requires_auth(self, client, db):
        """Test engagement endpoint requires authentication."""
        response = client.post(
            "/api/video-pipeline/engagement",
            json={"video_path": "/tmp/test.mp4"}
        )
        assert response.status_code == 401

    def test_feedback_requires_auth(self, client, db):
        """Test feedback endpoint requires authentication."""
        response = client.post(
            "/api/video-pipeline/feedback",
            json={
                "video_fingerprint": "abc123",
                "approved": True,
                "engagement_score": 0.8,
                "engagement_confidence": 0.9
            }
        )
        assert response.status_code == 401

    def test_stats_requires_auth(self, client, db):
        """Test stats endpoint requires authentication."""
        response = client.get("/api/video-pipeline/stats")
        assert response.status_code == 401

    def test_thumbnails_requires_auth(self, client, db):
        """Test thumbnails endpoint requires authentication."""
        response = client.post(
            "/api/video-pipeline/thumbnails",
            json={"video_path": "/tmp/test.mp4"}
        )
        assert response.status_code == 401

    def test_batch_requires_auth(self, client, db):
        """Test batch endpoint requires authentication."""
        response = client.post(
            "/api/video-pipeline/batch",
            json={"video_paths": ["/tmp/test.mp4"]}
        )
        assert response.status_code == 401

    def test_export_all_requires_auth(self, client, db):
        """Test export-all endpoint requires authentication."""
        response = client.post(
            "/api/video-pipeline/export-all",
            json={"video_path": "/tmp/test.mp4"}
        )
        assert response.status_code == 401

    def test_watermark_requires_auth(self, client, db):
        """Test watermark endpoint requires authentication."""
        response = client.post(
            "/api/video-pipeline/watermark",
            json={
                "video_path": "/tmp/test.mp4",
                "watermark_path": "/tmp/logo.png"
            }
        )
        assert response.status_code == 401

    def test_caption_requires_auth(self, client, db):
        """Test caption endpoint requires authentication."""
        response = client.post(
            "/api/video-pipeline/caption",
            json={"video_path": "/tmp/test.mp4"}
        )
        assert response.status_code == 401


class TestReviewQueueEndpoints:
    """Test review queue endpoints."""

    def test_review_queue_requires_auth(self, client, db):
        """Test review queue endpoint requires authentication."""
        response = client.get("/api/video-pipeline/review")
        assert response.status_code == 401

    def test_approve_review_requires_auth(self, client, db):
        """Test approve review endpoint requires authentication."""
        response = client.post("/api/video-pipeline/review/test.mp4/approve")
        assert response.status_code == 401

    def test_reject_review_requires_auth(self, client, db):
        """Test reject review endpoint requires authentication."""
        response = client.delete("/api/video-pipeline/review/test.mp4")
        assert response.status_code == 401


class TestPathValidation:
    """Test path validation security."""

    def _get_auth_headers(self, client, test_user):
        """Helper to login and get auth headers."""
        response = client.post(
            "/api/auth/login/json",
            json={"email": test_user.email, "password": "testpassword123"},
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_invalid_file_extension(self, client, test_user):
        """Test that invalid file extensions are rejected."""
        headers = self._get_auth_headers(client, test_user)
        response = client.post(
            "/api/video-pipeline/highlight",
            json={"video_path": "/tmp/test.txt", "duration": 60},
            headers=headers
        )
        # Should get 400 for invalid extension (if workers available) or 503 if not
        assert response.status_code in [400, 503]

    def test_path_traversal_blocked(self, client, test_user):
        """Test that path traversal is blocked."""
        headers = self._get_auth_headers(client, test_user)
        response = client.post(
            "/api/video-pipeline/highlight",
            json={"video_path": "/etc/passwd", "duration": 60},
            headers=headers
        )
        # Should get 400 for invalid extension or 403 for path access denied or 503 if workers unavailable
        assert response.status_code in [400, 403, 503]


class TestSSEProgressStream:
    """Test Server-Sent Events progress streaming."""

    @pytest.mark.skip(reason="SSE streaming test requires async handling, skip for CI")
    def test_progress_stream_endpoint_exists(self, client, db):
        """Test that the SSE progress stream endpoint returns correct content type."""
        # SSE endpoints stream indefinitely, so we just verify it starts correctly
        # by checking the response object properties (not consuming the stream)
        with client:
            # The endpoint should respond with SSE media type
            response = client.get("/api/video-pipeline/progress/test-job/stream", stream=True)
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            # Close immediately to avoid hanging
            response.close()
