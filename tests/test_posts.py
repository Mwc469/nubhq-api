"""
Tests for posts endpoints.
"""
import pytest
from app.models.post import Post


class TestPostsEndpoints:
    """Test posts endpoints."""

    def test_create_post(self, client, test_user, db):
        """Test creating a post."""
        # Login to get token
        login_response = client.post(
            "/api/auth/login/json",
            json={"email": "test@example.com", "password": "testpassword123"},
        )
        token = login_response.json()["access_token"]

        response = client.post(
            "/api/posts",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "content": "Test post content",
                "platform": "instagram",
                "post_type": "image",
                "hashtags": ["test", "nubhq"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Test post content"
        assert data["platform"] == "instagram"
        assert data["status"] == "draft"

    def test_create_post_unauthenticated(self, client):
        """Test creating a post without auth fails."""
        response = client.post(
            "/api/posts",
            json={
                "content": "Test content",
            },
        )
        assert response.status_code == 401

    def test_get_posts_empty(self, client, test_user):
        """Test getting posts when none exist."""
        login_response = client.post(
            "/api/auth/login/json",
            json={"email": "test@example.com", "password": "testpassword123"},
        )
        token = login_response.json()["access_token"]

        response = client.get(
            "/api/posts",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json() == []

    def test_get_posts(self, client, test_user, db):
        """Test getting posts."""
        # Create a post first
        post = Post(
            user_id=test_user.id,
            content="Test post",
            platform="instagram",
            post_type="image",
            status="draft",
        )
        db.add(post)
        db.commit()

        # Login and get posts
        login_response = client.post(
            "/api/auth/login/json",
            json={"email": "test@example.com", "password": "testpassword123"},
        )
        token = login_response.json()["access_token"]

        response = client.get(
            "/api/posts",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["content"] == "Test post"

    def test_get_post_by_id(self, client, test_user, db):
        """Test getting a specific post."""
        post = Post(
            user_id=test_user.id,
            content="Specific post",
            platform="twitter",
            post_type="text",
            status="draft",
        )
        db.add(post)
        db.commit()

        login_response = client.post(
            "/api/auth/login/json",
            json={"email": "test@example.com", "password": "testpassword123"},
        )
        token = login_response.json()["access_token"]

        response = client.get(
            f"/api/posts/{post.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Specific post"

    def test_get_post_not_found(self, client, test_user):
        """Test getting non-existent post."""
        login_response = client.post(
            "/api/auth/login/json",
            json={"email": "test@example.com", "password": "testpassword123"},
        )
        token = login_response.json()["access_token"]

        response = client.get(
            "/api/posts/99999",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404

    def test_update_post(self, client, test_user, db):
        """Test updating a post."""
        post = Post(
            user_id=test_user.id,
            content="Original content",
            platform="instagram",
            post_type="image",
            status="draft",
        )
        db.add(post)
        db.commit()

        login_response = client.post(
            "/api/auth/login/json",
            json={"email": "test@example.com", "password": "testpassword123"},
        )
        token = login_response.json()["access_token"]

        response = client.patch(
            f"/api/posts/{post.id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"content": "Updated content"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Updated content"

    def test_delete_post(self, client, test_user, db):
        """Test deleting a post."""
        post = Post(
            user_id=test_user.id,
            content="To be deleted",
            platform="instagram",
            post_type="image",
            status="draft",
        )
        db.add(post)
        db.commit()
        post_id = post.id

        login_response = client.post(
            "/api/auth/login/json",
            json={"email": "test@example.com", "password": "testpassword123"},
        )
        token = login_response.json()["access_token"]

        response = client.delete(
            f"/api/posts/{post_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

        # Verify deleted
        response = client.get(
            f"/api/posts/{post_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404

    def test_data_isolation(self, client, db):
        """Test that users can only see their own posts."""
        from app.models.user import User
        from app.auth import get_password_hash

        # Create two users
        user1 = User(
            email="user1@example.com",
            hashed_password=get_password_hash("password1"),
            display_name="User 1",
        )
        user2 = User(
            email="user2@example.com",
            hashed_password=get_password_hash("password2"),
            display_name="User 2",
        )
        db.add_all([user1, user2])
        db.commit()

        # Create posts for each user
        post1 = Post(user_id=user1.id, content="User 1 post", platform="instagram", post_type="image", status="draft")
        post2 = Post(user_id=user2.id, content="User 2 post", platform="instagram", post_type="image", status="draft")
        db.add_all([post1, post2])
        db.commit()

        # User 1 logs in and should only see their post
        login_response1 = client.post(
            "/api/auth/login/json",
            json={"email": "user1@example.com", "password": "password1"},
        )
        token1 = login_response1.json()["access_token"]

        response = client.get(
            "/api/posts",
            headers={"Authorization": f"Bearer {token1}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["content"] == "User 1 post"

        # User 2 logs in and should only see their post
        login_response2 = client.post(
            "/api/auth/login/json",
            json={"email": "user2@example.com", "password": "password2"},
        )
        token2 = login_response2.json()["access_token"]

        response = client.get(
            "/api/posts",
            headers={"Authorization": f"Bearer {token2}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["content"] == "User 2 post"

        # User 1 cannot access User 2's post directly
        response = client.get(
            f"/api/posts/{post2.id}",
            headers={"Authorization": f"Bearer {token1}"},
        )
        assert response.status_code == 404
