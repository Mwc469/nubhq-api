"""Seed script to populate database with sample data"""
from datetime import datetime, timedelta
from app.database import SessionLocal, engine, Base
from app.models import (
    Approval, FanMessage, UserSettings, ScheduledPost, TrainingExample,
    Post, Template, Activity, EmailCampaign, Media
)

# Create tables
Base.metadata.create_all(bind=engine)

db = SessionLocal()

# Clear existing data
db.query(Approval).delete()
db.query(FanMessage).delete()
db.query(UserSettings).delete()
db.query(ScheduledPost).delete()
db.query(TrainingExample).delete()
db.query(Post).delete()
db.query(Template).delete()
db.query(Activity).delete()
db.query(EmailCampaign).delete()
db.query(Media).delete()

now = datetime.utcnow()

# Sample approvals
approvals = [
    Approval(
        type="message",
        content="Hey! Thanks for subscribing. I really appreciate your support!",
        recipient="user_123",
        status="pending"
    ),
    Approval(
        type="message",
        content="Special content coming your way this weekend!",
        recipient="user_456",
        status="pending"
    ),
    Approval(
        type="message",
        content="Thanks for the tip! You are amazing.",
        recipient="user_789",
        status="pending"
    ),
    Approval(
        type="message",
        content="Welcome to the exclusive tier! You now have access to all content.",
        recipient="user_101",
        status="pending"
    ),
    Approval(
        type="message",
        content="Hope you enjoyed today's stream!",
        recipient="user_202",
        status="approved"
    ),
]

# Sample fan messages
fan_messages = [
    FanMessage(
        sender_id="fan_001",
        sender_name="JohnDoe",
        content="Love your content! Keep it up!",
        is_read=False
    ),
    FanMessage(
        sender_id="fan_002",
        sender_name="JaneSmith",
        content="When is the next live stream?",
        is_read=False
    ),
    FanMessage(
        sender_id="fan_003",
        sender_name="CoolFan42",
        content="You're my favorite creator!",
        is_read=True
    ),
    FanMessage(
        sender_id="fan_004",
        sender_name="SuperSupporter",
        content="Just subscribed to the top tier!",
        is_read=False
    ),
]

# Sample scheduled posts (calendar)
scheduled_posts = [
    ScheduledPost(
        title="New Year Special Post",
        content="Celebrating the new year with you all!",
        scheduled_at=now + timedelta(days=1, hours=10),
        status="scheduled"
    ),
    ScheduledPost(
        title="Behind the Scenes",
        content="A look at my creative process",
        scheduled_at=now + timedelta(days=3, hours=14),
        status="scheduled"
    ),
    ScheduledPost(
        title="Q&A Session",
        content="Answer your questions live",
        scheduled_at=now + timedelta(days=5, hours=18),
        status="scheduled"
    ),
    ScheduledPost(
        title="Exclusive Content Drop",
        content="New exclusive content for subscribers",
        scheduled_at=now + timedelta(days=7, hours=12),
        status="scheduled"
    ),
    ScheduledPost(
        title="Weekend Livestream",
        content="Join me for a fun weekend stream",
        scheduled_at=now + timedelta(days=10, hours=20),
        status="scheduled"
    ),
    ScheduledPost(
        title="Monthly Recap",
        content="Highlights from this month",
        scheduled_at=now + timedelta(days=14, hours=16),
        status="scheduled"
    ),
]

# Sample training examples
training_examples = [
    TrainingExample(
        category="greeting",
        input_message="Hey! Just subscribed!",
        response="Welcome to the family! So glad to have you here. You're going to love it!"
    ),
    TrainingExample(
        category="greeting",
        input_message="Hi, I'm new here",
        response="Hey there! Welcome! Feel free to ask me anything. I'm excited to connect with you!"
    ),
    TrainingExample(
        category="thanks",
        input_message="Thanks for the amazing content!",
        response="Aww thank you so much! Comments like this keep me going. You're the best!"
    ),
    TrainingExample(
        category="thanks",
        input_message="Just wanted to say I appreciate you",
        response="This means the world to me! Thank you for being such an amazing supporter!"
    ),
    TrainingExample(
        category="question",
        input_message="When do you usually post?",
        response="I try to post new content 3-4 times a week, usually in the evenings. Stay tuned!"
    ),
    TrainingExample(
        category="question",
        input_message="Do you do custom content?",
        response="Yes! DM me with what you have in mind and we can discuss the details."
    ),
    TrainingExample(
        category="promo",
        input_message="Any deals coming up?",
        response="Keep an eye out this weekend! I might have something special for my loyal fans."
    ),
    TrainingExample(
        category="compliment",
        input_message="You're so beautiful!",
        response="You're too sweet! Thank you for making me smile today!"
    ),
]

# Sample posts (Post Studio)
posts = [
    Post(
        content="Behind the scenes 📸 #content #creator",
        platform="instagram",
        post_type="image",
        status="published",
        hashtags=["content", "creator"],
        media_urls=["https://picsum.photos/400/400?random=1"],
        published_at=now - timedelta(days=2),
        engagement={"likes": 1234, "comments": 56, "shares": 23},
    ),
    Post(
        content="New content dropping soon! 🔥",
        platform="twitter",
        post_type="text",
        status="scheduled",
        hashtags=[],
        media_urls=[],
        scheduled_at=now + timedelta(days=3),
    ),
    Post(
        content="Thank you all for 10K! 🎉",
        platform="instagram",
        post_type="reel",
        status="draft",
        hashtags=["milestone", "thankyou"],
        media_urls=[],
    ),
    Post(
        content="Check out my latest video! Link in bio 👆",
        platform="tiktok",
        post_type="video",
        status="published",
        hashtags=["newvideo", "content"],
        media_urls=["https://picsum.photos/400/400?random=2"],
        published_at=now - timedelta(days=5),
        engagement={"likes": 5678, "comments": 234, "shares": 89, "views": 45000},
    ),
    Post(
        content="Big announcement coming this week! Stay tuned 👀",
        platform="twitter",
        post_type="text",
        status="scheduled",
        hashtags=["announcement"],
        media_urls=[],
        scheduled_at=now + timedelta(days=1),
    ),
]

# Sample templates
templates = [
    Template(
        name="Flirty Response",
        content="Hey cutie! Thanks for the love 💕 You always know how to make my day...",
        category="responses",
        is_favorite=True,
        use_count=234,
    ),
    Template(
        name="New Post Alert",
        content="🔥 Something spicy just dropped! Check my latest...",
        category="captions",
        is_favorite=True,
        use_count=189,
    ),
    Template(
        name="Thank You Message",
        content="You are amazing! Thank you so much for your support...",
        category="greetings",
        is_favorite=False,
        use_count=156,
    ),
    Template(
        name="Special Offer",
        content="🎉 Exclusive deal just for you! For the next 24 hours...",
        category="promotions",
        is_favorite=False,
        use_count=98,
    ),
    Template(
        name="Welcome New Fan",
        content="Welcome to the crew! 🎉 So happy to have you here...",
        category="greetings",
        is_favorite=True,
        use_count=312,
    ),
    Template(
        name="Behind the Scenes",
        content="Want to see how I make my content? 👀 Here's a peek...",
        category="captions",
        platform="instagram",
        hashtags="#bts #behindthescenes #content",
        is_favorite=False,
        use_count=45,
    ),
]

# Sample activities
activities = [
    Activity(
        activity_type="approval",
        title="Message approved",
        description="Reply to @superfan123 approved and sent",
        extra_data={"recipient": "@superfan123"},
    ),
    Activity(
        activity_type="ai",
        title="AI generated response",
        description="Created reply for message from @newbie_fan",
        extra_data={"model": "nub-ai-v1"},
    ),
    Activity(
        activity_type="post",
        title="Post scheduled",
        description="Instagram post scheduled for tomorrow at 2:00 PM",
        extra_data={"platform": "instagram", "scheduled_for": "2:00 PM"},
    ),
    Activity(
        activity_type="message",
        title="New fan message",
        description="Received message from @loyal_supporter",
        extra_data={"sender": "@loyal_supporter"},
    ),
    Activity(
        activity_type="approval",
        title="Message rejected",
        description="Reply to @spam_user rejected",
        extra_data={"recipient": "@spam_user", "reason": "inappropriate content"},
    ),
    Activity(
        activity_type="ai",
        title="AI training updated",
        description="Voice profile updated with 5 new samples",
        extra_data={"samples_added": 5},
    ),
    Activity(
        activity_type="post",
        title="Post published",
        description='Twitter post "Behind the scenes 📸" went live',
        extra_data={"platform": "twitter", "post_id": 1},
    ),
    Activity(
        activity_type="system",
        title="System backup",
        description="Daily backup completed successfully",
        extra_data={"backup_size": "2.3GB"},
    ),
]

# Sample email campaigns
email_campaigns = [
    EmailCampaign(
        name="Weekly Newsletter",
        subject="This Week's Exclusive Content",
        content="<h1>Hey there!</h1><p>Check out what I've been up to this week...</p>",
        status="sent",
        recipient_count=2450,
        sent_at=now - timedelta(days=2),
        stats={"opens": 1666, "clicks": 588, "open_rate": 68, "click_rate": 24},
    ),
    EmailCampaign(
        name="New Content Alert",
        subject="Something Special Just Dropped!",
        content="<h1>You won't want to miss this!</h1><p>I just posted something amazing...</p>",
        status="scheduled",
        recipient_count=1890,
        scheduled_at=now + timedelta(days=1, hours=9),
    ),
    EmailCampaign(
        name="Exclusive Offer",
        subject="24 Hours Only - Don't Miss Out!",
        content="<h1>Limited Time Offer</h1><p>For my most loyal fans...</p>",
        status="draft",
        recipient_count=0,
    ),
    EmailCampaign(
        name="Monthly Recap",
        subject="Your Monthly Content Digest",
        content="<h1>What a month!</h1><p>Here's everything you might have missed...</p>",
        status="sent",
        recipient_count=3200,
        sent_at=now - timedelta(days=7),
        stats={"opens": 2240, "clicks": 640, "open_rate": 70, "click_rate": 20},
    ),
]

# Sample media
media_items = [
    Media(
        name="banner.jpg",
        url="https://picsum.photos/800/400?random=1",
        media_type="image",
        size=245000,
        mime_type="image/jpeg",
        width=800,
        height=400,
    ),
    Media(
        name="profile.png",
        url="https://picsum.photos/400/400?random=2",
        media_type="image",
        size=128000,
        mime_type="image/png",
        width=400,
        height=400,
    ),
    Media(
        name="post-1.jpg",
        url="https://picsum.photos/600/600?random=3",
        media_type="image",
        size=512000,
        mime_type="image/jpeg",
        width=600,
        height=600,
    ),
    Media(
        name="thumbnail.jpg",
        url="https://picsum.photos/300/300?random=4",
        media_type="image",
        size=89000,
        mime_type="image/jpeg",
        width=300,
        height=300,
    ),
    Media(
        name="cover.png",
        url="https://picsum.photos/1200/630?random=5",
        media_type="image",
        size=324000,
        mime_type="image/png",
        width=1200,
        height=630,
    ),
    Media(
        name="story.jpg",
        url="https://picsum.photos/1080/1920?random=6",
        media_type="image",
        size=198000,
        mime_type="image/jpeg",
        width=1080,
        height=1920,
    ),
]

# Default settings
settings = UserSettings(
    user_id="default",
    display_name="Creator",
    push_notifications=True,
    email_notifications=True
)

# Add all data
db.add_all(approvals)
db.add_all(fan_messages)
db.add_all(scheduled_posts)
db.add_all(training_examples)
db.add_all(posts)
db.add_all(templates)
db.add_all(activities)
db.add_all(email_campaigns)
db.add_all(media_items)
db.add(settings)
db.commit()

print("🌱 Database seeded successfully!")
print(f"  ✓ {len(approvals)} approvals")
print(f"  ✓ {len(fan_messages)} fan messages")
print(f"  ✓ {len(scheduled_posts)} scheduled posts")
print(f"  ✓ {len(training_examples)} training examples")
print(f"  ✓ {len(posts)} posts")
print(f"  ✓ {len(templates)} templates")
print(f"  ✓ {len(activities)} activities")
print(f"  ✓ {len(email_campaigns)} email campaigns")
print(f"  ✓ {len(media_items)} media items")
print(f"  ✓ Default settings created")

db.close()
