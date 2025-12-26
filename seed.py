from datetime import datetime, timedelta
from app.database import SessionLocal, engine, Base
from app.models import Approval, FanMessage, UserSettings, ScheduledPost, TrainingExample

# Create tables
Base.metadata.create_all(bind=engine)

db = SessionLocal()

# Clear existing data
db.query(Approval).delete()
db.query(FanMessage).delete()
db.query(UserSettings).delete()
db.query(ScheduledPost).delete()
db.query(TrainingExample).delete()

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

# Sample scheduled posts
now = datetime.utcnow()
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

# Default settings
settings = UserSettings(
    user_id="default",
    display_name="Creator",
    push_notifications=True,
    email_notifications=True
)

db.add_all(approvals)
db.add_all(fan_messages)
db.add_all(scheduled_posts)
db.add_all(training_examples)
db.add(settings)
db.commit()

print("Database seeded successfully!")
print(f"  - {len(approvals)} approvals")
print(f"  - {len(fan_messages)} fan messages")
print(f"  - {len(scheduled_posts)} scheduled posts")
print(f"  - {len(training_examples)} training examples")
print(f"  - Default settings created")

db.close()
