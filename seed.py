from app.database import SessionLocal, engine, Base
from app.models import Approval, FanMessage, UserSettings

# Create tables
Base.metadata.create_all(bind=engine)

db = SessionLocal()

# Clear existing data
db.query(Approval).delete()
db.query(FanMessage).delete()
db.query(UserSettings).delete()

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

# Default settings
settings = UserSettings(
    user_id="default",
    display_name="Creator",
    push_notifications=True,
    email_notifications=True
)

db.add_all(approvals)
db.add_all(fan_messages)
db.add(settings)
db.commit()

print("Database seeded successfully!")
print(f"  - {len(approvals)} approvals")
print(f"  - {len(fan_messages)} fan messages")
print(f"  - Default settings created")

db.close()
