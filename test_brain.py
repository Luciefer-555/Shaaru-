from shaaru_brain import chat_with_riley, build_riley_context
import logging
logging.basicConfig(level=logging.INFO)
print("Building context...")
ctx = build_riley_context("demo_user_001", "Mumbai")

user_message = "I'm going to the mall and I'm wearing a Leather Jacket. What should I pair it with?"
print(f"User: {user_message}")

print("Thinking...")
response = chat_with_riley("demo_user_001", user_message, [])

print(f"\nShaaru: {response}")
