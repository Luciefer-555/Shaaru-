from dotenv import load_dotenv
load_dotenv()
from riley_brain import riley_think
import json

tests = [
  ('I really like this blazer', 'demo_user'),
  ('I want this made', 'demo_user'),
  ('what is trending in indian ethnic right now', 'demo_user'),
  ('find me something similar to a sage kurta palazzo', 'demo_user'),
  ('you are literally so good at this omg', 'demo_user'),
]

for msg, uid in tests:
  print(f'INPUT: {msg}')
  result = riley_think(msg, uid)
  print(f'REPLY: {result["reply"][:150]}')
  tools = result['tool_calls']
  print(f'TOOLS: {[t["tool"] for t in tools] if tools else "none"}')
  print()
