import json
import logging
import os
import re

from shaaru_brain import _get_db, _get_client, MODEL_TEXT
from shaaru_retry import nvidia_call
from pipeline.knowledge.graph_query import (
    get_brands_by_vibe,
    get_brands_by_occasion,
)

logger = logging.getLogger(__name__)

_SIMPLE_RE = re.compile(
    r'^\s*(hi+|hey+|hello|thanks?|thank you|ok+|okay|cool|great|nice|'
    r'lol+|haha|yes|no|yep|nope|sure|wow|omg|perfect|awesome|love it|'
    r'what\'?s up|how are you|good (morning|night|evening)|bye|goodbye)\s*[!?.]*\s*$',
    re.IGNORECASE
)

_TOOL_KEYWORDS = frozenset([
    'find', 'show', 'search', 'buy', 'shop', 'get', 'want',
    'tailor', 'make', 'stitch', 'custom', 'design', 'alter',
    'trend', 'style', 'recommend', 'suggest', 'outfit', 'wear',
    'fabric', 'cloth', 'kurta', 'saree', 'lehenga', 'dress',
    'shirt', 'pant', 'dupatta', 'anarkali', 'blazer', 'denim',
    'jacket', 'top', 'skirt', 'coord', 'look', 'vibe', 'aesthetic',
])

def _needs_tools(message: str) -> bool:
    if _SIMPLE_RE.match(message.strip()):
        return False
    words = set(message.lower().split())
    if len(words) <= 6 and not (words & _TOOL_KEYWORDS):
        return False
    return True

SHAARU_TOOLS = [
  {
    'type': 'function',
    'function': {
      'name': 'search_products',
      'description': 'Search the product catalog for garments matching a description, aesthetic, or style. Call this when user expresses admiration for a specific item or asks to find something similar.',
      'parameters': {
        'type': 'object',
        'properties': {
          'query': {'type': 'string', 'description': 'search query'},
          'filters': {'type': 'object', 'description': 'optional filters like price, color, occasion'}
        },
        'required': ['query']
      }
    }
  },
  {
    'type': 'function', 
    'function': {
      'name': 'trigger_tailor_flow',
      'description': 'Launch the custom tailor flow when user explicitly wants a garment made or tailored. ALWAYS ask for confirmation before calling. Never call without confirmed=true.',
      'parameters': {
        'type': 'object',
        'properties': {
          'garment_description': {'type': 'string'},
          'confirmed': {'type': 'boolean', 'description': 'must be true — user confirmed they want to start tailor flow'}
        },
        'required': ['garment_description', 'confirmed']
      }
    }
  },
  {
    'type': 'function',
    'function': {
      'name': 'get_user_taste_profile', 
      'description': 'Fetch the user taste profile including saved aesthetics, body measurements, city, past orders. Call when personalizing recommendations.',
      'parameters': {
        'type': 'object',
        'properties': {
          'user_id': {'type': 'string'}
        },
        'required': ['user_id']
      }
    }
  },
  {
    'type': 'function',
    'function': {
      'name': 'search_trends',
      'description': 'Get current fashion trends, trending aesthetics, or what is popular right now. Call when user asks about trends or what is in style.',
      'parameters': {
        'type': 'object', 
        'properties': {
          'category': {'type': 'string', 'description': 'trend category: indian_ethnic, western, streetwear, bridal, festive'},
          'city': {'type': 'string'}
        },
        'required': ['category']
      }
    }
  },
  {
    'type': 'function',
    'function': {
      'name': 'semantic_search',
      'description': 'Search the Neo4j aesthetic knowledge graph for similar aesthetics, vibes, or style matches. Use for vague style questions or aesthetic exploration.',
      'parameters': {
        'type': 'object',
        'properties': {
          'aesthetic_query': {'type': 'string'},
          'top_k': {'type': 'integer', 'default': 5}
        },
        'required': ['aesthetic_query']
      }
    }
  },
  {
    'type': 'function',
    'function': {
      'name': 'get_brands_by_vibe',
      'description': 'Query Neo4j knowledge graph to find real Indian brands matching a vibe or occasion. Use this whenever user asks for brand recommendations, where to shop, or what brands match their style. Always prefer this over guessing brand names.',
      'parameters': {
        'type': 'object',
        'properties': {
          'vibe': {
            'type': 'string',
            'description': 'The vibe or aesthetic to match. One of: streetwear, maximalist, minimal, avant_garde, ethnic, editorial, clean, genderfluid, dark, handcrafted, fast_fashion'
          },
          'region': {
            'type': 'string',
            'description': 'Optional city filter: Mumbai, Delhi, Chennai, Chandigarh, Bangalore'
          }
        },
        'required': ['vibe']
      }
    }
  },
  {
    'type': 'function',
    'function': {
      'name': 'lookup_brand_catalog',
      'description': 'Search the real Indian brand catalog (57 luxury, high street, and indie designers across 7 categories) by name, aesthetic hint, or category keyword to ground brand recommendations.',
      'parameters': {
        'type': 'object',
        'properties': {
          'query': {
            'type': 'string',
            'description': 'Brand name, aesthetic keyword, or category (e.g. Raw Mango, minimal, streetwear, bridal)'
          }
        },
        'required': ['query']
      }
    }
  },
  {
    'type': 'function',
    'function': {
      'name': 'query_fashion_knowledge',
      'description': 'Query the four-layer fashion knowledge fallback system (Neo4j -> verification -> web search -> model knowledge) for deep information on any brand, designer, fabric, or aesthetic.',
      'parameters': {
        'type': 'object',
        'properties': {
          'query': {'type': 'string', 'description': 'The fashion entity name or topic to look up.'},
          'entity_type': {'type': 'string', 'description': 'One of: brand, designer, fabric, aesthetic, general'}
        },
        'required': ['query']
      }
    }
  }
]

def execute_tool(tool_name: str, arguments: dict, user_id: str) -> str:
  
  if tool_name == 'search_products':
    db = _get_db()
    query = arguments.get('query', '')

    results: list[dict] = []
    try:
        from product_embeddings import search_products_semantic
        results = search_products_semantic(query, limit=5)
    except Exception as e:
        logger.warning(f"Vector search unavailable, using text fallback: {e}")

    if not results:
        results = list(db['products'].find(
            {'$text': {'$search': query}},
            {'_id': 0, 'name': 1, 'product_name': 1, 'price': 1, 'pricing': 1,
             'brand': 1, 'image_url': 1, 'product_url': 1}
        ).limit(5))

    # Normalize field names (product_name → name, pricing.price_inr → price)
    for r in results:
        if 'product_name' in r and 'name' not in r:
            r['name'] = r.pop('product_name')
        if isinstance(r.get('pricing'), dict) and 'price' not in r:
            r['price'] = r['pricing'].get('price_inr')
            del r['pricing']

    return json.dumps({'products': results, 'query': query})

  elif tool_name == 'trigger_tailor_flow':
    confirmed = arguments.get('confirmed', False)
    if not confirmed:
      return json.dumps({
        'status': 'needs_confirmation',
        'message': 'Ask user to confirm before launching'
      })
    return json.dumps({
      'status': 'tailor_flow_triggered',
      'garment': arguments.get('garment_description'),
      'redirect': '/tailor',
      'message': 'Tailor flow ready to launch'
    })

  elif tool_name == 'get_user_taste_profile':
    db = _get_db()
    profile = db['users'].find_one(
      {'user_id': user_id},
      {'_id': 0}
    )
    if not profile:
      profile = {
        'user_id': user_id,
        'city': 'bengaluru',
        'aesthetics': [],
        'measurements': {}
      }
    return json.dumps(profile, default=str)

  elif tool_name == 'search_trends':
    db = _get_db()
    category = arguments.get('category', 'western')
    trends = list(db['trends'].find(
      {'category': category},
      {'_id': 0}
    ).sort('score', -1).limit(5))
    return json.dumps({'trends': trends, 'category': category})

  elif tool_name == 'semantic_search':
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
      os.getenv('NEO4J_URI'),
      auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )
    query = arguments.get('aesthetic_query', '')
    top_k = arguments.get('top_k', 5)
    with driver.session() as session:
      result = session.run(
        """MATCH (a:Aesthetic) 
           WHERE toLower(a.name) CONTAINS toLower($q)
           OR toLower(a.description) CONTAINS toLower($q)
           RETURN a.name, a.description, a.vibe_tags
           LIMIT $top_k""",
        q=query, top_k=int(top_k)
      )
      aesthetics = [dict(r) for r in result]
    driver.close()
    return json.dumps({'aesthetics': aesthetics})

  elif tool_name == 'get_brands_by_vibe':
    vibe = arguments.get('vibe', '')
    region = arguments.get('region')
    brands = get_brands_by_vibe(vibe, region)
    return json.dumps({'brands': brands, 'vibe': vibe})

  elif tool_name == 'lookup_brand_catalog':
    query = arguments.get('query', '').lower()
    designers_path = os.path.join(os.path.dirname(__file__), 'pipeline', 'config', 'designers.json')
    try:
      with open(designers_path, encoding='utf-8') as f:
        catalog = json.load(f)
      matches = []
      for d in catalog:
        if not d.get('active', True):
          continue
        if (query in d.get('name', '').lower() or
            query in d.get('aesthetic_hint', '').lower() or
            query in d.get('category', '').lower() or
            query in d.get('id', '').lower() or
            query in d.get('region', '').lower()):
          matches.append(d)
      if not matches:
        from knowledge_fallback import resolve_fashion_knowledge
        fb = resolve_fashion_knowledge(query, 'brand', user_id)
        if fb.get('results'):
          return json.dumps({'brands': fb['results'], 'query': query, 'resolved_via': fb.get('source')})
        matches = catalog[:10]
      return json.dumps({'brands': matches[:10], 'query': query})
    except Exception as e:
      logger.warning(f"Could not read designers.json: {e}")
      return json.dumps({'error': 'Catalog unavailable', 'query': query})

  elif tool_name == 'query_fashion_knowledge':
    from knowledge_fallback import resolve_fashion_knowledge
    query = arguments.get('query', '')
    entity_type = arguments.get('entity_type', 'general')
    res = resolve_fashion_knowledge(query, entity_type, user_id)
    return json.dumps(res, default=str)

  return json.dumps({'error': f'Unknown tool: {tool_name}'})

def riley_think(user_message: str, user_id: str, conversation_history: list = None, image_base64: str = None) -> dict:

  # --- inject live user context ---
  user_context_block = ""
  try:
    db = _get_db()
    if db is not None:
      lines: list[str] = []
      user = db['users'].find_one({'user_id': user_id}, {'_id': 0})
      if user:
        style_dna: dict = user.get('style_dna', {})
        if style_dna:
          top = sorted(style_dna.items(), key=lambda x: x[1], reverse=True)[:3]
          lines.append("Aesthetic scores: " + ", ".join(f"{k}={v:.2f}" for k, v in top if v > 0))

        fabric_prefs: dict = user.get('fabric_preferences', {})
        if fabric_prefs:
          top_fabrics = sorted(fabric_prefs.items(), key=lambda x: x[1], reverse=True)[:3]
          lines.append("Fabric preferences: " + ", ".join(f"{k}={v:.2f}" for k, v in top_fabrics))

        avoid: list = user.get('avoid', [])
        if avoid:
          lines.append(f"Avoids: {', '.join(avoid)}")

        physical: dict = user.get('physical', {})
        if physical.get('body_type'):
          lines.append(f"Body type: {physical['body_type']}")

        taste: dict = user.get('taste', {})
        if taste.get('color_palette'):
          lines.append(f"Color palette: {', '.join(taste['color_palette'])}")

      recent_scans = list(db['scanned_items'].find(
        {'user_id': user_id},
        {'_id': 0, 'summary': 1}
      ).sort('_id', -1).limit(3))
      if recent_scans:
        scan_summaries = [s.get('summary') for s in recent_scans if s.get('summary')]
        if scan_summaries:
          lines.append("Recent Lens Scans (reference only if relevant to user request): " + "; ".join(scan_summaries))

      if lines:
        user_context_block = "\n\nUSER CONTEXT — use this to personalize every response:\n" + \
                             "\n".join(f"- {l}" for l in lines)
  except Exception as e:
    logger.warning(f"Could not load user context for {user_id}: {e}")
  # ---------------------------------

  SHAARU_SYSTEM = f"""You are SHAARU — a sharp, warm, expert AI stylist for Indian fashion. You know fabrics, construction, sourcing, and aesthetics deeply.

Your personality: direct, bestie-coded, never generic. You speak like a friend who happens to be a fashion expert.

You have tools available. Use them when needed:
- search_products: when user wants to find or buy something
- trigger_tailor_flow: ONLY when user explicitly wants something MADE. Always confirm first.
- get_user_taste_profile: when you need to personalize
- search_trends: when asked about trends
- semantic_search: for aesthetic/vibe exploration
- lookup_brand_catalog: to ground brand/designer references against real Indian designers

CRITICAL RULES:
- Never call trigger_tailor_flow without confirmed=true
- Don't over-trigger tools — a compliment is just a compliment, not a search query
- Plain conversation needs no tool calls
- Never proactively mention or volunteer past Lens scans unless the user's message refers to the garment, fabric, tailoring, or styling of that item

TAILOR FLOW RULES:
Before calling trigger_tailor_flow, you MUST have collected:
1. Garment type (be specific — kurta, co-ord set, blazer, lehenga, etc.)
2. Occasion (casual daily, festive, wedding, work, party)
3. Fabric preference or "open to suggestions"
4. Budget range in INR
5. Any reference image or vibe description

Ask these naturally across 1-2 messages, not as a numbered list. Sound like a friend, not a form.
Only once you have all 5, ask for confirmation and then call trigger_tailor_flow.
Pass the full garment_description as a summary of all 5 points.

NEW USER ONBOARDING RULES:
If user_context_block shows no name or taste data, this is a new user. Ask exactly these questions, one per message, in this exact order:

1. "What should I call you? and real quick — what are your pronouns? (she/her, he/him, they/them) 😊"

After they answer, extract both name and pronouns from their reply. Then continue with questions 2, 3, 4 as before:
2. "okay [name]! what does your everyday uniform look like? like what are you actually wearing on a regular day"
3. "love that. cozy rainy day — what's the fit?"
4. "okay last one — you're front row at an editorial fashion week show. what are you wearing?"

After they answer question 4, respond with exactly this and nothing else:
"okay [name] I have one more thing for you — pick everything that feels like your vibe 👇 VIBE_PICKER_READY:[pronouns]"
For example: "VIBE_PICKER_READY:he/him" or "VIBE_PICKER_READY:she/her" or "VIBE_PICKER_READY:they/them"

RULES:
- One question per message, never combine
- Use their name once you have it
- VIBE_PICKER_READY:[pronouns] must appear at the end of your message after question 4 answer
- Do not ask anything else during onboarding

- Always respond in SHAARU's warm, direct voice{user_context_block}"""

  messages = [{'role': 'system', 'content': SHAARU_SYSTEM}]
  
  if conversation_history:
    messages.extend(conversation_history[-6:])
  
  if image_base64:
    messages.append({
      'role': 'user',
      'content': [
        {'type': 'text', 'text': user_message or "What do you think of this?"},
        {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{image_base64}"}}
      ]
    })
  else:
    messages.append({'role': 'user', 'content': user_message})

  client = _get_client()

  # ── Vision path: extract image description, then proceed to tool evaluation ──
  if image_base64:
    vision_desc = None
    try:
      vision_resp = client.chat.completions.create(
        model='meta/llama-3.2-90b-vision-instruct',
        messages=messages,
        max_tokens=300,
        temperature=0.7,
        timeout=25
      )
      vision_desc = vision_resp.choices[0].message.content or "love this look! tell me what you want to make ✨"
    except Exception as e:
      logger.warning(f"Vision 90b failed, trying 11b: {e}")
      try:
        vision_resp = client.chat.completions.create(
          model='meta/llama-3.2-11b-vision-instruct',
          messages=messages,
          max_tokens=300,
          temperature=0.7,
          timeout=25
        )
        vision_desc = vision_resp.choices[0].message.content or "love this look! tell me what you want to make ✨"
      except Exception as e2:
        logger.warning(f"Vision 11b failed: {e2}")
        return {
          'reply': "I got your image but had trouble seeing it clearly — describe what catches your eye about it! ✨",
          'tool_calls': [],
          'tailor_flow': False,
          'model': 'vision-error'
        }

    if vision_desc:
      user_text = user_message or "What do you think of this?"
      messages[-1] = {
        'role': 'user',
        'content': f"{user_text}\n\n[Attached Image Analysis by Vision Model:\n{vision_desc}]"
      }

  # ── Fast path: skip tool overhead for simple conversational messages ──
  if not image_base64 and not _needs_tools(user_message):
    try:
      fast_resp = client.chat.completions.create(
        model=MODEL_TEXT,
        messages=messages,
        max_tokens=180,
        timeout=18
      )
      return {
        'reply': fast_resp.choices[0].message.content or "hey! what's on your mind?",
        'tool_calls': [],
        'tailor_flow': False,
        'model': 'fast'
      }
    except Exception as e:
      logger.warning(f"Fast path failed, continuing to full path: {e}")

  # TURN 1 — model decides what to do
  def _call_turn1():
    return client.chat.completions.create(
      model=MODEL_TEXT,
      messages=messages,
      tools=SHAARU_TOOLS,
      tool_choice='auto',
      max_tokens=300,
      timeout=25
    )
  def _fallback_turn1():
    return client.chat.completions.create(
      model='meta/llama-3.1-8b-instruct',
      messages=messages,
      tools=SHAARU_TOOLS,
      tool_choice='auto',
      max_tokens=300,
      timeout=25
    )
  try:
    try:
      response = _call_turn1()
    except Exception:
      response = _fallback_turn1()
  except Exception as e:
    logger.warning(f"Turn 1 timeout or error: {e}")
    return {
        'reply': "give me a sec, thinking...",
        'tool_calls': [],
        'model': 'timeout-fallback'
    }

  choice = response.choices[0]
  tool_calls_made = []

  # TURN 2 — if model called tools, execute them
  if choice.finish_reason == 'tool_calls':
    messages.append(choice.message)
    
    for tool_call in choice.message.tool_calls:
      tool_name = tool_call.function.name
      try:
        arguments = json.loads(tool_call.function.arguments)
      except:
        arguments = {}
      
      tool_result = execute_tool(tool_name, arguments, user_id)
      
      tool_calls_made.append({
        'tool': tool_name,
        'arguments': arguments,
        'result': json.loads(tool_result)
      })
      
      messages.append({
        'role': 'tool',
        'tool_call_id': tool_call.id,
        'content': tool_result
      })
      
      logger.info(f'[TOOL] {tool_name} called with {arguments}')

    # TURN 3 — model writes final reply with tool results
    def _call_turn3():
      return client.chat.completions.create(
        model=MODEL_TEXT,
        messages=messages,
        max_tokens=600,
        timeout=45
      )
    def _fallback_turn3():
      return client.chat.completions.create(
        model='meta/llama-3.1-8b-instruct',
        messages=messages,
        max_tokens=600,
        timeout=45
      )
    try:
      try:
        final_response = _call_turn3()
      except Exception:
        final_response = _fallback_turn3()
      final_text = final_response.choices[0].message.content
    except Exception as e:
      logger.warning(f"Turn 3 timeout or error: {e}")
      return {
          'reply': "give me a sec, thinking...",
          'tool_calls': tool_calls_made,
          'model': 'timeout-fallback'
      }

  else:
    # No tool calls — direct reply
    final_text = choice.message.content

  tailor_triggered = any(
    tc.get('tool') == 'trigger_tailor_flow'
    for tc in tool_calls_made
  )
  return {
    'reply': final_text,
    'tool_calls': tool_calls_made,
    'tailor_flow': tailor_triggered,
    'model': 'meta/llama-3.3-70b-instruct'
  }
