import json
from shaaru_brain import nvidia_call, _get_client

def estimate_brief_tokens(brief: dict) -> int:
    return len(json.dumps(brief)) // 4

def compress_riley_brief(brief: dict, token_ceiling: int = 800) -> dict:
    tokens = estimate_brief_tokens(brief)
    if tokens <= token_ceiling:
        return brief
        
    prompt = f"""You are compressing a fashion AI stylist's context brief. 
Preserve: user's core style identity, active occasion, measurements, top 3 trend signals.
Drop: repetitive signals, old session notes, low-confidence items.
Return ONLY valid JSON matching the input structure, compressed under {token_ceiling} tokens.

Brief to compress:
{json.dumps(brief, indent=2)}"""

    try:
        client = _get_client()
        response_text = nvidia_call(
            client=client, 
            model="meta/llama-3.1-70b-instruct", 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.3
        )
        json_str = response_text[response_text.find("{"):response_text.rfind("}")+1]
        compressed = json.loads(json_str)
        
        new_tokens = estimate_brief_tokens(compressed)
        print(f"[OK] Compressed brief from {tokens} to {new_tokens} tokens")
        return compressed
    except Exception as e:
        print(f"[FAIL] Compression failed: {e}")
        return brief
