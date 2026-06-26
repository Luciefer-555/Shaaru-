import json
from config.models import get_client

TEXT_PROMPT = """You are an Indian fashion cataloguing expert. Given the product title, description, 
and tags from an Indian designer's website, extract the following structured information.
Be specific. Do not invent information not present in the source text. 
If a field cannot be determined from the text, return null.

Product Title: {title}
Description: {description}
Tags: {tags}

Return ONLY a valid JSON object:

{{
  "confirmed_fabrics": [],
  "confirmed_techniques": [],
  "collection_name": "",
  "designer_notes": "",
  "care_instructions": "",
  "region_of_craft": "",
  "price_tier": ""
}}"""

def extract_text(title: str, description: str, tags: list, model_config: dict):
    """
    Extracts structured data from product text.
    """
    client = get_client(model_config["provider"])
    
    prompt = TEXT_PROMPT.format(title=title, description=description, tags=tags)
    
    kwargs = {
        "model": model_config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.1
    }
    
    if model_config.get("json_mode"):
        kwargs["response_format"] = {"type": "json_object"}
        
    tokens = {"input": 0, "output": 0}
    try:
        response = client.chat.completions.create(**kwargs)
        
        usage = response.usage
        tokens = {"input": usage.prompt_tokens, "output": usage.completion_tokens}
        
        raw_text = response.choices[0].message.content
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0]
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0]
            
        data = json.loads(raw_text.strip())
        return data, tokens
    except Exception as e:
        print(f"Text extraction failed for {model_config['model']}: {e}")
        return None, tokens
