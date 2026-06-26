import os

NIM_BASE_URL     = "https://integrate.api.nvidia.com/v1"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

MODELS = {
    "vision_primary": {
        "model": "gemini-2.5-flash",
        "provider": "google"
    },
    "vision_secondary": {
        "model": "meta/llama-3.2-11b-vision-instruct",
        "provider": "nim"
    },
    "text_extractor": {
        "model": "meta/llama-3.1-70b-instruct",
        "provider": "nim",
        "json_mode": True
    },
    "classifier": {
        "model": "meta/llama-3.1-70b-instruct",
        "provider": "nim",
        "json_mode": True
    },
    "caption": {
        "model": "meta/llama-3.1-70b-instruct",
        "provider": "nim",
        "json_mode": True
    },
    "embeddings": {
        "model": "nvidia/nv-embedqa-e5-v5",
        "provider": "nim"
    }
}

GOOGLE_BASE_URL  = "https://generativelanguage.googleapis.com/v1beta/openai/"

def get_client(provider: str):
    from openai import OpenAI
    if provider == "nim":
        key = os.getenv("NVIDIA_API_KEY")
        if not key:
            raise ValueError("NVIDIA_API_KEY not found")
        return OpenAI(base_url=NIM_BASE_URL, api_key=key, timeout=45.0, max_retries=0)
    elif provider == "together":
        key = os.getenv("TOGETHER_API_KEY")
        if not key:
            raise ValueError("TOGETHER_API_KEY not found")
        return OpenAI(base_url=TOGETHER_BASE_URL, api_key=key, timeout=45.0, max_retries=0)
    elif provider == "google":
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not found")
        return OpenAI(base_url=GOOGLE_BASE_URL, api_key=key, timeout=45.0, max_retries=0)
    else:
        raise ValueError(f"Unknown provider: {provider}")

def get_async_client(provider: str):
    from openai import AsyncOpenAI
    if provider == "nim":
        key = os.getenv("NVIDIA_API_KEY")
        return AsyncOpenAI(base_url=NIM_BASE_URL, api_key=key, timeout=45.0, max_retries=0)
    elif provider == "together":
        key = os.getenv("TOGETHER_API_KEY")
        return AsyncOpenAI(base_url=TOGETHER_BASE_URL, api_key=key, timeout=45.0, max_retries=0)
    elif provider == "google":
        key = os.getenv("GOOGLE_API_KEY")
        return AsyncOpenAI(base_url=GOOGLE_BASE_URL, api_key=key, timeout=45.0, max_retries=0)
    else:
        raise ValueError(f"Unknown provider: {provider}")
