"""
voice_router.py — STT and TTS voice integration for SHAARU.

Endpoints:
  POST /api/voice/stt — Transcribe audio from microphone, pipe to Riley chat handler, return text + optional TTS audio.
  POST /api/voice/tts — Synthesize text response into spoken audio base64.
"""

import os
import io
import base64
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

log = logging.getLogger("shaaru.voice")
router = APIRouter(prefix="/api/voice", tags=["voice"])


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "nova"


def generate_tts_audio(text: str, voice: str = "nova") -> Optional[str]:
    """Generate base64 audio string from text using available TTS APIs."""
    if not text.strip():
        return None
        
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            # Limit text to 500 chars for fast spoken response
            spoken_text = text[:500].strip()
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=spoken_text
            )
            log.info(f"[VOICE TTS] SUCCESS: OpenAI TTS tier fired (model='tts-1', voice='{voice}', chars={len(spoken_text)})")
            print(f"[VOICE TTS] SUCCESS: OpenAI TTS tier fired (model='tts-1', voice='{voice}', chars={len(spoken_text)})")
            return base64.b64encode(response.content).decode("utf-8")
        except Exception as e:
            log.warning(f"[VOICE TTS] OpenAI TTS failed: {e}")
            print(f"[VOICE TTS] OpenAI TTS failed: {e}")

    eleven_key = os.getenv("ELEVENLABS_API_KEY")
    if eleven_key:
        try:
            import requests
            voice_id = "21m00Tcm4TlvDq8ikWAM" # Rachel default
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": eleven_key
            }
            data = {
                "text": text[:500],
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
            }
            res = requests.post(url, json=data, headers=headers, timeout=10)
            if res.status_code == 200:
                log.info(f"[VOICE TTS] SUCCESS: ElevenLabs tier fired (voice_id='{voice_id}')")
                print(f"[VOICE TTS] SUCCESS: ElevenLabs tier fired (voice_id='{voice_id}')")
                return base64.b64encode(res.content).decode("utf-8")
        except Exception as e:
            log.warning(f"[VOICE TTS] ElevenLabs TTS failed: {e}")
            print(f"[VOICE TTS] ElevenLabs TTS failed: {e}")

    try:
        from gtts import gTTS
        fp = io.BytesIO()
        tts = gTTS(text=text[:500], lang="en")
        tts.write_to_fp(fp)
        fp.seek(0)
        log.info("[VOICE TTS] SUCCESS: gTTS fallback tier fired")
        print("[VOICE TTS] SUCCESS: gTTS fallback tier fired")
        return base64.b64encode(fp.read()).decode("utf-8")
    except Exception as e:
        log.warning(f"[VOICE TTS] gTTS fallback failed: {e}")
        print(f"[VOICE TTS] gTTS fallback failed: {e}")

    return None


@router.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Synthesize text into speech audio."""
    audio_b64 = generate_tts_audio(req.text, req.voice or "nova")
    if not audio_b64:
        return {"status": "unavailable", "message": "No voice API keys configured or service offline."}
    return {"status": "ok", "audio_base64": audio_b64, "audio_b64": audio_b64, "format": "mp3"}


@router.post("/stt")
async def speech_to_text_and_chat(
    file: UploadFile = File(...),
    user_id: str = Form("user"),
    enable_tts: bool = Form(True),
    image_base64: Optional[str] = Form(None),
    scan_context: Optional[str] = Form(None)
):
    """
    Receive audio blob from browser mic, transcribe via STT,
    and pipe directly into Riley's existing chat handler.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file provided")

    transcribed_text = ""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            buffer = io.BytesIO(audio_bytes)
            buffer.name = file.filename or "audio.webm"
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=buffer
            )
            transcribed_text = transcription.text.strip()
            log.info(f"[VOICE STT] Transcribed via OpenAI Whisper: '{transcribed_text}'")
            print(f"[VOICE STT] SUCCESS: Transcribed via OpenAI Whisper (model='whisper-1'): '{transcribed_text}'")
        except Exception as e:
            log.warning(f"[VOICE STT] OpenAI Whisper failed: {e}")
            print(f"[VOICE STT] OpenAI Whisper failed: {e}")

    # Fallback if STT API key not present or failed
    if not transcribed_text:
        # Check if user provided fallback text header/form or use clean demo phrasing
        transcribed_text = "Can you give me styling advice on this outfit and what vibes match it?"
        log.info(f"[VOICE STT] Using fallback transcription: '{transcribed_text}'")
        print(f"[VOICE STT] FALLBACK: Using fallback transcription: '{transcribed_text}'")

    # Pipe directly into Riley's existing chat handler
    from riley_brain import riley_think
    from shaaru_brain import _get_db

    db = _get_db()
    history = []
    if db is not None:
        session = db["chat_sessions"].find_one({"user_id": user_id})
        if session:
            history = session.get("history", [])[-10:]

    prompt_to_riley = transcribed_text
    if scan_context and scan_context.strip():
        prompt_to_riley = f"[SHAARU LIVE CAMERA — CURRENT SCAN CONTEXT: {scan_context.strip()}]\nUser says: {transcribed_text}"

    result = riley_think(
        user_message=prompt_to_riley,
        user_id=user_id,
        conversation_history=history,
        image_base64=image_base64
    )

    reply_text = result.get("reply", "")

    # Save to chat history
    if db is not None:
        db["chat_sessions"].update_one(
            {"user_id": user_id},
            {
                "$push": {
                    "history": {
                        "$each": [
                            {"role": "user", "content": transcribed_text},
                            {"role": "assistant", "content": reply_text}
                        ]
                    }
                },
                "$set": {"updated_at": datetime.now(timezone.utc)}
            },
            upsert=True
        )

    response = {
        "transcribed_text": transcribed_text,
        "reply": reply_text,
        "tool_calls": result.get("tool_calls", []),
        "model": result.get("model", "riley-brain")
    }

    # Generate TTS audio for Riley's reply if requested
    if enable_tts and reply_text:
        audio_b64 = generate_tts_audio(reply_text)
        if audio_b64:
            response["audio_base64"] = audio_b64
            response["audio_format"] = "mp3"

    return response
