"""
tailor_router.py
SHAARU — Tailor Feature API Routes
Mount in api.py: app.include_router(tailor_router)
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
import os
from pymongo import MongoClient
from bson import ObjectId
import json

def clean_mongo_doc(obj):
    if isinstance(obj, dict):
        return {k: clean_mongo_doc(v) for k, v in obj.items() if k != '_id'}
    elif isinstance(obj, list):
        return [clean_mongo_doc(i) for i in obj]
    else:
        try:
            json.dumps(obj)
            return obj
        except:
            return str(obj)

from tailor_engine import (
    analyze_garment_deep, 
    extract_unclear_dimensions, 
    generate_questions_for_gaps, 
    generate_universal_brief, 
    generate_and_save_sketch_bg,
    process_reference_with_modification
)
from tailor_db import get_session, update_session, save_project
from auth import get_current_user

tailor_router = APIRouter(prefix="/tailor", tags=["tailor"])

def require_same_user(requested_user_id: str, token_user: dict) -> None:
    token_user_id = token_user.get("user_id") or token_user.get("sub")
    if requested_user_id != token_user_id:
        raise HTTPException(status_code=403, detail="Cannot access another user's data")

class StartTailorRequest(BaseModel):
    user_id: str
    project_id: str
    image_b64: Optional[str] = None
    garment_type_override: Optional[str] = None

class SelectOptionRequest(BaseModel):
    user_id: str
    session_id: str
    dimension_id: str
    option_id: str

def get_db():
    from shaaru_brain import _get_db
    return _get_db()

@tailor_router.post("/start")
async def start_tailor_session(req: StartTailorRequest, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    require_same_user(req.user_id, user)
    db = get_db()
    
    if req.image_b64:
        analysis = analyze_garment_deep(req.image_b64)
    elif req.garment_type_override:
        analysis = {
            "garment_type": req.garment_type_override,
            "tradition": "unknown",
            "occasion": "unknown",
            "gender_expression": "unknown",
            "replication_complexity": "unknown",
            "tailor_notes": "",
            "silhouette": {"overall_shape": {"value": "", "confidence": "unclear"}}
        }
    else:
        raise HTTPException(status_code=400, detail="Must provide image_b64 or garment_type_override")
        
    garment_type = analysis.get("garment_type", "Garment")
    tradition = analysis.get("tradition", "unknown")
    
    gaps = extract_unclear_dimensions(analysis)
    questions = generate_questions_for_gaps(gaps, garment_type, tradition)
    
    session_doc = {
        "user_id": req.user_id,
        "project_id": req.project_id,
        "analysis": analysis,
        "questions": questions,
        "user_answers": {},
        "status": "active",
        "current_question_index": 0,
        "total_questions": len(questions),
        "created_at": datetime.now(timezone.utc)
    }
    
    result = db["tailor_sessions"].insert_one(session_doc)
    session_id = str(result.inserted_id)
    
    if len(questions) > 0:
        return {
            "status": "session_started",
            "session_id": session_id,
            "garment_name": garment_type,
            "progress": {"answered": 0, "total": len(questions)},
            "question": questions[0],
            "brief": None,
            "opening_message": f"I've analyzed the {garment_type}. Just need to clarify a few details to create your spec sheet."
        }
    else:
        # Generate brief immediately if no gaps
        profile = db["users"].find_one({"user_id": req.user_id}) or {}
        brief = generate_universal_brief(analysis, {}, profile)
        background_tasks.add_task(generate_and_save_sketch_bg, session_id, brief)
        update_session(session_id, {"status": "briefed", "brief": brief}, db)
        save_project(req.user_id, brief, session_id, db)
        
        return {
            "status": "complete",
            "session_id": session_id,
            "garment_name": garment_type,
            "progress": {"answered": 0, "total": 0},
            "question": None,
            "brief": brief,
            "opening_message": "Perfect. The image has all the details I need. Generating your spec sheet now."
        }

@tailor_router.post("/select")
async def select_option(req: SelectOptionRequest, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    require_same_user(req.user_id, user)
    db = get_db()
    session = get_session(req.session_id, db)
    
    if not session or session.get("user_id") != req.user_id:
        raise HTTPException(status_code=404, detail="Session not found")
        
    if session.get("status") != "active":
        return {"status": session.get("status"), "brief": session.get("brief")}
        
    questions = session.get("questions", [])
    idx = session.get("current_question_index", 0)
    
    if idx >= len(questions):
        raise HTTPException(status_code=400, detail="All questions already answered")
        
    q = questions[idx]
    if q.get("id") != req.dimension_id:
        # Flexible validation just in case UI is out of sync
        pass
        
    # Find option label
    option_label = req.option_id
    for opt in q.get("options", []):
        if opt.get("id") == req.option_id:
            option_label = opt.get("label")
            break
            
    # Save answer
    user_answers = session.get("user_answers", {})
    user_answers[req.dimension_id] = option_label
    
    next_idx = idx + 1
    
    if next_idx >= len(questions):
        # All done, generate brief
        profile = db["users"].find_one({"user_id": req.user_id}) or {}
        analysis = session.get("analysis", {})
        
        brief = generate_universal_brief(analysis, user_answers, profile)
        background_tasks.add_task(generate_and_save_sketch_bg, req.session_id, brief)
        
        update_session(req.session_id, {
            "user_answers": user_answers,
            "current_question_index": next_idx,
            "status": "briefed",
            "brief": brief
        }, db)
        
        save_project(req.user_id, brief, req.session_id, db)
        
        return {
            "status": "complete",
            "session_id": req.session_id,
            "progress": {"answered": next_idx, "total": len(questions)},
            "question": None,
            "brief": brief
        }
    else:
        # Next question
        update_session(req.session_id, {
            "user_answers": user_answers,
            "current_question_index": next_idx
        }, db)
        
        return {
            "status": "session_started",
            "session_id": req.session_id,
            "progress": {"answered": next_idx, "total": len(questions)},
            "question": questions[next_idx],
            "brief": None
        }

@tailor_router.get("/brief/{session_id}")
async def get_brief(session_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    session = get_session(session_id, db)
    if not session or session.get("status") != "briefed":
        raise HTTPException(status_code=404, detail="Brief not found or not ready")
    require_same_user(session.get("user_id"), user)
    return session.get("brief")

@tailor_router.get("/session/{session_id}")
async def get_session_state(session_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    session = get_session(session_id, db)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    require_same_user(session.get("user_id"), user)
        
    session["_id"] = str(session["_id"])
    return session

class ReferenceModificationRequest(BaseModel):
    user_id: str
    project_id: str
    image_b64: str
    user_message: str
    product_url: Optional[str] = None

@tailor_router.post("/reference")
async def process_reference(req: ReferenceModificationRequest, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    require_same_user(req.user_id, user)
    db = get_db()
    
    profile = db["users"].find_one({"user_id": req.user_id}) or {}
    
    brief = process_reference_with_modification(req.image_b64, req.user_message, profile, req.product_url)
    brief = clean_mongo_doc(brief)
    
    session_doc = {
        "user_id": req.user_id,
        "project_id": req.project_id,
        "type": "reference_modification",
        "status": "briefed",
        "brief": brief,
        "created_at": datetime.now(timezone.utc)
    }
    
    result = db["tailor_sessions"].insert_one(session_doc)
    session_id = str(result.inserted_id)
    
    background_tasks.add_task(generate_and_save_sketch_bg, session_id, brief)
    
    from shaaru_brain import nvidia_call, _get_client
    client = _get_client()
    prompt = f"You are Riley, SHAARU's AI stylist. The user just sent a reference image and wants to recreate it with modifications. Here's what they're making: {brief.get('garment_name')} with {brief.get('modification_summary')}.\nWrite one sentence in Riley's voice — warm, direct, bestie-coded — acknowledging what they're building. Under 30 words."
    
    try:
        opening_message = nvidia_call(client, "meta/llama-3.1-70b-instruct", [{"role": "user", "content": prompt}], temperature=0.7)
    except:
        opening_message = "Got it! Let's build this together."
        
    return {
        "status": "complete",
        "session_id": session_id,
        "brief": brief,
        "message": opening_message
    }
