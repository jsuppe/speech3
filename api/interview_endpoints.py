"""
AI Interview Practice endpoints.
Real-time conversational interview simulation with LLM.
"""

import os
import json
import tempfile
import logging
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("speechscore.interview")

router = APIRouter(prefix="/interview", tags=["Interview"])

# In-memory session storage (for MVP - could move to Redis/DB later)
_interview_sessions = {}


class InterviewSession:
    def __init__(self, session_id: str, user_id: int):
        self.session_id = session_id
        self.user_id = user_id
        self.job_description = ""
        self.resume = ""
        self.company_name = "the company"
        self.role_name = "this position"
        self.conversation_history: List[dict] = []
        self.started_at = datetime.utcnow()
        self.ended_at = None
        
    def add_turn(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_context_prompt(self) -> str:
        """Build the system prompt with JD and resume context."""
        context_parts = []
        
        if self.job_description:
            context_parts.append(f"JOB DESCRIPTION:\n{self.job_description[:3000]}")
        
        if self.resume:
            context_parts.append(f"CANDIDATE'S RESUME:\n{self.resume[:3000]}")
        
        context = "\n\n".join(context_parts) if context_parts else "No specific job or resume provided."
        
        return f"""You are an experienced hiring manager conducting a job interview. You work at {self.company_name} and are interviewing a candidate for {self.role_name}.

{context}

INTERVIEW GUIDELINES:
- Be professional but warm and conversational
- Ask relevant questions based on the job requirements and candidate's background
- Start with an introduction and warm-up question
- Progress to behavioral questions (STAR method)
- Include technical/skill-based questions relevant to the role
- Ask follow-up questions based on their answers
- Keep your responses concise (2-4 sentences typically)
- After 5-8 questions, wrap up the interview naturally

Remember: You are the INTERVIEWER, not the candidate. Ask questions and respond to their answers."""

    def build_conversation_for_llm(self) -> str:
        """Build the conversation history for the LLM."""
        parts = []
        for turn in self.conversation_history:
            role_label = "Interviewer" if turn["role"] == "interviewer" else "Candidate"
            parts.append(f"{role_label}: {turn['content']}")
        return "\n\n".join(parts)


def get_session(session_id: str) -> Optional[InterviewSession]:
    return _interview_sessions.get(session_id)


@router.post("/start")
async def start_interview(
    job_description: Optional[str] = Form(None),
    resume: Optional[str] = Form(None),
    company_name: Optional[str] = Form(None),
    role_name: Optional[str] = Form(None),
    user_id: int = Depends(lambda: 0),  # Will be replaced with actual auth
):
    """
    Start a new AI interview session.
    Returns session_id and the interviewer's opening statement.
    """
    import requests
    import uuid
    
    session_id = str(uuid.uuid4())[:8]
    
    # Create session
    session = InterviewSession(session_id, user_id)
    session.job_description = job_description or ""
    session.resume = resume or ""
    session.company_name = company_name or "the company"
    session.role_name = role_name or "this position"
    
    _interview_sessions[session_id] = session
    
    # Generate opening statement
    system_prompt = session.get_context_prompt()
    
    opening_prompt = f"""{system_prompt}

Start the interview now. Introduce yourself briefly and ask your first question. Keep it warm and professional."""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:latest",
                "prompt": opening_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "num_predict": 150,
                }
            },
            timeout=15,
        )
        
        if response.status_code == 200:
            result = response.json()
            opening = result.get("response", "").strip()
        else:
            opening = f"Hello! Thank you for coming in today. I'm excited to learn more about your background and experience. Let's start with a simple question: Can you tell me a bit about yourself and what interests you about {session.role_name}?"
            
    except Exception as e:
        logger.warning(f"Interview start error: {e}")
        opening = f"Hello! Thank you for coming in today. I'm excited to learn more about your background and experience. Let's start with a simple question: Can you tell me a bit about yourself and what interests you about {session.role_name}?"
    
    # Add to conversation history
    session.add_turn("interviewer", opening)
    
    return {
        "session_id": session_id,
        "interviewer_response": opening,
        "turn_count": 1,
    }


@router.post("/turn")
async def interview_turn(
    session_id: str = Form(...),
    candidate_response: str = Form(...),
):
    """
    Process a turn in the interview conversation.
    Takes the candidate's spoken response, returns the interviewer's next question/response.
    """
    import requests
    
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    # Add candidate's response to history
    session.add_turn("candidate", candidate_response)
    
    # Build prompt for interviewer response
    system_prompt = session.get_context_prompt()
    conversation = session.build_conversation_for_llm()
    
    turn_count = len([t for t in session.conversation_history if t["role"] == "interviewer"])
    
    # Check if we should wrap up
    should_wrap_up = turn_count >= 6
    
    prompt = f"""{system_prompt}

CONVERSATION SO FAR:
{conversation}

{"This is getting towards the end of the interview. Start wrapping up naturally - perhaps one final question or closing remarks." if should_wrap_up else "Continue the interview naturally. Respond to what they said, then ask your next question."}

Your response as the interviewer (be concise, 2-4 sentences):"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:latest",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "num_predict": 150,
                }
            },
            timeout=15,
        )
        
        if response.status_code == 200:
            result = response.json()
            interviewer_response = result.get("response", "").strip()
        else:
            interviewer_response = "That's interesting, thank you for sharing. Could you tell me more about a specific challenge you faced in that situation?"
            
    except Exception as e:
        logger.warning(f"Interview turn error: {e}")
        interviewer_response = "I appreciate that response. Let me ask you another question - what would you say is your greatest strength?"
    
    # Add interviewer response to history
    session.add_turn("interviewer", interviewer_response)
    
    new_turn_count = len([t for t in session.conversation_history if t["role"] == "interviewer"])
    
    return {
        "session_id": session_id,
        "interviewer_response": interviewer_response,
        "turn_count": new_turn_count,
        "is_wrapping_up": new_turn_count >= 6,
    }


@router.post("/end")
async def end_interview(
    session_id: str = Form(...),
):
    """
    End the interview session and get comprehensive feedback.
    """
    import requests
    
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    session.ended_at = datetime.utcnow()
    
    # Generate feedback
    conversation = session.build_conversation_for_llm()
    
    feedback_prompt = f"""You are an interview coach reviewing a practice interview session.

JOB BEING INTERVIEWED FOR: {session.role_name} at {session.company_name}

INTERVIEW TRANSCRIPT:
{conversation}

Provide constructive feedback on the candidate's interview performance. Include:

1. **Overall Impression** (1-2 sentences)
2. **Strengths** (2-3 specific things they did well)
3. **Areas for Improvement** (2-3 specific suggestions)
4. **Best Answer** (which response was strongest and why)
5. **Tip for Next Time** (one actionable piece of advice)

Be encouraging but honest. This is practice to help them improve."""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:latest",
                "prompt": feedback_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 400,
                }
            },
            timeout=30,
        )
        
        if response.status_code == 200:
            result = response.json()
            feedback = result.get("response", "").strip()
        else:
            feedback = "Great practice session! Review the transcript to see where you can add more specific examples using the STAR method."
            
    except Exception as e:
        logger.warning(f"Interview feedback error: {e}")
        feedback = "Great effort! Practice makes perfect. Focus on using specific examples from your experience."
    
    # Calculate some stats
    candidate_turns = [t for t in session.conversation_history if t["role"] == "candidate"]
    total_words = sum(len(t["content"].split()) for t in candidate_turns)
    
    # Clean up session
    del _interview_sessions[session_id]
    
    return {
        "session_id": session_id,
        "feedback": feedback,
        "stats": {
            "questions_answered": len(candidate_turns),
            "total_words_spoken": total_words,
            "duration_seconds": (session.ended_at - session.started_at).total_seconds(),
        },
        "transcript": session.conversation_history,
    }


@router.post("/transcribe-turn")
async def transcribe_and_respond(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    Combined endpoint: transcribe audio chunk and get interviewer response.
    For real-time conversation flow.
    """
    import requests
    from .pipeline_runner import run_whisper_stt
    
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    # Save audio to temp file
    suffix = os.path.splitext(audio.filename or "audio.m4a")[1] or ".m4a"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Transcribe
        result = run_whisper_stt(tmp_path)
        transcript = result.get("text", "").strip() if result else ""
        
        if not transcript:
            return {
                "session_id": session_id,
                "candidate_said": "",
                "interviewer_response": "",
                "needs_more_input": True,
            }
        
        # Process as a turn
        session.add_turn("candidate", transcript)
        
        # Get interviewer response
        system_prompt = session.get_context_prompt()
        conversation = session.build_conversation_for_llm()
        turn_count = len([t for t in session.conversation_history if t["role"] == "interviewer"])
        should_wrap_up = turn_count >= 6
        
        prompt = f"""{system_prompt}

CONVERSATION SO FAR:
{conversation}

{"This is getting towards the end of the interview. Start wrapping up naturally." if should_wrap_up else "Continue the interview naturally. Respond briefly, then ask your next question."}

Your response as the interviewer (be concise, 2-3 sentences):"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:latest",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "num_predict": 100,
                }
            },
            timeout=10,
        )
        
        if response.status_code == 200:
            llm_result = response.json()
            interviewer_response = llm_result.get("response", "").strip()
        else:
            interviewer_response = "Thank you for that. Tell me more about your experience."
        
        session.add_turn("interviewer", interviewer_response)
        
        return {
            "session_id": session_id,
            "candidate_said": transcript,
            "interviewer_response": interviewer_response,
            "turn_count": len([t for t in session.conversation_history if t["role"] == "interviewer"]),
            "is_wrapping_up": turn_count >= 6,
            "needs_more_input": False,
        }
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/extract-text")
async def extract_text_from_file(
    file: UploadFile = File(...),
):
    """
    Extract text from uploaded document (PDF, DOC, DOCX, TXT).
    Used for uploading job descriptions and resumes.
    """
    import subprocess
    
    filename = file.filename or "document"
    suffix = os.path.splitext(filename)[1].lower()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        extracted_text = ""
        
        if suffix == ".txt":
            # Plain text - just read it
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read()
                
        elif suffix == ".pdf":
            # Use pdftotext (from poppler-utils)
            try:
                result = subprocess.run(
                    ["pdftotext", "-layout", tmp_path, "-"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    extracted_text = result.stdout
                else:
                    logger.warning(f"pdftotext failed: {result.stderr}")
            except FileNotFoundError:
                # pdftotext not installed, try PyPDF2 as fallback
                try:
                    import PyPDF2
                    with open(tmp_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            extracted_text += page.extract_text() + "\n"
                except ImportError:
                    logger.warning("Neither pdftotext nor PyPDF2 available")
                    
        elif suffix in [".doc", ".docx"]:
            # Use antiword for .doc, python-docx for .docx
            if suffix == ".docx":
                try:
                    from docx import Document
                    doc = Document(tmp_path)
                    extracted_text = "\n".join([p.text for p in doc.paragraphs])
                except ImportError:
                    logger.warning("python-docx not installed")
            else:
                # .doc - use antiword or catdoc
                try:
                    result = subprocess.run(
                        ["antiword", tmp_path],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        extracted_text = result.stdout
                except FileNotFoundError:
                    logger.warning("antiword not installed for .doc files")
        
        # Clean up text
        extracted_text = extracted_text.strip()
        
        # Limit to reasonable size
        if len(extracted_text) > 10000:
            extracted_text = extracted_text[:10000] + "\n... [truncated]"
        
        return {
            "text": extracted_text,
            "filename": filename,
            "chars": len(extracted_text),
        }
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
