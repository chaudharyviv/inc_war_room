# ============================================================
# main.py — FastAPI Backend with Async Support
# ============================================================

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
from datetime import datetime
import logging
import asyncio
from contextlib import asynccontextmanager

from models import Incident, Message
from state import IncidentStateManager
from agents import OrchestratorAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
state = IncidentStateManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup/shutdown"""
    logger.info("Starting War Room API")
    yield
    logger.info("Shutting down War Room API")

app = FastAPI(title="War Room Incident API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Error Handler ──────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )

# ── Incidents ──────────────────────────────────────────────

@app.post("/incidents")
async def create_incident(payload: dict):
    """Create a new incident"""
    try:
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
        
        incident = Incident(
            id=incident_id,
            title=payload.get("title", "P1 Incident"),
            description=payload.get("description", ""),
            severity=payload.get("severity", "P1"),
            affected_system=payload.get("affected_system", "Unknown"),
        )
        
        await state.create_incident(incident)
        
        orchestrator = OrchestratorAgent(incident_id, state)
        opening_msgs = await orchestrator.open_incident(incident)
        
        return {
            "incident_id": incident_id,
            "incident": incident.dict(),
            "opening_messages": [m.dict() for m in opening_msgs]
        }
    except Exception as e:
        logger.error(f"Error creating incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/incidents")
async def list_incidents():
    """List all incidents"""
    try:
        incidents = await state.list_incidents()
        return [i.dict() for i in incidents]
    except Exception as e:
        logger.error(f"Error listing incidents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/incidents/{incident_id}")
async def get_incident(incident_id: str):
    """Get incident by ID"""
    try:
        incident = await state.get_incident(incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        return incident.dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/incidents/{incident_id}/resolve")
async def resolve_incident(incident_id: str):
    """Resolve an incident"""
    try:
        incident = await state.get_incident(incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        orchestrator = OrchestratorAgent(incident_id, state)
        msg = await orchestrator.resolve_incident()
        return {"message": msg.dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Messages ───────────────────────────────────────────────

@app.get("/incidents/{incident_id}/messages/{thread}")
async def get_messages(incident_id: str, thread: str):
    """Get messages from a thread"""
    try:
        msgs = await state.get_messages(incident_id, thread)
        return [m.dict() for m in msgs]
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/incidents/{incident_id}/messages/{thread}")
async def post_message(incident_id: str, thread: str, payload: dict):
    """Post a message to a thread"""
    try:
        # Validate incident exists
        incident = await state.get_incident(incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        # Validate payload
        content = payload.get("content", "").strip()
        engineer = payload.get("sender", "Engineer").strip()
        
        if not content:
            raise HTTPException(status_code=400, detail="Content required")
        
        if len(content) > 10000:
            content = content[:10000]
        
        # Ensure thread exists
        await state.ensure_thread_exists(incident_id, thread)
        
        # Store engineer message
        eng_msg = Message(
            incident_id=incident_id,
            thread=thread,
            sender=engineer,
            sender_type="engineer",
            content=content
        )
        await state.add_message(incident_id, thread, eng_msg)
        
        # Process through orchestrator
        orchestrator = OrchestratorAgent(incident_id, state)
        result = await orchestrator.process_engineer_input(
            thread=thread,
            engineer_name=engineer,
            content=content
        )
        
        response = {
            "engineer_message": eng_msg.dict(),
            "agent_reply": result["agent_reply"].dict() if result["agent_reply"] else None,
            "summary_update": result["summary_update"].dict() if result["summary_update"] else None,
            "profile_built": result.get("profile_built", False),
            "hypothesis_updated": result["summary_update"] is not None,
            "evidence_scored": result.get("evidence_scored", False),
            "auto_close_suggested": result.get("auto_close_suggested", False)
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error posting message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Threads ────────────────────────────────────────────────

@app.get("/incidents/{incident_id}/threads")
async def get_threads(incident_id: str):
    """Get all threads for an incident"""
    try:
        incident = await state.get_incident(incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        return {"threads": incident.threads}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting threads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Profiles ───────────────────────────────────────────────

@app.get("/incidents/{incident_id}/profiles")
async def get_profiles(incident_id: str):
    """Get all engineer profiles"""
    try:
        profiles = await state.get_all_profiles(incident_id)
        return [p.dict() for p in profiles]
    except Exception as e:
        logger.error(f"Error getting profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/incidents/{incident_id}/profiles/{thread}/{engineer}")
async def get_profile(incident_id: str, thread: str, engineer: str):
    """Get specific engineer profile"""
    try:
        profile = await state.get_profile(incident_id, thread, engineer)
        if not profile:
            return {"profiled": False}
        return {"profiled": True, "profile": profile.dict()}
    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Findings ───────────────────────────────────────────────

@app.get("/incidents/{incident_id}/findings")
async def get_findings(incident_id: str):
    """Get all findings"""
    try:
        findings = await state.get_findings(incident_id)
        return [f.dict() for f in findings]
    except Exception as e:
        logger.error(f"Error getting findings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Health Check ───────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)