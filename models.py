# ============================================================
# models.py — Pydantic data models with validation
# ============================================================

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import re


class EngineerProfile(BaseModel):
    """
    Built from the engineer's intro message when they first join a thread.
    Drives how the agent speaks to this specific person throughout the incident.
    """
    name: str
    role: str
    team: str
    seniority: str
    experience_summary: str
    systems_known: List[str] = []
    current_access: str = "unknown"
    shift_context: str = ""
    communication_style: str = "standard"
    raw_intro: str = ""
    joined_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Reliability tracking
    reliability_score: float = 0.5
    findings_contributed: int = 0
    accurate_findings: int = 0
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) < 1:
            raise ValueError('Name cannot be empty')
        return v.strip()
    
    @validator('seniority')
    def validate_seniority(cls, v):
        valid = ['junior', 'mid', 'senior', 'lead', 'architect', 'manager']
        if v not in valid:
            return 'mid'  # Default
        return v
    
    @validator('communication_style')
    def validate_style(cls, v):
        valid = ['guided', 'standard', 'peer', 'executive']
        if v not in valid:
            return 'standard'
        return v


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    incident_id: str
    thread: str
    sender: str
    sender_type: str
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    @validator('content')
    def validate_content(cls, v):
        # Remove control characters but keep newlines/tabs
        v = ''.join(char for char in v if ord(char) >= 32 or char in '\n\r\t')
        # Truncate if too long
        if len(v) > 10000:
            v = v[:10000]
        return v
    
    @validator('sender_type')
    def validate_sender_type(cls, v):
        valid = ['engineer', 'agent', 'orchestrator']
        if v not in valid:
            return 'engineer'
        return v


class EvidenceScore(BaseModel):
    overall: float = 0.5
    factors: Dict[str, float] = {}
    breakdown: str = ""
    
    @validator('overall')
    def validate_overall(cls, v):
        return max(0.0, min(1.0, v))


class Finding(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    thread: str
    engineer: str
    raw_text: str
    signal_type: str
    entities: dict = {}
    confidence: float = 0.5
    evidence: Optional[EvidenceScore] = None
    verified: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    @validator('signal_type')
    def validate_signal_type(cls, v):
        valid = ['root_cause_candidate', 'symptom', 'clear', 'action_complete', 'blocker', 'new_finding', 'informational']
        if v not in valid:
            return 'informational'
        return v


class HypothesisVersion(BaseModel):
    version: int
    root_cause: str
    confidence: float
    causal_chain: str
    formed_at: str
    supporting_findings: List[str] = []
    opposing_findings: List[str] = []
    proposed_by: List[str] = []


class Hypothesis(BaseModel):
    version: int = 1
    root_cause: str = "Under investigation"
    confidence: float = 0.5
    causal_chain: str = ""
    confirmed_by: List[str] = []
    cleared: List[str] = []
    formed_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Stability tracking
    stable: bool = False
    first_proposed_at: Optional[str] = None
    mention_count: int = 0
    avg_confidence: float = 0.0
    version_history: List[HypothesisVersion] = []


class Action(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    team: str
    description: str
    status: str = "assigned"
    assigned_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    @validator('status')
    def validate_status(cls, v):
        valid = ['assigned', 'in_progress', 'complete', 'blocked']
        if v not in valid:
            return 'assigned'
        return v


class TimelineEvent(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    event_type: str
    description: str
    thread: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('event_type')
    def validate_event_type(cls, v):
        valid = ['alert', 'finding', 'hypothesis', 'action', 'resolution', 'evidence']
        if v not in valid:
            return 'finding'
        return v


class Incident(BaseModel):
    id: str
    title: str
    description: str
    severity: str = "P1"
    affected_system: str
    opened_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    resolved_at: Optional[str] = None
    status: str = "active"
    threads: List[str] = Field(
        default_factory=lambda: ["unix", "storage", "database", "application", "network", "summary"]
    )
    hypothesis: Optional[Hypothesis] = None
    timeline: List[TimelineEvent] = []
    actions: List[Action] = []
    findings: List[Finding] = []
    engineer_profiles: dict = Field(default_factory=dict)
    
    @validator('severity')
    def validate_severity(cls, v):
        valid = ['P1', 'P2', 'P3', 'P4']
        if v not in valid:
            return 'P1'
        return v
    
    @validator('status')
    def validate_status(cls, v):
        valid = ['active', 'resolved']
        if v not in valid:
            return 'active'
        return v