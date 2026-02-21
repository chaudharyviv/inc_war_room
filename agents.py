# ============================================================
# agents.py — IMPROVED: Full LLM Power + Fixed Core Issues
# ============================================================

import os
import json
import asyncio
import re
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any
from openai import AsyncOpenAI
import logging

from models import (
    Message, Finding, EngineerProfile, Hypothesis, 
    HypothesisVersion, Action, TimelineEvent, Incident, EvidenceScore
)
from state import IncidentStateManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.environ.get("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key) if api_key else None
MODEL = "gpt-4o"

# ─────────────────────────────────────────────────────────────
# HELPER: Safe OpenAI Call
# ─────────────────────────────────────────────────────────────

async def safe_llm_call(messages: List[dict], response_format=None, max_tokens=800) -> Optional[str]:
    """Safe LLM call with fallback"""
    if not client:
        logger.warning("No OpenAI client - returning None")
        return None
    
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,  # Slightly creative for natural responses
            max_tokens=max_tokens,
            response_format=response_format
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# PROFILER AGENT (Simplified but Reliable)
# ─────────────────────────────────────────────────────────────

class ProfilerAgent:
    """Extract engineer profile - simple regex + LLM fallback"""
    
    def __init__(self, state: IncidentStateManager, incident_id: str):
        self.state = state
        self.incident_id = incident_id
    
    async def build_profile(self, thread: str, engineer_name: str, intro_text: str) -> EngineerProfile:
        """Build profile from intro"""
        
        # PHASE 1: Extract name reliably (regex first, then LLM)
        name_match = re.search(r"I'?m\s+(\w+)", intro_text, re.IGNORECASE)
        extracted_name = name_match.group(1) if name_match else None
        
        # If regex fails, try LLM
        if not extracted_name:
            prompt = f"Extract only the person's first name from: '{intro_text}'. Return just the name, nothing else."
            llm_result = await safe_llm_call(
                [{"role": "user", "content": prompt}],
                max_tokens=50
            )
            extracted_name = llm_result.strip() if llm_result else engineer_name
        
        # PHASE 2: Let LLM infer seniority and style naturally
        profile_prompt = f"""Analyze this engineer's introduction and infer their experience level:
"{intro_text}"

Based on their language, determine:
1. Seniority: junior/mid/senior/lead
2. Communication style preference: guided/standard/peer

Return JSON:
{{"seniority": "...", "communication_style": "..."}}"""

        llm_profile = await safe_llm_call(
            [{"role": "user", "content": profile_prompt}],
            response_format={"type": "json_object"},
            max_tokens=100
        )
        
        if llm_profile:
            try:
                profile_data = json.loads(llm_profile)
                seniority = profile_data.get("seniority", "mid")
                comm_style = profile_data.get("communication_style", "standard")
            except:
                seniority = "mid"
                comm_style = "standard"
        else:
            # Fallback to simple pattern matching
            intro_lower = intro_text.lower()
            if any(word in intro_lower for word in ["junior", "new", "first", "guide me"]):
                seniority = "junior"
                comm_style = "guided"
            elif any(word in intro_lower for word in ["senior", "lead", "architect", "years"]):
                seniority = "senior"
                comm_style = "peer"
            else:
                seniority = "mid"
                comm_style = "standard"
        
        # Build profile
        profile = EngineerProfile(
            name=extracted_name,
            role=f"{thread.title()} Engineer",
            team=thread,
            seniority=seniority,
            experience_summary=intro_text[:200],
            raw_intro=intro_text,
            communication_style=comm_style
        )
        
        await self.state.save_profile(self.incident_id, thread, extracted_name, profile)
        logger.info(f"Profile created: {extracted_name} ({seniority}, {comm_style}) in {thread}")
        
        return profile

# ─────────────────────────────────────────────────────────────
# THREAD AGENT (Full LLM Intelligence + Context Awareness)
# ─────────────────────────────────────────────────────────────

class ThreadAgent:
    """Context-aware thread agent that leverages full LLM capabilities"""
    
    def __init__(self, incident_id: str, thread: str, state: IncidentStateManager):
        self.incident_id = incident_id
        self.thread = thread
        self.state = state
    
    async def generate_response(
        self, 
        engineer_name: str, 
        engineer_input: str,
        profile: Optional[EngineerProfile] = None
    ) -> Message:
        """Generate intelligent, context-aware response using full LLM power"""
        
        # Get conversation history
        messages = await self.state.get_messages(self.incident_id, self.thread)
        
        # Get incident details
        incident = await self.state.get_incident(self.incident_id)
        
        # Get ALL findings (cross-team visibility)
        all_findings = await self.state.get_findings(self.incident_id)
        
        # Build rich context
        context = self._build_rich_context(
            incident=incident,
            messages=messages,
            all_findings=all_findings,
            engineer_input=engineer_input
        )
        
        # Generate response with full LLM intelligence
        response_text = await self._generate_intelligent_response(
            engineer_name=engineer_name,
            engineer_input=engineer_input,
            context=context,
            profile=profile
        )
        
        if not response_text:
            response_text = f"Acknowledged {engineer_name}. Continue investigation."
        
        # Create message
        msg = Message(
            incident_id=self.incident_id,
            thread=self.thread,
            sender=f"{self.thread.title()} Agent",
            sender_type="agent",
            content=response_text
        )
        
        await self.state.add_message(self.incident_id, self.thread, msg)
        return msg
    
    def _build_rich_context(
        self,
        incident: Incident,
        messages: List[Message],
        all_findings: List[Finding],
        engineer_input: str
    ) -> str:
        """Build comprehensive context for LLM"""
        
        # Incident overview
        incident_context = f"""INCIDENT: {incident.title}
Severity: {incident.severity}
Affected System: {incident.affected_system}
Description: {incident.description}"""
        
        # Conversation history (last 8 messages for more context)
        recent = messages[-8:] if len(messages) > 8 else messages
        conversation = "\n".join([
            f"[{m.timestamp[11:19]}] {m.sender}: {m.content[:200]}"
            for m in recent
        ])
        
        # Current thread's findings
        thread_findings = [f for f in all_findings if f.thread == self.thread]
        thread_summary = "\n".join([
            f"- {f.engineer}: {f.raw_text[:150]} [{f.signal_type}]"
            for f in thread_findings[-5:]
        ]) if thread_findings else "No findings yet in this thread"
        
        # OTHER teams' findings (CRITICAL for cross-team awareness)
        other_findings = [f for f in all_findings if f.thread != self.thread]
        other_summary = "\n".join([
            f"- [{f.thread.upper()}] {f.engineer}: {f.raw_text[:150]} [{f.signal_type}]"
            for f in other_findings[-10:]  # More context from other teams
        ]) if other_findings else "No updates from other teams yet"
        
        # Current hypothesis (if any)
        hypothesis_context = "No hypothesis yet"
        if incident.hypothesis:
            hyp = incident.hypothesis
            hypothesis_context = f"""Current Hypothesis (v{hyp.version}):
Root Cause: {hyp.root_cause}
Confidence: {int(hyp.confidence*100)}%
Confirmed by: {', '.join(hyp.confirmed_by) if hyp.confirmed_by else 'none'}
Cleared: {', '.join(hyp.cleared) if hyp.cleared else 'none'}"""
        
        return f"""{incident_context}

═══════════════════════════════════════════════════
CONVERSATION HISTORY (This {self.thread} thread):
═══════════════════════════════════════════════════
{conversation}

═══════════════════════════════════════════════════
FINDINGS FROM THIS THREAD ({self.thread}):
═══════════════════════════════════════════════════
{thread_summary}

═══════════════════════════════════════════════════
FINDINGS FROM OTHER TEAMS:
═══════════════════════════════════════════════════
{other_summary}

═══════════════════════════════════════════════════
CURRENT HYPOTHESIS:
═══════════════════════════════════════════════════
{hypothesis_context}

═══════════════════════════════════════════════════
CURRENT ENGINEER MESSAGE:
═══════════════════════════════════════════════════
{engineer_input}"""
    
    async def _generate_intelligent_response(
        self,
        engineer_name: str,
        engineer_input: str,
        context: str,
        profile: Optional[EngineerProfile]
    ) -> Optional[str]:
        """Let LLM use its full intelligence to respond appropriately"""
        
        # Build communication style guidance
        style_guidance = ""
        if profile:
            if profile.communication_style == "guided":
                style_guidance = "Be supportive and provide clear step-by-step guidance. Explain commands and what to look for."
            elif profile.communication_style == "peer":
                style_guidance = "Communicate as a technical peer. Be concise and trust their expertise."
            else:
                style_guidance = "Be clear and professional. Provide direction without over-explaining."
        
        # System prompt - give LLM full autonomy within guardrails
        system_prompt = f"""You are an expert Site Reliability Engineer helping investigate a {self.thread} issue in a War Room incident.

Your role:
- You specialize in {self.thread} systems but understand the full stack
- You coordinate with other teams (Unix, Storage, Network, Database, Application, etc.)
- You can handle ANY technology: SMB, Windows Servers, VMware, VDI, cloud, containers, etc.

CRITICAL RULES (read conversation history carefully):
1. NEVER repeat questions already answered in the conversation
2. If another team found the root cause, acknowledge it and coordinate accordingly
3. If engineer says "no issue" or "all clear", confirm and stand by
4. Reference previous conversation when relevant
5. Always address the engineer by name: {engineer_name}
6. Cross-reference findings from other teams before asking questions
7. If stuck or engineer pushes back on a suggestion, adapt your approach

{style_guidance}

Response guidelines:
- Keep responses under 100 words unless detailed explanation needed
- One focused question or directive at a time
- If engineer found something important, acknowledge and dig deeper
- If investigation is blocked, suggest alternatives
- If root cause is found, help verify and coordinate resolution"""

        user_prompt = f"""{context}

Based on ALL the context above, respond intelligently to {engineer_name}'s latest message.

Think step by step:
1. What has already been discussed in THIS thread?
2. What have OTHER teams found?
3. What is {engineer_name} saying now?
4. What's the most helpful next step?

Respond naturally and intelligently. Don't be robotic."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return await safe_llm_call(messages, max_tokens=400)

# ─────────────────────────────────────────────────────────────
# SIGNAL CLASSIFIER (LLM-Powered for Flexibility)
# ─────────────────────────────────────────────────────────────

class SignalClassifier:
    """Classify engineer messages - LLM-powered for maximum flexibility"""
    
    async def classify(self, content: str, thread: str) -> Tuple[str, dict]:
        """Classify message signal using LLM intelligence"""
        
        prompt = f"""Classify this engineer's message from the {thread} team:

"{content}"

Signal types:
- "root_cause_candidate": They believe they found the root cause
- "clear": No issues found, team is cleared
- "blocker": They're blocked or need access/help
- "new_finding": They found something relevant (error, metric, log entry)
- "informational": Just providing updates or asking clarification

Return JSON:
{{"signal_type": "...", "confidence": 0.0-1.0, "key_entities": []}}

Be generous with root_cause_candidate - if they sound confident about finding something, mark it."""

        llm_result = await safe_llm_call(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=200
        )
        
        if llm_result:
            try:
                data = json.loads(llm_result)
                signal_type = data.get("signal_type", "informational")
                entities = {"confidence": data.get("confidence", 0.5)}
                return signal_type, entities
            except:
                pass
        
        # Fallback: Simple pattern matching
        content_lower = content.lower()
        
        if any(phrase in content_lower for phrase in ["no issue", "all clear", "looks good", "fine here"]):
            return "clear", {}
        
        if any(phrase in content_lower for phrase in ["found", "this is", "root cause", "bug", "issue is"]):
            return "root_cause_candidate", {}
        
        if any(phrase in content_lower for phrase in ["cannot", "blocked", "need access", "no permission"]):
            return "blocker", {}
        
        if any(phrase in content_lower for phrase in ["seeing", "error", "warning", "high", "spike"]):
            return "new_finding", {}
        
        return "informational", {}

# ─────────────────────────────────────────────────────────────
# ORCHESTRATOR AGENT
# ─────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """Main orchestrator with LLM-powered hypothesis management"""
    
    def __init__(self, incident_id: str, state: IncidentStateManager):
        self.incident_id = incident_id
        self.state = state
        self.profiler = ProfilerAgent(state, incident_id)
        self.classifier = SignalClassifier()
    
    async def open_incident(self, incident: Incident) -> List[Message]:
        """Open incident and create welcome messages"""
        messages = []
        
        welcome_template = f"""👋 Welcome to {{thread}} thread
{incident.severity}: {incident.title}
Affected: {incident.affected_system}

Please introduce yourself: name, role, experience.
Examples:
• "I'm Raj, senior Unix admin, 3 years on this cluster"
• "I'm Priya, junior DBA, first P1 — guide me"
• "I'm Tom, storage architect, know this array"

ℹ️ Any team can join - just type your team name."""
        
        for thread in incident.threads:
            if thread != "summary":
                msg = Message(
                    incident_id=self.incident_id,
                    thread=thread,
                    sender="WarRoom Bot",
                    sender_type="orchestrator",
                    content=welcome_template.format(thread=thread.title())
                )
                await self.state.add_message(self.incident_id, thread, msg)
                messages.append(msg)
        
        return messages
    
    async def process_engineer_input(
        self,
        thread: str,
        engineer_name: str,
        content: str
    ) -> Dict[str, Any]:
        """Process engineer input with full LLM intelligence"""
        
        result = {
            "agent_reply": None,
            "summary_update": None,
            "profile_built": False,
            "evidence_scored": False,
            "auto_close_suggested": False
        }
        
        # Check if new engineer (needs profiling)
        is_new = await self.state.is_new_engineer(self.incident_id, thread, engineer_name)
        profile = None
        
        if is_new:
            # Build profile
            profile = await self.profiler.build_profile(thread, engineer_name, content)
            result["profile_built"] = True
            
            # Welcome message
            welcome = f"Welcome {profile.name}! Let's start investigating the {thread} system."
            reply_msg = Message(
                incident_id=self.incident_id,
                thread=thread,
                sender=f"{thread.title()} Agent",
                sender_type="agent",
                content=welcome
            )
            await self.state.add_message(self.incident_id, thread, reply_msg)
            result["agent_reply"] = reply_msg
            return result
        else:
            profile = await self.state.get_profile(self.incident_id, thread, engineer_name)
        
        # Classify signal using LLM
        signal_type, entities = await self.classifier.classify(content, thread)
        
        # Create finding
        finding = Finding(
            thread=thread,
            engineer=engineer_name,
            raw_text=content,
            signal_type=signal_type,
            entities=entities,
            confidence=0.8 if signal_type in ["root_cause_candidate", "clear"] else 0.6
        )
        await self.state.add_finding(self.incident_id, finding)
        result["evidence_scored"] = True
        
        # Generate intelligent, context-aware response
        agent = ThreadAgent(self.incident_id, thread, self.state)
        reply_msg = await agent.generate_response(engineer_name, content, profile)
        result["agent_reply"] = reply_msg
        
        # Update hypothesis if root cause found
        if signal_type == "root_cause_candidate":
            summary_msg = await self._update_hypothesis(finding)
            result["summary_update"] = summary_msg
        
        # Check if all teams clear
        if signal_type == "clear":
            all_clear = await self._check_all_clear()
            if all_clear:
                result["auto_close_suggested"] = True
        
        return result
    
    async def _update_hypothesis(self, finding: Finding) -> Optional[Message]:
        """Update hypothesis using LLM intelligence"""
        
        incident = await self.state.get_incident(self.incident_id)
        all_findings = await self.state.get_findings(self.incident_id)
        
        # Let LLM synthesize hypothesis from all findings
        findings_summary = "\n".join([
            f"[{f.thread.upper()}] {f.engineer}: {f.raw_text[:200]} [{f.signal_type}]"
            for f in all_findings[-15:]  # Last 15 findings
        ])
        
        hypothesis_prompt = f"""Analyze these findings from a multi-team incident investigation:

{findings_summary}

Current hypothesis: {incident.hypothesis.root_cause if incident.hypothesis else "None"}

Synthesize a clear, concise root cause hypothesis.
Consider:
- What do multiple teams agree on?
- Are there clear causal chains?
- What has highest confidence?

Return JSON:
{{
    "root_cause": "concise root cause statement",
    "confidence": 0.0-1.0,
    "confirmed_by": ["team1", "team2"],
    "causal_chain": "brief explanation of cause -> effect"
}}"""

        llm_result = await safe_llm_call(
            [{"role": "user", "content": hypothesis_prompt}],
            response_format={"type": "json_object"},
            max_tokens=300
        )
        
        if llm_result:
            try:
                data = json.loads(llm_result)
                
                if not incident.hypothesis:
                    incident.hypothesis = Hypothesis(
                        version=1,
                        root_cause=data.get("root_cause", finding.raw_text[:200]),
                        confidence=float(data.get("confidence", 0.7)),
                        causal_chain=data.get("causal_chain", ""),
                        confirmed_by=data.get("confirmed_by", [finding.thread])
                    )
                else:
                    # Update existing hypothesis
                    incident.hypothesis.version += 1
                    incident.hypothesis.root_cause = data.get("root_cause", incident.hypothesis.root_cause)
                    incident.hypothesis.confidence = float(data.get("confidence", incident.hypothesis.confidence))
                    incident.hypothesis.causal_chain = data.get("causal_chain", incident.hypothesis.causal_chain)
                    
                    # Merge confirmed_by lists
                    for team in data.get("confirmed_by", []):
                        if team not in incident.hypothesis.confirmed_by:
                            incident.hypothesis.confirmed_by.append(team)
                
                await self.state.update_incident(incident)
                
            except Exception as e:
                logger.error(f"Failed to parse hypothesis: {e}")
        
        # Create summary message
        if incident.hypothesis:
            hyp = incident.hypothesis
            summary_text = f"""📊 **HYPOTHESIS v{hyp.version}**

**Root Cause:** {hyp.root_cause}
**Confidence:** {int(hyp.confidence*100)}%
**Confirmed by:** {', '.join(hyp.confirmed_by)}
**Analysis:** {hyp.causal_chain}

Teams: Continue verification and gather additional evidence."""
            
            summary_msg = Message(
                incident_id=self.incident_id,
                thread="summary",
                sender="WarRoom Bot",
                sender_type="orchestrator",
                content=summary_text
            )
            await self.state.add_message(self.incident_id, "summary", summary_msg)
            
            return summary_msg
        
        return None
    
    async def _check_all_clear(self) -> bool:
        """Check if all teams cleared"""
        findings = await self.state.get_findings(self.incident_id)
        reported_threads = set(f.thread for f in findings)
        clear_threads = set(f.thread for f in findings if f.signal_type == "clear")
        return len(reported_threads) > 0 and reported_threads == clear_threads
    
    async def resolve_incident(self) -> Message:
        """Resolve incident with LLM-generated RCA"""
        
        incident = await self.state.get_incident(self.incident_id)
        all_findings = await self.state.get_findings(self.incident_id)
        all_messages = await self.state.get_all_messages(self.incident_id)
        
        incident.status = "resolved"
        incident.resolved_at = datetime.now().isoformat()
        
        opened = datetime.fromisoformat(incident.opened_at)
        resolved = datetime.fromisoformat(incident.resolved_at)
        duration_mins = int((resolved - opened).total_seconds() / 60)
        
        # Generate comprehensive RCA using LLM
        findings_summary = "\n".join([
            f"[{f.thread.upper()}] {f.engineer}: {f.raw_text[:200]}"
            for f in all_findings[-20:]
        ])
        
        rca_prompt = f"""Generate a concise Root Cause Analysis for this incident:

INCIDENT: {incident.title}
Duration: {duration_mins} minutes
Hypothesis: {incident.hypothesis.root_cause if incident.hypothesis else "Unresolved"}

Key Findings:
{findings_summary}

Create a brief RCA with:
1. Root cause (2-3 sentences)
2. Impact summary
3. Resolution steps taken
4. Preventive measures (optional)

Keep it clear and actionable."""

        llm_rca = await safe_llm_call(
            [{"role": "user", "content": rca_prompt}],
            max_tokens=500
        )
        
        root = llm_rca if llm_rca else (incident.hypothesis.root_cause if incident.hypothesis else "Under investigation")
        
        rca_text = f"""✅ **INCIDENT CLOSED — {incident.id}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Duration: {duration_mins} minutes
Severity: {incident.severity}
Affected: {incident.affected_system}

**ROOT CAUSE ANALYSIS:**
{root}

**TEAMS INVOLVED:**
{', '.join(incident.hypothesis.confirmed_by) if incident.hypothesis else 'Multiple teams'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        msg = Message(
            incident_id=self.incident_id,
            thread="summary",
            sender="WarRoom Bot",
            sender_type="orchestrator",
            content=rca_text
        )
        
        await self.state.add_message(self.incident_id, "summary", msg)
        await self.state.update_incident(incident)
        
        return msg
