# ============================================================
# agents.py — Optimized with Error Handling & Retries
# ============================================================

import os
import json
import asyncio
import re
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_sleep_log
)
import logging

from models import (
    Message, Finding, EngineerProfile, Hypothesis, 
    HypothesisVersion, Action, TimelineEvent, Incident, EvidenceScore
)
from state import IncidentStateManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.warning("OPENAI_API_KEY not set - using mock responses for testing")

client = AsyncOpenAI(api_key=api_key)
MODEL = "gpt-4o"

# Constants
CONTEXT_WINDOW = 20
EVIDENCE_THRESHOLD = 0.6
STABILITY_MIN_MENTIONS = 3
STABILITY_MIN_MINUTES = 10
STABILITY_MIN_CONFIDENCE = 0.7
MAX_RETRIES = 3
CACHE_MAX_SIZE = 100

# ─────────────────────────────────────────────────────────────
# TEAM DOMAIN KNOWLEDGE
# ─────────────────────────────────────────────────────────────

TEAM_DOMAINS = {
    "unix": {"name": "Unix/Linux", "focus": "CPU, memory, IO wait, syslog/dmesg errors", "tools": "top, iostat, vmstat, dmesg", "common_issues": "IO wait spikes, filesystem full, OOM killer"},
    "storage": {"name": "Storage", "focus": "LUN/volume utilisation, array health, I/O latency", "tools": "iostat, multipath, array console", "common_issues": "LUN capacity exhaustion, disk failure"},
    "database": {"name": "Database", "focus": "Alert log errors, tablespace utilisation, blocking sessions", "tools": "Oracle EM, pg_stat_activity, SQL Server DMVs", "common_issues": "Tablespace full, archive log full, blocking locks"},
    "application": {"name": "Application", "focus": "Error rates, response times, recent deployments", "tools": "APM, application logs", "common_issues": "Bad deployment, config change"},
    "network": {"name": "Network", "focus": "Latency, packet loss, interface errors, BGP/routing", "tools": "ping, traceroute, netstat", "common_issues": "Network partition, high latency"},
    "windows": {"name": "Windows", "focus": "Event logs, Windows services, IIS, Active Directory", "tools": "Event Viewer, PowerShell", "common_issues": "Service crash, AD replication"},
    "middleware": {"name": "Middleware", "focus": "App server health, queue depths, thread pools, JVM heap", "tools": "WebLogic Console, JMX, GC logs", "common_issues": "JVM OOM, thread pool exhaustion"},
    "security": {"name": "Security/Firewall", "focus": "Firewall rules, ACLs, IDS/IPS alerts, certificates", "tools": "Firewall console, SIEM", "common_issues": "Firewall rule blocking, expired cert"},
    "cloud": {"name": "Cloud Infrastructure", "focus": "Cloud resource health, auto-scaling, IAM permissions", "tools": "AWS Console, CloudWatch", "common_issues": "Service limit hit, IAM errors"},
    "vendor": {"name": "Vendor/Third-Party", "focus": "Third-party system status, vendor support", "tools": "Vendor portal", "common_issues": "Vendor outage, API limits"},
}
GENERIC_DOMAIN = {"name": "Specialist", "focus": "Domain-specific investigation", "tools": "Domain-specific tooling", "common_issues": "Domain-specific issues"}

# ─────────────────────────────────────────────────────────────
# COMMUNICATION STYLE GUIDES
# ─────────────────────────────────────────────────────────────

STYLE_GUIDES = {
    "guided": "Use warm, encouraging tone. Explain WHAT and HOW. Give exact commands.",
    "standard": "Clear direction without over-explaining. Mention tools by name.",
    "peer": "Be concise, speak as technical equal. Share reasoning briefly.",
    "executive": "Plain English only. Focus on impact, status, ETA."
}

# ─────────────────────────────────────────────────────────────
# RETRY DECORATOR FOR OPENAI CALLS
# ─────────────────────────────────────────────────────────────

def with_retry():
    """Decorator for retrying OpenAI API calls"""
    return retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )

# ─────────────────────────────────────────────────────────────
# SAFE OPENAI CALL WITH FALLBACK
# ─────────────────────────────────────────────────────────────

async def safe_openai_call(messages, response_format=None, default=None, **kwargs):
    """Safe OpenAI call with fallback and error handling"""
    if not api_key:
        logger.warning("No API key - returning default response")
        return default
    
    try:
        @with_retry()
        async def _call():
            return await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=kwargs.get('temperature', 0),
                max_tokens=kwargs.get('max_tokens', 300),
                response_format=response_format
            )
        return await _call()
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}")
        return default

# ─────────────────────────────────────────────────────────────
# EVIDENCE SCORING ENGINE (with LRU Cache)
# ─────────────────────────────────────────────────────────────

class EvidenceScoringEngine:
    """Scores evidence with LRU caching"""

    def __init__(self, state: IncidentStateManager, incident_id: str):
        self.state = state
        self.incident_id = incident_id
        self._cache = {}
        self._cache_access = []  # LRU tracking
        self._max_cache = CACHE_MAX_SIZE
        self._lock = asyncio.Lock()

    async def score_finding(self, finding: Finding, profile: Optional[EngineerProfile] = None) -> EvidenceScore:
        """Score a finding with parallel factor computation"""
        
        # Run factor computations in parallel
        seniority_task = self._score_seniority(profile)
        command_task = self._score_command_output(finding.raw_text)
        error_task = self._score_error_codes(finding.raw_text)
        metrics_task = self._score_metrics(finding.raw_text)
        specificity_task = self._score_specificity(finding)
        correlation_task = self._calculate_correlation(finding)
        
        # Gather all results
        results = await asyncio.gather(
            seniority_task, command_task, error_task, 
            metrics_task, specificity_task, correlation_task,
            return_exceptions=True
        )
        
        # Handle any errors in factor computation
        factors = {}
        factor_names = ['seniority_weight', 'has_command_output', 'has_error_codes',
                       'has_metrics', 'specificity', 'correlation_with_others']
        
        for name, result in zip(factor_names, results):
            if isinstance(result, Exception):
                logger.error(f"Error computing {name}: {result}")
                factors[name] = 0.5  # Default on error
            else:
                factors[name] = result
        
        # Calculate weighted overall score
        weights = {
            'seniority_weight': 0.25,
            'has_command_output': 0.30,
            'has_error_codes': 0.15,
            'has_metrics': 0.10,
            'specificity': 0.10,
            'correlation_with_others': 0.10
        }
        
        overall = sum(factors.get(k, 0.5) * weights[k] for k in weights)
        overall = max(0.0, min(1.0, overall))
        
        # Generate breakdown
        breakdown = self._generate_breakdown(factors, overall)
        
        return EvidenceScore(
            overall=overall,
            factors=factors,
            breakdown=breakdown
        )
    
    async def _score_seniority(self, profile: Optional[EngineerProfile]) -> float:
        """Score based on engineer seniority"""
        if not profile:
            return 0.5
        
        weights = {
            'junior': 0.3, 'mid': 0.5, 'senior': 0.8,
            'lead': 0.9, 'architect': 0.95, 'manager': 0.4
        }
        base = weights.get(profile.seniority, 0.5)
        
        if profile.findings_contributed > 0:
            accuracy = profile.accurate_findings / profile.findings_contributed
            return (base + accuracy) / 2
        return base
    
    async def _score_command_output(self, text: str) -> float:
        """Check for command output"""
        if '`' in text or '```' in text:
            count = text.count('`')
            if count >= 4:
                return 0.8
            elif count >= 2:
                return 0.5
        return 0.0
    
    async def _score_error_codes(self, text: str) -> float:
        """Check for error codes"""
        patterns = [
            r'ORA-\d{5}', r'ERR\d{4}', r'error \d+', r'status: \d+',
            r'exit code \d+', r'failed with \d+', r'[A-Z]{2,}-\d{4,}'
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 0.7
        return 0.0
    
    async def _score_metrics(self, text: str) -> float:
        """Check for metrics"""
        patterns = [r'\d+%', r'\d+ms', r'\d+s', r'\d+MB', r'\d+GB']
        for pattern in patterns:
            if re.search(pattern, text):
                return 0.6
        return 0.0
    
    async def _score_specificity(self, finding: Finding) -> float:
        """Score based on entity specificity"""
        entities = (finding.entities.get('hostnames', []) + 
                   finding.entities.get('devices', []) + 
                   finding.entities.get('services', []))
        return min(len(entities) * 0.2, 0.8) if entities else 0.2
    
    async def _calculate_correlation(self, finding: Finding) -> float:
        """Calculate correlation with LRU cache"""
        cache_key = f"{finding.thread}_{finding.engineer}_{finding.timestamp}"
        
        async with self._lock:
            # Check cache
            if cache_key in self._cache:
                self._cache_access.remove(cache_key)
                self._cache_access.append(cache_key)
                return self._cache[cache_key]
        
        # Get other findings
        other_findings = await self.state.get_findings(self.incident_id)
        if not other_findings:
            return 0.0
        
        # Extract entities
        my_entities = set()
        for entity_list in finding.entities.values():
            if isinstance(entity_list, list):
                my_entities.update(str(e).lower() for e in entity_list)
        
        if not my_entities:
            return 0.0
        
        # Calculate correlation
        matches = 0
        total = 0
        for other in other_findings[-10:]:
            if other.thread == finding.thread:
                continue
            
            other_entities = set()
            for entity_list in other.entities.values():
                if isinstance(entity_list, list):
                    other_entities.update(str(e).lower() for e in entity_list)
            
            if other_entities & my_entities:
                matches += 1
            total += 1
        
        score = matches / total if total > 0 else 0.0
        
        # Update cache with LRU
        async with self._lock:
            if len(self._cache) >= self._max_cache:
                oldest = self._cache_access.pop(0)
                del self._cache[oldest]
            
            self._cache[cache_key] = score
            self._cache_access.append(cache_key)
        
        return score
    
    def _generate_breakdown(self, factors: dict, overall: float) -> str:
        """Generate human-readable breakdown"""
        lines = [f"Evidence Score: {overall:.0%}"]
        
        if factors.get('has_command_output', 0) > 0.5:
            lines.append("✓ Includes command output")
        if factors.get('has_error_codes', 0) > 0.5:
            lines.append("✓ Contains error codes")
        if factors.get('has_metrics', 0) > 0.5:
            lines.append("✓ Includes metrics")
        if factors.get('seniority_weight', 0) > 0.7:
            lines.append("✓ From senior engineer")
        elif factors.get('seniority_weight', 0) < 0.4:
            lines.append("⚠ From junior - verify")
        
        if overall < 0.4:
            lines.append("⚠ Low confidence - needs verification")
        elif overall > 0.8:
            lines.append("✅ High confidence evidence")
        
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# HYPOTHESIS STABILITY MANAGER (Thread-safe)
# ─────────────────────────────────────────────────────────────

class HypothesisStabilityManager:
    """Thread-safe hypothesis stability manager"""

    def __init__(self):
        self.candidates = {}
        self.stable_hypothesis = None
        self.stable_since = None
        self._lock = asyncio.Lock()
    
    async def add_candidate(self, root_cause: str, confidence: float) -> Optional[str]:
        """Add a candidate with thread safety"""
        async with self._lock:
            # Normalize
            key = root_cause.lower().strip().rstrip('.!?')
            if not key:
                return None
            
            # Update candidate
            if key not in self.candidates:
                self.candidates[key] = {
                    'count': 1,
                    'first_seen': datetime.now(),
                    'confidences': [confidence],
                    'original_text': root_cause
                }
            else:
                self.candidates[key]['count'] += 1
                self.candidates[key]['confidences'].append(confidence)
            
            # Check each candidate for stability
            for cand_key, data in self.candidates.items():
                if self.stable_hypothesis and cand_key != self.stable_hypothesis.lower():
                    continue
                    
                avg_conf = sum(data['confidences']) / len(data['confidences'])
                time_seen = (datetime.now() - data['first_seen']).total_seconds() / 60
                
                if (data['count'] >= STABILITY_MIN_MENTIONS and
                    time_seen >= STABILITY_MIN_MINUTES and
                    avg_conf >= STABILITY_MIN_CONFIDENCE):
                    
                    self.stable_hypothesis = data['original_text']
                    self.stable_since = datetime.now()
                    return data['original_text']
            
            return None
    
    async def get_stable_hypothesis(self) -> Optional[str]:
        """Get current stable hypothesis"""
        async with self._lock:
            return self.stable_hypothesis
    
    async def reset(self):
        """Reset stability tracking"""
        async with self._lock:
            self.candidates = {}
            self.stable_hypothesis = None
            self.stable_since = None


# ─────────────────────────────────────────────────────────────
# ROLLING CONTEXT SUMMARISER
# ─────────────────────────────────────────────────────────────

async def summarise_old_messages(messages: List[Message]) -> str:
    """Summarise old messages with batching"""
    if not messages:
        return ""
    
    # For large contexts, batch process
    if len(messages) > 50:
        batch_size = 25
        batches = [messages[i:i+batch_size] for i in range(0, len(messages), batch_size)]
        
        # Summarize batches in parallel (max 4)
        batch_tasks = []
        for batch in batches[:4]:
            batch_tasks.append(_summarise_batch(batch))
        
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Filter out errors
        valid_results = [r for r in batch_results if not isinstance(r, Exception)]
        if valid_results:
            combined = "\n".join(valid_results)
            return await _final_summary(combined)
    
    # Direct summarization for smaller contexts
    return await _direct_summarise(messages)


async def _summarise_batch(messages: List[Message]) -> str:
    """Summarize a batch of messages"""
    conversation = "\n".join([
        f"{m.sender}: {m.content[:200]}"
        for m in messages
    ])
    
    prompt = f"Summarize these incident messages in 2-3 lines:\n{conversation}"
    
    resp = await safe_openai_call(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        default=None
    )
    
    if resp and resp.choices:
        return resp.choices[0].message.content.strip()
    return ""


async def _final_summary(combined: str) -> str:
    """Create final summary"""
    prompt = f"Combine these into a single digest. Start with 'EARLIER CONTEXT:':\n{combined}"
    
    resp = await safe_openai_call(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        default=None
    )
    
    if resp and resp.choices:
        return resp.choices[0].message.content.strip()
    return "EARLIER CONTEXT: No summary available"


async def _direct_summarise(messages: List[Message]) -> str:
    """Direct summarization"""
    conversation = "\n".join([
        f"{m.sender}: {m.content}"
        for m in messages
    ])
    
    prompt = f"""Summarise this incident conversation into key findings.
Focus on what was checked, found, ruled out.
Start with 'EARLIER CONTEXT:':

{conversation}"""
    
    resp = await safe_openai_call(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        default=None
    )
    
    if resp and resp.choices:
        return resp.choices[0].message.content.strip()
    return "EARLIER CONTEXT: Unable to summarize"


def split_context(messages: List[Message]) -> Tuple[List[Message], List[Message]]:
    """Split messages into old and recent"""
    if len(messages) <= CONTEXT_WINDOW:
        return [], messages
    split = len(messages) - CONTEXT_WINDOW
    return messages[:split], messages[split:]


# ─────────────────────────────────────────────────────────────
# PROFILE BUILDER
# ─────────────────────────────────────────────────────────────

async def build_engineer_profile(thread: str, intro_text: str) -> EngineerProfile:
    """Parse engineer intro with error handling"""
    
    prompt = f"""Parse this engineer intro into JSON profile:

Thread: {thread}
Intro: "{intro_text}"

Return JSON:
{{
  "name": "<first name>",
  "role": "<job title>",
  "team": "{thread}",
  "seniority": "junior|mid|senior|lead|architect|manager",
  "experience_summary": "<summary>",
  "systems_known": ["<systems>"],
  "current_access": "confirmed|limited|none|unknown",
  "shift_context": "<handover notes>",
  "communication_style": "guided|standard|peer|executive",
  "raw_intro": "{intro_text}"
}}"""

    resp = await safe_openai_call(
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=300,
        default=None
    )
    
    # Default profile if API fails
    default_profile = {
        "name": "Engineer",
        "role": "Engineer",
        "team": thread,
        "seniority": "mid",
        "experience_summary": "Engineer",
        "systems_known": [],
        "current_access": "unknown",
        "shift_context": "",
        "communication_style": "standard",
        "raw_intro": intro_text
    }
    
    if not resp or not resp.choices:
        return EngineerProfile(**default_profile)
    
    try:
        content = resp.choices[0].message.content
        # Clean JSON
        content = content.replace('```json', '').replace('```', '').strip()
        data = json.loads(content)
        
        # Ensure required fields
        for key in default_profile:
            if key not in data:
                data[key] = default_profile[key]
        
        return EngineerProfile(**data)
    except Exception as e:
        logger.error(f"Profile parsing error: {e}")
        return EngineerProfile(**default_profile)


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

def build_system_prompt(profile: EngineerProfile, incident_title: str,
                        affected_system: str, earlier_context: str = "") -> str:
    """Build personalised system prompt"""
    domain = TEAM_DOMAINS.get(profile.team.lower(), GENERIC_DOMAIN)
    style = STYLE_GUIDES.get(profile.communication_style, STYLE_GUIDES["standard"])
    
    systems = ""
    if profile.systems_known:
        systems = f"\nKnows: {', '.join(profile.systems_known)}"
    
    shift = f"\nShift context: {profile.shift_context}" if profile.shift_context else ""
    earlier = f"\n\n{earlier_context}" if earlier_context else ""
    
    return f"""You are the {domain['name']} agent in a P1 incident.

INCIDENT: {incident_title}
AFFECTED: {affected_system}

ENGINEER:
- {profile.name}, {profile.role} ({profile.seniority})
- {profile.experience_summary}{systems}{shift}{earlier}

DOMAIN:
- Focus: {domain['focus']}
- Tools: {domain['tools']}

STYLE: {style}

RULES:
1. Address by name
2. One question at a time
3. Acknowledge findings
4. Say "CLEAR" if no issues
5. Prefix "ROOT CAUSE CANDIDATE:"
6. Keep under 6 lines
7. Never re-ask
8. Engage with their ideas"""


# ─────────────────────────────────────────────────────────────
# SIGNAL EXTRACTOR
# ─────────────────────────────────────────────────────────────

async def extract_signal(thread: str, engineer: str, text: str,
                         profile: Optional[EngineerProfile] = None,
                         state: Optional[IncidentStateManager] = None,
                         incident_id: Optional[str] = None) -> Finding:
    """Extract finding with parallel processing"""
    
    # Start LLM extraction
    role = f"Role: {profile.role}, Seniority: {profile.seniority}" if profile else "Role: unknown"
    
    prompt = f"""Extract finding from engineer message:

Thread: {thread}
Engineer: {engineer} ({role})
Message: "{text}"

Return JSON:
{{
  "signal_type": "root_cause_candidate|symptom|clear|action_complete|blocker|new_finding|informational",
  "entities": {{
    "hostnames": [], "error_codes": [], "metrics": {{}},
    "devices": [], "services": [], "other": []
  }},
  "summary": "<one line>"
}}"""

    llm_task = safe_openai_call(
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=200,
        default=None
    )
    
    # Create finding placeholder
    finding = Finding(
        thread=thread,
        engineer=engineer,
        raw_text=text,
        signal_type="informational",
        entities={},
        confidence=0.5
    )
    
    # Start evidence scoring in parallel
    evidence_task = None
    if state and incident_id:
        scorer = EvidenceScoringEngine(state, incident_id)
        evidence_task = scorer.score_finding(finding, profile)
    
    # Wait for LLM
    resp = await llm_task
    
    # Parse LLM response
    if resp and resp.choices:
        try:
            content = resp.choices[0].message.content
            content = content.replace('```json', '').replace('```', '').strip()
            data = json.loads(content)
            finding.signal_type = data.get("signal_type", "informational")
            finding.entities = data.get("entities", {})
        except Exception as e:
            logger.error(f"LLM parse error: {e}")
    
    # Get evidence score
    if evidence_task:
        try:
            evidence = await evidence_task
            finding.evidence = evidence
            finding.confidence = evidence.overall
            finding.verified = evidence.overall >= EVIDENCE_THRESHOLD
        except Exception as e:
            logger.error(f"Evidence scoring error: {e}")
    
    return finding


# ─────────────────────────────────────────────────────────────
# AUTO CLOSE DETECTOR
# ─────────────────────────────────────────────────────────────

def all_teams_green(incident: Incident) -> bool:
    """Check if all teams report green"""
    if not incident:
        return False
    
    active = [t for t in incident.threads if t != "summary"]
    if not active:
        return False
    
    cleared = set(incident.hypothesis.cleared) if incident.hypothesis else set()
    confirmed = set(incident.hypothesis.confirmed_by) if incident.hypothesis else set()
    
    for f in incident.findings:
        if f.verified and f.signal_type == "clear":
            cleared.add(f.thread)
        if f.verified and f.signal_type == "action_complete":
            confirmed.add(f.thread)
    
    resolved = cleared | confirmed
    return all(t in resolved for t in active)


# ─────────────────────────────────────────────────────────────
# TEAM AGENT
# ─────────────────────────────────────────────────────────────

class TeamAgent:
    """Thread-specific agent"""

    def __init__(self, thread: str, incident_id: str, state: IncidentStateManager):
        self.thread = thread
        self.incident_id = incident_id
        self.state = state

    def greeting_message(self, incident: Incident) -> str:
        """Opening message"""
        domain = TEAM_DOMAINS.get(self.thread.lower(), GENERIC_DOMAIN)
        return (
            f"👋 **Welcome to {domain['name']} thread**\n\n"
            f"**{incident.severity}: {incident.title}**\n"
            f"Affected: `{incident.affected_system}`\n\n"
            f"Please introduce yourself: name, role, experience.\n\n"
            f"Examples:\n"
            f"• \"I'm Raj, senior Unix admin, 3 years on this cluster\"\n"
            f"• \"I'm Priya, junior DBA, first P1 — guide me\"\n"
            f"• \"I'm Tom, storage architect, know this array\"\n\n"
            f"ℹ️ Any team can join - just type your team name."
        )

    async def post_intro_kickoff(self, profile: EngineerProfile, incident: Incident) -> str:
        """First personalised message"""
        domain = TEAM_DOMAINS.get(self.thread.lower(), GENERIC_DOMAIN)
        findings = await self.state.get_findings_summary(self.incident_id)
        
        context = ""
        if "No structured findings" not in findings:
            context = f"\n\nOther findings:\n{findings}"
        
        prompt = build_system_prompt(profile, incident.title, incident.affected_system)
        user = f"Engineer intro: \"{profile.raw_intro}\"\nDomain: {domain['name']}{context}\n\nGenerate opening with first question."
        
        resp = await safe_openai_call(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user}
            ],
            temperature=0.4,
            max_tokens=350,
            default=None
        )
        
        if resp and resp.choices:
            return resp.choices[0].message.content.strip()
        
        return f"Thanks {profile.name}. What's the current status on {incident.affected_system}?"

    async def handle_new_team_joining(self, profile: EngineerProfile, incident: Incident) -> str:
        """Brief new team"""
        findings = await self.state.get_findings_summary(self.incident_id)
        hyp = incident.hypothesis
        hyp_text = f"\nHypothesis: {hyp.root_cause}" if hyp else ""
        
        prompt = build_system_prompt(profile, incident.title, incident.affected_system)
        user = f"{profile.name} just joined.\nFindings:\n{findings}{hyp_text}\n\nWelcome and ask first question."
        
        resp = await safe_openai_call(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user}
            ],
            temperature=0.4,
            max_tokens=400,
            default=None
        )
        
        if resp and resp.choices:
            return resp.choices[0].message.content.strip()
        
        return f"Welcome {profile.name}. Current status: {findings[:100]}... Can you check {incident.affected_system}?"

    async def get_response(self, engineer_name: str, message: str,
                           profile: EngineerProfile) -> str:
        """Generate response with context"""
        incident = await self.state.get_incident(self.incident_id)
        history = await self.state.get_messages(self.incident_id, self.thread)
        
        old_msgs, recent_msgs = split_context(history)
        
        # Start summarization if needed
        summarise_task = None
        if old_msgs:
            summarise_task = summarise_old_messages(old_msgs)
        
        # Build base prompt
        system = build_system_prompt(profile, incident.title, incident.affected_system, "")
        
        # Wait for summary
        earlier = ""
        if summarise_task:
            earlier = await summarise_task
            system = build_system_prompt(profile, incident.title, incident.affected_system, earlier)
        
        # Build message list
        msgs = [{"role": "system", "content": system}]
        for m in recent_msgs:
            role = "assistant" if m.sender_type in ("agent", "orchestrator") else "user"
            msgs.append({"role": role, "content": f"{m.sender}: {m.content}"})
        
        msgs.append({"role": "user", "content": f"{engineer_name}: {message}"})
        
        resp = await safe_openai_call(
            messages=msgs,
            temperature=0.3,
            max_tokens=300,
            default=None
        )
        
        if resp and resp.choices:
            return resp.choices[0].message.content.strip()
        
        return f"Thanks for the update, {engineer_name}. Let me analyze that."


# ─────────────────────────────────────────────────────────────
# ORCHESTRATOR AGENT
# ─────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """Main orchestrator with parallel processing"""

    def __init__(self, incident_id: str, state: IncidentStateManager):
        self.incident_id = incident_id
        self.state = state
        self.stability_manager = HypothesisStabilityManager()

    async def open_incident(self, incident: Incident) -> List[Message]:
        """Open incident with parallel greetings"""
        messages = []
        
        # Summary message
        summary = (
            f"📌 **War Room Opened: {incident.id}**\n"
            f"**{incident.severity} — {incident.title}**\n"
            f"Affected: `{incident.affected_system}`\n\n"
            f"Teams: Unix · Storage · Database · Application · Network\n\n"
            f"ℹ️ Any engineer can join any thread.\n"
            f"Type a custom team name to create a new thread.\n"
            f"First message in any thread is your intro."
        )
        
        summary_msg = Message(
            incident_id=self.incident_id, thread="summary",
            sender="WarRoom Bot", sender_type="orchestrator",
            content=summary
        )
        await self.state.add_message(self.incident_id, "summary", summary_msg)
        messages.append(summary_msg)
        
        # Create greeting tasks
        tasks = []
        for thread in [t for t in incident.threads if t != "summary"]:
            agent = TeamAgent(thread, self.incident_id, self.state)
            content = agent.greeting_message(incident)
            msg = Message(
                incident_id=self.incident_id, thread=thread,
                sender="WarRoom Bot", sender_type="agent",
                content=content
            )
            tasks.append(self.state.add_message(self.incident_id, thread, msg))
            messages.append(msg)
        
        # Wait for all greetings
        if tasks:
            await asyncio.gather(*tasks)
        
        # Update timeline
        incident.timeline.append(TimelineEvent(
            event_type="alert",
            description=f"{incident.severity} opened — {incident.title}"
        ))
        await self.state.update_incident(incident)
        
        return messages

    async def process_engineer_input(self, thread: str, engineer_name: str, content: str) -> dict:
        """Process engineer input with true parallelism"""
        
        result = {
            "agent_reply": None,
            "summary_update": None,
            "profile_built": False,
            "auto_close_suggested": False,
            "evidence_scored": False,
            "hypothesis_stable": False
        }
        
        # Get incident
        incident = await self.state.get_incident(self.incident_id)
        if not incident:
            result["agent_reply"] = Message(
                incident_id=self.incident_id, thread=thread,
                sender="WarRoom Bot", sender_type="agent",
                content="Error: Incident not found"
            )
            return result
        
        # Check if new
        is_new = await self.state.is_new_engineer(self.incident_id, thread, engineer_name)
        is_new_thread = thread not in incident.threads
        
        # Create thread if needed
        if is_new_thread:
            await self.state.ensure_thread_exists(self.incident_id, thread)
            incident = await self.state.get_incident(self.incident_id)
            incident.timeline.append(TimelineEvent(
                event_type="finding",
                description=f"New team joined: {thread.upper()} ({engineer_name})"
            ))
            await self.state.update_incident(incident)
        
        agent = TeamAgent(thread, self.incident_id, self.state)
        
        # ── NEW ENGINEER ───────────────────────────────────
        if is_new:
            # Build profile
            profile = await build_engineer_profile(thread, content)
            await self.state.save_profile(self.incident_id, thread, engineer_name, profile)
            
            # Get findings count
            findings = await self.state.get_findings(self.incident_id)
            has_findings = len(findings) > 0
            
            # Generate response
            if has_findings or is_new_thread:
                reply_text = await agent.handle_new_team_joining(profile, incident)
            else:
                reply_text = await agent.post_intro_kickoff(profile, incident)
            
            # Summary update
            all_profiles = await self.state.get_all_profiles(self.incident_id)
            all_names = ", ".join([f"{p.name} ({p.role})" for p in all_profiles])
            
            summary = (
                f"👤 **{profile.name}** ({profile.role}, {profile.seniority}) "
                f"joined **{thread.upper()}**.\n"
                f"Engineers: {all_names}"
            )
            
            summary_msg = Message(
                incident_id=self.incident_id, thread="summary",
                sender="WarRoom Bot", sender_type="orchestrator",
                content=summary
            )
            await self.state.add_message(self.incident_id, "summary", summary_msg)
            
            result["summary_update"] = summary_msg
            result["profile_built"] = True
            
            # Timeline
            incident.timeline.append(TimelineEvent(
                event_type="finding",
                description=f"{profile.name} ({profile.role}) joined {thread.upper()}",
                thread=thread
            ))
            await self.state.update_incident(incident)
        
        # ── RETURNING ENGINEER ─────────────────────────────
        else:
            profile = await self.state.get_profile(self.incident_id, thread, engineer_name)
            
            # Start parallel tasks
            finding_task = extract_signal(thread, engineer_name, content, 
                                         profile, self.state, self.incident_id)
            response_task = asyncio.create_task(
                agent.get_response(engineer_name, content, profile)
            )
            
            # Wait for finding
            finding = await finding_task
            
            # Process finding
            if finding.verified:
                await self.state.add_finding(self.incident_id, finding)
                incident = await self.state.get_incident(self.incident_id)
                incident.findings.append(finding)
                result["evidence_scored"] = True
                
                if profile:
                    profile.findings_contributed += 1
                    await self.state.save_profile(self.incident_id, thread, engineer_name, profile)
                
                if finding.signal_type in ("root_cause_candidate", "clear", "action_complete", "blocker"):
                    incident.timeline.append(TimelineEvent(
                        event_type="finding",
                        description=f"{thread.upper()} — {engineer_name}: {content[:80]}",
                        thread=thread
                    ))
                await self.state.update_incident(incident)
                
                # Check for hypothesis update
                all_findings = await self.state.get_findings(self.incident_id)
                root_candidates = [f for f in all_findings 
                                 if f.signal_type == "root_cause_candidate" and f.verified]
                
                if len(root_candidates) >= 2 or finding.signal_type == "root_cause_candidate":
                    hypothesis_task = asyncio.create_task(
                        self._update_hypothesis_with_stability(incident)
                    )
                    
                    # Wait for response
                    reply_text = await response_task
                    
                    # Then hypothesis
                    summary_msg, stable = await hypothesis_task
                    result["summary_update"] = summary_msg
                    result["hypothesis_stable"] = stable
                else:
                    reply_text = await response_task
            else:
                # Low evidence - custom response
                reply_text = (
                    f"Thanks {profile.name}. This has low evidence ({finding.confidence:.0%}). "
                    f"Can you provide command output, error codes, or metrics?"
                )
                
                if not response_task.done():
                    response_task.cancel()
                
                incident.timeline.append(TimelineEvent(
                    event_type="evidence",
                    description=f"{thread.upper()} — Low evidence ({finding.confidence:.0%})",
                    thread=thread
                ))
                await self.state.update_incident(incident)
            
            # Check auto-close
            incident = await self.state.get_incident(self.incident_id)
            if all_teams_green(incident) and incident.status == "active":
                close_msg = await self._suggest_auto_close(incident)
                result["auto_close_suggested"] = True
                if not result["summary_update"]:
                    result["summary_update"] = close_msg
        
        # Store reply
        agent_reply = Message(
            incident_id=self.incident_id, thread=thread,
            sender="WarRoom Bot", sender_type="agent",
            content=reply_text
        )
        await self.state.add_message(self.incident_id, thread, agent_reply)
        result["agent_reply"] = agent_reply
        
        return result

    async def _update_hypothesis_with_stability(self, incident: Incident) -> tuple:
        """Update hypothesis with stability"""
        
        # Get verified findings
        verified = [f for f in incident.findings if f.verified]
        if not verified:
            return None, False
        
        # Build prompt
        findings_text = "\n".join([
            f"[{f.thread.upper()}] {f.engineer}: {f.raw_text[:100]} (conf: {f.confidence:.0%})"
            for f in verified[-10:]
        ])
        
        profiles = await self.state.get_all_profiles(self.incident_id)
        engineers = "\n".join([
            f"- {p.name} ({p.role}, {p.seniority}, reliability: {p.reliability_score:.0%})"
            for p in profiles
        ])
        
        prompt = f"""Correlate findings:

Incident: {incident.title}
Affected: {incident.affected_system}

Engineers:
{engineers}

Findings:
{findings_text}

Return JSON:
{{
  "root_cause": "<description>",
  "confidence": 0.0-1.0,
  "causal_chain": "<A caused B>",
  "confirmed_by": ["<thread>"],
  "cleared": ["<thread>"],
  "actions": [{{"team": "<thread>", "description": "<action>"}}]
}}"""

        resp = await safe_openai_call(
            messages=[
                {"role": "system", "content": "You are an incident orchestrator. Return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
            default=None
        )
        
        if not resp or not resp.choices:
            return None, False
        
        try:
            content = resp.choices[0].message.content
            content = content.replace('```json', '').replace('```', '').strip()
            data = json.loads(content)
        except:
            return None, False
        
        version = (incident.hypothesis.version + 1) if incident.hypothesis else 1
        
        # Check stability
        stable_cause = await self.stability_manager.add_candidate(
            data.get("root_cause", "Under investigation"),
            float(data.get("confidence", 0.5))
        )
        
        if stable_cause or version == 1:
            # Create version entry
            version_entry = HypothesisVersion(
                version=version,
                root_cause=data.get("root_cause", "Under investigation"),
                confidence=float(data.get("confidence", 0.5)),
                causal_chain=data.get("causal_chain", ""),
                formed_at=datetime.now().isoformat(),
                supporting_findings=[f.id for f in verified if f.signal_type == "root_cause_candidate"],
                proposed_by=list(set(f.thread for f in verified if f.signal_type == "root_cause_candidate"))
            )
            
            if not incident.hypothesis:
                incident.hypothesis = Hypothesis(
                    version=version,
                    root_cause=data.get("root_cause", "Under investigation"),
                    confidence=float(data.get("confidence", 0.5)),
                    causal_chain=data.get("causal_chain", ""),
                    confirmed_by=data.get("confirmed_by", []),
                    cleared=data.get("cleared", []),
                    stable=bool(stable_cause),
                    first_proposed_at=version_entry.formed_at,
                    mention_count=1,
                    avg_confidence=float(data.get("confidence", 0.5)),
                    version_history=[version_entry]
                )
            else:
                incident.hypothesis.version = version
                incident.hypothesis.root_cause = data.get("root_cause", incident.hypothesis.root_cause)
                incident.hypothesis.confidence = float(data.get("confidence", incident.hypothesis.confidence))
                incident.hypothesis.causal_chain = data.get("causal_chain", incident.hypothesis.causal_chain)
                incident.hypothesis.confirmed_by = data.get("confirmed_by", incident.hypothesis.confirmed_by)
                incident.hypothesis.cleared = data.get("cleared", incident.hypothesis.cleared)
                incident.hypothesis.stable = bool(stable_cause)
                incident.hypothesis.mention_count += 1
                incident.hypothesis.avg_confidence = (
                    (incident.hypothesis.avg_confidence + float(data.get("confidence", 0.5))) / 2
                )
                incident.hypothesis.version_history.append(version_entry)
            
            # Create actions
            existing = {a.team for a in incident.actions if a.status != "complete"}
            for a in data.get("actions", []):
                team = a.get("team", "")
                if team and team not in existing:
                    action = Action(team=team, description=a.get("description", ""))
                    incident.actions.append(action)
                    existing.add(team)
                    
                    # Post to thread
                    msg = Message(
                        incident_id=self.incident_id, thread=team,
                        sender="WarRoom Bot", sender_type="orchestrator",
                        content=f"⚡ **ACTION:** {action.description}"
                    )
                    await self.state.add_message(self.incident_id, team, msg)
            
            # Update timeline
            status = "STABLE" if stable_cause else "TENTATIVE"
            incident.timeline.append(TimelineEvent(
                event_type="hypothesis",
                description=f"Hypothesis v{version} ({status}): {data.get('root_cause', '')[:50]}"
            ))
            await self.state.update_incident(incident)
            
            # Create summary
            hyp = incident.hypothesis
            summary = (
                f"📊 **HYPOTHESIS v{version}** — {'✅' if hyp.stable else '🔄'} "
                f"{'STABLE' if hyp.stable else 'TENTATIVE'}\n"
                f"Confidence: {int(hyp.confidence*100)}% | Mentions: {hyp.mention_count}\n"
                f"**Root Cause:** {hyp.root_cause}\n"
                f"**Confirmed:** {', '.join(hyp.confirmed_by) or 'gathering'}\n"
                f"**Cleared:** {', '.join(hyp.cleared) or 'none'}"
            )
            
            summary_msg = Message(
                incident_id=self.incident_id, thread="summary",
                sender="WarRoom Bot", sender_type="orchestrator",
                content=summary
            )
            await self.state.add_message(self.incident_id, "summary", summary_msg)
            
            return summary_msg, bool(stable_cause)
        
        return None, False

    async def _suggest_auto_close(self, incident: Incident) -> Message:
        """Suggest auto-close"""
        content = (
            f"🟢 **ALL TEAMS GREEN**\n\n"
            f"Every thread reports clear or complete.\n\n"
            f"**Hypothesis:** {incident.hypothesis.root_cause if incident.hypothesis else 'N/A'}\n"
            f"**Confidence:** {incident.hypothesis.confidence:.0% if incident.hypothesis else 'N/A'}\n\n"
            f"👉 Click **Mark Resolved** to close and generate RCA."
        )
        
        msg = Message(
            incident_id=self.incident_id, thread="summary",
            sender="WarRoom Bot", sender_type="orchestrator",
            content=content
        )
        await self.state.add_message(self.incident_id, "summary", msg)
        
        incident.timeline.append(TimelineEvent(
            event_type="action",
            description="All teams green - awaiting resolution"
        ))
        await self.state.update_incident(incident)
        
        return msg

    async def resolve_incident(self) -> Message:
        """Close incident and generate RCA"""
        incident = await self.state.get_incident(self.incident_id)
        incident.status = "resolved"
        incident.resolved_at = datetime.now().isoformat()
        
        opened = datetime.fromisoformat(incident.opened_at)
        resolved = datetime.fromisoformat(incident.resolved_at)
        mins = int((resolved - opened).total_seconds() / 60)
        
        root = incident.hypothesis.root_cause if incident.hypothesis else "Under investigation"
        
        # Build RCA
        content = f"""✅ **INCIDENT CLOSED — {incident.id}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Duration: {mins} minutes
Severity: {incident.severity}
Affected: {incident.affected_system}

**ROOT CAUSE:**
  {root}

**TIMELINE:**
{self._format_timeline(incident.timeline)}

**ACTIONS:**
{self._format_actions(incident.actions)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        msg = Message(
            incident_id=self.incident_id, thread="summary",
            sender="WarRoom Bot", sender_type="orchestrator",
            content=content
        )
        await self.state.add_message(self.incident_id, "summary", msg)
        
        incident.timeline.append(TimelineEvent(
            event_type="resolution",
            description=f"Incident resolved after {mins} minutes"
        ))
        await self.state.update_incident(incident)
        
        return msg
    
    def _format_timeline(self, events: List[TimelineEvent]) -> str:
        """Format timeline for RCA"""
        lines = []
        for e in events[-20:]:
            time = e.timestamp[11:19]
            lines.append(f"  {time} [{e.event_type.upper()}] {e.description[:50]}")
        return "\n".join(lines)
    
    def _format_actions(self, actions: List[Action]) -> str:
        """Format actions for RCA"""
        lines = []
        for a in actions:
            lines.append(f"  • [{a.status}] {a.team.upper()}: {a.description[:50]}")
        return "\n".join(lines) if lines else "  • No actions"