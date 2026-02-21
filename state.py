# ============================================================
# state.py — Thread-safe Incident State Manager
# ============================================================

from typing import Optional, List, Dict
import asyncio
import copy
from models import Incident, Message, Finding, EngineerProfile


class IncidentStateManager:
    """Thread-safe state manager with per-incident locks"""

    def __init__(self):
        self._incidents: Dict[str, Incident] = {}
        self._messages: Dict[str, Dict[str, List[Message]]] = {}
        self._findings: Dict[str, List[Finding]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def _get_lock(self, incident_id: str) -> asyncio.Lock:
        """Get or create a lock for an incident"""
        async with self._global_lock:
            if incident_id not in self._locks:
                self._locks[incident_id] = asyncio.Lock()
            return self._locks[incident_id]

    # ── Incidents ──────────────────────────────────────────

    async def create_incident(self, incident: Incident):
        """Create a new incident"""
        lock = await self._get_lock(incident.id)
        async with lock:
            self._incidents[incident.id] = incident
            self._messages[incident.id] = {t: [] for t in incident.threads}
            self._findings[incident.id] = []

    async def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID (returns a copy to avoid mutation issues)"""
        lock = await self._get_lock(incident_id)
        async with lock:
            incident = self._incidents.get(incident_id)
            if incident:
                # Return a deep copy to prevent external mutation
                return copy.deepcopy(incident)
            return None

    async def update_incident(self, incident: Incident):
        """Update an existing incident"""
        lock = await self._get_lock(incident.id)
        async with lock:
            # Merge with existing to prevent data loss
            existing = self._incidents.get(incident.id)
            if existing:
                # Update only changed fields
                for field, value in incident.dict().items():
                    if value != getattr(existing, field, None):
                        setattr(existing, field, value)
                self._incidents[incident.id] = existing
            else:
                self._incidents[incident.id] = incident

    async def list_incidents(self) -> List[Incident]:
        """List all incidents"""
        async with self._global_lock:
            return [copy.deepcopy(i) for i in self._incidents.values()]

    # ── Dynamic thread registration ────────────────────────

    async def ensure_thread_exists(self, incident_id: str, thread: str):
        """Create thread if it doesn't exist"""
        lock = await self._get_lock(incident_id)
        async with lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return
            
            # Add to threads list if new
            if thread not in incident.threads:
                if "summary" in incident.threads:
                    idx = incident.threads.index("summary")
                    incident.threads.insert(idx, thread)
                else:
                    incident.threads.append(thread)
            
            # Initialize messages dict
            if incident_id not in self._messages:
                self._messages[incident_id] = {}
            if thread not in self._messages[incident_id]:
                self._messages[incident_id][thread] = []

    # ── Messages ───────────────────────────────────────────

    async def add_message(self, incident_id: str, thread: str, message: Message):
        """Add a message to a thread"""
        await self.ensure_thread_exists(incident_id, thread)
        lock = await self._get_lock(incident_id)
        async with lock:
            self._messages[incident_id][thread].append(message)

    async def get_messages(self, incident_id: str, thread: str) -> List[Message]:
        """Get messages from a thread"""
        lock = await self._get_lock(incident_id)
        async with lock:
            msgs = self._messages.get(incident_id, {}).get(thread, [])
            return [copy.deepcopy(m) for m in msgs]

    async def get_all_messages(self, incident_id: str) -> List[Message]:
        """Get all messages across all threads"""
        lock = await self._get_lock(incident_id)
        async with lock:
            all_msgs = []
            for thread_msgs in self._messages.get(incident_id, {}).values():
                all_msgs.extend(thread_msgs)
            return sorted(copy.deepcopy(all_msgs), key=lambda m: m.timestamp)

    # ── Findings ───────────────────────────────────────────

    async def add_finding(self, incident_id: str, finding: Finding):
        """Add a finding"""
        lock = await self._get_lock(incident_id)
        async with lock:
            if incident_id not in self._findings:
                self._findings[incident_id] = []
            self._findings[incident_id].append(finding)

    async def get_findings(self, incident_id: str) -> List[Finding]:
        """Get all findings"""
        lock = await self._get_lock(incident_id)
        async with lock:
            findings = self._findings.get(incident_id, [])
            return [copy.deepcopy(f) for f in findings]

    async def get_findings_summary(self, incident_id: str) -> str:
        """Get findings summary text"""
        findings = await self.get_findings(incident_id)
        if not findings:
            return "No structured findings yet."
        lines = []
        for f in findings[-20:]:  # Last 20 findings
            verified = "✓" if f.verified else "?"
            lines.append(
                f"[{verified}] [{f.thread.upper()}] {f.engineer}: {f.raw_text[:100]} "
                f"(type={f.signal_type}, conf={f.confidence:.0%})"
            )
        return "\n".join(lines)

    # ── Engineer Profiles ──────────────────────────────────

    async def save_profile(self, incident_id: str, thread: str,
                           engineer_name: str, profile: EngineerProfile):
        """Save engineer profile"""
        lock = await self._get_lock(incident_id)
        async with lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return
            key = f"{thread}:{engineer_name}"
            incident.engineer_profiles[key] = profile.dict()

    async def get_profile(self, incident_id: str, thread: str,
                          engineer_name: str) -> Optional[EngineerProfile]:
        """Get engineer profile"""
        lock = await self._get_lock(incident_id)
        async with lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return None
            key = f"{thread}:{engineer_name}"
            data = incident.engineer_profiles.get(key)
            return EngineerProfile(**data) if data else None

    async def get_all_profiles(self, incident_id: str) -> List[EngineerProfile]:
        """Get all engineer profiles"""
        lock = await self._get_lock(incident_id)
        async with lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return []
            return [EngineerProfile(**v) for v in incident.engineer_profiles.values()]

    async def is_new_engineer(self, incident_id: str, thread: str, engineer_name: str) -> bool:
        """Check if engineer is new to this thread"""
        profile = await self.get_profile(incident_id, thread, engineer_name)
        return profile is None