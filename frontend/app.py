# ============================================================
# app.py — Streamlit Frontend for Cloud Deployment
# ============================================================

import streamlit as st
import asyncio
import nest_asyncio
import httpx
import os
import re
from datetime import datetime
import threading
import time

# Apply nest_asyncio
nest_asyncio.apply()

# ── Configuration ──────────────────────────────────────────
# Get backend URL from environment variable
BACKEND_URL = os.environ.get("BACKEND_URL", "").rstrip('/')

if not BACKEND_URL:
    st.error("""
    🚨 **BACKEND_URL environment variable not set!**
    
    Please set the BACKEND_URL in your Streamlit Cloud secrets.
    Example: `https://war-room-backend.onrender.com`
    """)
    st.stop()

# Timeout for API calls (Render cold start can take up to 60s)
TIMEOUT = 60.0

# ── Async Helpers ──────────────────────────────────────────

def run_async(coro):
    """Run async coroutine in Streamlit's sync environment"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Run in thread
            result = []
            def run():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result.append(new_loop.run_until_complete(coro))
                finally:
                    new_loop.close()
            thread = threading.Thread(target=run)
            thread.start()
            thread.join()
            return result[0] if result else None
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
    except Exception as e:
        st.error(f"Async error: {e}")
        return None

# ── API Functions ──────────────────────────────────────────

async def api_get_async(path: str):
    """Async GET request to backend"""
    url = f"{BACKEND_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        st.error(f"🚨 Cannot connect to backend at {BACKEND_URL}")
        st.info("Make sure your backend is running and the URL is correct")
        return None
    except httpx.TimeoutException:
        st.error("⏱️ Backend timeout - please try again")
        return None
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        st.error(f"API error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

async def api_post_async(path: str, payload: dict):
    """Async POST request to backend"""
    url = f"{BACKEND_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        st.error(f"🚨 Cannot connect to backend at {BACKEND_URL}")
        return None
    except httpx.TimeoutException:
        st.error("⏱️ Backend timeout - please try again")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API error: {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

# Sync wrappers
def api_get(path: str):
    """Sync GET wrapper"""
    return run_async(api_get_async(path))

def api_post(path: str, payload: dict):
    """Sync POST wrapper"""
    return run_async(api_post_async(path, payload))

# ── Health Check ───────────────────────────────────────────

def check_backend_health():
    """Check if backend is reachable"""
    health = api_get("/health")
    if health:
        st.sidebar.success("✅ Connected to backend")
        return True
    else:
        st.sidebar.error("❌ Backend not reachable")
        return False

# ── UI Constants ───────────────────────────────────────────

KNOWN_THREADS = {
    "unix": {"label": "🐧 Unix", "color": "#f59e0b"},
    "storage": {"label": "💾 Storage", "color": "#3b82f6"},
    "database": {"label": "🗄️ DB", "color": "#8b5cf6"},
    "application": {"label": "⚙️ App", "color": "#ec4899"},
    "network": {"label": "🌐 Network", "color": "#14b8a6"},
    "windows": {"label": "🪟 Windows", "color": "#0ea5e9"},
    "middleware": {"label": "⚡ Middleware", "color": "#f97316"},
    "security": {"label": "🔐 Security", "color": "#ef4444"},
    "cloud": {"label": "☁️ Cloud", "color": "#06b6d4"},
    "vendor": {"label": "🤝 Vendor", "color": "#84cc16"},
    "summary": {"label": "📣 Summary", "color": "#a855f7"},
}

SENIORITY_COLORS = {
    "junior": "#fbbf24", "mid": "#60a5fa", "senior": "#34d399",
    "lead": "#a78bfa", "architect": "#f472b6", "manager": "#fb923c"
}

def thread_label(thread: str) -> str:
    """Get display label for thread"""
    return KNOWN_THREADS.get(thread, {}).get("label", f"🔧 {thread.title()}")

# ── Message Rendering ──────────────────────────────────────

def render_message(msg: dict):
    """Render a chat message"""
    stype = msg.get("sender_type", "engineer")
    sender = msg.get("sender", "")
    content = msg.get("content", "")
    ts = msg.get("timestamp", "")[:19].replace("T", " ")
    
    # Map CSS classes
    css_map = {
        "engineer": "msg-engineer",
        "agent": "msg-agent"
    }
    css = css_map.get(stype, "msg-orchestrator")
    
    scls_map = {
        "engineer": "s-eng",
        "agent": "s-agt"
    }
    scls = scls_map.get(stype, "s-orc")
    
    # Format content
    content = content.replace('\n', '<br>')
    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'`(.*?)`', r'<code style="background:#2d3748;padding:1px 5px;border-radius:3px;">\1</code>', content)
    
    st.markdown(f"""
    <div class="{css}">
        <div class="msg-sender {scls}">{sender} · {ts}</div>
        <div style="font-size:14px;line-height:1.5;">{content}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Page Config ────────────────────────────────────────────

st.set_page_config(
    page_title="War Room",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────

st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #e6edf3; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background-color: #161b22; border-radius: 6px 6px 0 0;
    color: #8b949e; font-weight: 600; padding: 8px 14px; font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background-color: #1f2937 !important;
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff;
}

/* Chat bubbles */
.msg-engineer  { background:#1a2332; border-left:3px solid #3b82f6; padding:10px 14px; border-radius:0 8px 8px 0; margin:5px 0; }
.msg-agent     { background:#0f2820; border-left:3px solid #10b981; padding:10px 14px; border-radius:0 8px 8px 0; margin:5px 0; }
.msg-orchestrator { background:#1a0f2e; border-left:3px solid #a855f7; padding:10px 14px; border-radius:0 8px 8px 0; margin:5px 0; }

.msg-sender { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; margin-bottom:4px; }
.s-eng  { color:#60a5fa; }
.s-agt  { color:#34d399; }
.s-orc  { color:#c084fc; }

/* Profile card */
.profile-card {
    background:#161b22; border:1px solid #30363d; border-radius:8px;
    padding:10px 14px; margin:5px 0; font-size:12px;
}

/* Hypothesis card */
.hyp-card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:14px; margin-bottom:10px; }

/* Timeline */
.tl-item { border-left:2px solid #30363d; padding:3px 10px; margin:3px 0; font-size:12px; color:#8b949e; }
.tl-item strong { color:#e6edf3; }

/* Badges */
.badge-p1 { background:#dc2626; color:#fff; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:700; }
.badge-resolved { background:#059669; color:#fff; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────

defaults = {
    "active_incident_id": None,
    "engineer_name": "",
    "my_team": "",
    "last_thread_count": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Check Backend ──────────────────────────────────────────

backend_ok = check_backend_health()

if not backend_ok:
    st.stop()

# ── Sidebar ────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚨 War Room")
    st.markdown(f"**Backend:** `{BACKEND_URL}`")
    st.markdown("---")
    
    # Identity
    st.markdown("### Your Identity")
    st.session_state.engineer_name = st.text_input(
        "Your name", 
        value=st.session_state.engineer_name,
        placeholder="e.g. Raj",
        key="name_input"
    )
    
    # Team selection
    st.markdown("### Your Team")
    known = ["unix", "storage", "database", "application",
             "network", "windows", "middleware", "security", "cloud", "vendor"]
    
    team_choice = st.selectbox(
        "Select team",
        options=[""] + known,
        format_func=lambda x: thread_label(x) if x else "— select —"
    )
    
    custom = st.text_input(
        "Or custom team",
        placeholder="e.g. firewall, sap"
    )
    
    if custom.strip():
        st.session_state.my_team = custom.strip().lower().replace(" ", "-")
    elif team_choice:
        st.session_state.my_team = team_choice
    
    if st.session_state.my_team and st.session_state.engineer_name:
        st.success(f"You are in {thread_label(st.session_state.my_team)}")
    
    st.markdown("---")
    
    # New incident
    st.markdown("### New Incident")
    with st.form("new_incident"):
        inc_title = st.text_input("Title", placeholder="Order Service down")
        inc_system = st.text_input("System", placeholder="app-prod")
        inc_sev = st.selectbox("Severity", ["P1", "P2", "P3"])
        inc_desc = st.text_area("Description", height=70)
        
        if st.form_submit_button("🚨 Open War Room", use_container_width=True):
            if inc_title and inc_system:
                with st.spinner("Opening..."):
                    result = api_post("/incidents", {
                        "title": inc_title,
                        "affected_system": inc_system,
                        "severity": inc_sev,
                        "description": inc_desc
                    })
                if result:
                    st.session_state.active_incident_id = result["incident_id"]
                    st.rerun()
    
    st.markdown("---")
    
    # Active incidents
    st.markdown("### Active Incidents")
    incidents = api_get("/incidents") or []
    for inc in incidents:
        icon = "🔴" if inc["status"] == "active" else "✅"
        if st.button(f"{icon} {inc['id']}\n{inc['title'][:28]}",
                     key=f"inc_{inc['id']}", use_container_width=True):
            st.session_state.active_incident_id = inc["id"]
            st.rerun()
    
    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)

# ── Main Content ───────────────────────────────────────────

if not st.session_state.active_incident_id:
    st.markdown("""
    <div style="text-align:center;padding:80px 40px;color:#8b949e;">
        <div style="font-size:56px;">🚨</div>
        <h2>War Room</h2>
        <p>Open a new incident from the sidebar to begin.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load incident
inc_id = st.session_state.active_incident_id
incident = api_get(f"/incidents/{inc_id}")

if not incident:
    st.error("Could not load incident")
    st.stop()

# ── Header ─────────────────────────────────────────────────

col1, col2, col3 = st.columns([4, 1.5, 1.5])

with col1:
    badge = '<span class="badge-p1">P1</span>' if incident["status"] == "active" \
            else '<span class="badge-resolved">RESOLVED</span>'
    st.markdown(f"### {badge} &nbsp; {incident['title']}", unsafe_allow_html=True)
    st.caption(f"{inc_id} · {incident['affected_system']} · {incident['opened_at'][:19]}")

with col2:
    if incident["status"] == "active":
        if st.button("✅ Resolve", use_container_width=True):
            with st.spinner("Generating RCA..."):
                api_post(f"/incidents/{inc_id}/resolve", {})
            st.rerun()

with col3:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

st.markdown("---")

# ── Two Column Layout ──────────────────────────────────────

chat_col, intel_col = st.columns([3, 1])

# ── Intelligence Panel ─────────────────────────────────────

with intel_col:
    # Engineers
    st.markdown("#### 👥 In the Room")
    profiles = api_get(f"/incidents/{inc_id}/profiles") or []
    if profiles:
        for p in profiles:
            color = SENIORITY_COLORS.get(p.get("seniority", "mid"), "#8b949e")
            st.markdown(f"""
            <div class="profile-card">
                <div style="font-weight:700;">{p['name']}
                    <span style="color:{color};margin-left:6px;">{p['seniority'].upper()}</span>
                </div>
                <div style="color:#8b949e;">{p['role']}</div>
                <div style="color:#58a6ff;">{thread_label(p['team'])}</div>
                <div style="font-size:10px;color:#10b981;">Reliability: {p.get('reliability_score', 0.5):.0%}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No engineers yet")
    
    # Evidence Quality
    st.markdown("#### 🔍 Evidence")
    findings = api_get(f"/incidents/{inc_id}/findings") or []
    if findings:
        verified = [f for f in findings if f.get('verified')]
        v_count = len(verified)
        t_count = len(findings)
        v_pct = int(v_count/t_count*100) if t_count > 0 else 0
        
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;">
                <span>Verified</span>
                <span style="color:#10b981;">{v_count}/{t_count} ({v_pct}%)</span>
            </div>
            <div style="height:4px;background:#30363d;border-radius:2px;">
                <div style="height:4px;width:{v_pct}%;background:#10b981;border-radius:2px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Recent Scores"):
            for f in findings[-5:]:
                conf = f.get('confidence', 0)
                color = "#10b981" if conf >= 0.7 else "#f59e0b" if conf >= 0.4 else "#ef4444"
                icon = "✅" if f.get('verified') else "⚠️"
                st.markdown(f"""
                <div style="font-size:11px;">
                    {icon} <b>{f['thread'].upper()}</b>: {f['raw_text'][:30]}...
                    <span style="color:{color};">({conf:.0%})</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.caption("No findings yet")
    
    # Hypothesis
    st.markdown("#### 🧠 Hypothesis")
    hyp = incident.get("hypothesis")
    if hyp:
        conf = int(hyp["confidence"] * 100)
        conf_color = "#ef4444" if conf < 50 else "#f59e0b" if conf < 75 else "#10b981"
        stable = hyp.get("stable", False)
        stability = "✅ STABLE" if stable else "🔄 TENTATIVE"
        stability_color = "#10b981" if stable else "#f59e0b"
        
        st.markdown(f"""
        <div class="hyp-card">
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#8b949e;">v{hyp['version']}</span>
                <span style="color:{stability_color};">{stability}</span>
            </div>
            <div style="font-weight:700;margin:5px 0;">{hyp['root_cause']}</div>
            <div style="font-size:11px;color:#8b949e;">{hyp.get('causal_chain', '')}</div>
            <div style="margin-top:8px;">
                <div style="color:{conf_color};">Confidence: {conf}%</div>
                <div style="height:3px;background:#30363d;">
                    <div style="height:3px;width:{conf}%;background:{conf_color};"></div>
                </div>
            </div>
            <div style="margin-top:8px;font-size:11px;">
                ✅ Confirmed: {', '.join(hyp.get('confirmed_by', [])) or 'gathering'}<br>
                ✅ Cleared: {', '.join(hyp.get('cleared', [])) or 'none'}<br>
                📊 Mentions: {hyp.get('mention_count', 1)}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No hypothesis yet - waiting for findings")
    
    # Timeline
    st.markdown("#### ⏱️ Timeline")
    timeline = incident.get("timeline", [])
    if timeline:
        icons = {"alert": "🚨", "finding": "🔍", "hypothesis": "🧠", 
                 "action": "⚡", "resolution": "✅", "evidence": "📊"}
        for e in reversed(timeline[-10:]):
            icon = icons.get(e["event_type"], "•")
            ts = e["timestamp"][11:19]
            desc = e["description"][:50]
            st.markdown(f'<div class="tl-item">{icon} <strong>{ts}</strong> {desc}</div>', 
                       unsafe_allow_html=True)
    else:
        st.caption("No events")

# ── Chat Threads ───────────────────────────────────────────

with chat_col:
    # Get threads
    threads_data = api_get(f"/incidents/{inc_id}/threads") or {}
    threads = threads_data.get("threads", incident.get("threads", []))
    
    # Ensure current team is in list
    if st.session_state.my_team and st.session_state.my_team not in threads:
        threads = [t for t in threads if t != "summary"] + [st.session_state.my_team, "summary"]
    
    # Create tabs
    tabs = st.tabs([thread_label(t) for t in threads])
    
    for tab, thread in zip(tabs, threads):
        with tab:
            # Get messages
            messages = api_get(f"/incidents/{inc_id}/messages/{thread}") or []
            
            # Show intro banner if needed
            is_my_thread = (thread == st.session_state.my_team)
            
            if is_my_thread and st.session_state.engineer_name and thread != "summary":
                profile_data = api_get(f"/incidents/{inc_id}/profiles/{thread}/{st.session_state.engineer_name}")
                
                if profile_data and not profile_data.get("profiled"):
                    st.info("""
                    👋 **Your first message will be your intro.**
                    
                    Tell the agent: your name, role, and experience.
                    
                    Examples:
                    • "I'm Raj, senior Unix admin, 3 years on this cluster"
                    • "I'm Priya, junior DBA, first P1 - guide me"
                    """)
                elif profile_data and profile_data.get("profiled"):
                    p = profile_data.get("profile", {})
                    st.success(f"✅ Profiled as {p.get('name')} ({p.get('seniority')})")
            
            # Chat history
            chat = st.container(height=400)
            with chat:
                if not messages:
                    st.caption("Waiting for engineers to join...")
                else:
                    for msg in messages:
                        render_message(msg)
            
            # Input
            if thread == "summary":
                st.caption("📣 Read-only - summary channel")
            else:
                if not st.session_state.engineer_name:
                    st.warning("Enter your name in sidebar")
                elif not st.session_state.my_team:
                    st.info("Select your team in sidebar")
                else:
                    # Check if profiled
                    profile_data = api_get(f"/incidents/{inc_id}/profiles/{thread}/{st.session_state.engineer_name}")
                    is_profiled = profile_data and profile_data.get("profiled", False)
                    
                    placeholder = (
                        "Introduce yourself..." 
                        if not is_profiled and thread == st.session_state.my_team
                        else f"Post your {thread} findings..."
                    )
                    
                    with st.form(f"chat_{thread}", clear_on_submit=True):
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            user_input = st.text_input(
                                "msg", placeholder=placeholder,
                                label_visibility="collapsed"
                            )
                        with col2:
                            sent = st.form_submit_button("Send", use_container_width=True)
                        
                        if sent and user_input.strip():
                            engineer = st.session_state.engineer_name
                            
                            with st.spinner("Agent thinking..."):
                                result = api_post(
                                    f"/incidents/{inc_id}/messages/{thread}",
                                    {"content": user_input, "sender": engineer}
                                )
                            
                            if result:
                                if result.get("profile_built"):
                                    st.toast("✅ Profile built!", icon="👤")
                                if result.get("evidence_scored"):
                                    st.toast("📊 Evidence scored", icon="📊")
                                if result.get("hypothesis_updated"):
                                    st.toast("🧠 Hypothesis updated", icon="🧠")
                                if result.get("auto_close_suggested"):
                                    st.toast("🟢 All teams green!", icon="🟢")
                                st.rerun()

# ── Auto-refresh ───────────────────────────────────────────

if auto_refresh and incident.get("status") == "active":
    time.sleep(5)
    st.rerun()