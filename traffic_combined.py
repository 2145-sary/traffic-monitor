"""
IoT-Based Intelligent Traffic Monitoring & Congestion Prediction System
K.R. Mangalam University | Department of Computer Science & Engineering
Developer: Sarthak Mishra | Roll No: 2301010232 | B.Tech CSE | 2025-26
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import cv2
import os
import time
import sqlite3
import hashlib
import tempfile
import pandas as pd
import streamlit as st
from datetime import datetime
from collections import deque
from ultralytics import YOLO

try:
    import paho.mqtt.client as mqtt_client
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════
# PART 1 — DATABASE
# ════════════════════════════════════════════════════════════════════

DB_FILE = "traffic_data.db"

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def init_db():
    conn = sqlite3.connect(DB_FILE)

    # Traffic log table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS traffic_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT,
            lane1_count   INTEGER,
            lane2_count   INTEGER,
            lane1_density TEXT,
            lane2_density TEXT,
            lane1_signal  TEXT,
            lane2_signal  TEXT
        )
    """)

    # Users table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role     TEXT
        )
    """)

    # Seed default users if table is empty
    existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if existing == 0:
        conn.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
                     ("admin", hash_pw("krmu2025"), "Admin"))
        conn.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
                     ("sarthak", hash_pw("pass123"), "Viewer"))

    conn.commit()
    conn.close()

def check_login(username, password):
    conn = sqlite3.connect(DB_FILE)
    row = conn.execute(
        "SELECT role FROM users WHERE username=? AND password=?",
        (username, hash_pw(password))
    ).fetchone()
    conn.close()
    return row[0] if row else None

def save_to_db(l1c, l2c, l1d, l2d, l1s, l2s):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO traffic_log
        (timestamp, lane1_count, lane2_count,
         lane1_density, lane2_density, lane1_signal, lane2_signal)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          l1c, l2c, l1d, l2d, l1s, l2s))
    conn.commit()
    conn.close()

def read_db(limit=50):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(
        f"SELECT * FROM traffic_log ORDER BY id DESC LIMIT {limit}", conn
    )
    conn.close()
    return df

def get_total_records():
    conn = sqlite3.connect(DB_FILE)
    count = conn.execute("SELECT COUNT(*) FROM traffic_log").fetchone()[0]
    conn.close()
    return count

def get_db_stats():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM traffic_log", conn)
    conn.close()
    return df

init_db()


# ════════════════════════════════════════════════════════════════════
# PART 2 — YOLOV8 DETECTION
# ════════════════════════════════════════════════════════════════════

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
_model = None

def load_yolo_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model

def get_density(count):
    if count <= 3:   return "Low"
    elif count <= 7: return "Medium"
    else:            return "High"

def get_signal(d1, d2):
    rank = {"High": 3, "Medium": 2, "Low": 1}
    if rank[d1] >= rank[d2]:
        return "Lane 1 GREEN", "Lane 2 RED"
    return "Lane 1 RED", "Lane 2 GREEN"

def get_congestion_score(count, density):
    """Returns 0-100 congestion score"""
    base = {"Low": 10, "Medium": 45, "High": 75}
    score = base[density] + min(count * 2, 25)
    return min(score, 100)

def detect_frame(frame, conf=0.4):
    mdl = load_yolo_model()
    h, w = frame.shape[:2]
    mid = w // 2
    annotated = frame.copy()
    cv2.line(annotated, (mid, 0), (mid, h), (0, 212, 255), 2)
    cv2.line(annotated, (mid-1, 0), (mid-1, h), (0, 80, 120), 1)
    cv2.line(annotated, (mid+1, 0), (mid+1, h), (0, 80, 120), 1)
    cv2.rectangle(annotated, (5, 5), (155, 44), (0,0,0), -1)
    cv2.putText(annotated, "LANE 1", (10, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 212, 255), 2)
    cv2.rectangle(annotated, (mid+5, 5), (mid+155, 44), (0,0,0), -1)
    cv2.putText(annotated, "LANE 2", (mid+10, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 160, 0), 2)
    results = mdl(frame, conf=conf, verbose=False)[0]
    l1_cnt = l2_cnt = 0
    conf_scores = []
    vehicle_types = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        label = VEHICLE_CLASSES[cls_id]
        cs = float(box.conf[0])
        conf_scores.append(cs)
        vehicle_types[label] += 1
        if cx < mid:
            l1_cnt += 1
            color = (0, 230, 118)
        else:
            l2_cnt += 1
            color = (255, 160, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        ltext = f"{label} {cs:.0%}"
        (tw, th), _ = cv2.getTextSize(ltext, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(annotated, ltext, (x1+2, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,0,0), 1)
    l1_den = get_density(l1_cnt)
    l2_den = get_density(l2_cnt)
    l1_sig, l2_sig = get_signal(l1_den, l2_den)
    avg_conf = round(sum(conf_scores) / len(conf_scores) * 100, 1) if conf_scores else 0.0
    return annotated, l1_cnt, l2_cnt, l1_den, l2_den, l1_sig, l2_sig, avg_conf, vehicle_types


# ════════════════════════════════════════════════════════════════════
# PART 3 — IoT PUBLISHER
# ════════════════════════════════════════════════════════════════════

def publish_iot(l1c, l2c, l1d, l2d, l1s, l2s):
    save_to_db(l1c, l2c, l1d, l2d, l1s, l2s)


# ════════════════════════════════════════════════════════════════════
# PART 4 — PAGE CONFIG
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="IoT Traffic Monitor | KRM University",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ════════════════════════════════════════════════════════════════════
# PART 5 — SESSION STATE
# ════════════════════════════════════════════════════════════════════

defaults = {
    "logged_in":    False,
    "username":     "",
    "role":         "",
    "dark_mode":    True,
    "running":      False,
    "l1_hist":      deque(maxlen=40),
    "l2_hist":      deque(maxlen=40),
    "d1_hist":      deque(maxlen=40),
    "d2_hist":      deque(maxlen=40),
    "total_frames": 0,
    "session_start": None,
    "session_high_events": 0,
    "session_total_vehicles": 0,
    "last_frame":   None,
    "last_l1": 0, "last_l2": 0,
    "last_d1": "Low", "last_d2": "Low",
    "last_s1": "Lane 1 GREEN", "last_s2": "Lane 2 RED",
    "last_conf": 0.0,
    "last_vtypes": {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0},
    "show_summary": False,
    "active_tab":   "🎯  Live Monitor",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

DENSITY_NUM = {"Low": 1, "Medium": 2, "High": 3}

# ── Theme colors ──────────────────────────────────────────────────
def theme():
    if st.session_state.dark_mode:
        return {
            "bg":      "#060810",
            "panel":   "#0b0d18",
            "card":    "#0f1220",
            "border":  "#1a1f35",
            "text1":   "#eef0f8",
            "text2":   "#8890aa",
            "text3":   "#444d6a",
        }
    else:
        return {
            "bg":      "#f4f6fb",
            "panel":   "#ffffff",
            "card":    "#ffffff",
            "border":  "#e0e4f0",
            "text1":   "#0f1220",
            "text2":   "#4a5280",
            "text3":   "#8890aa",
        }


# ════════════════════════════════════════════════════════════════════
# PART 6 — MASTER CSS (theme-aware)
# ════════════════════════════════════════════════════════════════════

def inject_css():
    t = theme()
    dm = st.session_state.dark_mode
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {{
    --bg:        {t["bg"]};
    --panel:     {t["panel"]};
    --card:      {t["card"]};
    --border:    {t["border"]};
    --text1:     {t["text1"]};
    --text2:     {t["text2"]};
    --text3:     {t["text3"]};
    --blue:      #00d4ff;
    --cyan:      #00fff0;
    --green:     #00e676;
    --orange:    #ff9800;
    --red:       #ff3d57;
    --yellow:    #ffd600;
    --purple:    #b44dff;
    --fd:        'Syne', sans-serif;
    --fb:        'DM Sans', sans-serif;
    --r-sm:      8px;
    --r-md:      14px;
    --r-lg:      20px;
    --r-xl:      28px;
    --shadow:    0 4px 24px rgba(0,0,0,{0.5 if dm else 0.08});
    --glow-b:    0 0 30px rgba(0,212,255,{0.2 if dm else 0.12});
    --glow-g:    0 0 30px rgba(0,230,118,{0.2 if dm else 0.12});
    --glow-o:    0 0 30px rgba(255,152,0,{0.2 if dm else 0.12});
    --glow-r:    0 0 30px rgba(255,61,87,{0.3 if dm else 0.15});
}}

*,*::before,*::after{{box-sizing:border-box;margin:0;}}
html,body,.stApp{{
    background:var(--bg) !important;
    color:var(--text1) !important;
    font-family:var(--fb) !important;
}}
#MainMenu,footer,header{{visibility:hidden;}}
.stDeployButton{{display:none;}}
.block-container{{padding:0 2rem 4rem !important;max-width:1700px !important;}}
::-webkit-scrollbar{{width:5px;}}
::-webkit-scrollbar-track{{background:var(--bg);}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:3px;}}
::-webkit-scrollbar-thumb:hover{{background:var(--blue);}}

/* MESH BG */
.stApp::before{{
    content:'';position:fixed;top:0;left:0;right:0;bottom:0;
    background:
        radial-gradient(ellipse 800px 500px at 10% 10%,rgba(0,212,255,{0.03 if dm else 0.04}) 0%,transparent 60%),
        radial-gradient(ellipse 600px 400px at 90% 80%,rgba(255,152,0,{0.03 if dm else 0.03}) 0%,transparent 60%),
        radial-gradient(ellipse 400px 300px at 50% 50%,rgba(180,77,255,{0.02 if dm else 0.02}) 0%,transparent 70%);
    pointer-events:none;z-index:0;
}}

/* ── KILL DEFAULT STREAMLIT PADDING ON LOGIN ── */
.login-page-active .block-container{{
    padding:0 !important;
    max-width:100% !important;
}}

/* LOGIN PAGE */
.login-full{{
    position:fixed;top:0;left:0;right:0;bottom:0;
    background:#060810;
    display:flex;
    z-index:9999;
    overflow:hidden;
}}
/* Left panel — branding side */
.login-left{{
    flex:1;
    background:linear-gradient(145deg,#060810 0%,#0a0e1a 40%,#060d1a 100%);
    display:flex;flex-direction:column;
    align-items:flex-start;justify-content:center;
    padding:60px 56px;
    position:relative;overflow:hidden;
    border-right:1px solid rgba(0,212,255,0.08);
}}
.login-left::before{{
    content:'';position:absolute;top:-100px;right:-100px;
    width:500px;height:500px;border-radius:50%;
    background:radial-gradient(circle,rgba(0,212,255,0.05) 0%,transparent 65%);
    pointer-events:none;
}}
.login-left::after{{
    content:'';position:absolute;bottom:-80px;left:-80px;
    width:400px;height:400px;border-radius:50%;
    background:radial-gradient(circle,rgba(255,152,0,0.04) 0%,transparent 65%);
    pointer-events:none;
}}
/* Decorative grid lines on left panel */
.login-grid{{
    position:absolute;top:0;left:0;right:0;bottom:0;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px,transparent 1px),
        linear-gradient(90deg,rgba(0,212,255,0.03) 1px,transparent 1px);
    background-size:60px 60px;
    pointer-events:none;
}}
.login-brand-wrap{{position:relative;z-index:1;}}
.login-brand-icon{{
    font-size:3.5rem;display:block;margin-bottom:28px;
    filter:drop-shadow(0 0 20px rgba(0,212,255,0.3));
}}
.login-brand-univ{{
    font-family:var(--fd);font-size:.62rem;font-weight:700;
    letter-spacing:.22em;text-transform:uppercase;
    background:linear-gradient(90deg,var(--blue),var(--cyan));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;margin-bottom:16px;
    display:flex;align-items:center;gap:10px;
}}
.login-brand-univ::before{{
    content:'';display:block;width:24px;height:1px;
    background:linear-gradient(90deg,var(--blue),var(--cyan));
    flex-shrink:0;
}}
.login-brand-title{{
    font-family:var(--fd);font-size:2.4rem;font-weight:800;
    color:#eef0f8;line-height:1.15;letter-spacing:-.03em;
    margin-bottom:16px;
}}
.login-brand-title .t-blue{{
    background:linear-gradient(135deg,var(--blue),var(--cyan));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;
}}
.login-brand-title .t-orange{{color:var(--orange);}}
.login-brand-desc{{
    font-size:.82rem;color:#444d6a;line-height:1.8;
    max-width:340px;margin-bottom:40px;
}}
.login-features{{display:flex;flex-direction:column;gap:12px;}}
.login-feat{{
    display:flex;align-items:center;gap:12px;
    font-size:.75rem;color:#4a5570;
}}
.login-feat-dot{{
    width:6px;height:6px;border-radius:50%;background:var(--blue);
    box-shadow:0 0 8px var(--blue);flex-shrink:0;
}}

/* Right panel — form side */
.login-right{{
    width:420px;flex-shrink:0;
    background:#0b0d18;
    display:flex;flex-direction:column;
    align-items:center;justify-content:center;
    padding:50px 44px;
    position:relative;overflow:hidden;
}}
.login-right::before{{
    content:'';position:absolute;top:0;left:0;bottom:0;width:1px;
    background:linear-gradient(180deg,transparent,rgba(0,212,255,0.15),rgba(0,212,255,0.08),transparent);
}}
.login-form-wrap{{width:100%;}}
.login-form-header{{margin-bottom:32px;}}
.login-form-title{{
    font-family:var(--fd);font-size:1.4rem;font-weight:800;
    color:#eef0f8;letter-spacing:-.02em;margin-bottom:6px;
}}
.login-form-sub{{font-size:.72rem;color:#444d6a;letter-spacing:.04em;}}
.login-label{{
    font-family:var(--fd);font-size:.6rem;font-weight:700;
    letter-spacing:.14em;text-transform:uppercase;color:#3a4260;
    margin-bottom:6px;display:block;
}}
.login-divider{{
    height:1px;margin:28px 0;
    background:linear-gradient(90deg,transparent,rgba(0,212,255,0.12),transparent);
}}
.login-hint{{
    margin-top:20px;padding:14px 16px;
    background:rgba(0,212,255,0.04);
    border:1px solid rgba(0,212,255,0.1);
    border-radius:var(--r-sm);
    font-size:.67rem;color:#3a4260;line-height:1.9;
}}
.login-hint b{{color:rgba(0,212,255,0.6);}}

/* HEADER */
.site-header{{
    background:linear-gradient(135deg,{
        "rgba(10,13,24,.98),rgba(14,18,32,.98),rgba(10,13,24,.98)" if dm
        else "rgba(255,255,255,.98),rgba(248,250,255,.98),rgba(255,255,255,.98)"
    });
    border-bottom:1px solid var(--border);
    padding:18px 40px 16px;margin:0 -2rem 0;
    display:flex;align-items:center;justify-content:space-between;
    position:relative;overflow:hidden;
}}
.site-header::before{{
    content:'';position:absolute;top:0;left:0;right:0;bottom:0;
    background:
        radial-gradient(ellipse 700px 150px at 15% 50%,rgba(0,212,255,.05) 0%,transparent 70%),
        radial-gradient(ellipse 500px 150px at 85% 50%,rgba(255,152,0,.04) 0%,transparent 70%);
    pointer-events:none;
}}
.site-header::after{{
    content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(0,212,255,.4),rgba(255,152,0,.3),transparent);
}}
.header-left{{display:flex;flex-direction:column;gap:2px;}}
.univ-badge{{
    font-family:var(--fd);font-size:.65rem;font-weight:700;
    letter-spacing:.2em;text-transform:uppercase;
    background:linear-gradient(90deg,var(--blue),var(--cyan));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.header-title{{
    font-family:var(--fd);font-size:1.7rem;font-weight:800;
    color:var(--text1);line-height:1.15;letter-spacing:-.03em;
}}
.header-title .hl-b{{
    background:linear-gradient(135deg,var(--blue),var(--cyan));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.header-title .hl-o{{color:var(--orange);}}
.header-sub{{font-size:.75rem;color:var(--text3);letter-spacing:.05em;margin-top:2px;font-style:italic;}}
.header-right{{display:flex;flex-direction:column;align-items:flex-end;gap:8px;}}
.status-pill{{
    display:inline-flex;align-items:center;gap:8px;
    background:rgba(0,230,118,.08);border:1px solid rgba(0,230,118,.25);
    border-radius:100px;padding:6px 16px;font-size:.7rem;font-weight:700;
    color:var(--green);letter-spacing:.1em;text-transform:uppercase;
    box-shadow:0 0 20px rgba(0,230,118,.1);
}}
.status-dot{{
    width:7px;height:7px;background:var(--green);border-radius:50%;
    box-shadow:0 0 8px var(--green);animation:pdot 2s ease-in-out infinite;
}}
@keyframes pdot{{
    0%,100%{{opacity:1;transform:scale(1);}}
    50%{{opacity:.5;transform:scale(.8);}}
}}
.dept-tag{{font-size:.65rem;color:var(--text3);letter-spacing:.06em;}}

/* NAV TABS */
.nav-wrap{{
    display:flex;align-items:center;gap:4px;
    background:var(--panel);border-bottom:1px solid var(--border);
    padding:0 40px;margin:0 -2rem 2rem;overflow-x:auto;
}}

/* SEC LABEL */
.sec-label{{
    font-family:var(--fd);font-size:.62rem;font-weight:700;
    letter-spacing:.2em;text-transform:uppercase;color:var(--text3);
    margin-bottom:10px;display:flex;align-items:center;gap:8px;
}}
.sec-label::after{{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);}}

/* STAT CARDS */
.stat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;}}
.stat-card{{
    background:var(--card);border:1px solid var(--border);
    border-radius:var(--r-md);padding:20px 20px 16px;
    position:relative;overflow:hidden;
    transition:all .3s cubic-bezier(.4,0,.2,1);box-shadow:var(--shadow);
}}
.stat-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;}}
.stat-card.l1::before{{background:linear-gradient(90deg,var(--blue),var(--cyan),transparent);}}
.stat-card.l2::before{{background:linear-gradient(90deg,var(--orange),#ffcc02,transparent);}}
.stat-card.l1{{border-top-color:rgba(0,212,255,.3);}}
.stat-card.l2{{border-top-color:rgba(255,152,0,.3);}}
.stat-card.l1:hover{{border-color:rgba(0,212,255,.3);box-shadow:var(--glow-b);transform:translateY(-2px);}}
.stat-card.l2:hover{{border-color:rgba(255,152,0,.3);box-shadow:var(--glow-o);transform:translateY(-2px);}}
.lane-label{{
    font-family:var(--fd);font-size:.62rem;font-weight:700;
    letter-spacing:.18em;text-transform:uppercase;margin-bottom:12px;
    display:flex;align-items:center;gap:6px;
}}
.lane-label.c1{{color:var(--blue);}} .lane-label.c2{{color:var(--orange);}}
.ldot{{width:6px;height:6px;border-radius:50%;animation:pdot 2s ease-in-out infinite;}}
.c1 .ldot{{background:var(--blue);box-shadow:0 0 6px var(--blue);}}
.c2 .ldot{{background:var(--orange);box-shadow:0 0 6px var(--orange);}}
.count-num{{font-family:var(--fd);font-size:3.2rem;font-weight:800;line-height:1;margin-bottom:2px;letter-spacing:-.04em;}}
.count-sub{{font-size:.65rem;color:var(--text3);letter-spacing:.08em;margin-bottom:16px;text-transform:uppercase;}}
.den-wrap{{margin-bottom:12px;}}
.den-row{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}}
.den-key{{font-size:.65rem;color:var(--text2);letter-spacing:.06em;text-transform:uppercase;}}
.den-val{{font-size:.7rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;font-family:var(--fd);}}
.den-val.Low{{color:var(--green);}} .den-val.Medium{{color:var(--yellow);}} .den-val.High{{color:var(--red);}}
.den-bar{{height:5px;background:var(--border);border-radius:3px;overflow:hidden;}}
.den-fill{{height:100%;border-radius:3px;transition:width .8s cubic-bezier(.4,0,.2,1);}}
.den-fill.Low{{background:linear-gradient(90deg,var(--green),#00ff99);width:28%;box-shadow:0 0 8px rgba(0,230,118,.4);}}
.den-fill.Medium{{background:linear-gradient(90deg,#e6a800,var(--yellow));width:60%;box-shadow:0 0 8px rgba(255,214,0,.4);}}
.den-fill.High{{background:linear-gradient(90deg,var(--red),#ff6b35);width:95%;box-shadow:0 0 8px rgba(255,61,87,.5);}}
.sig-badge{{
    display:inline-flex;align-items:center;gap:8px;padding:8px 16px;
    border-radius:100px;font-size:.7rem;font-weight:800;
    letter-spacing:.12em;text-transform:uppercase;font-family:var(--fd);
}}
.sig-badge.green{{background:rgba(0,230,118,.1);border:1px solid rgba(0,230,118,.3);color:var(--green);box-shadow:0 0 16px rgba(0,230,118,.15);}}
.sig-badge.red{{background:rgba(255,61,87,.08);border:1px solid rgba(255,61,87,.2);color:var(--red);}}
.sdot{{width:8px;height:8px;border-radius:50%;}}
.sig-badge.green .sdot{{background:var(--green);box-shadow:0 0 10px var(--green);animation:pdot 1.2s ease-in-out infinite;}}
.sig-badge.red .sdot{{background:var(--red);opacity:.7;}}

/* CONGESTION SCORE */
.cong-row{{display:flex;gap:10px;margin-top:10px;}}
.cong-box{{
    flex:1;background:var(--panel);border:1px solid var(--border);
    border-radius:var(--r-sm);padding:10px 12px;text-align:center;
}}
.cong-num{{font-family:var(--fd);font-size:1.4rem;font-weight:800;}}
.cong-lbl{{font-size:.58rem;color:var(--text3);text-transform:uppercase;letter-spacing:.1em;margin-top:2px;}}
.conf-badge{{
    display:inline-flex;align-items:center;gap:6px;
    padding:5px 12px;border-radius:100px;margin-top:8px;
    background:rgba(180,77,255,.1);border:1px solid rgba(180,77,255,.25);
    font-size:.68rem;font-weight:700;color:var(--purple);letter-spacing:.06em;
    font-family:var(--fd);
}}

/* TOTALS BAR */
.tot-bar{{
    background:var(--card);border:1px solid var(--border);
    border-radius:var(--r-md);padding:14px 20px;
    display:flex;justify-content:space-around;align-items:center;
    margin-bottom:12px;position:relative;overflow:hidden;
}}
.tot-bar::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,var(--blue),var(--orange),transparent);opacity:.4;
}}
.tot-item{{text-align:center;}}
.tot-num{{font-family:var(--fd);font-size:1.5rem;font-weight:800;letter-spacing:-.02em;}}
.tot-lbl{{font-size:.58rem;color:var(--text3);text-transform:uppercase;letter-spacing:.12em;margin-top:2px;}}
.tot-div{{width:1px;height:36px;background:linear-gradient(180deg,transparent,var(--border),transparent);}}

/* FEED PANEL */
.feed-panel{{background:var(--card);border:1px solid var(--border);border-radius:var(--r-lg);overflow:hidden;box-shadow:var(--shadow);}}
.feed-bar{{
    background:{"rgba(11,13,24,.9)" if dm else "rgba(248,250,255,.9)"};
    border-bottom:1px solid var(--border);padding:11px 18px;
    display:flex;align-items:center;justify-content:space-between;
}}
.feed-title{{font-family:var(--fd);font-size:.68rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--text2);display:flex;align-items:center;gap:8px;}}
.live-dot{{width:7px;height:7px;background:var(--red);border-radius:50%;box-shadow:0 0 8px var(--red);animation:pdot 1s ease-in-out infinite;}}
.feed-badge{{font-size:.6rem;color:var(--text3);background:var(--border);padding:3px 10px;border-radius:100px;letter-spacing:.06em;}}

/* ALERT */
.alert-box{{
    background:linear-gradient(135deg,rgba(255,61,87,.12),rgba(255,100,50,.08));
    border:1px solid rgba(255,61,87,.35);border-left:4px solid var(--red);
    border-radius:var(--r-md);padding:14px 20px;margin-bottom:16px;
    display:flex;align-items:center;gap:14px;
    animation:alert-pulse 2s ease-in-out infinite;box-shadow:var(--glow-r);
}}
@keyframes alert-pulse{{
    0%,100%{{border-color:rgba(255,61,87,.35);box-shadow:var(--glow-r);}}
    50%{{border-color:rgba(255,61,87,.7);box-shadow:0 0 40px rgba(255,61,87,.4);}}
}}
.alert-icon{{font-size:1.4rem;}}
.alert-title{{font-family:var(--fd);font-size:.82rem;font-weight:700;color:var(--red);letter-spacing:.04em;}}
.alert-sub{{font-size:.7rem;color:rgba(255,100,100,.7);margin-top:2px;}}

/* SESSION SUMMARY */
.summary-card{{
    background:linear-gradient(135deg,rgba(0,212,255,.08),rgba(180,77,255,.06));
    border:1px solid rgba(0,212,255,.25);border-radius:var(--r-lg);
    padding:30px 36px;margin-bottom:20px;
}}
.summary-title{{font-family:var(--fd);font-size:1rem;font-weight:800;color:var(--blue);margin-bottom:20px;letter-spacing:.04em;}}
.summary-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}}
.summary-item{{text-align:center;padding:14px;background:{"rgba(0,0,0,.2)" if dm else "rgba(0,0,0,.04)"};border-radius:var(--r-sm);}}
.summary-num{{font-family:var(--fd);font-size:1.6rem;font-weight:800;}}
.summary-lbl{{font-size:.62rem;color:var(--text3);text-transform:uppercase;letter-spacing:.1em;margin-top:3px;}}

/* VEHICLE BREAKDOWN */
.vtype-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px;}}
.vtype-card{{
    background:var(--card);border:1px solid var(--border);
    border-radius:var(--r-sm);padding:10px 12px;text-align:center;
    transition:all .3s ease;
}}
.vtype-card:hover{{transform:translateY(-2px);box-shadow:var(--glow-b);}}
.vtype-icon{{font-size:1.2rem;display:block;margin-bottom:4px;}}
.vtype-num{{font-family:var(--fd);font-size:1.1rem;font-weight:800;color:var(--text1);}}
.vtype-name{{font-size:.6rem;color:var(--text3);text-transform:uppercase;letter-spacing:.08em;margin-top:2px;}}

/* OVERVIEW CARDS */
.ov-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px;}}
.ov-card{{background:var(--card);border:1px solid var(--border);border-radius:var(--r-md);padding:20px 22px;position:relative;overflow:hidden;transition:all .3s ease;}}
.ov-card:hover{{transform:translateY(-3px);box-shadow:var(--shadow);}}
.ov-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;}}
.ov-card:nth-child(1)::before{{background:linear-gradient(90deg,var(--blue),transparent);}}
.ov-card:nth-child(2)::before{{background:linear-gradient(90deg,var(--green),transparent);}}
.ov-card:nth-child(3)::before{{background:linear-gradient(90deg,var(--orange),transparent);}}
.ov-card:nth-child(4)::before{{background:linear-gradient(90deg,var(--purple),transparent);}}
.ov-icon{{font-size:1.6rem;margin-bottom:10px;display:block;}}
.ov-num{{font-family:var(--fd);font-size:2rem;font-weight:800;letter-spacing:-.03em;}}
.ov-lbl{{font-size:.65rem;color:var(--text3);text-transform:uppercase;letter-spacing:.1em;margin-top:3px;}}

/* ABOUT PAGE */
.about-hero{{
    background:linear-gradient(135deg,rgba(0,212,255,.06),rgba(255,152,0,.04));
    border:1px solid var(--border);border-radius:var(--r-xl);
    padding:50px 50px;margin-bottom:24px;position:relative;overflow:hidden;
}}
.about-hero::before{{
    content:'';position:absolute;top:-50px;right:-50px;
    width:300px;height:300px;border-radius:50%;
    background:radial-gradient(circle,rgba(0,212,255,.06),transparent 70%);
}}
.about-title{{font-family:var(--fd);font-size:2rem;font-weight:800;letter-spacing:-.03em;margin-bottom:10px;}}
.about-title span{{
    background:linear-gradient(135deg,var(--blue),var(--cyan));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.about-sub{{font-size:.9rem;color:var(--text2);line-height:1.8;max-width:600px;}}
.info-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:24px;}}
.info-card{{background:var(--card);border:1px solid var(--border);border-radius:var(--r-md);padding:22px 24px;}}
.info-card-title{{font-family:var(--fd);font-size:.72rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--text3);margin-bottom:14px;}}
.info-row{{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid var(--border);}}
.info-row:last-child{{border-bottom:none;}}
.info-key{{font-size:.72rem;color:var(--text2);}}
.info-val{{font-size:.72rem;color:var(--text1);font-weight:500;text-align:right;}}
.tech-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}}
.tech-pill{{background:var(--card);border:1px solid var(--border);border-radius:var(--r-sm);padding:12px 14px;text-align:center;transition:all .3s ease;}}
.tech-pill:hover{{border-color:var(--blue);box-shadow:var(--glow-b);transform:translateY(-2px);}}
.tech-icon{{font-size:1.4rem;display:block;margin-bottom:6px;}}
.tech-name{{font-family:var(--fd);font-size:.65rem;font-weight:700;color:var(--text2);letter-spacing:.08em;}}
.dev-card{{background:linear-gradient(135deg,rgba(0,212,255,.06),rgba(180,77,255,.04));border:1px solid rgba(0,212,255,.2);border-radius:var(--r-lg);padding:30px 36px;display:flex;align-items:center;gap:30px;}}
.dev-avatar{{width:70px;height:70px;border-radius:50%;background:linear-gradient(135deg,var(--blue),var(--purple));display:flex;align-items:center;justify-content:center;font-size:1.8rem;flex-shrink:0;box-shadow:0 0 24px rgba(0,212,255,.25);}}
.dev-name{{font-family:var(--fd);font-size:1.3rem;font-weight:800;letter-spacing:-.02em;margin-bottom:4px;}}
.dev-role{{font-size:.76rem;color:var(--text2);margin-bottom:10px;}}
.dev-tags{{display:flex;gap:8px;flex-wrap:wrap;}}
.dev-tag-pill{{font-size:.62rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;padding:4px 12px;border-radius:100px;background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.2);color:var(--blue);}}

/* DB PAGE */
.db-stat-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;}}
.db-stat-card{{background:var(--card);border:1px solid var(--border);border-radius:var(--r-md);padding:16px 18px;text-align:center;}}
.db-stat-num{{font-family:var(--fd);font-size:1.6rem;font-weight:800;}}
.db-stat-lbl{{font-size:.62rem;color:var(--text3);text-transform:uppercase;letter-spacing:.1em;margin-top:3px;}}

/* UPLOAD */
.upload-box{{background:var(--card);border:1.5px dashed var(--border);border-radius:var(--r-xl);padding:70px 40px;text-align:center;transition:border-color .3s ease;}}
.upload-box:hover{{border-color:var(--blue);}}
.up-icon{{font-size:2.8rem;margin-bottom:14px;opacity:.4;display:block;}}
.up-title{{font-family:var(--fd);font-size:1.1rem;font-weight:800;color:var(--text2);margin-bottom:8px;}}
.up-sub{{font-size:.78rem;color:var(--text3);line-height:1.8;}}
.up-sub b{{color:var(--blue);}}

/* TABLE */
.stDataFrame{{border:none !important;}}
.stDataFrame table{{font-family:var(--fb) !important;font-size:.76rem !important;border-collapse:collapse !important;width:100% !important;}}
.stDataFrame thead th{{background:var(--panel) !important;color:var(--text3) !important;font-size:.62rem !important;letter-spacing:.12em !important;text-transform:uppercase !important;font-weight:700 !important;padding:10px 14px !important;border-bottom:1px solid var(--border) !important;font-family:var(--fd) !important;}}
.stDataFrame tbody td{{padding:9px 14px !important;border-bottom:1px solid var(--border) !important;color:var(--text2) !important;}}
.stDataFrame tbody tr:hover td{{background:{"#141828" if dm else "#f0f4ff"} !important;}}

/* SIDEBAR */
[data-testid="stSidebar"]{{background:var(--panel) !important;border-right:1px solid var(--border) !important;}}
[data-testid="stSidebar"] .block-container{{padding:1.8rem 1.4rem !important;}}
.sb-brand{{text-align:center;padding:0 0 20px;border-bottom:1px solid var(--border);margin-bottom:20px;}}
.sb-brand-icon{{font-size:2rem;margin-bottom:6px;display:block;}}
.sb-brand-name{{font-family:var(--fd);font-size:1rem;font-weight:800;background:linear-gradient(135deg,var(--blue),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}}
.sb-brand-sub{{font-size:.62rem;color:var(--text3);letter-spacing:.08em;margin-top:3px;}}
.sb-sec{{font-family:var(--fd);font-size:.58rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--text3);margin:18px 0 8px;display:flex;align-items:center;gap:6px;}}
.sb-sec::after{{content:'';flex:1;height:1px;background:var(--border);}}
.sb-info{{background:{"rgba(0,0,0,.2)" if dm else "rgba(0,0,0,.03)"};border:1px solid var(--border);border-radius:var(--r-sm);padding:12px 14px;margin-top:14px;font-size:.62rem;color:var(--text3);line-height:1.9;font-family:var(--fb);}}
.sb-info b{{color:var(--text2);}}
.sb-dev{{margin-top:12px;text-align:center;font-size:.6rem;color:var(--text3);font-family:'Courier New',monospace;letter-spacing:.06em;line-height:1.8;}}

/* WIDGETS */
.stSlider>label,.stNumberInput>label,.stCheckbox>label,.stSelectbox>label{{font-family:var(--fb) !important;font-size:.76rem !important;color:var(--text2) !important;}}
.stButton button{{font-family:var(--fd) !important;font-weight:700 !important;letter-spacing:.08em !important;border-radius:var(--r-sm) !important;transition:all .25s ease !important;}}
.stButton button[kind="primary"]{{background:linear-gradient(135deg,#00b8d9,var(--blue)) !important;color:#000 !important;border:none !important;box-shadow:0 4px 15px rgba(0,212,255,.25) !important;}}
.stButton button[kind="primary"]:hover{{background:linear-gradient(135deg,var(--blue),var(--cyan)) !important;box-shadow:0 6px 25px rgba(0,212,255,.4) !important;transform:translateY(-1px) !important;}}
.stTextInput input{{background:var(--card) !important;border:1px solid var(--border) !important;border-radius:var(--r-sm) !important;color:var(--text1) !important;font-family:var(--fb) !important;}}
.stTextInput input:focus{{border-color:var(--blue) !important;box-shadow:0 0 0 2px rgba(0,212,255,.15) !important;}}

/* FOOTER */
.site-footer{{border-top:1px solid var(--border);margin-top:3rem;padding:20px 0 8px;display:flex;justify-content:space-between;align-items:flex-end;position:relative;}}
.site-footer::before{{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(0,212,255,.3),rgba(255,152,0,.2),transparent);}}
.footer-l{{font-size:.66rem;color:var(--text3);line-height:1.8;}}
.footer-l b{{color:var(--text2);}}
.footer-r{{font-size:.64rem;color:var(--text3);text-align:right;line-height:1.8;font-style:italic;}}
.footer-r span{{color:var(--blue);font-style:normal;font-weight:500;}}

/* MOBILE RESPONSIVE */
@media (max-width: 768px) {{
    .site-header{{padding:14px 16px;flex-direction:column;gap:12px;align-items:flex-start;}}
    .header-title{{font-size:1.2rem;}}
    .header-right{{align-items:flex-start;}}
    .block-container{{padding:0 1rem 3rem !important;}}
    .stat-grid{{grid-template-columns:1fr;}}
    .ov-grid{{grid-template-columns:repeat(2,1fr);}}
    .info-grid{{grid-template-columns:1fr;}}
    .tech-grid{{grid-template-columns:repeat(2,1fr);}}
    .summary-grid{{grid-template-columns:1fr;}}
    .vtype-grid{{grid-template-columns:repeat(2,1fr);}}
    .dev-card{{flex-direction:column;text-align:center;}}
    .db-stat-row{{grid-template-columns:repeat(2,1fr);}}
    .nav-wrap{{padding:0 16px;}}
}}
</style>
""", unsafe_allow_html=True)

inject_css()


# ════════════════════════════════════════════════════════════════════
# PART 7 — LOGIN PAGE
# ════════════════════════════════════════════════════════════════════

def show_login():
    st.markdown("""
    <style>
    [data-testid="stSidebar"]{visibility:hidden;width:0px !important;min-width:0px !important;}
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div style="text-align:center; padding: 40px 20px 10px;">
        <div style="font-size:3rem; margin-bottom:12px;">🚦</div>
        <div style="font-family:'Syne',sans-serif; font-size:0.65rem; font-weight:700;
                    letter-spacing:0.2em; text-transform:uppercase;
                    background:linear-gradient(90deg,#00d4ff,#00fff0);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; margin-bottom:8px;">
            K.R. Mangalam University &nbsp;·&nbsp; School of Engineering & Technology
        </div>
        <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800;
                    color:#eef0f8; letter-spacing:-0.02em; margin-bottom:6px;">
            IoT <span style="color:#00d4ff;">Intelligent Traffic</span>
            <span style="color:#ff9800;"> Monitoring</span> System
        </div>
        <div style="font-size:0.78rem; color:#8890aa; margin-bottom:30px;">
            Congestion Prediction &nbsp;·&nbsp; Lane Analysis &nbsp;·&nbsp; YOLOv8 Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Login card centered
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("""
        <div style="background:#0f1220; border:1px solid #1a1f35;
                    border-radius:20px; padding:32px 28px;
                    box-shadow:0 8px 40px rgba(0,0,0,0.5);">
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem;
                        font-weight:800; color:#eef0f8; margin-bottom:4px;">
                Welcome Back
            </div>
            <div style="font-size:0.75rem; color:#444d6a; margin-bottom:24px;">
                Sign in to access the dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", placeholder="Enter password",
                                     type="password")
            submitted = st.form_submit_button("Sign In →",
                                              use_container_width=True,
                                              type="primary")
            if submitted:
                role = check_login(username.strip(), password)
                if role:
                    st.session_state.logged_in = True
                    st.session_state.username  = username.strip()
                    st.session_state.role      = role
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")

        st.markdown("""
        <div style="margin-top:16px; padding:12px 16px;
                    background:rgba(0,212,255,0.06);
                    border:1px solid rgba(0,212,255,0.15);
                    border-radius:10px; font-size:0.7rem;
                    color:#8890aa; line-height:1.9; text-align:center;">
            <b style="color:#00d4ff;">Admin</b> &nbsp;→&nbsp; admin / krmu2025
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <b style="color:#00d4ff;">Viewer</b> &nbsp;→&nbsp; sarthak / pass123
        </div>
        """, unsafe_allow_html=True)

    # Features row below
    st.markdown("""
    <div style="display:flex; justify-content:center; gap:24px;
                margin-top:30px; flex-wrap:wrap; padding:0 20px 40px;">
        <div style="display:flex; align-items:center; gap:8px;
                    font-size:0.72rem; color:#8890aa;">
            <div style="width:8px; height:8px; border-radius:50%;
                        background:#00d4ff; box-shadow:0 0 8px #00d4ff;"></div>
            YOLOv8 Neural Detection
        </div>
        <div style="display:flex; align-items:center; gap:8px;
                    font-size:0.72rem; color:#8890aa;">
            <div style="width:8px; height:8px; border-radius:50%;
                        background:#ff9800; box-shadow:0 0 8px #ff9800;"></div>
            Real-Time Lane Analysis
        </div>
        <div style="display:flex; align-items:center; gap:8px;
                    font-size:0.72rem; color:#8890aa;">
            <div style="width:8px; height:8px; border-radius:50%;
                        background:#00e676; box-shadow:0 0 8px #00e676;"></div>
            SQLite IoT Database
        </div>
        <div style="display:flex; align-items:center; gap:8px;
                    font-size:0.72rem; color:#8890aa;">
            <div style="width:8px; height:8px; border-radius:50%;
                        background:#b44dff; box-shadow:0 0 8px #b44dff;"></div>
            Live Analytics Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

if not st.session_state.logged_in:
    show_login()
    st.stop()

# Force sidebar visible after login
st.markdown("""
<style>
[data-testid="stSidebar"] { display: block !important; }
[data-testid="collapsedControl"] { display: block !important; }
section[data-testid="stSidebar"] { width: 300px !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PART 8 — HEADER (shown after login)
# ════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="site-header">
    <div class="header-left">
        <div class="univ-badge">K.R. Mangalam University &nbsp;·&nbsp; School of Engineering & Technology</div>
        <div class="header-title">
            IoT <span class="hl-b">Intelligent Traffic</span>
            <span class="hl-o"> Monitoring</span> System
        </div>
        <div class="header-sub">
            Congestion Prediction &nbsp;·&nbsp; Real-Time Lane Analysis &nbsp;·&nbsp; YOLOv8 Neural Detection
        </div>
    </div>
    <div class="header-right">
        <div class="status-pill"><div class="status-dot"></div>System Active</div>
        <div class="dept-tag">
            👤 {st.session_state.username} ({st.session_state.role})
            &nbsp;·&nbsp; B.Tech CSE &nbsp;·&nbsp; 2025–26
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PART 9 — NAV TABS
# ════════════════════════════════════════════════════════════════════

tabs_list = ["🎯  Live Monitor", "📊  Analytics", "🗄️  Database", "ℹ️  About"]
st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
tab_cols = st.columns(len(tabs_list))
for i, tab in enumerate(tabs_list):
    with tab_cols[i]:
        is_active = st.session_state.active_tab == tab
        if st.button(tab, key=f"tab_{i}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.active_tab = tab
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
active = st.session_state.active_tab


# ════════════════════════════════════════════════════════════════════
# PART 10 — SIDEBAR
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <span class="sb-brand-icon">🚦</span>
        <div class="sb-brand-name">TrafficVision</div>
        <div class="sb-brand-sub">IoT Intelligence Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Input Mode</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Select input source",
        ["📁 Upload Video", "📸 Camera Capture"],
        label_visibility="collapsed"
    )

    default_video  = "traffic.mp4"
    uploaded_file  = None
    camera_photo   = None

    if input_mode == "📁 Upload Video":
        uploaded_file = st.file_uploader(
            "Upload Traffic Video", type=["mp4","avi","mov"])
        if os.path.exists(default_video) and uploaded_file is None:
            st.success("✅ traffic.mp4 detected")
        video_ready = uploaded_file is not None or os.path.exists(default_video)
        if not video_ready:
            st.warning("Upload a video to begin")

    elif input_mode == "📸 Camera Capture":
        st.info("📱 Works on phone + deployed!\nPoint camera at traffic and capture.")
        camera_photo = st.camera_input("Take a photo")
        video_ready = camera_photo is not None

    st.markdown('<div class="sb-sec">Detection</div>', unsafe_allow_html=True)
    conf_val    = st.slider("Confidence", 0.1, 0.9, 0.4, 0.05)
    pub_every   = st.slider("DB save every N frames", 1, 30, 10)
    skip_frames = st.slider("Frame skip (speed)", 1, 5, 2)

    st.markdown('<div class="sb-sec">Controls</div>', unsafe_allow_html=True)
    if input_mode == "📸 Camera Capture":
        st.info("📸 No start needed — just take a photo above")
        start = False
        stop  = st.button("🔄 Reset Results", use_container_width=True)
    else:
        start = st.button("▶  Start Analysis", use_container_width=True,
                          type="primary", disabled=not video_ready)
        stop  = st.button("⏹  Stop",           use_container_width=True)

    st.markdown('<div class="sb-sec">Appearance</div>', unsafe_allow_html=True)
    dark_toggle = st.toggle("Dark Mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

    st.markdown('<div class="sb-sec">Session</div>', unsafe_allow_html=True)
    if st.button("🚪 Logout", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown("""
    <div class="sb-info">
        <b>Stack</b><br>
        Python · YOLOv8 · OpenCV<br>
        Streamlit · SQLite · MQTT<br><br>
        <b>Model</b><br>
        YOLOv8n · COCO Dataset<br><br>
        <b>Database</b><br>
        SQLite · traffic_data.db
    </div>
    <div class="sb-dev">
        Developed by<br>
        <span style="color:#00d4ff;font-style:normal;font-size:.68rem;font-weight:600;">
            Sarthak Mishra
        </span><br>
        B.Tech CSE · KRM University<br>
        <span style="color:#222840;">uid · 2301010232</span>
    </div>
    """, unsafe_allow_html=True)

# Controls
if start:
    st.session_state.running      = True
    st.session_state.show_summary = False
    st.session_state.session_start = datetime.now().strftime("%H:%M:%S")
    st.session_state.session_high_events = 0
    st.session_state.session_total_vehicles = 0

if stop and st.session_state.running:
    st.session_state.running      = False
    st.session_state.show_summary = True
    st.session_state.l1_hist.clear()
    st.session_state.l2_hist.clear()


# ════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE MONITOR
# ════════════════════════════════════════════════════════════════════

if active == "🎯  Live Monitor":

    alert_ph = st.empty()

    # Session summary
    if st.session_state.show_summary and st.session_state.total_frames > 0:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-title">📋 Session Summary</div>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-num" style="color:var(--blue)">
                        {st.session_state.total_frames}
                    </div>
                    <div class="summary-lbl">Frames Processed</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num" style="color:var(--green)">
                        {st.session_state.session_total_vehicles}
                    </div>
                    <div class="summary-lbl">Vehicles Detected</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num" style="color:var(--red)">
                        {st.session_state.session_high_events}
                    </div>
                    <div class="summary-lbl">High Density Events</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num" style="color:var(--purple)">
                        {get_total_records()}
                    </div>
                    <div class="summary-lbl">DB Records Saved</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num" style="font-size:1rem;color:var(--text2)">
                        {st.session_state.session_start or "—"}
                    </div>
                    <div class="summary-lbl">Session Started</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num" style="font-size:1rem;color:var(--text2)">
                        {datetime.now().strftime("%H:%M:%S")}
                    </div>
                    <div class="summary-lbl">Session Ended</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    feed_col, stat_col = st.columns([3, 2], gap="large")

    with feed_col:
        st.markdown('<div class="sec-label">Live Detection Feed</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="feed-panel">
            <div class="feed-bar">
                <div class="feed-title"><div class="live-dot"></div>LIVE PROCESSING</div>
                <div class="feed-badge">YOLOv8n · COCO · 80 classes</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        frame_ph = st.empty()

        # Snapshot button
        snap_col1, snap_col2 = st.columns([1, 3])
        with snap_col1:
            snap_btn = st.button("📸 Snapshot", use_container_width=True)
        snap_msg = st.empty()

    with stat_col:
        st.markdown('<div class="sec-label">Lane Intelligence</div>',
                    unsafe_allow_html=True)
        stat_ph   = st.empty()
        totals_ph = st.empty()

        st.markdown('<div class="sec-label">Vehicle Breakdown</div>',
                    unsafe_allow_html=True)
        vtype_ph = st.empty()

    st.markdown("---")
    st.markdown('<div class="sec-label">Quick Charts</div>',
                unsafe_allow_html=True)
    qc1, qc2 = st.columns(2, gap="medium")
    with qc1:
        st.caption("Vehicle Count — Last 40 Frames")
        qchart1_ph = st.empty()
    with qc2:
        st.caption("Density Level — Last 40 Frames")
        qchart2_ph = st.empty()

    # ── helpers ──
    def show_alert(l1d, l2d):
        if l1d == "High" or l2d == "High":
            lanes = " & ".join(
                [l for l, d in [("Lane 1", l1d), ("Lane 2", l2d)] if d == "High"])
            alert_ph.markdown(f"""
            <div class="alert-box">
                <div class="alert-icon">🚨</div>
                <div>
                    <div class="alert-title">HIGH TRAFFIC ALERT — {lanes}</div>
                    <div class="alert-sub">
                        Critical density · Automatic signal adjustment active
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
            st.session_state.session_high_events += 1
        else:
            alert_ph.empty()

    def show_stats(l1c, l2c, l1d, l2d, l1s, l2s, avg_conf):
        s1  = "green" if "GREEN" in l1s else "red"
        s2  = "green" if "GREEN" in l2s else "red"
        s1t = "🟢 GREEN" if "GREEN" in l1s else "🔴 RED"
        s2t = "🟢 GREEN" if "GREEN" in l2s else "🔴 RED"
        cs1 = get_congestion_score(l1c, l1d)
        cs2 = get_congestion_score(l2c, l2d)
        c1col = "var(--green)" if cs1 < 40 else "var(--yellow)" if cs1 < 70 else "var(--red)"
        c2col = "var(--green)" if cs2 < 40 else "var(--yellow)" if cs2 < 70 else "var(--red)"

        stat_ph.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card l1">
                <div class="lane-label c1"><div class="ldot"></div>Lane 1 · Left</div>
                <div class="count-num" style="color:var(--blue)">{l1c}</div>
                <div class="count-sub">vehicles detected</div>
                <div class="den-wrap">
                    <div class="den-row">
                        <span class="den-key">Density</span>
                        <span class="den-val {l1d}">{l1d}</span>
                    </div>
                    <div class="den-bar"><div class="den-fill {l1d}"></div></div>
                </div>
                <div class="sig-badge {s1}"><div class="sdot"></div>{s1t}</div>
                <div class="cong-row">
                    <div class="cong-box">
                        <div class="cong-num" style="color:{c1col}">{cs1}</div>
                        <div class="cong-lbl">Congestion /100</div>
                    </div>
                </div>
            </div>
            <div class="stat-card l2">
                <div class="lane-label c2"><div class="ldot"></div>Lane 2 · Right</div>
                <div class="count-num" style="color:var(--orange)">{l2c}</div>
                <div class="count-sub">vehicles detected</div>
                <div class="den-wrap">
                    <div class="den-row">
                        <span class="den-key">Density</span>
                        <span class="den-val {l2d}">{l2d}</span>
                    </div>
                    <div class="den-bar"><div class="den-fill {l2d}"></div></div>
                </div>
                <div class="sig-badge {s2}"><div class="sdot"></div>{s2t}</div>
                <div class="cong-row">
                    <div class="cong-box">
                        <div class="cong-num" style="color:{c2col}">{cs2}</div>
                        <div class="cong-lbl">Congestion /100</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="conf-badge">🎯 Avg Confidence: {avg_conf}%</div>
        """, unsafe_allow_html=True)

        totals_ph.markdown(f"""
        <div class="tot-bar">
            <div class="tot-item">
                <div class="tot-num" style="color:var(--blue)">{l1c+l2c}</div>
                <div class="tot-lbl">Total Vehicles</div>
            </div>
            <div class="tot-div"></div>
            <div class="tot-item">
                <div class="tot-num" style="color:var(--purple)">
                    {st.session_state.total_frames}
                </div>
                <div class="tot-lbl">Frames</div>
            </div>
            <div class="tot-div"></div>
            <div class="tot-item">
                <div class="tot-num" style="font-size:.85rem;color:var(--text2)">
                    {datetime.now().strftime("%H:%M:%S")}
                </div>
                <div class="tot-lbl">Time</div>
            </div>
            <div class="tot-div"></div>
            <div class="tot-item">
                <div class="tot-num" style="color:var(--green)">{get_total_records()}</div>
                <div class="tot-lbl">DB Records</div>
            </div>
        </div>""", unsafe_allow_html=True)

    def show_vtypes(vtypes):
        icons = {"Car": "🚗", "Motorcycle": "🏍️", "Bus": "🚌", "Truck": "🚛"}
        html = '<div class="vtype-grid">'
        for vt, icon in icons.items():
            html += f"""
            <div class="vtype-card">
                <span class="vtype-icon">{icon}</span>
                <div class="vtype-num">{vtypes.get(vt, 0)}</div>
                <div class="vtype-name">{vt}</div>
            </div>"""
        html += '</div>'
        vtype_ph.markdown(html, unsafe_allow_html=True)

    def show_quick_charts():
        if len(st.session_state.l1_hist) < 2:
            return
        df_c = pd.DataFrame({
            "Frame": list(range(len(st.session_state.l1_hist))),
            "Lane 1": list(st.session_state.l1_hist),
            "Lane 2": list(st.session_state.l2_hist),
        })
        qchart1_ph.line_chart(df_c.set_index("Frame"),
                              color=["#00d4ff","#ff9800"],
                              height=160, use_container_width=True)
        df_d = pd.DataFrame({
            "Frame": list(range(len(st.session_state.d1_hist))),
            "Lane 1": [DENSITY_NUM[d] for d in st.session_state.d1_hist],
            "Lane 2": [DENSITY_NUM[d] for d in st.session_state.d2_hist],
        })
        qchart2_ph.line_chart(df_d.set_index("Frame"),
                              color=["#00e676","#ffd600"],
                              height=160, use_container_width=True)

    # ── helper: process one frame ──
    def process_one_frame(frame, fn):
        frame = cv2.resize(frame, (960, 540))
        (ann, l1c, l2c, l1d, l2d,
         l1s, l2s, avg_conf, vtypes) = detect_frame(frame, conf_val)
        st.session_state.last_frame  = ann.copy()
        st.session_state.last_conf   = avg_conf
        st.session_state.last_vtypes = vtypes
        st.session_state.session_total_vehicles += (l1c + l2c)
        st.session_state.total_frames += 1
        frame_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                       channels="RGB", use_container_width=True)
        if snap_btn and st.session_state.last_frame is not None:
            fname = f"snapshot_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(fname, st.session_state.last_frame)
            snap_msg.success(f"📸 Saved: {fname}")
        st.session_state.l1_hist.append(l1c)
        st.session_state.l2_hist.append(l2c)
        st.session_state.d1_hist.append(l1d)
        st.session_state.d2_hist.append(l2d)
        st.session_state.last_l1 = l1c
        st.session_state.last_l2 = l2c
        st.session_state.last_d1 = l1d
        st.session_state.last_d2 = l2d
        st.session_state.last_s1 = l1s
        st.session_state.last_s2 = l2s
        show_alert(l1d, l2d)
        show_stats(l1c, l2c, l1d, l2d, l1s, l2s, avg_conf)
        show_vtypes(vtypes)
        show_quick_charts()
        if fn % pub_every == 0:
            publish_iot(l1c, l2c, l1d, l2d, l1s, l2s)
        return l1c, l2c

    # ════════════════════════
    # MODE 1 — VIDEO UPLOAD
    # ════════════════════════
    if st.session_state.running and input_mode == "📁 Upload Video":

        if uploaded_file is not None:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tf.write(uploaded_file.read())
            vpath = tf.name
        else:
            vpath = default_video
        cap = cv2.VideoCapture(vpath)

        if not cap.isOpened():
            st.error("❌ Cannot open video. Please re-upload.")
            st.session_state.running = False
            st.stop()

        fn = 0
        try:
            while st.session_state.running:
                for _ in range(skip_frames - 1):
                    cap.read()
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                fn += 1
                process_one_frame(frame, fn)
                time.sleep(0.03)
        finally:
            cap.release()

    # ═══════════════════════════
    # MODE 2 — CAMERA CAPTURE
    # (phone + deployed friendly)
    # ═══════════════════════════
    elif input_mode == "📸 Camera Capture":
        if camera_photo is not None:
            import numpy as np
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(camera_photo.getvalue()))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            process_one_frame(frame, 1)
            st.success("✅ Frame analysed! Take another photo to update.")
        else:
            frame_ph.markdown("""
            <div class="upload-box">
                <span class="up-icon">📸</span>
                <div class="up-title">Camera Capture Mode</div>
                <div class="up-sub">
                    Click the camera button above to take a photo<br>
                    Works on <b>phone browser</b> and deployed link<br>
                    Point your camera at a traffic scene
                </div>
            </div>""", unsafe_allow_html=True)
            show_stats(
                st.session_state.last_l1, st.session_state.last_l2,
                st.session_state.last_d1, st.session_state.last_d2,
                st.session_state.last_s1, st.session_state.last_s2,
                st.session_state.last_conf
            )
            show_vtypes(st.session_state.last_vtypes)
            show_quick_charts()

    else:
        # Idle state
        if not video_ready:
            frame_ph.markdown("""
            <div class="upload-box">
                <span class="up-icon">🎬</span>
                <div class="up-title">Select an Input Mode</div>
                <div class="up-sub">
                    Choose <b>Upload Video</b>
                    or <b>Camera Capture</b> in the sidebar
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            frame_ph.markdown("""
            <div class="upload-box">
                <span class="up-icon">✅</span>
                <div class="up-title">Ready — Click Start Analysis</div>
                <div class="up-sub">
                    Press <b>▶ Start Analysis</b> in the sidebar
                </div>
            </div>""", unsafe_allow_html=True)

        show_stats(
            st.session_state.last_l1, st.session_state.last_l2,
            st.session_state.last_d1, st.session_state.last_d2,
            st.session_state.last_s1, st.session_state.last_s2,
            st.session_state.last_conf
        )
        show_vtypes(st.session_state.last_vtypes)
        show_quick_charts()


# ════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ════════════════════════════════════════════════════════════════════

elif active == "📊  Analytics":
    st.markdown('<div class="sec-label">Traffic Analytics Overview</div>',
                unsafe_allow_html=True)
    df_all = get_db_stats()

    if df_all.empty:
        st.info("No data yet. Run Live Monitor first to collect data.")
    else:
        total  = len(df_all)
        avg_l1 = round(df_all["lane1_count"].mean(), 1)
        avg_l2 = round(df_all["lane2_count"].mean(), 1)
        high_c = len(df_all[
            (df_all["lane1_density"] == "High") |
            (df_all["lane2_density"] == "High")
        ])

        st.markdown(f"""
        <div class="ov-grid">
            <div class="ov-card">
                <span class="ov-icon">🗄️</span>
                <div class="ov-num" style="color:var(--blue)">{total}</div>
                <div class="ov-lbl">Total Records</div>
            </div>
            <div class="ov-card">
                <span class="ov-icon">🚗</span>
                <div class="ov-num" style="color:var(--green)">{avg_l1}</div>
                <div class="ov-lbl">Avg Lane 1</div>
            </div>
            <div class="ov-card">
                <span class="ov-icon">🚙</span>
                <div class="ov-num" style="color:var(--orange)">{avg_l2}</div>
                <div class="ov-lbl">Avg Lane 2</div>
            </div>
            <div class="ov-card">
                <span class="ov-icon">🚨</span>
                <div class="ov-num" style="color:var(--red)">{high_c}</div>
                <div class="ov-lbl">High Density Events</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Vehicle count over time
        st.markdown('<div class="sec-label">Vehicle Count Over Time</div>',
                    unsafe_allow_html=True)
        df_plot = df_all[["id","lane1_count","lane2_count"]].set_index("id")
        df_plot.columns = ["Lane 1","Lane 2"]
        st.line_chart(df_plot, color=["#00d4ff","#ff9800"],
                      height=220, use_container_width=True)

        # Density distribution
        st.markdown('<div class="sec-label">Density Distribution</div>',
                    unsafe_allow_html=True)
        dc1, dc2 = st.columns(2, gap="medium")
        with dc1:
            st.caption("Lane 1 Density Breakdown")
            st.bar_chart(df_all["lane1_density"].value_counts(),
                         color="#00d4ff", height=200, use_container_width=True)
        with dc2:
            st.caption("Lane 2 Density Breakdown")
            st.bar_chart(df_all["lane2_density"].value_counts(),
                         color="#ff9800", height=200, use_container_width=True)

        # Peak hour analysis
        st.markdown('<div class="sec-label">Peak Hour Analysis</div>',
                    unsafe_allow_html=True)
        if "timestamp" in df_all.columns and not df_all["timestamp"].isnull().all():
            df_all["hour"] = pd.to_datetime(
                df_all["timestamp"], errors="coerce"
            ).dt.hour
            df_hour = df_all.groupby("hour")[["lane1_count","lane2_count"]].mean().round(1)
            df_hour.columns = ["Lane 1 Avg","Lane 2 Avg"]
            if not df_hour.empty:
                st.bar_chart(df_hour, color=["#00d4ff","#ff9800"],
                             height=220, use_container_width=True)
                peak_hour = df_hour["Lane 1 Avg"].add(df_hour["Lane 2 Avg"]).idxmax()
                st.caption(f"🔴 Peak traffic hour: **{peak_hour}:00 – {peak_hour+1}:00**")
        else:
            st.info("Timestamp data needed for peak hour chart. Run analysis first.")

        # Signal decision history
        st.markdown('<div class="sec-label">Signal Decision History</div>',
                    unsafe_allow_html=True)
        sig_counts = df_all["lane1_signal"].value_counts().reset_index()
        sig_counts.columns = ["Signal","Count"]
        sc1, sc2 = st.columns([1,2], gap="medium")
        with sc1:
            st.dataframe(sig_counts, use_container_width=True, hide_index=True)
        with sc2:
            st.bar_chart(sig_counts.set_index("Signal"),
                         color="#b44dff", height=180, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 3 — DATABASE
# ════════════════════════════════════════════════════════════════════

elif active == "🗄️  Database":
    st.markdown('<div class="sec-label">SQLite Database — traffic_data.db</div>',
                unsafe_allow_html=True)

    df = read_db(limit=100)

    if df.empty:
        st.info("Database is empty. Start Live Monitor to populate records.")
    else:
        total  = get_total_records()
        avg_l1 = round(df["lane1_count"].mean(), 1)
        avg_l2 = round(df["lane2_count"].mean(), 1)
        latest = df.iloc[0]["timestamp"] if not df.empty else "—"

        st.markdown(f"""
        <div class="db-stat-row">
            <div class="db-stat-card">
                <div class="db-stat-num" style="color:var(--blue)">{total}</div>
                <div class="db-stat-lbl">Total Records</div>
            </div>
            <div class="db-stat-card">
                <div class="db-stat-num" style="color:var(--green)">{avg_l1}</div>
                <div class="db-stat-lbl">Avg Lane 1</div>
            </div>
            <div class="db-stat-card">
                <div class="db-stat-num" style="color:var(--orange)">{avg_l2}</div>
                <div class="db-stat-lbl">Avg Lane 2</div>
            </div>
            <div class="db-stat-card">
                <div class="db-stat-num"
                     style="font-size:1rem;color:var(--text2);margin-top:6px">
                    {latest[:16]}
                </div>
                <div class="db-stat-lbl">Latest Record</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-label">Records</div>',
                    unsafe_allow_html=True)
        limit_opt = st.selectbox("Show records", [10, 25, 50, 100], index=0)
        df_show = read_db(limit=limit_opt)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-label">Export</div>',
                    unsafe_allow_html=True)
        csv = df_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download as CSV",
            data=csv,
            file_name=f"traffic_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        st.caption("💡 Open traffic_data.db with DB Browser for SQLite → sqlitebrowser.org")

        # Users table (admin only)
        if st.session_state.role == "Admin":
            st.markdown('<div class="sec-label" style="margin-top:20px">Users Table (Admin Only)</div>',
                        unsafe_allow_html=True)
            conn = sqlite3.connect(DB_FILE)
            users_df = pd.read_sql(
                "SELECT id, username, role FROM users", conn)
            conn.close()
            st.dataframe(users_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════════════════════════════════

elif active == "ℹ️  About":

    st.markdown("""
    <div class="about-hero">
        <div class="about-title">
            IoT <span>Intelligent Traffic</span> Monitoring System
        </div>
        <div class="about-sub">
            A real-time intelligent traffic monitoring and congestion prediction
            system built using YOLOv8 neural detection, IoT simulation, SQLite
            database and secure login authentication. Designed as a pre-final
            year B.Tech project at K.R. Mangalam University.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Project Information</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-grid">
        <div class="info-card">
            <div class="info-card-title">📋 Project Details</div>
            <div class="info-row">
                <span class="info-key">Title</span>
                <span class="info-val">IoT Intelligent Traffic Monitoring</span>
            </div>
            <div class="info-row">
                <span class="info-key">Type</span>
                <span class="info-val">IoT Based</span>
            </div>
            <div class="info-row">
                <span class="info-key">Year</span>
                <span class="info-val">2025 – 2026</span>
            </div>
            <div class="info-row">
                <span class="info-key">Semester</span>
                <span class="info-val">Pre-Final Year</span>
            </div>
            <div class="info-row">
                <span class="info-key">Status</span>
                <span class="info-val" style="color:#00e676">✅ Approved</span>
            </div>
        </div>
        <div class="info-card">
            <div class="info-card-title">🎓 Academic Details</div>
            <div class="info-row">
                <span class="info-key">University</span>
                <span class="info-val">K.R. Mangalam University</span>
            </div>
            <div class="info-row">
                <span class="info-key">School</span>
                <span class="info-val">Engineering & Technology</span>
            </div>
            <div class="info-row">
                <span class="info-key">Program</span>
                <span class="info-val">B.Tech CSE</span>
            </div>
            <div class="info-row">
                <span class="info-key">Problem Type</span>
                <span class="info-val">IoT Based</span>
            </div>
            <div class="info-row">
                <span class="info-key">Tech Stack</span>
                <span class="info-val">Python</span>
            </div>
        </div>
        <div class="info-card">
            <div class="info-card-title">🔧 System Specs</div>
            <div class="info-row">
                <span class="info-key">AI Model</span>
                <span class="info-val">YOLOv8 Nano</span>
            </div>
            <div class="info-row">
                <span class="info-key">Dataset</span>
                <span class="info-val">COCO (80 classes)</span>
            </div>
            <div class="info-row">
                <span class="info-key">Database</span>
                <span class="info-val">SQLite (2 tables)</span>
            </div>
            <div class="info-row">
                <span class="info-key">Auth</span>
                <span class="info-val">SHA-256 Hashed</span>
            </div>
            <div class="info-row">
                <span class="info-key">Deployment</span>
                <span class="info-val">Streamlit Cloud</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Technology Stack</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="tech-grid">
        <div class="tech-pill"><span class="tech-icon">🐍</span><div class="tech-name">Python 3.11</div></div>
        <div class="tech-pill"><span class="tech-icon">🤖</span><div class="tech-name">YOLOv8</div></div>
        <div class="tech-pill"><span class="tech-icon">📷</span><div class="tech-name">OpenCV</div></div>
        <div class="tech-pill"><span class="tech-icon">🌐</span><div class="tech-name">Streamlit</div></div>
        <div class="tech-pill"><span class="tech-icon">🗄️</span><div class="tech-name">SQLite</div></div>
        <div class="tech-pill"><span class="tech-icon">📡</span><div class="tech-name">MQTT</div></div>
        <div class="tech-pill"><span class="tech-icon">🐼</span><div class="tech-name">Pandas</div></div>
        <div class="tech-pill"><span class="tech-icon">☁️</span><div class="tech-name">Streamlit Cloud</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label" style="margin-top:24px">Developer</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="dev-card">
        <div class="dev-avatar">👨‍💻</div>
        <div>
            <div class="dev-name">Sarthak Mishra</div>
            <div class="dev-role">
                B.Tech Computer Science & Engineering
                · K.R. Mangalam University
            </div>
            <div class="dev-tags">
                <span class="dev-tag-pill">IoT Systems</span>
                <span class="dev-tag-pill">Computer Vision</span>
                <span class="dev-tag-pill">Python</span>
                <span class="dev-tag-pill">Machine Learning</span>
                <span class="dev-tag-pill">SQLite</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="site-footer">
    <div class="footer-l">
        <b>IoT-Based Intelligent Traffic Monitoring & Congestion Prediction System</b><br>
        K.R. Mangalam University &nbsp;·&nbsp; School of Engineering & Technology
        &nbsp;·&nbsp; B.Tech CSE &nbsp;·&nbsp; 2025–26
    </div>
    <div class="footer-r">
        Designed & Developed by<br>
        <span>Sarthak Mishra</span> &nbsp;·&nbsp; B.Tech CSE<br>
        K.R. Mangalam University
    </div>
</div>
""", unsafe_allow_html=True)
