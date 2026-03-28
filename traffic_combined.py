"""
IoT-Based Intelligent Traffic Monitoring & Congestion Prediction System
K.R. Mangalam University | Department of Computer Science & Engineering
Developer: Sarthak Mishra | Roll No: 2301010232 | B.Tech CSE | 2025-26
v2.0 — Multi-Lane | Dynamic Signal Timer | 3D UI
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import cv2
import os
import time
import sqlite3
import hashlib
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from PIL import Image
import io

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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS traffic_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT,
            lane1_count   INTEGER,
            lane2_count   INTEGER,
            lane3_count   INTEGER,
            total_count   INTEGER,
            overall_density TEXT,
            signal_verdict  TEXT,
            signal_duration INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role     TEXT
        )
    """)
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

def save_to_db(l1c, l2c, l3c, total, density, verdict, duration):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO traffic_log
        (timestamp, lane1_count, lane2_count, lane3_count,
         total_count, overall_density, signal_verdict, signal_duration)
        VALUES (?,?,?,?,?,?,?,?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          l1c, l2c, l3c, total, density, verdict, duration))
    conn.commit()
    conn.close()

def read_db(limit=50):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(
        f"SELECT * FROM traffic_log ORDER BY id DESC LIMIT {limit}", conn)
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
# PART 2 — MULTI-LANE YOLOV8 DETECTION
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
    elif count <= 8: return "Medium"
    else:            return "High"

def get_overall_density(total):
    if total <= 5:    return "Low"
    elif total <= 15: return "Medium"
    else:             return "High"

def get_signal_verdict(density):
    """
    Realistic signal logic for full intersection:
    GREEN  = traffic is manageable, vehicles can go
    YELLOW = moderate, signal about to change
    RED    = high congestion, stop
    """
    if density == "Low":    return "GREEN",  15
    elif density == "Medium": return "GREEN", 30
    else:                     return "RED",   45

def get_congestion_score(count, density):
    base = {"Low": 10, "Medium": 45, "High": 78}
    score = base[density] + min(count * 2, 22)
    return min(score, 100)

def detect_frame_multilane(frame, conf=0.4):
    """
    3-lane detection system.
    Frame divided into 3 equal vertical zones.
    Returns per-lane counts + annotated frame.
    """
    mdl = load_yolo_model()
    h, w = frame.shape[:2]
    z1 = w // 3
    z2 = (w * 2) // 3

    annotated = frame.copy()

    # Draw lane dividers
    cv2.line(annotated, (z1, 0), (z1, h), (0, 212, 255), 2)
    cv2.line(annotated, (z2, 0), (z2, h), (0, 212, 255), 2)

    # Lane labels with background
    for pos, label, color in [
        (10,       "LANE 1", (0, 212, 255)),
        (z1+10,    "LANE 2", (0, 230, 118)),
        (z2+10,    "LANE 3", (255, 160, 0)),
    ]:
        cv2.rectangle(annotated, (pos, 5), (pos+140, 42), (0,0,0), -1)
        cv2.putText(annotated, label, (pos+5, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2)

    results = mdl(frame, conf=conf, verbose=False)[0]
    l1, l2, l3 = 0, 0, 0
    conf_scores = []
    vtypes = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}

    LANE_COLORS = [(0, 212, 255), (0, 230, 118), (255, 160, 0)]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        label = VEHICLE_CLASSES[cls_id]
        cs = float(box.conf[0])
        conf_scores.append(cs)
        vtypes[label] += 1

        if cx < z1:
            l1 += 1
            color = LANE_COLORS[0]
        elif cx < z2:
            l2 += 1
            color = LANE_COLORS[1]
        else:
            l3 += 1
            color = LANE_COLORS[2]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        ltext = f"{label} {cs:.0%}"
        (tw, th), _ = cv2.getTextSize(ltext, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(annotated, ltext, (x1+2, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,0,0), 1)

    total = l1 + l2 + l3
    avg_conf = round(sum(conf_scores)/len(conf_scores)*100, 1) if conf_scores else 0.0
    overall_den = get_overall_density(total)
    verdict, duration = get_signal_verdict(overall_den)

    # Draw verdict overlay on frame
    verdict_color = (0, 230, 80) if verdict == "GREEN" else (0, 0, 220)
    cv2.rectangle(annotated, (w-220, 5), (w-5, 50), (0,0,0), -1)
    cv2.putText(annotated, f"SIGNAL: {verdict}", (w-215, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, verdict_color, 2)

    return (annotated, l1, l2, l3, total,
            overall_den, verdict, duration, avg_conf, vtypes)


# ════════════════════════════════════════════════════════════════════
# PART 3 — IoT PUBLISHER
# ════════════════════════════════════════════════════════════════════

def publish_iot(l1c, l2c, l3c, total, density, verdict, duration):
    save_to_db(l1c, l2c, l3c, total, density, verdict, duration)


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
    "l3_hist":      deque(maxlen=40),
    "total_hist":   deque(maxlen=40),
    "den_hist":     deque(maxlen=40),
    "total_frames": 0,
    "session_start": None,
    "session_high":  0,
    "session_vehicles": 0,
    "last_frame":   None,
    "last_l1": 0, "last_l2": 0, "last_l3": 0,
    "last_total": 0,
    "last_den": "Low",
    "last_verdict": "GREEN",
    "last_duration": 15,
    "last_conf": 0.0,
    "last_vtypes": {"Car":0,"Motorcycle":0,"Bus":0,"Truck":0},
    "show_summary": False,
    "active_tab":   "🎯  Live Monitor",
    # Signal timer state
    "signal_verdict":   "GREEN",
    "signal_remaining": 15,
    "signal_duration":  15,
    "signal_last_update": 0.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

DENSITY_NUM = {"Low": 1, "Medium": 2, "High": 3}


# ════════════════════════════════════════════════════════════════════
# PART 6 — CSS (Dark 3D Glassmorphism)
# ════════════════════════════════════════════════════════════════════

def inject_css():
    dm = st.session_state.dark_mode
    bg     = "#06080f" if dm else "#f0f2f8"
    panel  = "#0b0d1a" if dm else "#ffffff"
    card   = "#0e1120" if dm else "#ffffff"
    border = "#1a1f38" if dm else "#e0e4f0"
    t1     = "#eef0f8" if dm else "#0f1220"
    t2     = "#8890aa" if dm else "#4a5280"
    t3     = "#444d6a" if dm else "#8890aa"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {{
    --bg:{bg}; --panel:{panel}; --card:{card}; --border:{border};
    --t1:{t1}; --t2:{t2}; --t3:{t3};
    --blue:#00d4ff; --cyan:#00fff0; --green:#00e676;
    --orange:#ff9800; --red:#ff3d57; --yellow:#ffd600;
    --purple:#b44dff; --pink:#ff4da6;
    --fd:'Syne',sans-serif; --fb:'DM Sans',sans-serif;
    --r-sm:10px; --r-md:16px; --r-lg:22px; --r-xl:30px;
}}

*,*::before,*::after{{box-sizing:border-box;margin:0;}}
html,body,.stApp{{background:var(--bg) !important;color:var(--t1) !important;font-family:var(--fb) !important;}}
#MainMenu,footer,header{{visibility:hidden;}}
.stDeployButton{{display:none;}}
.block-container{{padding:0 2rem 4rem !important;max-width:1800px !important;}}
::-webkit-scrollbar{{width:4px;}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px;}}

/* ── 3D MESH BACKGROUND ── */
.stApp::before{{
    content:'';position:fixed;top:0;left:0;right:0;bottom:0;
    background:
        radial-gradient(ellipse 900px 600px at 5% 5%,rgba(0,212,255,0.07) 0%,transparent 55%),
        radial-gradient(ellipse 700px 500px at 95% 90%,rgba(255,152,0,0.06) 0%,transparent 55%),
        radial-gradient(ellipse 500px 400px at 50% 50%,rgba(180,77,255,0.04) 0%,transparent 60%),
        radial-gradient(ellipse 300px 300px at 80% 20%,rgba(0,230,118,0.04) 0%,transparent 50%);
    pointer-events:none;z-index:0;
}}

/* ── HEADER ── */
.site-header{{
    background:linear-gradient(135deg,rgba(6,8,15,0.97),rgba(11,13,26,0.97));
    border-bottom:1px solid var(--border);
    padding:16px 40px 14px;margin:0 -2rem 0;
    display:flex;align-items:center;justify-content:space-between;
    position:relative;overflow:hidden;
    box-shadow:0 4px 30px rgba(0,0,0,0.5);
}}
.site-header::after{{
    content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,var(--blue),var(--orange),transparent);
}}
.univ-badge{{
    font-family:var(--fd);font-size:.65rem;font-weight:700;
    letter-spacing:.2em;text-transform:uppercase;
    background:linear-gradient(90deg,var(--blue),var(--cyan));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.header-title{{
    font-family:var(--fd);font-size:1.6rem;font-weight:800;
    color:var(--t1);line-height:1.15;letter-spacing:-.03em;
}}
.header-title .hl-b{{
    background:linear-gradient(135deg,var(--blue),var(--cyan));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.header-title .hl-o{{color:var(--orange);}}
.header-sub{{font-size:.73rem;color:var(--t3);letter-spacing:.05em;margin-top:2px;font-style:italic;}}
.status-pill{{
    display:inline-flex;align-items:center;gap:8px;
    background:rgba(0,230,118,.08);border:1px solid rgba(0,230,118,.25);
    border-radius:100px;padding:6px 16px;font-size:.7rem;font-weight:700;
    color:var(--green);letter-spacing:.1em;text-transform:uppercase;
    box-shadow:0 0 20px rgba(0,230,118,.12);
}}
.status-dot{{
    width:7px;height:7px;background:var(--green);border-radius:50%;
    box-shadow:0 0 10px var(--green);animation:pdot 2s ease-in-out infinite;
}}
@keyframes pdot{{0%,100%{{opacity:1;transform:scale(1);}}50%{{opacity:.4;transform:scale(.75);}}}}

/* ── NAV ── */
.nav-wrap{{
    display:flex;align-items:center;gap:4px;
    background:var(--panel);border-bottom:1px solid var(--border);
    padding:0 40px;margin:0 -2rem 2rem;overflow-x:auto;
}}

/* ── SEC LABEL ── */
.sec-label{{
    font-family:var(--fd);font-size:.62rem;font-weight:700;
    letter-spacing:.2em;text-transform:uppercase;color:var(--t3);
    margin-bottom:10px;display:flex;align-items:center;gap:8px;
}}
.sec-label::after{{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);}}

/* ── 3D GLASS CARDS ── */
.glass-card{{
    background:{'rgba(14,17,32,0.8)' if dm else 'rgba(255,255,255,0.85)'};
    backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
    border:1px solid {'rgba(255,255,255,0.06)' if dm else 'rgba(0,0,0,0.08)'};
    border-radius:var(--r-lg);
    box-shadow:
        0 8px 32px rgba(0,0,0,{'0.5' if dm else '0.12'}),
        0 2px 8px rgba(0,0,0,{'0.3' if dm else '0.08'}),
        inset 0 1px 0 rgba(255,255,255,{'0.06' if dm else '0.8'});
    transition:transform .3s cubic-bezier(.4,0,.2,1),box-shadow .3s ease;
    position:relative;overflow:hidden;
}}
.glass-card::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,{'0.12' if dm else '0.6'}),transparent);
}}
.glass-card:hover{{
    transform:translateY(-4px) scale(1.01);
    box-shadow:
        0 20px 60px rgba(0,0,0,{'0.6' if dm else '0.2'}),
        0 8px 20px rgba(0,212,255,0.1);
}}

/* ── LANE CARDS ── */
.lane-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:14px;}}
.lane-card{{
    background:{'rgba(14,17,32,0.85)' if dm else 'rgba(255,255,255,0.9)'};
    backdrop-filter:blur(20px);
    border:1px solid var(--border);
    border-radius:var(--r-md);padding:18px 16px;
    position:relative;overflow:hidden;
    transition:all .3s cubic-bezier(.4,0,.2,1);
    box-shadow:0 4px 20px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.05);
}}
.lane-card::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:3px;
}}
.lane-card.l1::before{{background:linear-gradient(90deg,var(--blue),var(--cyan));}}
.lane-card.l2::before{{background:linear-gradient(90deg,var(--green),#00ff99);}}
.lane-card.l3::before{{background:linear-gradient(90deg,var(--orange),var(--yellow));}}
.lane-card:hover{{
    transform:translateY(-6px);
    box-shadow:0 16px 40px rgba(0,0,0,.5);
}}
.lane-card.l1:hover{{box-shadow:0 16px 40px rgba(0,212,255,.2);}}
.lane-card.l2:hover{{box-shadow:0 16px 40px rgba(0,230,118,.2);}}
.lane-card.l3:hover{{box-shadow:0 16px 40px rgba(255,152,0,.2);}}
.lane-lbl{{
    font-family:var(--fd);font-size:.6rem;font-weight:700;
    letter-spacing:.18em;text-transform:uppercase;
    margin-bottom:10px;display:flex;align-items:center;gap:6px;
}}
.lane-lbl.c1{{color:var(--blue);}}
.lane-lbl.c2{{color:var(--green);}}
.lane-lbl.c3{{color:var(--orange);}}
.ldot{{width:6px;height:6px;border-radius:50%;animation:pdot 2s ease-in-out infinite;}}
.c1 .ldot{{background:var(--blue);box-shadow:0 0 8px var(--blue);}}
.c2 .ldot{{background:var(--green);box-shadow:0 0 8px var(--green);}}
.c3 .ldot{{background:var(--orange);box-shadow:0 0 8px var(--orange);}}
.count-big{{
    font-family:var(--fd);font-size:3rem;font-weight:800;
    line-height:1;letter-spacing:-.04em;margin-bottom:2px;
}}
.count-sub{{font-size:.62rem;color:var(--t3);letter-spacing:.08em;margin-bottom:12px;text-transform:uppercase;}}
.den-row{{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;}}
.den-key{{font-size:.62rem;color:var(--t2);letter-spacing:.05em;text-transform:uppercase;}}
.den-val{{font-size:.68rem;font-weight:800;font-family:var(--fd);text-transform:uppercase;letter-spacing:.08em;}}
.den-val.Low{{color:var(--green);}}
.den-val.Medium{{color:var(--yellow);}}
.den-val.High{{color:var(--red);}}
.den-bar{{height:5px;background:var(--border);border-radius:3px;overflow:hidden;margin-bottom:10px;}}
.den-fill{{height:100%;border-radius:3px;}}
.den-fill.Low{{background:linear-gradient(90deg,var(--green),#00ff99);width:25%;box-shadow:0 0 8px rgba(0,230,118,.5);}}
.den-fill.Medium{{background:linear-gradient(90deg,#e6a800,var(--yellow));width:58%;box-shadow:0 0 8px rgba(255,214,0,.5);}}
.den-fill.High{{background:linear-gradient(90deg,var(--red),#ff6b35);width:95%;box-shadow:0 0 8px rgba(255,61,87,.6);}}
.cong-score{{
    display:flex;align-items:center;justify-content:space-between;
    background:{'rgba(0,0,0,0.2)' if dm else 'rgba(0,0,0,0.05)'};
    border-radius:var(--r-sm);padding:8px 12px;
}}
.cong-num{{font-family:var(--fd);font-size:1.3rem;font-weight:800;}}
.cong-lbl{{font-size:.58rem;color:var(--t3);text-transform:uppercase;letter-spacing:.1em;}}

/* ── SIGNAL VERDICT CARD ── */
.verdict-card{{
    background:{'rgba(14,17,32,0.9)' if dm else 'rgba(255,255,255,0.95)'};
    backdrop-filter:blur(30px);
    border-radius:var(--r-xl);
    padding:28px 24px;
    margin-bottom:14px;
    text-align:center;
    position:relative;overflow:hidden;
    transition:all .5s ease;
    box-shadow:0 8px 40px rgba(0,0,0,.5),inset 0 1px 0 rgba(255,255,255,.06);
}}
.verdict-card.green{{
    border:2px solid rgba(0,230,118,.4);
    box-shadow:0 8px 40px rgba(0,0,0,.5),0 0 60px rgba(0,230,118,.15),inset 0 1px 0 rgba(255,255,255,.06);
}}
.verdict-card.red{{
    border:2px solid rgba(255,61,87,.4);
    box-shadow:0 8px 40px rgba(0,0,0,.5),0 0 60px rgba(255,61,87,.15),inset 0 1px 0 rgba(255,255,255,.06);
}}
.verdict-icon{{font-size:3rem;margin-bottom:8px;display:block;}}
.verdict-label{{
    font-family:var(--fd);font-size:2rem;font-weight:800;
    letter-spacing:.1em;margin-bottom:4px;
}}
.verdict-label.green{{
    background:linear-gradient(135deg,var(--green),#00ff99);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.verdict-label.red{{
    background:linear-gradient(135deg,var(--red),#ff6b35);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.verdict-sub{{font-size:.78rem;color:var(--t2);margin-bottom:16px;}}

/* ── TIMER ── */
.timer-wrap{{
    display:flex;align-items:center;justify-content:center;gap:12px;
    margin-top:8px;
}}
.timer-num{{
    font-family:var(--fd);font-size:2.8rem;font-weight:800;
    color:var(--t1);letter-spacing:-.04em;line-height:1;
}}
.timer-lbl{{font-size:.65rem;color:var(--t3);text-transform:uppercase;letter-spacing:.12em;}}
.timer-bar-wrap{{
    height:6px;background:var(--border);border-radius:3px;
    overflow:hidden;margin-top:12px;
}}
.timer-bar{{height:100%;border-radius:3px;transition:width .5s linear;}}
.timer-bar.green{{background:linear-gradient(90deg,var(--green),#00ff99);box-shadow:0 0 10px rgba(0,230,118,.5);}}
.timer-bar.red{{background:linear-gradient(90deg,var(--red),#ff6b35);box-shadow:0 0 10px rgba(255,61,87,.5);}}

/* ── TOTALS BAR ── */
.tot-bar{{
    background:{'rgba(14,17,32,0.85)' if dm else 'rgba(255,255,255,0.9)'};
    backdrop-filter:blur(20px);
    border:1px solid var(--border);
    border-radius:var(--r-md);padding:14px 20px;
    display:flex;justify-content:space-around;align-items:center;
    margin-bottom:12px;
    box-shadow:0 4px 20px rgba(0,0,0,.3),inset 0 1px 0 rgba(255,255,255,.05);
    position:relative;overflow:hidden;
}}
.tot-bar::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,var(--blue),var(--orange),transparent);opacity:.5;
}}
.tot-item{{text-align:center;}}
.tot-num{{font-family:var(--fd);font-size:1.5rem;font-weight:800;letter-spacing:-.02em;}}
.tot-lbl{{font-size:.58rem;color:var(--t3);text-transform:uppercase;letter-spacing:.12em;margin-top:2px;}}
.tot-div{{width:1px;height:36px;background:linear-gradient(180deg,transparent,var(--border),transparent);}}

/* ── FEED PANEL ── */
.feed-panel{{
    background:{'rgba(14,17,32,0.8)' if dm else 'rgba(255,255,255,0.85)'};
    backdrop-filter:blur(20px);
    border:1px solid var(--border);
    border-radius:var(--r-lg);overflow:hidden;
    box-shadow:0 8px 32px rgba(0,0,0,.5),inset 0 1px 0 rgba(255,255,255,.05);
}}
.feed-bar{{
    background:{'rgba(6,8,15,0.9)' if dm else 'rgba(248,250,255,0.9)'};
    border-bottom:1px solid var(--border);padding:10px 18px;
    display:flex;align-items:center;justify-content:space-between;
}}
.feed-title{{
    font-family:var(--fd);font-size:.65rem;font-weight:700;
    letter-spacing:.18em;text-transform:uppercase;color:var(--t2);
    display:flex;align-items:center;gap:8px;
}}
.live-dot{{
    width:7px;height:7px;background:var(--red);border-radius:50%;
    box-shadow:0 0 10px var(--red);animation:pdot 1s ease-in-out infinite;
}}
.feed-badge{{
    font-size:.58rem;color:var(--t3);
    background:rgba(255,255,255,.04);border:1px solid var(--border);
    padding:3px 10px;border-radius:100px;letter-spacing:.05em;
}}

/* ── ALERT ── */
.alert-box{{
    background:linear-gradient(135deg,rgba(255,61,87,.1),rgba(255,100,50,.06));
    border:1px solid rgba(255,61,87,.35);border-left:4px solid var(--red);
    border-radius:var(--r-md);padding:14px 20px;margin-bottom:16px;
    display:flex;align-items:center;gap:14px;
    animation:alert-pulse 2s ease-in-out infinite;
    box-shadow:0 0 30px rgba(255,61,87,.2);
}}
@keyframes alert-pulse{{
    0%,100%{{box-shadow:0 0 30px rgba(255,61,87,.2);}}
    50%{{box-shadow:0 0 50px rgba(255,61,87,.4);}}
}}
.alert-title{{font-family:var(--fd);font-size:.8rem;font-weight:700;color:var(--red);}}
.alert-sub{{font-size:.68rem;color:rgba(255,100,100,.7);margin-top:2px;}}

/* ── VTYPE ── */
.vtype-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px;}}
.vtype-card{{
    background:{'rgba(14,17,32,0.7)' if dm else 'rgba(255,255,255,0.8)'};
    backdrop-filter:blur(10px);
    border:1px solid var(--border);
    border-radius:var(--r-sm);padding:10px 8px;text-align:center;
    transition:all .3s ease;
    box-shadow:0 2px 10px rgba(0,0,0,.3);
}}
.vtype-card:hover{{transform:translateY(-3px);box-shadow:0 8px 20px rgba(0,212,255,.15);}}
.vtype-icon{{font-size:1.2rem;display:block;margin-bottom:4px;}}
.vtype-num{{font-family:var(--fd);font-size:1.1rem;font-weight:800;color:var(--t1);}}
.vtype-name{{font-size:.58rem;color:var(--t3);text-transform:uppercase;letter-spacing:.08em;margin-top:2px;}}

/* ── SUMMARY CARD ── */
.summary-card{{
    background:linear-gradient(135deg,rgba(0,212,255,.07),rgba(180,77,255,.05));
    border:1px solid rgba(0,212,255,.2);border-radius:var(--r-lg);
    padding:28px 32px;margin-bottom:18px;
    box-shadow:0 8px 32px rgba(0,0,0,.4);
}}
.summary-title{{font-family:var(--fd);font-size:1rem;font-weight:800;color:var(--blue);margin-bottom:18px;}}
.summary-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}}
.summary-item{{
    text-align:center;padding:14px;
    background:{'rgba(0,0,0,.2)' if dm else 'rgba(0,0,0,.04)'};
    border-radius:var(--r-sm);
}}
.summary-num{{font-family:var(--fd);font-size:1.6rem;font-weight:800;}}
.summary-lbl{{font-size:.6rem;color:var(--t3);text-transform:uppercase;letter-spacing:.1em;margin-top:2px;}}

/* ── OVERVIEW CARDS ── */
.ov-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:22px;}}
.ov-card{{
    background:{'rgba(14,17,32,0.85)' if dm else 'rgba(255,255,255,0.9)'};
    backdrop-filter:blur(20px);
    border:1px solid var(--border);border-radius:var(--r-md);
    padding:20px 18px;position:relative;overflow:hidden;
    transition:all .3s ease;
    box-shadow:0 4px 20px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.05);
}}
.ov-card:hover{{transform:translateY(-4px);}}
.ov-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;}}
.ov-card:nth-child(1)::before{{background:linear-gradient(90deg,var(--blue),transparent);}}
.ov-card:nth-child(2)::before{{background:linear-gradient(90deg,var(--green),transparent);}}
.ov-card:nth-child(3)::before{{background:linear-gradient(90deg,var(--orange),transparent);}}
.ov-card:nth-child(4)::before{{background:linear-gradient(90deg,var(--purple),transparent);}}
.ov-num{{font-family:var(--fd);font-size:2rem;font-weight:800;letter-spacing:-.03em;}}
.ov-lbl{{font-size:.62rem;color:var(--t3);text-transform:uppercase;letter-spacing:.1em;margin-top:4px;}}

/* ── ABOUT ── */
.about-hero{{
    background:linear-gradient(135deg,rgba(0,212,255,.06),rgba(255,152,0,.04));
    border:1px solid var(--border);border-radius:var(--r-xl);
    padding:48px 50px;margin-bottom:22px;position:relative;overflow:hidden;
    box-shadow:0 8px 32px rgba(0,0,0,.3);
}}
.about-title{{font-family:var(--fd);font-size:2rem;font-weight:800;letter-spacing:-.03em;margin-bottom:10px;}}
.about-title span{{
    background:linear-gradient(135deg,var(--blue),var(--cyan));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.about-sub{{font-size:.9rem;color:var(--t2);line-height:1.8;max-width:600px;}}
.info-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:22px;}}
.info-card{{
    background:{'rgba(14,17,32,0.85)' if dm else 'rgba(255,255,255,0.9)'};
    backdrop-filter:blur(20px);
    border:1px solid var(--border);border-radius:var(--r-md);padding:22px 22px;
    box-shadow:0 4px 20px rgba(0,0,0,.3);
}}
.info-card-title{{font-family:var(--fd);font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--t3);margin-bottom:14px;}}
.info-row{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--border);}}
.info-row:last-child{{border-bottom:none;}}
.info-key{{font-size:.72rem;color:var(--t2);}}
.info-val{{font-size:.72rem;color:var(--t1);font-weight:500;text-align:right;}}
.tech-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}}
.tech-pill{{
    background:{'rgba(14,17,32,0.85)' if dm else 'rgba(255,255,255,0.9)'};
    backdrop-filter:blur(10px);
    border:1px solid var(--border);border-radius:var(--r-sm);
    padding:12px 10px;text-align:center;transition:all .3s ease;
    box-shadow:0 2px 10px rgba(0,0,0,.3);
}}
.tech-pill:hover{{
    border-color:var(--blue);
    box-shadow:0 8px 24px rgba(0,212,255,.2);
    transform:translateY(-3px);
}}
.tech-icon{{font-size:1.4rem;display:block;margin-bottom:6px;}}
.tech-name{{font-family:var(--fd);font-size:.62rem;font-weight:700;color:var(--t2);letter-spacing:.06em;}}
.dev-card{{
    background:linear-gradient(135deg,rgba(0,212,255,.06),rgba(180,77,255,.04));
    border:1px solid rgba(0,212,255,.2);border-radius:var(--r-lg);
    padding:28px 34px;display:flex;align-items:center;gap:28px;
    box-shadow:0 8px 32px rgba(0,0,0,.3);
}}
.dev-avatar{{
    width:68px;height:68px;border-radius:50%;
    background:linear-gradient(135deg,var(--blue),var(--purple));
    display:flex;align-items:center;justify-content:center;
    font-size:1.8rem;flex-shrink:0;
    box-shadow:0 0 30px rgba(0,212,255,.3);
}}
.dev-name{{font-family:var(--fd);font-size:1.3rem;font-weight:800;margin-bottom:4px;}}
.dev-role{{font-size:.74rem;color:var(--t2);margin-bottom:10px;}}
.dev-tags{{display:flex;gap:8px;flex-wrap:wrap;}}
.dev-tag{{
    font-size:.6rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
    padding:4px 12px;border-radius:100px;
    background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.2);color:var(--blue);
}}

/* ── DB PAGE ── */
.db-stat-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px;}}
.db-stat-card{{
    background:{'rgba(14,17,32,0.85)' if dm else 'rgba(255,255,255,0.9)'};
    backdrop-filter:blur(20px);
    border:1px solid var(--border);border-radius:var(--r-md);padding:16px 18px;text-align:center;
    box-shadow:0 4px 16px rgba(0,0,0,.3);
}}
.db-stat-num{{font-family:var(--fd);font-size:1.6rem;font-weight:800;}}
.db-stat-lbl{{font-size:.6rem;color:var(--t3);text-transform:uppercase;letter-spacing:.1em;margin-top:3px;}}

/* ── UPLOAD ── */
.upload-box{{
    background:{'rgba(14,17,32,0.7)' if dm else 'rgba(255,255,255,0.8)'};
    backdrop-filter:blur(20px);
    border:1.5px dashed var(--border);border-radius:var(--r-xl);
    padding:70px 40px;text-align:center;
    transition:all .3s ease;
    box-shadow:0 4px 20px rgba(0,0,0,.2);
}}
.upload-box:hover{{border-color:var(--blue);box-shadow:0 0 40px rgba(0,212,255,.1);}}
.up-icon{{font-size:2.8rem;margin-bottom:14px;opacity:.4;display:block;}}
.up-title{{font-family:var(--fd);font-size:1.1rem;font-weight:800;color:var(--t2);margin-bottom:8px;}}
.up-sub{{font-size:.76rem;color:var(--t3);line-height:1.8;}}
.up-sub b{{color:var(--blue);}}

/* ── TABLE ── */
.stDataFrame{{border:none !important;}}
.stDataFrame table{{font-family:var(--fb) !important;font-size:.75rem !important;border-collapse:collapse !important;width:100% !important;}}
.stDataFrame thead th{{background:var(--panel) !important;color:var(--t3) !important;font-size:.6rem !important;letter-spacing:.12em !important;text-transform:uppercase !important;font-weight:700 !important;padding:10px 14px !important;border-bottom:1px solid var(--border) !important;font-family:var(--fd) !important;}}
.stDataFrame tbody td{{padding:8px 14px !important;border-bottom:1px solid var(--border) !important;color:var(--t2) !important;}}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"]{{
    display:flex !important;visibility:visible !important;opacity:1 !important;
    width:21rem !important;min-width:21rem !important;
    background:var(--panel) !important;border-right:1px solid var(--border) !important;
}}
[data-testid="stSidebar"] .block-container{{padding:1.8rem 1.4rem !important;}}
.sb-brand{{text-align:center;padding:0 0 18px;border-bottom:1px solid var(--border);margin-bottom:18px;}}
.sb-brand-icon{{font-size:2rem;margin-bottom:6px;display:block;}}
.sb-brand-name{{
    font-family:var(--fd);font-size:1rem;font-weight:800;
    background:linear-gradient(135deg,var(--blue),var(--cyan));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.sb-brand-sub{{font-size:.6rem;color:var(--t3);letter-spacing:.08em;margin-top:2px;}}
.sb-sec{{
    font-family:var(--fd);font-size:.57rem;font-weight:700;letter-spacing:.2em;
    text-transform:uppercase;color:var(--t3);margin:16px 0 8px;
    display:flex;align-items:center;gap:6px;
}}
.sb-sec::after{{content:'';flex:1;height:1px;background:var(--border);}}
.sb-info{{
    background:rgba(0,0,0,.2);border:1px solid var(--border);
    border-radius:var(--r-sm);padding:12px 14px;margin-top:14px;
    font-size:.6rem;color:var(--t3);line-height:1.9;font-family:var(--fb);
}}
.sb-info b{{color:var(--t2);}}
.sb-dev{{margin-top:10px;text-align:center;font-size:.58rem;color:var(--t3);font-family:'Courier New',monospace;letter-spacing:.05em;line-height:1.8;}}

/* ── WIDGETS ── */
.stSlider>label,.stNumberInput>label,.stCheckbox>label,.stSelectbox>label,.stRadio>label{{font-family:var(--fb) !important;font-size:.75rem !important;color:var(--t2) !important;}}
.stButton button{{font-family:var(--fd) !important;font-weight:700 !important;letter-spacing:.08em !important;border-radius:var(--r-sm) !important;transition:all .25s ease !important;}}
.stButton button[kind="primary"]{{background:linear-gradient(135deg,#00b8d9,var(--blue)) !important;color:#000 !important;border:none !important;box-shadow:0 4px 16px rgba(0,212,255,.3) !important;}}
.stButton button[kind="primary"]:hover{{background:linear-gradient(135deg,var(--blue),var(--cyan)) !important;box-shadow:0 8px 28px rgba(0,212,255,.45) !important;transform:translateY(-1px) !important;}}
.stTextInput input{{background:var(--card) !important;border:1px solid var(--border) !important;border-radius:var(--r-sm) !important;color:var(--t1) !important;font-family:var(--fb) !important;}}
.stTextInput input:focus{{border-color:var(--blue) !important;box-shadow:0 0 0 3px rgba(0,212,255,.15) !important;}}

/* ── FOOTER ── */
.site-footer{{
    border-top:1px solid var(--border);margin-top:3rem;
    padding:18px 0 8px;display:flex;justify-content:space-between;
    align-items:flex-end;position:relative;
}}
.site-footer::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(0,212,255,.3),rgba(255,152,0,.2),transparent);
}}
.footer-l{{font-size:.64rem;color:var(--t3);line-height:1.8;}}
.footer-l b{{color:var(--t2);}}
.footer-r{{font-size:.62rem;color:var(--t3);text-align:right;line-height:1.8;font-style:italic;}}
.footer-r span{{color:var(--blue);font-style:normal;font-weight:500;}}

/* ── MOBILE ── */
@media(max-width:768px){{
    .lane-grid{{grid-template-columns:1fr;}}
    .ov-grid{{grid-template-columns:repeat(2,1fr);}}
    .info-grid{{grid-template-columns:1fr;}}
    .tech-grid{{grid-template-columns:repeat(2,1fr);}}
    .summary-grid{{grid-template-columns:1fr;}}
    .vtype-grid{{grid-template-columns:repeat(2,1fr);}}
    .db-stat-row{{grid-template-columns:repeat(2,1fr);}}
    .dev-card{{flex-direction:column;text-align:center;}}
    .site-header{{padding:14px 16px;flex-direction:column;gap:10px;}}
    .header-title{{font-size:1.2rem;}}
    .block-container{{padding:0 1rem 3rem !important;}}
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
    section[data-testid="stSidebar"]{display:none !important;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:50px 20px 10px;">
        <div style="font-size:3.5rem;margin-bottom:14px;">🚦</div>
        <div style="font-family:'Syne',sans-serif;font-size:.65rem;font-weight:700;
                    letter-spacing:.2em;text-transform:uppercase;
                    background:linear-gradient(90deg,#00d4ff,#00fff0);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;margin-bottom:8px;">
            K.R. Mangalam University &nbsp;·&nbsp; School of Engineering & Technology
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
                    color:#eef0f8;letter-spacing:-.02em;margin-bottom:6px;">
            IoT <span style="color:#00d4ff;">Intelligent Traffic</span>
            <span style="color:#ff9800;"> Monitoring</span> System
        </div>
        <div style="font-size:.76rem;color:#444d6a;margin-bottom:36px;font-style:italic;">
            Multi-Lane Detection &nbsp;·&nbsp; Dynamic Signal Control &nbsp;·&nbsp; YOLOv8 AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.markdown("""
        <div style="background:rgba(14,17,32,0.9);backdrop-filter:blur(30px);
                    border:1px solid rgba(255,255,255,0.06);border-radius:24px;
                    padding:32px 28px;
                    box-shadow:0 20px 60px rgba(0,0,0,0.6),inset 0 1px 0 rgba(255,255,255,0.06);">
            <div style="font-family:'Syne',sans-serif;font-size:1.2rem;
                        font-weight:800;color:#eef0f8;margin-bottom:4px;">Welcome Back</div>
            <div style="font-size:.72rem;color:#444d6a;margin-bottom:24px;">
                Sign in to access the dashboard</div>
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
                    st.session_state.logged_in  = True
                    st.session_state.username   = username.strip()
                    st.session_state.role       = role
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")

        st.markdown("""
        <div style="margin-top:14px;padding:12px 16px;
                    background:rgba(0,212,255,0.05);
                    border:1px solid rgba(0,212,255,0.12);
                    border-radius:10px;font-size:.68rem;color:#8890aa;
                    line-height:1.9;text-align:center;">
            <b style="color:#00d4ff;">Admin</b> → admin / krmu2025
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <b style="color:#00d4ff;">Viewer</b> → sarthak / pass123
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;justify-content:center;gap:28px;
                margin-top:32px;flex-wrap:wrap;padding:0 20px 50px;">
        <div style="display:flex;align-items:center;gap:8px;font-size:.7rem;color:#8890aa;">
            <div style="width:8px;height:8px;border-radius:50%;background:#00d4ff;box-shadow:0 0 8px #00d4ff;"></div>
            3-Lane YOLOv8 Detection
        </div>
        <div style="display:flex;align-items:center;gap:8px;font-size:.7rem;color:#8890aa;">
            <div style="width:8px;height:8px;border-radius:50%;background:#ff9800;box-shadow:0 0 8px #ff9800;"></div>
            Dynamic Signal Timer
        </div>
        <div style="display:flex;align-items:center;gap:8px;font-size:.7rem;color:#8890aa;">
            <div style="width:8px;height:8px;border-radius:50%;background:#00e676;box-shadow:0 0 8px #00e676;"></div>
            SQLite IoT Database
        </div>
        <div style="display:flex;align-items:center;gap:8px;font-size:.7rem;color:#8890aa;">
            <div style="width:8px;height:8px;border-radius:50%;background:#b44dff;box-shadow:0 0 8px #b44dff;"></div>
            Mobile Camera Capture
        </div>
    </div>
    """, unsafe_allow_html=True)

if not st.session_state.logged_in:
    show_login()
    st.stop()


# ════════════════════════════════════════════════════════════════════
# PART 8 — HEADER
# ════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="site-header">
    <div class="header-left" style="display:flex;flex-direction:column;gap:2px;">
        <div class="univ-badge">K.R. Mangalam University &nbsp;·&nbsp; School of Engineering & Technology</div>
        <div class="header-title">
            IoT <span class="hl-b">Intelligent Traffic</span>
            <span class="hl-o"> Monitoring</span> System
        </div>
        <div class="header-sub">
            3-Lane Detection &nbsp;·&nbsp; Dynamic Signal Control &nbsp;·&nbsp; YOLOv8 Neural Detection
        </div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px;">
        <div class="status-pill"><div class="status-dot"></div>System Active</div>
        <div style="font-size:.63rem;color:#444d6a;letter-spacing:.05em;">
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
tc = st.columns(len(tabs_list))
for i, tab in enumerate(tabs_list):
    with tc[i]:
        active_tab = st.session_state.active_tab == tab
        if st.button(tab, key=f"tab_{i}", use_container_width=True,
                     type="primary" if active_tab else "secondary"):
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
    input_mode = st.radio("Select source",
        ["📁 Upload Video", "📸 Camera Capture"],
        label_visibility="collapsed")

    default_video = "traffic.mp4"
    uploaded_file = None
    camera_photo  = None

    if input_mode == "📁 Upload Video":
        uploaded_file = st.file_uploader("Upload Traffic Video",
                                          type=["mp4","avi","mov"])
        if os.path.exists(default_video) and uploaded_file is None:
            st.success("✅ traffic.mp4 detected")
        video_ready = uploaded_file is not None or os.path.exists(default_video)
        if not video_ready:
            st.warning("Upload a video to begin")
    else:
        st.info("📱 Works on phone + deployed!\nUses back camera for traffic capture.")
        st.markdown("""
        <script>
        function switchToBackCamera() {
            const videos = document.querySelectorAll('video');
            videos.forEach(v => {
                if(v.srcObject){v.srcObject.getTracks().forEach(t=>t.stop());}
            });
            navigator.mediaDevices.getUserMedia({
                video:{facingMode:{exact:"environment"}}
            }).then(stream=>{
                document.querySelectorAll('video').forEach(v=>{v.srcObject=stream;});
            }).catch(()=>{
                navigator.mediaDevices.getUserMedia({
                    video:{facingMode:"environment"}
                }).then(stream=>{
                    document.querySelectorAll('video').forEach(v=>{v.srcObject=stream;});
                });
            });
        }
        setTimeout(switchToBackCamera,1000);
        setTimeout(switchToBackCamera,2500);
        </script>
        """, unsafe_allow_html=True)
        camera_photo = st.camera_input("📷 Take photo of traffic")
        video_ready  = camera_photo is not None

    st.markdown('<div class="sb-sec">Detection</div>', unsafe_allow_html=True)
    conf_val    = st.slider("Confidence", 0.1, 0.9, 0.4, 0.05)
    pub_every   = st.slider("DB save every N frames", 1, 30, 10)
    skip_frames = st.slider("Frame skip (speed)", 1, 5, 2)

    st.markdown('<div class="sb-sec">Controls</div>', unsafe_allow_html=True)
    if input_mode == "📸 Camera Capture":
        st.info("📸 Just take a photo above")
        start = False
        stop  = st.button("🔄 Reset", use_container_width=True)
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
        YOLOv8n · COCO Dataset<br>
        3-Lane Detection System<br><br>
        <b>Database</b><br>
        SQLite · traffic_data.db
    </div>
    <div class="sb-dev">
        Developed by<br>
        <span style="color:#00d4ff;font-size:.68rem;font-weight:600;">Sarthak Mishra</span><br>
        B.Tech CSE · KRM University<br>
        <span style="color:#1a2040;">uid·2301010232</span>
    </div>
    """, unsafe_allow_html=True)

# Controls
if start:
    st.session_state.running        = True
    st.session_state.show_summary   = False
    st.session_state.session_start  = datetime.now().strftime("%H:%M:%S")
    st.session_state.session_high   = 0
    st.session_state.session_vehicles = 0
    st.session_state.signal_last_update = time.time()

if stop and st.session_state.running:
    st.session_state.running      = False
    st.session_state.show_summary = True
    st.session_state.l1_hist.clear()
    st.session_state.l2_hist.clear()
    st.session_state.l3_hist.clear()
    st.session_state.total_hist.clear()


# ════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE MONITOR
# ════════════════════════════════════════════════════════════════════

if active == "🎯  Live Monitor":

    alert_ph = st.empty()

    # Session summary
    if st.session_state.show_summary and st.session_state.total_frames > 0:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-title">📋 Session Complete — Summary</div>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-num" style="color:var(--blue)">{st.session_state.total_frames}</div>
                    <div class="summary-lbl">Frames Processed</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num" style="color:var(--green)">{st.session_state.session_vehicles}</div>
                    <div class="summary-lbl">Vehicles Detected</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num" style="color:var(--red)">{st.session_state.session_high}</div>
                    <div class="summary-lbl">High Density Events</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num" style="color:var(--purple)">{get_total_records()}</div>
                    <div class="summary-lbl">DB Records</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num" style="font-size:1rem;color:var(--t2)">{st.session_state.session_start or '—'}</div>
                    <div class="summary-lbl">Started</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num" style="font-size:1rem;color:var(--t2)">{datetime.now().strftime("%H:%M:%S")}</div>
                    <div class="summary-lbl">Ended</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    feed_col, right_col = st.columns([3, 2], gap="large")

    with feed_col:
        st.markdown('<div class="sec-label">Live Detection Feed — 3 Lane View</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="feed-panel">
            <div class="feed-bar">
                <div class="feed-title"><div class="live-dot"></div>LIVE · 3-LANE PROCESSING</div>
                <div class="feed-badge">YOLOv8n · COCO · Lane 1 | 2 | 3</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        frame_ph = st.empty()

        # Snapshot
        snap_col, _ = st.columns([1, 3])
        with snap_col:
            snap_btn = st.button("📸 Snapshot", use_container_width=True)
        snap_msg = st.empty()

    with right_col:
        st.markdown('<div class="sec-label">Signal Verdict</div>',
                    unsafe_allow_html=True)
        verdict_ph = st.empty()

        st.markdown('<div class="sec-label">Lane Intelligence</div>',
                    unsafe_allow_html=True)
        lanes_ph = st.empty()

        st.markdown('<div class="sec-label">Vehicle Breakdown</div>',
                    unsafe_allow_html=True)
        vtype_ph = st.empty()

    st.markdown('<div class="sec-label">Session Statistics</div>',
                unsafe_allow_html=True)
    totals_ph = st.empty()

    st.markdown("---")
    st.markdown('<div class="sec-label">Live Charts</div>',
                unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)
    with ch1:
        st.caption("Total Vehicle Count — Last 40 Frames")
        chart1_ph = st.empty()
    with ch2:
        st.caption("Overall Density Level — Last 40 Frames")
        chart2_ph = st.empty()

    # ── HELPERS ──────────────────────────────────────────────────────
    def update_signal_timer(new_verdict, new_duration):
        now = time.time()
        elapsed = now - st.session_state.signal_last_update

        if new_verdict != st.session_state.signal_verdict:
            # Verdict changed — only switch if timer expired
            if st.session_state.signal_remaining <= 0:
                st.session_state.signal_verdict   = new_verdict
                st.session_state.signal_duration  = new_duration
                st.session_state.signal_remaining = new_duration
                st.session_state.signal_last_update = now
        else:
            # Same verdict — count down
            remaining = st.session_state.signal_remaining - elapsed
            st.session_state.signal_remaining = max(0, remaining)
            st.session_state.signal_last_update = now

            # When timer hits 0 — update duration based on current density
            if st.session_state.signal_remaining <= 0:
                st.session_state.signal_verdict   = new_verdict
                st.session_state.signal_duration  = new_duration
                st.session_state.signal_remaining = new_duration

    def render_verdict():
        v        = st.session_state.signal_verdict
        rem      = int(st.session_state.signal_remaining)
        dur      = st.session_state.signal_duration
        pct      = (rem / dur * 100) if dur > 0 else 0
        v_cls    = "green" if v == "GREEN" else "red"
        icon     = "🟢" if v == "GREEN" else "🔴"
        msg      = "Traffic flowing — vehicles may proceed" if v == "GREEN" else "High congestion — please wait"

        verdict_ph.markdown(f"""
        <div class="verdict-card {v_cls}">
            <span class="verdict-icon">{icon}</span>
            <div class="verdict-label {v_cls}">{v}</div>
            <div class="verdict-sub">{msg}</div>
            <div class="timer-wrap">
                <div>
                    <div class="timer-num">{rem:02d}s</div>
                    <div class="timer-lbl">remaining</div>
                </div>
            </div>
            <div class="timer-bar-wrap">
                <div class="timer-bar {v_cls}" style="width:{pct:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_lanes(l1c, l2c, l3c, overall_den, avg_conf):
        def cs(c, d):
            return get_congestion_score(c, d)
        d1 = get_density(l1c)
        d2 = get_density(l2c)
        d3 = get_density(l3c)
        cs1 = cs(l1c, d1)
        cs2 = cs(l2c, d2)
        cs3 = cs(l3c, d3)
        c1c = "var(--green)" if cs1<40 else "var(--yellow)" if cs1<70 else "var(--red)"
        c2c = "var(--green)" if cs2<40 else "var(--yellow)" if cs2<70 else "var(--red)"
        c3c = "var(--green)" if cs3<40 else "var(--yellow)" if cs3<70 else "var(--red)"

        lanes_ph.markdown(f"""
        <div class="lane-grid">
            <div class="lane-card l1">
                <div class="lane-lbl c1"><div class="ldot"></div>Lane 1 · Left</div>
                <div class="count-big" style="color:var(--blue)">{l1c}</div>
                <div class="count-sub">vehicles</div>
                <div class="den-row">
                    <span class="den-key">Density</span>
                    <span class="den-val {d1}">{d1}</span>
                </div>
                <div class="den-bar"><div class="den-fill {d1}"></div></div>
                <div class="cong-score">
                    <div><div class="cong-num" style="color:{c1c}">{cs1}</div><div class="cong-lbl">Score/100</div></div>
                </div>
            </div>
            <div class="lane-card l2">
                <div class="lane-lbl c2"><div class="ldot"></div>Lane 2 · Middle</div>
                <div class="count-big" style="color:var(--green)">{l2c}</div>
                <div class="count-sub">vehicles</div>
                <div class="den-row">
                    <span class="den-key">Density</span>
                    <span class="den-val {d2}">{d2}</span>
                </div>
                <div class="den-bar"><div class="den-fill {d2}"></div></div>
                <div class="cong-score">
                    <div><div class="cong-num" style="color:{c2c}">{cs2}</div><div class="cong-lbl">Score/100</div></div>
                </div>
            </div>
            <div class="lane-card l3">
                <div class="lane-lbl c3"><div class="ldot"></div>Lane 3 · Right</div>
                <div class="count-big" style="color:var(--orange)">{l3c}</div>
                <div class="count-sub">vehicles</div>
                <div class="den-row">
                    <span class="den-key">Density</span>
                    <span class="den-val {d3}">{d3}</span>
                </div>
                <div class="den-bar"><div class="den-fill {d3}"></div></div>
                <div class="cong-score">
                    <div><div class="cong-num" style="color:{c3c}">{cs3}</div><div class="cong-lbl">Score/100</div></div>
                </div>
            </div>
        </div>
        <div style="text-align:center;margin-top:4px;">
            <span style="font-size:.62rem;color:var(--t3);letter-spacing:.1em;text-transform:uppercase;">
                Avg Confidence
            </span>
            <span style="font-family:'Syne',sans-serif;font-size:.78rem;font-weight:700;
                         color:var(--purple);margin-left:8px;">
                🎯 {avg_conf}%
            </span>
        </div>
        """, unsafe_allow_html=True)

    def render_vtypes(vtypes):
        icons = {"Car":"🚗","Motorcycle":"🏍️","Bus":"🚌","Truck":"🚛"}
        html  = '<div class="vtype-grid">'
        for vt, icon in icons.items():
            html += f"""
            <div class="vtype-card">
                <span class="vtype-icon">{icon}</span>
                <div class="vtype-num">{vtypes.get(vt,0)}</div>
                <div class="vtype-name">{vt}</div>
            </div>"""
        html += "</div>"
        vtype_ph.markdown(html, unsafe_allow_html=True)

    def render_totals(l1c, l2c, l3c, total, overall_den):
        totals_ph.markdown(f"""
        <div class="tot-bar">
            <div class="tot-item">
                <div class="tot-num" style="color:var(--blue)">{total}</div>
                <div class="tot-lbl">Total Vehicles</div>
            </div>
            <div class="tot-div"></div>
            <div class="tot-item">
                <div class="tot-num" style="color:var(--purple)">{st.session_state.total_frames}</div>
                <div class="tot-lbl">Frames</div>
            </div>
            <div class="tot-div"></div>
            <div class="tot-item">
                <div class="tot-num" style="font-size:.85rem;color:var(--t2)">{datetime.now().strftime("%H:%M:%S")}</div>
                <div class="tot-lbl">Time</div>
            </div>
            <div class="tot-div"></div>
            <div class="tot-item">
                <div class="tot-num" style="color:var(--green)">{get_total_records()}</div>
                <div class="tot-lbl">DB Records</div>
            </div>
            <div class="tot-div"></div>
            <div class="tot-item">
                <div class="tot-num" style="font-size:.9rem;color:{'var(--red)' if overall_den=='High' else 'var(--yellow)' if overall_den=='Medium' else 'var(--green)'}">{overall_den}</div>
                <div class="tot-lbl">Overall Density</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_charts():
        if len(st.session_state.total_hist) < 2:
            return
        df_c = pd.DataFrame({
            "Frame": list(range(len(st.session_state.total_hist))),
            "Total Vehicles": list(st.session_state.total_hist),
        })
        chart1_ph.line_chart(df_c.set_index("Frame"),
                             color=["#00d4ff"], height=160,
                             use_container_width=True)
        df_d = pd.DataFrame({
            "Frame": list(range(len(st.session_state.den_hist))),
            "Density": [DENSITY_NUM[d] for d in st.session_state.den_hist],
        })
        chart2_ph.line_chart(df_d.set_index("Frame"),
                             color=["#ffd600"], height=160,
                             use_container_width=True)

    def show_alert(overall_den):
        if overall_den == "High":
            alert_ph.markdown("""
            <div class="alert-box">
                <div style="font-size:1.4rem;">🚨</div>
                <div>
                    <div class="alert-title">HIGH CONGESTION ALERT — Full Intersection</div>
                    <div class="alert-sub">Critical vehicle density detected · Signal extended automatically</div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            alert_ph.empty()

    # ── MAIN VIDEO LOOP ───────────────────────────────────────────────
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
                st.session_state.total_frames += 1
                frame = cv2.resize(frame, (960, 540))

                (ann, l1c, l2c, l3c, total,
                 overall_den, verdict, duration,
                 avg_conf, vtypes) = detect_frame_multilane(frame, conf_val)

                # Save state
                st.session_state.last_frame   = ann.copy()
                st.session_state.last_l1      = l1c
                st.session_state.last_l2      = l2c
                st.session_state.last_l3      = l3c
                st.session_state.last_total   = total
                st.session_state.last_den     = overall_den
                st.session_state.last_verdict = verdict
                st.session_state.last_duration= duration
                st.session_state.last_conf    = avg_conf
                st.session_state.last_vtypes  = vtypes
                st.session_state.session_vehicles += total
                if overall_den == "High":
                    st.session_state.session_high += 1

                # Update histories
                st.session_state.l1_hist.append(l1c)
                st.session_state.l2_hist.append(l2c)
                st.session_state.l3_hist.append(l3c)
                st.session_state.total_hist.append(total)
                st.session_state.den_hist.append(overall_den)

                # Update dynamic signal timer
                update_signal_timer(verdict, duration)

                # Render
                frame_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)

                if snap_btn and st.session_state.last_frame is not None:
                    fname = f"snapshot_{datetime.now().strftime('%H%M%S')}.jpg"
                    cv2.imwrite(fname, st.session_state.last_frame)
                    snap_msg.success(f"📸 Saved: {fname}")

                show_alert(overall_den)
                render_verdict()
                render_lanes(l1c, l2c, l3c, overall_den, avg_conf)
                render_vtypes(vtypes)
                render_totals(l1c, l2c, l3c, total, overall_den)
                render_charts()

                if fn % pub_every == 0:
                    publish_iot(l1c, l2c, l3c, total, overall_den, verdict, duration)

                time.sleep(0.03)
        finally:
            cap.release()

    # ── CAMERA CAPTURE MODE ───────────────────────────────────────────
    elif input_mode == "📸 Camera Capture":
        if camera_photo is not None:
            img   = Image.open(io.BytesIO(camera_photo.getvalue()))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            (ann, l1c, l2c, l3c, total,
             overall_den, verdict, duration,
             avg_conf, vtypes) = detect_frame_multilane(frame, conf_val)

            frame_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                           channels="RGB", use_container_width=True)

            update_signal_timer(verdict, duration)
            show_alert(overall_den)
            render_verdict()
            render_lanes(l1c, l2c, l3c, overall_den, avg_conf)
            render_vtypes(vtypes)
            render_totals(l1c, l2c, l3c, total, overall_den)
            publish_iot(l1c, l2c, l3c, total, overall_den, verdict, duration)
            st.success("✅ Analysed! Take another photo to update.")
        else:
            frame_ph.markdown("""
            <div class="upload-box">
                <span class="up-icon">📸</span>
                <div class="up-title">Camera Capture Mode</div>
                <div class="up-sub">
                    Click the camera above to take a photo<br>
                    Works on <b>phone browser</b> and deployed link<br>
                    Point at a real traffic scene
                </div>
            </div>""", unsafe_allow_html=True)
            render_verdict()
            render_lanes(st.session_state.last_l1, st.session_state.last_l2,
                         st.session_state.last_l3, st.session_state.last_den,
                         st.session_state.last_conf)
            render_vtypes(st.session_state.last_vtypes)

    # ── IDLE ─────────────────────────────────────────────────────────
    else:
        frame_ph.markdown("""
        <div class="upload-box">
            <span class="up-icon">🎬</span>
            <div class="up-title">Select Input Mode</div>
            <div class="up-sub">
                Choose <b>Upload Video</b> or <b>Camera Capture</b> in the sidebar<br>
                Free traffic videos → <b>pexels.com</b> → search "highway traffic"
            </div>
        </div>""", unsafe_allow_html=True)
        render_verdict()
        render_lanes(st.session_state.last_l1, st.session_state.last_l2,
                     st.session_state.last_l3, st.session_state.last_den,
                     st.session_state.last_conf)
        render_vtypes(st.session_state.last_vtypes)
        render_totals(0, 0, 0, 0, "Low")
        render_charts()


# ════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ════════════════════════════════════════════════════════════════════

elif active == "📊  Analytics":
    st.markdown('<div class="sec-label">Traffic Analytics Overview</div>',
                unsafe_allow_html=True)
    df_all = get_db_stats()

    if df_all.empty:
        st.info("No data yet. Run Live Monitor first.")
    else:
        total   = len(df_all)
        avg_tot = round(df_all["total_count"].mean(), 1) if "total_count" in df_all else 0
        high_c  = len(df_all[df_all["overall_density"] == "High"]) if "overall_density" in df_all else 0
        green_c = len(df_all[df_all["signal_verdict"] == "GREEN"]) if "signal_verdict" in df_all else 0

        st.markdown(f"""
        <div class="ov-grid">
            <div class="ov-card">
                <span style="font-size:1.6rem;display:block;margin-bottom:10px;">🗄️</span>
                <div class="ov-num" style="color:var(--blue)">{total}</div>
                <div class="ov-lbl">Total Records</div>
            </div>
            <div class="ov-card">
                <span style="font-size:1.6rem;display:block;margin-bottom:10px;">🚗</span>
                <div class="ov-num" style="color:var(--green)">{avg_tot}</div>
                <div class="ov-lbl">Avg Vehicles/Reading</div>
            </div>
            <div class="ov-card">
                <span style="font-size:1.6rem;display:block;margin-bottom:10px;">🚨</span>
                <div class="ov-num" style="color:var(--red)">{high_c}</div>
                <div class="ov-lbl">High Density Events</div>
            </div>
            <div class="ov-card">
                <span style="font-size:1.6rem;display:block;margin-bottom:10px;">🟢</span>
                <div class="ov-num" style="color:var(--green)">{green_c}</div>
                <div class="ov-lbl">GREEN Signal Count</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if "total_count" in df_all.columns:
            st.markdown('<div class="sec-label">Total Vehicle Count Over Time</div>',
                        unsafe_allow_html=True)
            st.line_chart(df_all[["id","total_count"]].set_index("id"),
                          color=["#00d4ff"], height=220, use_container_width=True)

        if all(c in df_all.columns for c in ["lane1_count","lane2_count","lane3_count"]):
            st.markdown('<div class="sec-label">Per Lane Vehicle Count</div>',
                        unsafe_allow_html=True)
            df_lanes = df_all[["id","lane1_count","lane2_count","lane3_count"]].set_index("id")
            df_lanes.columns = ["Lane 1","Lane 2","Lane 3"]
            st.line_chart(df_lanes, color=["#00d4ff","#00e676","#ff9800"],
                          height=220, use_container_width=True)

        if "overall_density" in df_all.columns:
            st.markdown('<div class="sec-label">Density Distribution</div>',
                        unsafe_allow_html=True)
            dc1, dc2 = st.columns(2, gap="medium")
            with dc1:
                st.caption("Overall Density Breakdown")
                st.bar_chart(df_all["overall_density"].value_counts(),
                             color="#ffd600", height=200, use_container_width=True)
            with dc2:
                st.caption("Signal Verdict History")
                if "signal_verdict" in df_all.columns:
                    st.bar_chart(df_all["signal_verdict"].value_counts(),
                                 color="#00e676", height=200, use_container_width=True)

        if "timestamp" in df_all.columns:
            st.markdown('<div class="sec-label">Peak Hour Analysis</div>',
                        unsafe_allow_html=True)
            df_all["hour"] = pd.to_datetime(
                df_all["timestamp"], errors="coerce").dt.hour
            df_hour = df_all.groupby("hour")["total_count"].mean().round(1)
            if not df_hour.empty:
                st.bar_chart(df_hour, color="#b44dff",
                             height=200, use_container_width=True)
                peak = df_hour.idxmax()
                st.caption(f"🔴 Peak traffic hour: **{peak}:00 – {peak+1}:00**")


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
        avg_v  = round(df["total_count"].mean(), 1) if "total_count" in df else 0
        latest = df.iloc[0]["timestamp"] if not df.empty else "—"
        green  = len(df[df["signal_verdict"]=="GREEN"]) if "signal_verdict" in df else 0

        st.markdown(f"""
        <div class="db-stat-row">
            <div class="db-stat-card">
                <div class="db-stat-num" style="color:var(--blue)">{total}</div>
                <div class="db-stat-lbl">Total Records</div>
            </div>
            <div class="db-stat-card">
                <div class="db-stat-num" style="color:var(--green)">{avg_v}</div>
                <div class="db-stat-lbl">Avg Vehicles</div>
            </div>
            <div class="db-stat-card">
                <div class="db-stat-num" style="color:var(--orange)">{green}</div>
                <div class="db-stat-lbl">GREEN Signals</div>
            </div>
            <div class="db-stat-card">
                <div class="db-stat-num" style="font-size:1rem;color:var(--t2);margin-top:6px">{latest[:16]}</div>
                <div class="db-stat-lbl">Latest Record</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        limit_opt = st.selectbox("Show records", [10, 25, 50, 100], index=0)
        df_show   = read_db(limit=limit_opt)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        csv = df_show.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️  Download as CSV", data=csv,
                           file_name=f"traffic_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv")

        if st.session_state.role == "Admin":
            st.markdown('<div class="sec-label" style="margin-top:20px">Users Table (Admin)</div>',
                        unsafe_allow_html=True)
            conn     = sqlite3.connect(DB_FILE)
            users_df = pd.read_sql("SELECT id,username,role FROM users", conn)
            conn.close()
            st.dataframe(users_df, use_container_width=True, hide_index=True)

        st.caption("💡 View database visually → sqlitebrowser.org → Open traffic_data.db")


# ════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════════════════════════════════

elif active == "ℹ️  About":
    st.markdown("""
    <div class="about-hero">
        <div class="about-title">IoT <span>Intelligent Traffic</span> Monitoring System</div>
        <div class="about-sub">
            A real-time multi-lane intelligent traffic monitoring and congestion prediction
            system built using YOLOv8 neural detection, dynamic signal control with timer,
            SQLite database, IoT simulation and secure login authentication.
            Deployed on Streamlit Cloud — accessible from any device.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Project Information</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-grid">
        <div class="info-card">
            <div class="info-card-title">📋 Project Details</div>
            <div class="info-row"><span class="info-key">Title</span><span class="info-val">IoT Intelligent Traffic Monitoring</span></div>
            <div class="info-row"><span class="info-key">Type</span><span class="info-val">IoT Based</span></div>
            <div class="info-row"><span class="info-key">Lanes</span><span class="info-val">3-Lane Detection System</span></div>
            <div class="info-row"><span class="info-key">Year</span><span class="info-val">2025 – 2026</span></div>
            <div class="info-row"><span class="info-key">Status</span><span class="info-val" style="color:#00e676">✅ Approved</span></div>
        </div>
        <div class="info-card">
            <div class="info-card-title">🎓 Academic Details</div>
            <div class="info-row"><span class="info-key">University</span><span class="info-val">K.R. Mangalam University</span></div>
            <div class="info-row"><span class="info-key">School</span><span class="info-val">Engineering & Technology</span></div>
            <div class="info-row"><span class="info-key">Program</span><span class="info-val">B.Tech CSE</span></div>
            <div class="info-row"><span class="info-key">Semester</span><span class="info-val">Pre-Final Year</span></div>
            <div class="info-row"><span class="info-key">Problem Type</span><span class="info-val">IoT Based</span></div>
        </div>
        <div class="info-card">
            <div class="info-card-title">🔧 System Specs</div>
            <div class="info-row"><span class="info-key">AI Model</span><span class="info-val">YOLOv8 Nano</span></div>
            <div class="info-row"><span class="info-key">Dataset</span><span class="info-val">COCO (80 classes)</span></div>
            <div class="info-row"><span class="info-key">Lanes</span><span class="info-val">3 Lanes (L/M/R)</span></div>
            <div class="info-row"><span class="info-key">Signal Timer</span><span class="info-val">Dynamic (15/30/45s)</span></div>
            <div class="info-row"><span class="info-key">Auth</span><span class="info-val">SHA-256 Hashed</span></div>
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
            <div class="dev-role">B.Tech Computer Science & Engineering · K.R. Mangalam University</div>
            <div class="dev-tags">
                <span class="dev-tag">IoT Systems</span>
                <span class="dev-tag">Computer Vision</span>
                <span class="dev-tag">Python</span>
                <span class="dev-tag">Machine Learning</span>
                <span class="dev-tag">Multi-Lane Detection</span>
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
