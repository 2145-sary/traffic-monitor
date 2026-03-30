"""
IoT-Based Intelligent Traffic Monitoring & Congestion Prediction System
K.R. Mangalam University | Department of Computer Science & Engineering
Developer: Sarthak Mishra | Roll No: 2301010232 | B.Tech CSE | 2025-26
v3.1 — 4-Way Intersection | Dynamic Signal Pair System | Fixed Sidebar + Premium UI
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
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT,
            north_count     INTEGER,
            south_count     INTEGER,
            east_count      INTEGER,
            west_count      INTEGER,
            total_count     INTEGER,
            green_pair      TEXT,
            signal_phase    TEXT,
            duration        INTEGER,
            overall_density TEXT
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

def check_login(u, p):
    conn = sqlite3.connect(DB_FILE)
    row  = conn.execute(
        "SELECT role FROM users WHERE username=? AND password=?",
        (u, hash_pw(p))).fetchone()
    conn.close()
    return row[0] if row else None

def save_to_db(nc, sc, ec, wc, total, green_pair, phase, dur, density):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO traffic_log
        (timestamp,north_count,south_count,east_count,west_count,
         total_count,green_pair,signal_phase,duration,overall_density)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          nc, sc, ec, wc, total, green_pair, phase, dur, density))
    conn.commit()
    conn.close()

def read_db(limit=50):
    conn = sqlite3.connect(DB_FILE)
    df   = pd.read_sql(f"SELECT * FROM traffic_log ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df

def get_total_records():
    conn  = sqlite3.connect(DB_FILE)
    count = conn.execute("SELECT COUNT(*) FROM traffic_log").fetchone()[0]
    conn.close()
    return count

def get_db_stats():
    conn = sqlite3.connect(DB_FILE)
    df   = pd.read_sql("SELECT * FROM traffic_log", conn)
    conn.close()
    return df

init_db()


# ════════════════════════════════════════════════════════════════════
# PART 2 — 4-WAY INTERSECTION DETECTION
# ════════════════════════════════════════════════════════════════════

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
_model = None

def load_yolo():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model

def get_density(count):
    if count <= 4:    return "Low"
    elif count <= 10: return "Medium"
    else:             return "High"

def get_overall_density(total):
    if total <= 6:    return "Low"
    elif total <= 18: return "Medium"
    else:             return "High"

def get_pair_score(c1, c2):
    """Combined congestion score for a signal pair"""
    total = c1 + c2
    if total <= 6:    den = "Low"
    elif total <= 18: den = "Medium"
    else:             den = "High"
    base = {"Low": 10, "Medium": 45, "High": 78}
    return base[den] + min(total * 1.5, 22), den

def get_timer(density):
    return {"Low": 15, "Medium": 30, "High": 45}[density]

def detect_4way(frame, conf=0.4):
    """
    4-way intersection detection.
    Frame divided into 4 quadrants:
      Top-half    = North (vehicles coming from north)
      Bottom-half = South
      Left-half   = West
      Right-half  = East

    Signal pairs:
      Pair A: North + South  (run together)
      Pair B: East  + West   (run together)
    """
    mdl  = load_yolo()
    h, w = frame.shape[:2]
    mh   = h // 2
    mw   = w // 2

    ann  = frame.copy()

    # Draw intersection lines
    cv2.line(ann, (0, mh), (w, mh), (255, 255, 255), 2)
    cv2.line(ann, (mw, 0), (mw, h), (255, 255, 255), 2)

    # Direction labels with background boxes
    zones = [
        (mw//2 - 60, 10,    "NORTH", (0, 180, 255)),
        (mw//2 - 60, mh+10, "SOUTH", (0, 220, 150)),
        (10,         mh//2, "WEST",  (255, 160, 0)),
        (mw+10,      mh//2, "EAST",  (220, 80,  255)),
    ]
    for lx, ly, label, color in zones:
        cv2.rectangle(ann, (lx-4, ly-4), (lx+100, ly+28), (0,0,0), -1)
        cv2.putText(ann, label, (lx, ly+20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    results = mdl(frame, conf=conf, verbose=False)[0]

    nc = sc = ec = wc = 0
    conf_scores = []
    vtypes = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}

    ZONE_COLORS = {
        "N": (0, 180, 255),
        "S": (0, 220, 150),
        "W": (255, 160, 0),
        "E": (220, 80, 255),
    }

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        label = VEHICLE_CLASSES[cls_id]
        cs    = float(box.conf[0])
        conf_scores.append(cs)
        vtypes[label] += 1

        # Determine quadrant
        if cy < mh and cx < mw:       zone = "N"  # top-left → North
        elif cy < mh and cx >= mw:    zone = "N"  # top-right → North
        elif cy >= mh and cx < mw:    zone = "S"  # bottom-left → South
        else:                          zone = "S"  # bottom-right → South

        # Refine left/right for East/West
        if cx < mw:  zone_lr = "W"
        else:         zone_lr = "E"

        # Use horizontal split for E/W, vertical for N/S
        # Blend: if cx is strongly left/right, use E/W
        # Use center strip as pure N/S
        strip = w // 6
        if cx < strip or cx > w - strip:
            # Clear left/right → definitely E or W
            if cx < mw: zone = "W"; wc += 1
            else:        zone = "E"; ec += 1
        else:
            # Center area → North or South based on vertical
            if cy < mh:  zone = "N"; nc += 1
            else:         zone = "S"; sc += 1

        color = ZONE_COLORS[zone]
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
        ltext = f"{label} {cs:.0%}"
        (tw, th), _ = cv2.getTextSize(ltext, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(ann, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(ann, ltext, (x1+2, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1)

    total    = nc + sc + ec + wc
    avg_conf = round(sum(conf_scores)/len(conf_scores)*100, 1) if conf_scores else 0.0

    # Signal pair decision
    ns_score, ns_den = get_pair_score(nc, sc)
    ew_score, ew_den = get_pair_score(ec, wc)

    if ns_score >= ew_score:
        green_pair = "NS"   # North-South gets GREEN
        green_den  = ns_den
    else:
        green_pair = "EW"   # East-West gets GREEN
        green_den  = ew_den

    duration       = get_timer(green_den)
    overall_den    = get_overall_density(total)

    return (ann, nc, sc, ec, wc, total,
            green_pair, green_den, duration,
            overall_den, avg_conf, vtypes)


# ════════════════════════════════════════════════════════════════════
# PART 3 — IoT PUBLISHER
# ════════════════════════════════════════════════════════════════════

def publish_iot(nc, sc, ec, wc, total, green_pair, phase, dur, density):
    save_to_db(nc, sc, ec, wc, total, green_pair, phase, dur, density)


# ════════════════════════════════════════════════════════════════════
# PART 4 — PAGE CONFIG
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TrafficVision | KRM University",
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
    "running":      False,
    "total_frames": 0,
    "session_start": None,
    "session_high":  0,
    "session_vehicles": 0,
    "last_frame":   None,
    "last_nc": 0, "last_sc": 0, "last_ec": 0, "last_wc": 0,
    "last_total": 0,
    "last_gp":  "NS",
    "last_den": "Low",
    "last_dur": 15,
    "last_conf": 0.0,
    "last_vtypes": {"Car":0,"Motorcycle":0,"Bus":0,"Truck":0},
    "show_summary": False,
    "active_tab":   "🎯  Live Monitor",
    "total_hist":   deque(maxlen=40),
    "den_hist":     deque(maxlen=40),
    # Signal timer
    "sig_verdict":   "NS",
    "sig_remaining": 15,
    "sig_duration":  15,
    "sig_phase":     "GREEN",   # GREEN / YELLOW
    "sig_last_t":    0.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

DENSITY_NUM = {"Low": 1, "Medium": 2, "High": 3}


# ════════════════════════════════════════════════════════════════════
# PART 6 — PREMIUM UI CSS (FIXED SIDEBAR + SEMI-3D DESIGN)
# ════════════════════════════════════════════════════════════════════

# ⚠️ SIDEBAR FIX: Force sidebar to be visible after login
SIDEBAR_FORCE_VISIBLE = """
<style>
/* CRITICAL: Override any hiding CSS from login page */
section[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 21rem !important;
    min-width: 21rem !important;
    max-width: 21rem !important;
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
    border-right: 3px solid #0f0e0b !important;
    box-shadow: 4px 0 24px rgba(15, 14, 11, 0.08) !important;
    z-index: 999999 !important;
    position: relative !important;
    pointer-events: all !important;
}

/* Force all child elements to be visible */
section[data-testid="stSidebar"] > div {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] * {
    visibility: visible !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.8rem 1.4rem !important;
}

/* Ensure sidebar content is fully rendered */
[data-testid="stSidebar"][aria-expanded="true"] {
    transform: translateX(0) !important;
}
</style>
"""

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Barlow+Condensed:wght@300;400;600;700;900&family=Barlow:wght@300;400;500&display=swap');

:root {
    --bg:        #f8f9fa;
    --bg2:       #eef0f2;
    --white:     #ffffff;
    --ink:       #0f0e0b;
    --ink2:      #2a2820;
    --ink3:      #6b6860;
    --ink4:      #a8a59e;
    --red:       #d32f2f;
    --red2:      #e63329;
    --green:     #2e7d32;
    --green2:    #22873f;
    --amber:     #f57c00;
    --amber2:    #e6920f;
    --blue:      #1976d2;
    --border:    #dee2e6;
    --border2:   #e8eaed;
    --shadow-sm: 0 2px 8px rgba(15, 14, 11, 0.06);
    --shadow-md: 0 4px 16px rgba(15, 14, 11, 0.1);
    --shadow-lg: 0 8px 32px rgba(15, 14, 11, 0.14);
    --shadow-xl: 0 12px 48px rgba(15, 14, 11, 0.18);
    --fd:        'Playfair Display', serif;
    --fc:        'Barlow Condensed', sans-serif;
    --fb:        'Barlow', sans-serif;
    --r-sm:      8px;
    --r-md:      12px;
    --r-lg:      16px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; }
html, body, .stApp {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: var(--fb) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding: 0 2rem 4rem !important; max-width: 1800px !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── PREMIUM 3D EFFECTS ── */
.depth-card {
    background: var(--white);
    border-radius: var(--r-md);
    box-shadow: var(--shadow-md);
    transform: translateZ(0);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

.depth-card:hover {
    transform: translateY(-2px) translateZ(0);
    box-shadow: var(--shadow-lg);
}

.depth-card-xl {
    background: var(--white);
    border-radius: var(--r-lg);
    box-shadow: var(--shadow-lg);
    border: 1px solid var(--border);
    transform: translateZ(0);
    position: relative;
}

/* Subtle gradient overlays for depth */
.gradient-overlay {
    background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
}

/* ── MASTHEAD / HEADER ── */
.masthead {
    border-bottom: 3px solid var(--ink);
    padding: 24px 0 20px;
    margin: 0 -2rem 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    background: linear-gradient(to bottom, var(--white) 0%, var(--bg) 100%);
    position: relative;
    box-shadow: var(--shadow-sm);
}

.masthead::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 5px;
    background: linear-gradient(90deg, var(--red), var(--red2), var(--amber));
}

.masthead-left { display: flex; flex-direction: column; gap: 4px; }

.masthead-flag {
    font-family: var(--fc);
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: .25em;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 6px;
}

.masthead-title {
    font-family: var(--fd);
    font-size: 2.8rem;
    font-weight: 900;
    color: var(--ink);
    line-height: 1;
    letter-spacing: -.02em;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.05);
}

.masthead-title em {
    font-style: italic;
    color: var(--red);
}

.masthead-sub {
    font-family: var(--fc);
    font-size: .75rem;
    font-weight: 400;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-top: 8px;
    border-top: 1px solid var(--border);
    padding-top: 8px;
}

.masthead-right {
    text-align: right;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 8px;
}

.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, var(--red) 0%, var(--red2) 100%);
    color: #fff;
    font-family: var(--fc);
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    padding: 8px 18px;
    border-radius: var(--r-sm);
    box-shadow: var(--shadow-md);
    transform: translateZ(0);
}

.live-dot-r {
    width: 7px; height: 7px;
    background: #fff;
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(255,255,255,0.8);
    animation: blink 1s ease-in-out infinite;
}

@keyframes blink { 0%,100%{opacity:1; transform: scale(1);} 50%{opacity:.3; transform: scale(0.9);} }

.masthead-meta {
    font-family: var(--fc);
    font-size: .65rem;
    font-weight: 400;
    letter-spacing: .08em;
    color: var(--ink3);
}

/* ── NAV BAR ── */
.nav-bar {
    display: flex;
    align-items: center;
    gap: 0;
    border-bottom: 2px solid var(--border);
    margin: 0 -2rem 2.5rem;
    padding: 0 2rem;
    background: var(--white);
    box-shadow: var(--shadow-sm);
}

/* ── RULE LABEL ── */
.rule-label {
    font-family: var(--fc);
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: .22em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.rule-label::after {
    content: '';
    flex: 1;
    height: 2px;
    background: linear-gradient(to right, var(--border), transparent);
}

/* ── SIGNAL CARD (4-WAY) - ENHANCED 3D ── */
.signal-board {
    border: 2px solid var(--border);
    background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%);
    border-radius: var(--r-lg);
    padding: 28px 24px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
    transform: translateZ(0);
    transition: all 0.3s ease;
}

.signal-board:hover {
    transform: translateY(-2px) translateZ(0);
    box-shadow: var(--shadow-xl);
}

.signal-board::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 6px;
    background: linear-gradient(90deg, var(--green), var(--green2));
}

.signal-board.green::before { 
    background: linear-gradient(90deg, var(--green), var(--green2)); 
    box-shadow: 0 2px 12px rgba(46, 125, 50, 0.3);
}

.signal-board.amber::before { 
    background: linear-gradient(90deg, var(--amber), var(--amber2)); 
    box-shadow: 0 2px 12px rgba(245, 124, 0, 0.3);
}

.signal-board.red-phase::before { 
    background: linear-gradient(90deg, var(--red), var(--red2)); 
    box-shadow: 0 2px 12px rgba(211, 47, 47, 0.3);
}

.signal-pair-label {
    font-family: var(--fc);
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .22em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-bottom: 12px;
}

.signal-verdict-text {
    font-family: var(--fd);
    font-size: 3.6rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -.02em;
    margin-bottom: 6px;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.08);
}

.signal-verdict-text.green { color: var(--green); }
.signal-verdict-text.amber { color: var(--amber); }
.signal-verdict-text.red   { color: var(--red); }

.signal-pair-dirs {
    font-family: var(--fc);
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--ink2);
    margin-bottom: 8px;
}

.signal-msg {
    font-family: var(--fb);
    font-size: .78rem;
    color: var(--ink3);
    margin-bottom: 18px;
    line-height: 1.6;
}

.timer-display {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-bottom: 12px;
}

.timer-big {
    font-family: var(--fc);
    font-size: 4.4rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -.04em;
    color: var(--ink);
    text-shadow: 2px 2px 8px rgba(0,0,0,0.06);
}

.timer-unit {
    font-family: var(--fc);
    font-size: .85rem;
    font-weight: 600;
    letter-spacing: .1em;
    color: var(--ink3);
    text-transform: uppercase;
}

.timer-track {
    height: 6px;
    background: var(--border2);
    border-radius: 3px;
    overflow: hidden;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}

.timer-fill {
    height: 100%;
    border-radius: 3px;
    transition: width .5s linear;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}

.timer-fill.green { background: linear-gradient(90deg, var(--green), var(--green2)); }
.timer-fill.amber { background: linear-gradient(90deg, var(--amber), var(--amber2)); }
.timer-fill.red   { background: linear-gradient(90deg, var(--red), var(--red2)); }

/* ── INTERSECTION DIAGRAM - ENHANCED 3D ── */
.intersection-map {
    display: grid;
    grid-template-columns: 1fr 90px 1fr;
    grid-template-rows:    1fr 90px 1fr;
    gap: 6px;
    margin-bottom: 16px;
    height: 240px;
}

.int-cell {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border: 2px solid var(--border);
    border-radius: var(--r-md);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 10px 6px;
    position: relative;
    transition: all .3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow-sm);
    transform: translateZ(0);
}

.int-cell:hover {
    transform: translateY(-2px) translateZ(0);
    box-shadow: var(--shadow-md);
}

.int-cell.active {
    border-color: var(--green);
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f4 100%);
    box-shadow: 0 4px 16px rgba(46, 125, 50, 0.2), inset 0 1px 0 rgba(255,255,255,0.8);
    transform: translateY(-2px) translateZ(0);
}

.int-cell.waiting {
    border-color: var(--border2);
    background: linear-gradient(135deg, #fafafa 0%, #f0f0f0 100%);
    opacity: 0.8;
}

.int-cell.yellow-phase {
    border-color: var(--amber);
    background: linear-gradient(135deg, #fff8e1 0%, #fffaf0 100%);
    box-shadow: 0 4px 16px rgba(245, 124, 0, 0.2), inset 0 1px 0 rgba(255,255,255,0.8);
}

.int-dir {
    font-family: var(--fc);
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-bottom: 5px;
}

.int-count {
    font-family: var(--fd);
    font-size: 2rem;
    font-weight: 900;
    line-height: 1;
    color: var(--ink);
    text-shadow: 1px 1px 3px rgba(0,0,0,0.05);
}

.int-den {
    font-family: var(--fc);
    font-size: .6rem;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-top: 3px;
}

.int-den.Low  { color: var(--green); }
.int-den.Medium { color: var(--amber); }
.int-den.High   { color: var(--red); }

.int-signal {
    position: absolute;
    bottom: 8px;
    font-size: .72rem;
    font-weight: 700;
    font-family: var(--fc);
    letter-spacing: .06em;
}

.int-signal.green { color: var(--green); text-shadow: 0 1px 3px rgba(46, 125, 50, 0.3); }
.int-signal.red   { color: var(--red); text-shadow: 0 1px 3px rgba(211, 47, 47, 0.3); }
.int-signal.amber { color: var(--amber); text-shadow: 0 1px 3px rgba(245, 124, 0, 0.3); }

.int-center {
    background: linear-gradient(135deg, var(--ink) 0%, var(--ink2) 100%);
    border-radius: var(--r-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    box-shadow: var(--shadow-md), inset 0 -2px 8px rgba(0,0,0,0.2);
}

.int-road-h {
    background: linear-gradient(135deg, var(--ink2) 0%, #3d3d3d 100%);
    border-radius: var(--r-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.3);
}

.int-road-v {
    background: linear-gradient(135deg, var(--ink2) 0%, #3d3d3d 100%);
    border-radius: var(--r-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.3);
}

/* ── STAT ROW - ENHANCED 3D ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0;
    border: 2px solid var(--border);
    border-radius: var(--r-lg);
    overflow: hidden;
    margin-bottom: 16px;
    background: var(--white);
    box-shadow: var(--shadow-md);
}

.stat-cell {
    padding: 18px 14px;
    border-right: 1px solid var(--border);
    text-align: center;
    position: relative;
    background: linear-gradient(to bottom, #ffffff, #fafbfc);
    transition: all 0.25s ease;
}

.stat-cell:hover {
    background: linear-gradient(to bottom, #fafbfc, #f5f7f9);
    transform: translateY(-1px);
}

.stat-cell:last-child { border-right: none; }

.stat-cell::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 4px;
}

.stat-cell:nth-child(1)::before { background: linear-gradient(90deg, var(--red), var(--red2)); }
.stat-cell:nth-child(2)::before { background: linear-gradient(90deg, var(--green), var(--green2)); }
.stat-cell:nth-child(3)::before { background: linear-gradient(90deg, var(--amber), var(--amber2)); }
.stat-cell:nth-child(4)::before { background: linear-gradient(90deg, var(--blue), #2196f3); }
.stat-cell:nth-child(5)::before { background: linear-gradient(90deg, var(--ink3), var(--ink2)); }

.stat-num {
    font-family: var(--fc);
    font-size: 2rem;
    font-weight: 900;
    line-height: 1;
    color: var(--ink);
    letter-spacing: -.02em;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.05);
}

.stat-lbl {
    font-family: var(--fc);
    font-size: .6rem;
    font-weight: 600;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-top: 4px;
}

/* ── FEED PANEL - ENHANCED ── */
.feed-wrap {
    border: 2px solid var(--border);
    border-radius: var(--r-lg);
    overflow: hidden;
    background: var(--white);
    box-shadow: var(--shadow-lg);
}

.feed-header {
    background: linear-gradient(135deg, var(--ink) 0%, var(--ink2) 100%);
    padding: 10px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

.feed-hl {
    font-family: var(--fc);
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 10px;
}

.feed-tag {
    font-family: var(--fc);
    font-size: .6rem;
    color: rgba(255,255,255,.6);
    letter-spacing: .08em;
}

/* ── VEHICLE BREAKDOWN - ENHANCED 3D ── */
.vt-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 16px;
}

.vt-cell {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border: 2px solid var(--border);
    border-radius: var(--r-md);
    padding: 14px 10px;
    text-align: center;
    transition: all .3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow-sm);
    transform: translateZ(0);
}

.vt-cell:hover {
    border-color: var(--ink);
    box-shadow: var(--shadow-md), 3px 3px 0 var(--ink);
    transform: translate(-2px, -2px) translateZ(0);
}

.vt-icon { 
    font-size: 1.4rem; 
    display: block; 
    margin-bottom: 6px; 
    filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.1));
}

.vt-num {
    font-family: var(--fc);
    font-size: 1.6rem;
    font-weight: 900;
    color: var(--ink);
    line-height: 1;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
}

.vt-name {
    font-family: var(--fc);
    font-size: .58rem;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-top: 3px;
}

/* ── ALERT - ENHANCED ── */
.alert-strip {
    background: linear-gradient(135deg, var(--red) 0%, var(--red2) 100%);
    color: #fff;
    padding: 14px 22px;
    margin-bottom: 16px;
    border-radius: var(--r-md);
    display: flex;
    align-items: center;
    gap: 14px;
    animation: pulse-red .8s ease-in-out infinite alternate;
    box-shadow: var(--shadow-lg), 0 0 24px rgba(211, 47, 47, 0.3);
}

@keyframes pulse-red { 
    from{opacity:.95; transform: scale(1);} 
    to{opacity:1; transform: scale(1.01);} 
}

.alert-text {
    font-family: var(--fc);
    font-size: .85rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
}

/* ── SUMMARY - ENHANCED ── */
.summary-wrap {
    border: 2px solid var(--border);
    border-radius: var(--r-lg);
    overflow: hidden;
    margin-bottom: 20px;
    background: var(--white);
    box-shadow: var(--shadow-lg);
}

.summary-head {
    background: linear-gradient(135deg, var(--ink) 0%, var(--ink2) 100%);
    padding: 12px 22px;
    font-family: var(--fc);
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

.summary-body {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
}

.summary-item {
    padding: 20px 18px;
    border-right: 1px solid var(--border);
    text-align: center;
    background: linear-gradient(to bottom, #ffffff, #fafbfc);
    transition: background 0.25s ease;
}

.summary-item:hover {
    background: linear-gradient(to bottom, #fafbfc, #f5f7f9);
}

.summary-item:last-child { border-right: none; }

.summary-num {
    font-family: var(--fc);
    font-size: 2.2rem;
    font-weight: 900;
    color: var(--ink);
    letter-spacing: -.02em;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.05);
}

.summary-lbl {
    font-family: var(--fc);
    font-size: .6rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-top: 4px;
}

/* ── ANALYTICS CARDS - ENHANCED ── */
.ov-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    border: 2px solid var(--border);
    border-radius: var(--r-lg);
    overflow: hidden;
    margin-bottom: 24px;
    background: var(--white);
    box-shadow: var(--shadow-md);
}

.ov-cell {
    padding: 24px 20px;
    border-right: 1px solid var(--border);
    background: linear-gradient(to bottom, #ffffff, #fafbfc);
    transition: all 0.25s ease;
}

.ov-cell:hover {
    background: linear-gradient(to bottom, #fafbfc, #f5f7f9);
    transform: translateY(-2px);
}

.ov-cell:last-child { border-right: none; }

.ov-icon { 
    font-size: 1.6rem; 
    margin-bottom: 10px; 
    display: block; 
    filter: drop-shadow(1px 1px 3px rgba(0,0,0,0.1));
}

.ov-num {
    font-family: var(--fc);
    font-size: 2.4rem;
    font-weight: 900;
    color: var(--ink);
    line-height: 1;
    letter-spacing: -.02em;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.05);
}

.ov-lbl {
    font-family: var(--fc);
    font-size: .6rem;
    font-weight: 600;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-top: 5px;
}

/* ── DB STATS - ENHANCED ── */
.db-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    border: 2px solid var(--border);
    border-radius: var(--r-lg);
    overflow: hidden;
    margin-bottom: 20px;
    background: var(--white);
    box-shadow: var(--shadow-md);
}

.db-cell {
    padding: 20px 16px;
    border-right: 1px solid var(--border);
    text-align: center;
    background: linear-gradient(to bottom, #ffffff, #fafbfc);
    transition: all 0.25s ease;
}

.db-cell:hover {
    background: linear-gradient(to bottom, #fafbfc, #f5f7f9);
}

.db-cell:last-child { border-right: none; }

.db-num {
    font-family: var(--fc);
    font-size: 2rem;
    font-weight: 900;
    color: var(--ink);
    line-height: 1;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.05);
}

.db-lbl {
    font-family: var(--fc);
    font-size: .6rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-top: 4px;
}

/* ── ABOUT - ENHANCED ── */
.about-masthead {
    border-bottom: 3px solid var(--ink);
    padding-bottom: 28px;
    margin-bottom: 32px;
}

.about-vol {
    font-family: var(--fc);
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .25em;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 10px;
}

.about-hed {
    font-family: var(--fd);
    font-size: 3.2rem;
    font-weight: 900;
    color: var(--ink);
    line-height: 1.05;
    letter-spacing: -.03em;
    margin-bottom: 16px;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.05);
}

.about-hed em { font-style: italic; color: var(--red); }

.about-dek {
    font-family: var(--fb);
    font-size: .95rem;
    color: var(--ink2);
    line-height: 1.8;
    max-width: 660px;
    border-left: 4px solid var(--red);
    padding-left: 18px;
}

.info-3col {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    border: 2px solid var(--border);
    border-radius: var(--r-lg);
    overflow: hidden;
    margin-bottom: 24px;
    background: var(--white);
    box-shadow: var(--shadow-md);
}

.info-col { 
    padding: 24px 20px; 
    border-right: 1px solid var(--border); 
    background: linear-gradient(to bottom, #ffffff, #fafbfc);
}

.info-col:last-child { border-right: none; }

.info-col-title {
    font-family: var(--fc);
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .22em;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--border);
}

.info-kv {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border2);
}

.info-kv:last-child { border-bottom: none; }

.info-k {
    font-family: var(--fc);
    font-size: .7rem;
    font-weight: 400;
    color: var(--ink3);
    letter-spacing: .04em;
}

.info-v {
    font-family: var(--fc);
    font-size: .7rem;
    font-weight: 700;
    color: var(--ink);
    text-align: right;
}

.tech-8 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}

.tech-item {
    border: 2px solid var(--border);
    border-radius: var(--r-md);
    padding: 14px 12px;
    text-align: center;
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    transition: all .3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
    box-shadow: var(--shadow-sm);
    transform: translateZ(0);
}

.tech-item:hover {
    border-color: var(--ink);
    box-shadow: var(--shadow-md), 3px 3px 0 var(--ink);
    transform: translate(-2px, -2px) translateZ(0);
}

.tech-ico { 
    font-size: 1.6rem; 
    display: block; 
    margin-bottom: 6px; 
    filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.1));
}

.tech-nm {
    font-family: var(--fc);
    font-size: .64rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--ink2);
}

.dev-strip {
    display: flex;
    align-items: center;
    gap: 28px;
    background: linear-gradient(135deg, var(--ink) 0%, var(--ink2) 100%);
    border-radius: var(--r-lg);
    padding: 28px 32px;
    margin-top: 24px;
    box-shadow: var(--shadow-lg);
}

.dev-av {
    width: 72px; height: 72px;
    background: linear-gradient(135deg, var(--red) 0%, var(--red2) 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    flex-shrink: 0;
    box-shadow: 0 4px 16px rgba(211, 47, 47, 0.4);
}

.dev-name {
    font-family: var(--fd);
    font-size: 1.5rem;
    font-weight: 900;
    color: #fff;
    margin-bottom: 5px;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
}

.dev-role { 
    font-family: var(--fc); 
    font-size: .75rem; 
    color: rgba(255,255,255,.65); 
    margin-bottom: 12px; 
    letter-spacing:.05em;
}

.dev-tag {
    display: inline-block;
    font-family: var(--fc);
    font-size: .6rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    padding: 4px 12px;
    border: 1px solid rgba(255,255,255,.3);
    border-radius: var(--r-sm);
    color: rgba(255,255,255,.8);
    margin-right: 8px;
    margin-bottom: 5px;
    transition: all 0.25s ease;
}

.dev-tag:hover {
    background: rgba(255,255,255,0.1);
    border-color: rgba(255,255,255,.5);
}

/* ── UPLOAD / IDLE - ENHANCED ── */
.idle-box {
    border: 3px dashed var(--border);
    border-radius: var(--r-lg);
    padding: 80px 50px;
    text-align: center;
    background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%);
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
}

.idle-box:hover { 
    border-color: var(--ink); 
    box-shadow: var(--shadow-md);
}

.idle-icon { 
    font-size: 2.6rem; 
    margin-bottom: 14px; 
    opacity: .45; 
    display: block; 
    filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.1));
}

.idle-title {
    font-family: var(--fd);
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--ink2);
    margin-bottom: 10px;
}

.idle-sub { 
    font-size: .78rem; 
    color: var(--ink3); 
    line-height: 1.8; 
}

.idle-sub b { color: var(--red); }

/* ── TABLE - ENHANCED ── */
.stDataFrame { border: none !important; }

.stDataFrame table { 
    font-family: var(--fb) !important; 
    font-size: .78rem !important; 
    box-shadow: var(--shadow-sm) !important;
}

.stDataFrame thead th {
    background: linear-gradient(135deg, var(--ink) 0%, var(--ink2) 100%) !important;
    color: #fff !important;
    font-size: .62rem !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    padding: 12px 16px !important;
    font-family: var(--fc) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
}

.stDataFrame tbody td {
    padding: 10px 16px !important;
    border-bottom: 1px solid var(--border2) !important;
    color: var(--ink2) !important;
    background: var(--white) !important;
}

.stDataFrame tbody tr:hover td { 
    background: var(--bg2) !important; 
}

/* ── SIDEBAR - FIXED & ENHANCED ── */
.sb-masthead {
    text-align: center;
    padding: 0 0 20px;
    border-bottom: 3px solid var(--ink);
    margin-bottom: 20px;
}

.sb-flag {
    font-family: var(--fc);
    font-size: .6rem;
    font-weight: 700;
    letter-spacing: .22em;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 5px;
}

.sb-name {
    font-family: var(--fd);
    font-size: 1.2rem;
    font-weight: 900;
    color: var(--ink);
    letter-spacing: -.01em;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
}

.sb-sub {
    font-family: var(--fc);
    font-size: .6rem;
    letter-spacing: .1em;
    color: var(--ink3);
    margin-top: 3px;
}

.sb-sec {
    font-family: var(--fc);
    font-size: .6rem;
    font-weight: 700;
    letter-spacing: .22em;
    text-transform: uppercase;
    color: var(--ink3);
    margin: 18px 0 10px;
    display: flex;
    align-items: center;
    gap: 10px;
    border-top: 2px solid var(--border);
    padding-top: 14px;
}

.sb-info {
    border: 2px solid var(--border);
    border-radius: var(--r-md);
    padding: 14px 16px;
    margin-top: 16px;
    font-size: .64rem;
    color: var(--ink3);
    line-height: 1.9;
    font-family: var(--fb);
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    box-shadow: var(--shadow-sm);
}

.sb-info b { color: var(--ink2); }

.sb-dev {
    margin-top: 12px;
    text-align: center;
    font-size: .6rem;
    color: var(--ink4);
    font-family: var(--fc);
    letter-spacing: .06em;
    line-height: 1.8;
}

/* ── WIDGETS - ENHANCED ── */
.stSlider>label, .stNumberInput>label, .stCheckbox>label,
.stSelectbox>label, .stRadio>label {
    font-family: var(--fb) !important;
    font-size: .76rem !important;
    color: var(--ink2) !important;
}

.stButton button {
    font-family: var(--fc) !important;
    font-weight: 700 !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    border-radius: var(--r-md) !important;
    transition: all .25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: var(--shadow-sm) !important;
}

.stButton button[kind="primary"] {
    background: linear-gradient(135deg, var(--ink) 0%, var(--ink2) 100%) !important;
    color: #fff !important;
    border: 2px solid var(--ink) !important;
}

.stButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--red) 0%, var(--red2) 100%) !important;
    border-color: var(--red) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md), 4px 4px 0 var(--ink) !important;
}

.stButton button[kind="secondary"] {
    background: var(--white) !important;
    color: var(--ink) !important;
    border: 2px solid var(--border) !important;
}

.stButton button[kind="secondary"]:hover {
    border-color: var(--ink) !important;
    box-shadow: var(--shadow-md) !important;
}

.stTextInput input {
    background: var(--white) !important;
    border: 2px solid var(--border) !important;
    border-radius: var(--r-md) !important;
    color: var(--ink) !important;
    font-family: var(--fb) !important;
    box-shadow: var(--shadow-sm) !important;
}

.stTextInput input:focus {
    border-color: var(--ink) !important;
    box-shadow: var(--shadow-md), 3px 3px 0 var(--ink) !important;
}

/* ── FOOTER - ENHANCED ── */
.footer-bar {
    border-top: 3px solid var(--ink);
    margin-top: 3.5rem;
    padding: 18px 0 10px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}

.footer-l { 
    font-family: var(--fc); 
    font-size: .64rem; 
    color: var(--ink3); 
    line-height: 1.8; 
}

.footer-l b { color: var(--ink); }

.footer-r { 
    font-family: var(--fc); 
    font-size: .62rem; 
    color: var(--ink4); 
    text-align: right; 
    line-height: 1.8; 
}

.footer-r span { color: var(--red); font-weight: 700; }

/* ── MOBILE RESPONSIVE ── */
@media (max-width: 768px) {
    .stat-row { grid-template-columns: repeat(3, 1fr); }
    .ov-row   { grid-template-columns: repeat(2, 1fr); }
    .info-3col { grid-template-columns: 1fr; }
    .tech-8   { grid-template-columns: repeat(2, 1fr); }
    .summary-body { grid-template-columns: 1fr; }
    .masthead { flex-direction: column; align-items: flex-start; gap: 14px; }
    .intersection-map { height: 200px; grid-template-columns: 1fr 70px 1fr; grid-template-rows: 1fr 70px 1fr; }
    .dev-strip { flex-direction: column; text-align: center; }
    .db-row   { grid-template-columns: repeat(2, 1fr); }
    .vt-row { grid-template-columns: repeat(2, 1fr); }
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PART 7 — LOGIN PAGE
# ════════════════════════════════════════════════════════════════════

def show_login():
    # ⚠️ FIX: Hide sidebar ONLY on login page
    # This CSS is ONLY injected when user is NOT logged in
    st.markdown("""
    <style>
    section[data-testid="stSidebar"]{
        display:none !important;
        visibility: hidden !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="max-width:480px;margin:90px auto 0;padding:0 20px;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:.62rem;
                    font-weight:700;letter-spacing:.25em;text-transform:uppercase;
                    color:#d32f2f;margin-bottom:10px;">
            K.R. Mangalam University
        </div>
        <div style="font-family:'Playfair Display',serif;font-size:2.8rem;
                    font-weight:900;color:#0f0e0b;line-height:1.05;
                    border-bottom:3px solid #0f0e0b;padding-bottom:18px;
                    margin-bottom:28px;text-shadow: 2px 2px 6px rgba(0,0,0,0.05);">
            Traffic<br><em style="color:#d32f2f;">Vision</em>
        </div>
        <div style="font-family:'Barlow',sans-serif;font-size:.85rem;
                    color:#6b6860;line-height:1.7;margin-bottom:32px;
                    border-left:4px solid #d32f2f;padding-left:16px;">
            4-Way Intersection Intelligence System.<br>
            Real-time congestion analysis for Indian roads.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        with st.form("lf"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", placeholder="Enter password",
                                     type="password")
            st.markdown("<br>", unsafe_allow_html=True)
            sub = st.form_submit_button("Sign In →",
                                        use_container_width=True,
                                        type="primary")
            if sub:
                role = check_login(username.strip(), password)
                if role:
                    st.session_state.logged_in = True
                    st.session_state.username  = username.strip()
                    st.session_state.role      = role
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        st.markdown("""
        <div style="margin-top:16px;font-family:'Barlow Condensed',sans-serif;
                    font-size:.68rem;color:#a8a59e;letter-spacing:.06em;
                    border-top:1px solid #d4d0c8;padding-top:14px;">
            admin / krmu2025 &nbsp;·&nbsp; sarthak / pass123
        </div>
        """, unsafe_allow_html=True)

if not st.session_state.logged_in:
    show_login()
    st.stop()

# ⚠️ CRITICAL FIX: After login, force sidebar to be visible
# This overrides the hiding CSS from the login page
st.markdown(SIDEBAR_FORCE_VISIBLE, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PART 8 — MASTHEAD
# ════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="masthead">
    <div class="masthead-left">
        <div class="masthead-flag">K.R. Mangalam University &nbsp;·&nbsp; School of Engineering & Technology</div>
        <div class="masthead-title">IoT <em>Traffic</em> Vision</div>
        <div class="masthead-sub">
            4-Way Intersection Intelligence &nbsp;·&nbsp; Dynamic Signal Control
            &nbsp;·&nbsp; YOLOv8 Neural Detection &nbsp;·&nbsp; Indian Road System
        </div>
    </div>
    <div class="masthead-right">
        <div class="live-badge"><div class="live-dot-r"></div>Live System</div>
        <div class="masthead-meta">
            {st.session_state.username} ({st.session_state.role})
            &nbsp;·&nbsp; B.Tech CSE &nbsp;·&nbsp; 2025–26
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PART 9 — NAV
# ════════════════════════════════════════════════════════════════════

tabs_list = ["🎯  Live Monitor", "📊  Analytics", "🗄️  Database", "ℹ️  About"]
st.markdown('<div class="nav-bar">', unsafe_allow_html=True)
nc = st.columns(len(tabs_list))
for i, tab in enumerate(tabs_list):
    with nc[i]:
        is_a = st.session_state.active_tab == tab
        if st.button(tab, key=f"t{i}", use_container_width=True,
                     type="primary" if is_a else "secondary"):
            st.session_state.active_tab = tab
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
active = st.session_state.active_tab


# ════════════════════════════════════════════════════════════════════
# PART 10 — SIDEBAR (ALREADY VISIBLE - NO CHANGES NEEDED)
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sb-masthead">
        <div class="sb-flag">TrafficVision</div>
        <div class="sb-name">Control Panel</div>
        <div class="sb-sub">IoT Intelligence Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Input Mode</div>', unsafe_allow_html=True)
    input_mode = st.radio("Source", ["📁 Upload Video", "📸 Camera Capture"],
                          label_visibility="collapsed")

    default_video = "traffic.mp4"
    uploaded_file = None
    camera_photo  = None

    if input_mode == "📁 Upload Video":
        uploaded_file = st.file_uploader("Upload Traffic Video",
                                          type=["mp4", "avi", "mov"])
        if os.path.exists(default_video) and uploaded_file is None:
            st.success("✅ traffic.mp4 detected")
        video_ready = uploaded_file is not None or os.path.exists(default_video)
        if not video_ready:
            st.warning("Upload a video to begin")
    else:
        st.info("📱 Back camera · Works on phone + deployed")
        st.markdown("""<script>
        function bCam(){const vs=document.querySelectorAll('video');
        vs.forEach(v=>{if(v.srcObject)v.srcObject.getTracks().forEach(t=>t.stop());});
        navigator.mediaDevices.getUserMedia({video:{facingMode:{exact:"environment"}})
        .then(s=>{document.querySelectorAll('video').forEach(v=>v.srcObject=s);})
        .catch(()=>navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"})
        .then(s=>{document.querySelectorAll('video').forEach(v=>v.srcObject=s);}));}
        setTimeout(bCam,1000);setTimeout(bCam,2500);
        </script>""", unsafe_allow_html=True)
        camera_photo = st.camera_input("📷 Capture intersection")
        video_ready  = camera_photo is not None

    st.markdown('<div class="sb-sec">Detection Settings</div>', unsafe_allow_html=True)
    conf_val    = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
    pub_every   = st.slider("Save to DB every N frames", 1, 30, 10)
    skip_frames = st.slider("Frame skip (speed)", 1, 5, 2)

    st.markdown('<div class="sb-sec">Controls</div>', unsafe_allow_html=True)
    if input_mode == "📸 Camera Capture":
        st.info("Capture above to analyse")
        start = False
        stop  = st.button("🔄 Reset Results", use_container_width=True)
    else:
        start = st.button("▶ Start Analysis", use_container_width=True,
                          type="primary", disabled=not video_ready)
        stop  = st.button("⏹ Stop",          use_container_width=True)

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
        <b>Intersection Model</b><br>
        4-Way: N / S / E / W<br>
        Pairs: NS ↔ EW<br>
        Timer: 15s / 30s / 45s<br><br>
        <b>Database</b><br>
        SQLite · traffic_data.db
    </div>
    <div class="sb-dev">
        Developed by<br>
        <b style="color:#d32f2f;font-size:.7rem;">Sarthak Mishra</b><br>
        B.Tech CSE · KRM University<br>
        <span style="color:#d4d0c8;">uid·2301010232</span>
    </div>
    """, unsafe_allow_html=True)

# Controls
if start:
    st.session_state.running          = True
    st.session_state.show_summary     = False
    st.session_state.session_start    = datetime.now().strftime("%H:%M:%S")
    st.session_state.session_high     = 0
    st.session_state.session_vehicles = 0
    st.session_state.sig_last_t       = time.time()

if stop and st.session_state.running:
    st.session_state.running      = False
    st.session_state.show_summary = True
    st.session_state.total_hist.clear()
    st.session_state.den_hist.clear()


# ════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE MONITOR
# ════════════════════════════════════════════════════════════════════

if active == "🎯  Live Monitor":

    alert_ph = st.empty()

    # Session summary
    if st.session_state.show_summary and st.session_state.total_frames > 0:
        st.markdown(f"""
        <div class="summary-wrap">
            <div class="summary-head">Session Complete — Summary Report</div>
            <div class="summary-body">
                <div class="summary-item">
                    <div class="summary-num">{st.session_state.total_frames}</div>
                    <div class="summary-lbl">Frames Processed</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num">{st.session_state.session_vehicles}</div>
                    <div class="summary-lbl">Vehicles Detected</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num">{st.session_state.session_high}</div>
                    <div class="summary-lbl">High Density Events</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num">{get_total_records()}</div>
                    <div class="summary-lbl">DB Records Saved</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num" style="font-size:1rem;">{st.session_state.session_start or '—'}</div>
                    <div class="summary-lbl">Started</div>
                </div>
                <div class="summary-item">
                    <div class="summary-num" style="font-size:1rem;">{datetime.now().strftime("%H:%M:%S")}</div>
                    <div class="summary-lbl">Ended</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    feed_col, ctrl_col = st.columns([3, 2], gap="large")

    with feed_col:
        st.markdown('<div class="rule-label">Live Detection Feed — 4-Way View</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="feed-wrap">
            <div class="feed-header">
                <div class="feed-hl">
                    <div class="live-dot-r" style="background:#ef4444;"></div>
                    Live · 4-Way Intersection Processing
                </div>
                <div class="feed-tag">YOLOv8n · N / S / E / W Detection</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        frame_ph = st.empty()
        sc1, sc2 = st.columns([1, 3])
        with sc1:
            snap_btn = st.button("📸 Snapshot", use_container_width=True)
        snap_msg = st.empty()

    with ctrl_col:
        st.markdown('<div class="rule-label">Signal Verdict</div>',
                    unsafe_allow_html=True)
        verdict_ph = st.empty()

        st.markdown('<div class="rule-label">Intersection Map</div>',
                    unsafe_allow_html=True)
        map_ph = st.empty()

        st.markdown('<div class="rule-label">Vehicle Breakdown</div>',
                    unsafe_allow_html=True)
        vtype_ph = st.empty()

    st.markdown('<div class="rule-label">Session Statistics</div>',
                unsafe_allow_html=True)
    stats_ph = st.empty()

    st.markdown("---")
    st.markdown('<div class="rule-label">Live Charts</div>',
                unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)
    with ch1:
        st.caption("Total Vehicles — Last 40 Readings")
        chart1_ph = st.empty()
    with ch2:
        st.caption("Overall Density — Last 40 Readings")
        chart2_ph = st.empty()

    # ── SIGNAL TIMER LOGIC ────────────────────────────────────────────
    def update_timer(new_gp, new_dur):
        now     = time.time()
        elapsed = now - st.session_state.sig_last_t

        if st.session_state.sig_phase == "YELLOW":
            # Counting down yellow (5s)
            rem = st.session_state.sig_remaining - elapsed
            if rem <= 0:
                # Switch to other pair GREEN
                other = "EW" if st.session_state.sig_verdict == "NS" else "NS"
                st.session_state.sig_verdict   = other
                st.session_state.sig_phase     = "GREEN"
                st.session_state.sig_duration  = new_dur
                st.session_state.sig_remaining = new_dur
            else:
                st.session_state.sig_remaining = rem
            st.session_state.sig_last_t = now
            return

        # GREEN phase
        rem = st.session_state.sig_remaining - elapsed
        if rem <= 0:
            # Start YELLOW for 5 seconds before switching
            st.session_state.sig_phase     = "YELLOW"
            st.session_state.sig_remaining = 5
        else:
            st.session_state.sig_remaining = rem

        st.session_state.sig_last_t = now

    # ── RENDER HELPERS ────────────────────────────────────────────────
    def render_verdict():
        gp    = st.session_state.sig_verdict
        phase = st.session_state.sig_phase
        rem   = int(st.session_state.sig_remaining)
        dur   = st.session_state.sig_duration

        if phase == "YELLOW":
            v_cls = "amber"
            v_txt = "YELLOW"
            dirs  = "↕ N·S &nbsp;·&nbsp; ↔ E·W"
            msg   = "Signal changing — prepare to stop"
            icon  = "🟡"
        elif gp == "NS":
            v_cls = "green"
            v_txt = "GREEN"
            dirs  = "↕ NORTH · SOUTH"
            msg   = "North–South corridor open · East–West waiting"
            icon  = "🟢"
        else:
            v_cls = "green"
            v_txt = "GREEN"
            dirs  = "↔ EAST · WEST"
            msg   = "East–West corridor open · North–South waiting"
            icon  = "🟢"

        pct = (rem / max(dur, 1)) * 100

        verdict_ph.markdown(f"""
        <div class="signal-board {v_cls}">
            <div class="signal-pair-label">Active Signal Pair</div>
            <div class="signal-verdict-text {v_cls}">{v_txt}</div>
            <div class="signal-pair-dirs">{dirs}</div>
            <div class="signal-msg">{msg}</div>
            <div class="timer-display">
                <div class="timer-big">{rem:02d}</div>
                <div class="timer-unit">sec</div>
            </div>
            <div class="timer-track">
                <div class="timer-fill {v_cls}" style="width:{pct:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_map(nc, sc, ec, wc, gp, phase):
        nd = get_density(nc)
        sd = get_density(sc)
        ed = get_density(ec)
        wd = get_density(wc)

        def sig_label(direction):
            if phase == "YELLOW":
                return '<span class="int-signal amber">●YLW</span>'
            if (direction in ["N","S"] and gp == "NS") or \
               (direction in ["E","W"] and gp == "EW"):
                return '<span class="int-signal green">●GRN</span>'
            return '<span class="int-signal red">●RED</span>'

        def active_cls(direction):
            if phase == "YELLOW": return "yellow-phase"
            if (direction in ["N","S"] and gp == "NS") or \
               (direction in ["E","W"] and gp == "EW"):
                return "active"
            return "waiting"

        map_ph.markdown(f"""
        <div class="intersection-map">
            <div class="int-cell {active_cls('N')}">
                <div class="int-dir">North</div>
                <div class="int-count">{nc}</div>
                <div class="int-den {nd}">{nd}</div>
                {sig_label('N')}
            </div>
            <div class="int-road-v"></div>
            <div class="int-cell {active_cls('N')}">
                <div class="int-dir">North ↓</div>
                <div class="int-count">{nc}</div>
                <div class="int-den {nd}">{nd}</div>
            </div>
            <div class="int-road-h"></div>
            <div class="int-center">🚦</div>
            <div class="int-road-h"></div>
            <div class="int-cell {active_cls('W')}">
                <div class="int-dir">West</div>
                <div class="int-count">{wc}</div>
                <div class="int-den {wd}">{wd}</div>
                {sig_label('W')}
            </div>
            <div class="int-road-v"></div>
            <div class="int-cell {active_cls('E')}">
                <div class="int-dir">East</div>
                <div class="int-count">{ec}</div>
                <div class="int-den {ed}">{ed}</div>
                {sig_label('E')}
            </div>
            <div class="int-cell {active_cls('S')}">
                <div class="int-dir">South ↑</div>
                <div class="int-count">{sc}</div>
                <div class="int-den {sd}">{sd}</div>
            </div>
            <div class="int-road-v"></div>
            <div class="int-cell {active_cls('S')}">
                <div class="int-dir">South</div>
                <div class="int-count">{sc}</div>
                <div class="int-den {sd}">{sd}</div>
                {sig_label('S')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_vtypes(vtypes):
        icons = {"Car": "🚗", "Motorcycle": "🏍️", "Bus": "🚌", "Truck": "🚛"}
        html  = '<div class="vt-row">'
        for vt, icon in icons.items():
            html += f"""
            <div class="vt-cell">
                <span class="vt-icon">{icon}</span>
                <div class="vt-num">{vtypes.get(vt, 0)}</div>
                <div class="vt-name">{vt}</div>
            </div>"""
        html += "</div>"
        vtype_ph.markdown(html, unsafe_allow_html=True)

    def render_stats(nc, sc, ec, wc, total, den):
        stats_ph.markdown(f"""
        <div class="stat-row">
            <div class="stat-cell">
                <div class="stat-num">{total}</div>
                <div class="stat-lbl">Total Vehicles</div>
            </div>
            <div class="stat-cell">
                <div class="stat-num">{nc}</div>
                <div class="stat-lbl">North</div>
            </div>
            <div class="stat-cell">
                <div class="stat-num">{sc}</div>
                <div class="stat-lbl">South</div>
            </div>
            <div class="stat-cell">
                <div class="stat-num">{ec}</div>
                <div class="stat-lbl">East</div>
            </div>
            <div class="stat-cell">
                <div class="stat-num">{wc}</div>
                <div class="stat-lbl">West</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_charts():
        if len(st.session_state.total_hist) < 2:
            return
        df_c = pd.DataFrame({
            "Frame": list(range(len(st.session_state.total_hist))),
            "Total": list(st.session_state.total_hist),
        })
        chart1_ph.line_chart(df_c.set_index("Frame"),
                             color=["#d32f2f"], height=150,
                             use_container_width=True)
        df_d = pd.DataFrame({
            "Frame": list(range(len(st.session_state.den_hist))),
            "Density": [DENSITY_NUM[d] for d in st.session_state.den_hist],
        })
        chart2_ph.line_chart(df_d.set_index("Frame"),
                             color=["#2e7d32"], height=150,
                             use_container_width=True)

    def show_alert(den):
        if den == "High":
            alert_ph.markdown("""
            <div class="alert-strip">
                <span style="font-size:1.3rem;">🚨</span>
                <div>
                    <div class="alert-text">High Congestion — Intersection Critical</div>
                    <div style="font-size:.7rem;opacity:.85;margin-top:3px;">
                        All directions heavily congested · Signal duration extended
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            alert_ph.empty()

    # ── VIDEO LOOP ────────────────────────────────────────────────────
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

                (ann, nc, sc, ec, wc, total,
                 gp, gden, dur, oden,
                 avg_conf, vtypes) = detect_4way(frame, conf_val)

                # Save state
                st.session_state.last_nc = nc
                st.session_state.last_sc = sc
                st.session_state.last_ec = ec
                st.session_state.last_wc = wc
                st.session_state.last_total = total
                st.session_state.last_gp    = gp
                st.session_state.last_den   = oden
                st.session_state.last_dur   = dur
                st.session_state.last_conf  = avg_conf
                st.session_state.last_vtypes = vtypes
                st.session_state.last_frame  = ann.copy()
                st.session_state.session_vehicles += total
                if oden == "High":
                    st.session_state.session_high += 1

                st.session_state.total_hist.append(total)
                st.session_state.den_hist.append(oden)

                # Update timer
                update_timer(gp, dur)

                # Render
                frame_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)
                if snap_btn and st.session_state.last_frame is not None:
                    fname = f"snapshot_{datetime.now().strftime('%H%M%S')}.jpg"
                    cv2.imwrite(fname, st.session_state.last_frame)
                    snap_msg.success(f"📸 Saved: {fname}")

                show_alert(oden)
                render_verdict()
                render_map(nc, sc, ec, wc, st.session_state.sig_verdict,
                           st.session_state.sig_phase)
                render_vtypes(vtypes)
                render_stats(nc, sc, ec, wc, total, oden)
                render_charts()

                if fn % pub_every == 0:
                    publish_iot(nc, sc, ec, wc, total, gp,
                                st.session_state.sig_phase, dur, oden)
                time.sleep(0.03)
        finally:
            cap.release()

    # ── CAMERA CAPTURE MODE ───────────────────────────────────────────
    elif input_mode == "📸 Camera Capture":
        if camera_photo is not None:
            img   = Image.open(io.BytesIO(camera_photo.getvalue()))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            (ann, nc, sc, ec, wc, total,
             gp, gden, dur, oden,
             avg_conf, vtypes) = detect_4way(frame, conf_val)

            frame_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                           channels="RGB", use_container_width=True)
            update_timer(gp, dur)
            show_alert(oden)
            render_verdict()
            render_map(nc, sc, ec, wc, st.session_state.sig_verdict,
                       st.session_state.sig_phase)
            render_vtypes(vtypes)
            render_stats(nc, sc, ec, wc, total, oden)
            publish_iot(nc, sc, ec, wc, total, gp,
                        st.session_state.sig_phase, dur, oden)
            st.success("✅ Analysed! Capture again to update.")
        else:
            frame_ph.markdown("""
            <div class="idle-box">
                <span class="idle-icon">📸</span>
                <div class="idle-title">Camera Capture Mode</div>
                <div class="idle-sub">
                    Click camera above to capture traffic<br>
                    Works on <b>phone browser</b> and deployed link
                </div>
            </div>""", unsafe_allow_html=True)
            render_verdict()
            render_map(st.session_state.last_nc, st.session_state.last_sc,
                       st.session_state.last_ec, st.session_state.last_wc,
                       st.session_state.last_gp, st.session_state.sig_phase)
            render_vtypes(st.session_state.last_vtypes)
    else:
        frame_ph.markdown("""
        <div class="idle-box">
            <span class="idle-icon">🎬</span>
            <div class="idle-title">Select Input Mode</div>
            <div class="idle-sub">
                Choose <b>Upload Video</b> or <b>Camera Capture</b><br>
                Free videos → <b>pexels.com</b> → search "highway traffic"
            </div>
        </div>""", unsafe_allow_html=True)
        render_verdict()
        render_map(0, 0, 0, 0, "NS", "GREEN")
        render_vtypes({"Car":0,"Motorcycle":0,"Bus":0,"Truck":0})
        render_stats(0, 0, 0, 0, 0, "Low")
        render_charts()


# ════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ════════════════════════════════════════════════════════════════════

elif active == "📊  Analytics":
    st.markdown('<div class="rule-label">Traffic Analytics</div>',
                unsafe_allow_html=True)
    df_all = get_db_stats()

    if df_all.empty:
        st.info("No data yet. Run Live Monitor first.")
    else:
        total  = len(df_all)
        avg_v  = round(df_all["total_count"].mean(), 1) if "total_count" in df_all else 0
        high_c = len(df_all[df_all["overall_density"]=="High"]) if "overall_density" in df_all else 0
        green  = len(df_all[df_all["signal_phase"]=="GREEN"]) if "signal_phase" in df_all else 0

        st.markdown(f"""
        <div class="ov-row">
            <div class="ov-cell">
                <span class="ov-icon">🗄️</span>
                <div class="ov-num">{total}</div>
                <div class="ov-lbl">Total Records</div>
            </div>
            <div class="ov-cell">
                <span class="ov-icon">🚗</span>
                <div class="ov-num">{avg_v}</div>
                <div class="ov-lbl">Avg Vehicles</div>
            </div>
            <div class="ov-cell">
                <span class="ov-icon">🚨</span>
                <div class="ov-num">{high_c}</div>
                <div class="ov-lbl">High Density</div>
            </div>
            <div class="ov-cell">
                <span class="ov-icon">🟢</span>
                <div class="ov-num">{green}</div>
                <div class="ov-lbl">GREEN Phases</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if "total_count" in df_all.columns:
            st.markdown('<div class="rule-label">Total Vehicle Count Over Time</div>',
                        unsafe_allow_html=True)
            st.line_chart(df_all[["id","total_count"]].set_index("id"),
                          color=["#d32f2f"], height=200, use_container_width=True)

        cols = ["north_count","south_count","east_count","west_count"]
        if all(c in df_all.columns for c in cols):
            st.markdown('<div class="rule-label">Per Direction — N / S / E / W</div>',
                        unsafe_allow_html=True)
            df_d = df_all[["id"]+cols].set_index("id")
            df_d.columns = ["North","South","East","West"]
            st.line_chart(df_d, color=["#d32f2f","#2e7d32","#f57c00","#1976d2"],
                          height=200, use_container_width=True)

        if "green_pair" in df_all.columns:
            st.markdown('<div class="rule-label">Signal Pair Distribution</div>',
                        unsafe_allow_html=True)
            a1, a2 = st.columns(2, gap="medium")
            with a1:
                st.caption("Green Pair (NS vs EW)")
                st.bar_chart(df_all["green_pair"].value_counts(),
                             color="#2e7d32", height=180, use_container_width=True)
            with a2:
                st.caption("Overall Density Distribution")
                if "overall_density" in df_all.columns:
                    st.bar_chart(df_all["overall_density"].value_counts(),
                                 color="#f57c00", height=180, use_container_width=True)

        if "timestamp" in df_all.columns:
            st.markdown('<div class="rule-label">Peak Hour Analysis</div>',
                        unsafe_allow_html=True)
            df_all["hour"] = pd.to_datetime(
                df_all["timestamp"], errors="coerce").dt.hour
            df_hr = df_all.groupby("hour")["total_count"].mean().round(1)
            if not df_hr.empty:
                st.bar_chart(df_hr, color="#1976d2",
                             height=180, use_container_width=True)
                pk = df_hr.idxmax()
                st.caption(f"🔴 Peak hour: **{pk}:00 – {pk+1}:00**")


# ════════════════════════════════════════════════════════════════════
# PAGE 3 — DATABASE
# ════════════════════════════════════════════════════════════════════

elif active == "🗄️  Database":
    st.markdown('<div class="rule-label">SQLite Database — traffic_data.db</div>',
                unsafe_allow_html=True)
    df = read_db(limit=100)

    if df.empty:
        st.info("Database is empty. Run Live Monitor to populate.")
    else:
        total  = get_total_records()
        avg_v  = round(df["total_count"].mean(), 1) if "total_count" in df else 0
        green  = len(df[df["signal_phase"]=="GREEN"]) if "signal_phase" in df else 0
        latest = df.iloc[0]["timestamp"][:16] if not df.empty else "—"

        st.markdown(f"""
        <div class="db-row">
            <div class="db-cell">
                <div class="db-num">{total}</div>
                <div class="db-lbl">Total Records</div>
            </div>
            <div class="db-cell">
                <div class="db-num">{avg_v}</div>
                <div class="db-lbl">Avg Vehicles</div>
            </div>
            <div class="db-cell">
                <div class="db-num">{green}</div>
                <div class="db-lbl">GREEN Phases</div>
            </div>
            <div class="db-cell">
                <div class="db-num" style="font-size:1rem;">{latest}</div>
                <div class="db-lbl">Latest Record</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        limit_opt = st.selectbox("Show records", [10, 25, 50, 100], index=0)
        df_show   = read_db(limit=limit_opt)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        csv = df_show.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv,
                           file_name=f"traffic_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv")

        if st.session_state.role == "Admin":
            st.markdown('<div class="rule-label" style="margin-top:20px">Users Table — Admin View</div>',
                        unsafe_allow_html=True)
            conn = sqlite3.connect(DB_FILE)
            u_df = pd.read_sql("SELECT id,username,role FROM users", conn)
            conn.close()
            st.dataframe(u_df, use_container_width=True, hide_index=True)

        st.caption("View raw database → sqlitebrowser.org → Open traffic_data.db")


# ════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════════════════════════════════

elif active == "ℹ️  About":
    st.markdown("""
    <div class="about-masthead">
        <div class="about-vol">K.R. Mangalam University · B.Tech CSE · Pre-Final Year · 2025–26</div>
        <div class="about-hed">IoT <em>Intelligent</em><br>Traffic Vision</div>
        <div class="about-dek">
            A real-time 4-way intersection traffic monitoring and congestion prediction system.
            Modelled on Indian road logic — North–South and East–West signal pairs run together,
            with dynamic timers based on live congestion analysis. Built with YOLOv8 AI,
            SQLite database, secure authentication, and deployed on Streamlit Cloud.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="rule-label">Project Information</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-3col">
        <div class="info-col">
            <div class="info-col-title">Project Details</div>
            <div class="info-kv"><span class="info-k">Title</span><span class="info-v">IoT Traffic Monitoring</span></div>
            <div class="info-kv"><span class="info-k">Type</span><span class="info-v">IoT Based</span></div>
            <div class="info-kv"><span class="info-k">Model</span><span class="info-v">4-Way Intersection</span></div>
            <div class="info-kv"><span class="info-k">Signal Logic</span><span class="info-v">NS ↔ EW Pair System</span></div>
            <div class="info-kv"><span class="info-k">Status</span><span class="info-v" style="color:#2e7d32;">✅ Approved</span></div>
        </div>
        <div class="info-col">
            <div class="info-col-title">Academic Details</div>
            <div class="info-kv"><span class="info-k">University</span><span class="info-v">K.R. Mangalam</span></div>
            <div class="info-kv"><span class="info-k">School</span><span class="info-v">Engg & Technology</span></div>
            <div class="info-kv"><span class="info-k">Program</span><span class="info-v">B.Tech CSE</span></div>
            <div class="info-kv"><span class="info-k">Semester</span><span class="info-v">Pre-Final Year</span></div>
            <div class="info-kv"><span class="info-k">Year</span><span class="info-v">2025–2026</span></div>
        </div>
        <div class="info-col">
            <div class="info-col-title">System Specs</div>
            <div class="info-kv"><span class="info-k">AI Model</span><span class="info-v">YOLOv8 Nano</span></div>
            <div class="info-kv"><span class="info-k">Dataset</span><span class="info-v">COCO 80 classes</span></div>
            <div class="info-kv"><span class="info-k">Signal Timer</span><span class="info-v">15s / 30s / 45s</span></div>
            <div class="info-kv"><span class="info-k">Yellow Phase</span><span class="info-v">5s transition</span></div>
            <div class="info-kv"><span class="info-k">Auth</span><span class="info-v">SHA-256 Hashed</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="rule-label">Technology Stack</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="tech-8">
        <div class="tech-item"><span class="tech-ico">🐍</span><div class="tech-nm">Python 3.11</div></div>
        <div class="tech-item"><span class="tech-ico">🤖</span><div class="tech-nm">YOLOv8</div></div>
        <div class="tech-item"><span class="tech-ico">📷</span><div class="tech-nm">OpenCV</div></div>
        <div class="tech-item"><span class="tech-ico">🌐</span><div class="tech-nm">Streamlit</div></div>
        <div class="tech-item"><span class="tech-ico">🗄️</span><div class="tech-nm">SQLite</div></div>
        <div class="tech-item"><span class="tech-ico">📡</span><div class="tech-nm">MQTT</div></div>
        <div class="tech-item"><span class="tech-ico">🐼</span><div class="tech-nm">Pandas</div></div>
        <div class="tech-item"><span class="tech-ico">☁️</span><div class="tech-nm">Streamlit Cloud</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dev-strip">
        <div class="dev-av">👨‍💻</div>
        <div>
            <div class="dev-name">Sarthak Mishra</div>
            <div class="dev-role">B.Tech Computer Science & Engineering · K.R. Mangalam University</div>
            <div>
                <span class="dev-tag">IoT Systems</span>
                <span class="dev-tag">Computer Vision</span>
                <span class="dev-tag">Python</span>
                <span class="dev-tag">4-Way Intersection Logic</span>
                <span class="dev-tag">Machine Learning</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer-bar">
    <div class="footer-l">
        <b>IoT-Based Intelligent Traffic Monitoring & Congestion Prediction System</b><br>
        K.R. Mangalam University · School of Engineering & Technology · B.Tech CSE · 2025–26
    </div>
    <div class="footer-r">
        Designed & Developed by<br>
        <span>Sarthak Mishra</span> · B.Tech CSE<br>
        K.R. Mangalam University
    </div>
</div>
""", unsafe_allow_html=True)
