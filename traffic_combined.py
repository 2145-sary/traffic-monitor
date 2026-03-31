"""
IoT-Based Intelligent Traffic Monitoring & Congestion Prediction System
K.R. Mangalam University | Department of Computer Science & Engineering
Developer: Sarthak Mishra | Roll No: 2301010232 | B.Tech CSE | 2025-26
v3.0 — 4-Way Intersection | Dynamic Signal Pair System | Editorial UI
\"\"\"

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

DB_FILE = \"traffic_data.db\"

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute(\"\"\"
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
    \"\"\")
    conn.execute(\"\"\"
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role     TEXT
        )
    \"\"\")
    existing = conn.execute(\"SELECT COUNT(*) FROM users\").fetchone()[0]
    if existing == 0:
        conn.execute(\"INSERT INTO users (username,password,role) VALUES (?,?,?)\",
                     (\"admin\", hash_pw(\"krmu2025\"), \"Admin\"))
        conn.execute(\"INSERT INTO users (username,password,role) VALUES (?,?,?)\",
                     (\"sarthak\", hash_pw(\"pass123\"), \"Viewer\"))
    conn.commit()
    conn.close()

def check_login(u, p):
    conn = sqlite3.connect(DB_FILE)
    row  = conn.execute(
        \"SELECT role FROM users WHERE username=? AND password=?\",
        (u, hash_pw(p))).fetchone()
    conn.close()
    return row[0] if row else None

def save_to_db(nc, sc, ec, wc, total, green_pair, phase, dur, density):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(\"\"\"
        INSERT INTO traffic_log
        (timestamp,north_count,south_count,east_count,west_count,
         total_count,green_pair,signal_phase,duration,overall_density)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    \"\"\", (datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\"),
          nc, sc, ec, wc, total, green_pair, phase, dur, density))
    conn.commit()
    conn.close()

def read_db(limit=50):
    conn = sqlite3.connect(DB_FILE)
    df   = pd.read_sql(f\"SELECT * FROM traffic_log ORDER BY id DESC LIMIT {limit}\", conn)
    conn.close()
    return df

def get_total_records():
    conn  = sqlite3.connect(DB_FILE)
    count = conn.execute(\"SELECT COUNT(*) FROM traffic_log\").fetchone()[0]
    conn.close()
    return count

def get_db_stats():
    conn = sqlite3.connect(DB_FILE)
    df   = pd.read_sql(\"SELECT * FROM traffic_log\", conn)
    conn.close()
    return df

init_db()


# ════════════════════════════════════════════════════════════════════
# PART 2 — 4-WAY INTERSECTION DETECTION
# ════════════════════════════════════════════════════════════════════

VEHICLE_CLASSES = {2: \"Car\", 3: \"Motorcycle\", 5: \"Bus\", 7: \"Truck\"}
_model = None

def load_yolo():
    global _model
    if _model is None:
        _model = YOLO(\"yolov8n.pt\")
    return _model

def get_density(count):
    if count <= 4:    return \"Low\"
    elif count <= 10: return \"Medium\"
    else:             return \"High\"

def get_overall_density(total):
    if total <= 6:    return \"Low\"
    elif total <= 18: return \"Medium\"
    else:             return \"High\"

def get_pair_score(c1, c2):
    \"\"\"Combined congestion score for a signal pair\"\"\"
    total = c1 + c2
    if total <= 6:    den = \"Low\"
    elif total <= 18: den = \"Medium\"
    else:             den = \"High\"
    base = {\"Low\": 10, \"Medium\": 45, \"High\": 78}
    return base[den] + min(total * 1.5, 22), den

def get_timer(density):
    return {\"Low\": 15, \"Medium\": 30, \"High\": 45}[density]

def detect_4way(frame, conf=0.4):
    \"\"\"
    4-way intersection detection.
    Frame divided into 4 quadrants:
      Top-half    = North (vehicles coming from north)
      Bottom-half = South
      Left-half   = West
      Right-half  = East

    Signal pairs:
      Pair A: North + South  (run together)
      Pair B: East  + West   (run together)
    \"\"\"
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
        (mw//2 - 60, 10,    \"NORTH\", (0, 180, 255)),
        (mw//2 - 60, mh+10, \"SOUTH\", (0, 220, 150)),
        (10,         mh//2, \"WEST\",  (255, 160, 0)),
        (mw+10,      mh//2, \"EAST\",  (220, 80,  255)),
    ]
    for lx, ly, label, color in zones:
        cv2.rectangle(ann, (lx-4, ly-4), (lx+100, ly+28), (0,0,0), -1)
        cv2.putText(ann, label, (lx, ly+20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    results = mdl(frame, conf=conf, verbose=False)[0]

    nc = sc = ec = wc = 0
    conf_scores = []
    vtypes = {\"Car\": 0, \"Motorcycle\": 0, \"Bus\": 0, \"Truck\": 0}

    ZONE_COLORS = {
        \"N\": (0, 180, 255),
        \"S\": (0, 220, 150),
        \"W\": (255, 160, 0),
        \"E\": (220, 80, 255),
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
        if cy < mh and cx < mw:       zone = \"N\"  # top-left → North
        elif cy < mh and cx >= mw:    zone = \"N\"  # top-right → North
        elif cy >= mh and cx < mw:    zone = \"S\"  # bottom-left → South
        else:                          zone = \"S\"  # bottom-right → South

        # Refine left/right for East/West
        if cx < mw:  zone_lr = \"W\"
        else:         zone_lr = \"E\"

        # Use horizontal split for E/W, vertical for N/S
        # Blend: if cx is strongly left/right, use E/W
        # Use center strip as pure N/S
        strip = w // 6
        if cx < strip or cx > w - strip:
            # Clear left/right → definitely E or W
            if cx < mw: zone = \"W\"; wc += 1
            else:        zone = \"E\"; ec += 1
        else:
            # Center area → North or South based on vertical
            if cy < mh:  zone = \"N\"; nc += 1
            else:         zone = \"S\"; sc += 1

        color = ZONE_COLORS[zone]
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
        ltext = f\"{label} {cs:.0%}\"
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
        green_pair = \"NS\"   # North-South gets GREEN
        green_den  = ns_den
    else:
        green_pair = \"EW\"   # East-West gets GREEN
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
    page_title=\"TrafficVision | KRM University\",
    page_icon=\"🚦\",
    layout=\"wide\",
    initial_sidebar_state=\"expanded\"
)


# ════════════════════════════════════════════════════════════════════
# PART 5 — SESSION STATE
# ════════════════════════════════════════════════════════════════════

defaults = {
    \"logged_in\":    False,
    \"username\":     \"\",
    \"role\":         \"\",
    \"running\":      False,
    \"total_frames\": 0,
    \"session_start\": None,
    \"session_high\":  0,
    \"session_vehicles\": 0,
    \"last_frame\":   None,
    \"last_nc\": 0, \"last_sc\": 0, \"last_ec\": 0, \"last_wc\": 0,
    \"last_total\": 0,
    \"last_gp\":  \"NS\",
    \"last_den\": \"Low\",
    \"last_dur\": 15,
    \"last_conf\": 0.0,
    \"last_vtypes\": {\"Car\":0,\"Motorcycle\":0,\"Bus\":0,\"Truck\":0},
    \"show_summary\": False,
    \"active_tab\":   \"🎯  Live Monitor\",
    \"total_hist\":   deque(maxlen=40),
    \"den_hist\":     deque(maxlen=40),
    # Signal timer
    \"sig_verdict\":   \"NS\",
    \"sig_remaining\": 15,
    \"sig_duration\":  15,
    \"sig_phase\":     \"GREEN\",   # GREEN / YELLOW
    \"sig_last_t\":    0.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

DENSITY_NUM = {\"Low\": 1, \"Medium\": 2, \"High\": 3}


# ════════════════════════════════════════════════════════════════════
# PART 6 — PREMIUM 3D GLASSMORPHISM CSS (Bold Light Theme)
# ════════════════════════════════════════════════════════════════════

st.markdown(\"\"\"
<style>
@import url('https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@400;500;700;800;900&family=IBM+Plex+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Barlow+Condensed:wght@300;400;600;700;900&family=Barlow:wght@300;400;500&display=swap');

:root {
    /* Core Colors */
    --bg:        #f5f2ed;
    --bg2:       #edeae3;
    --white:     #ffffff;
    --ink:       #0f0e0b;
    --ink2:      #2a2820;
    --ink3:      #6b6860;
    --ink4:      #a8a59e;
    --red:       #c8281e;
    --red2:      #e63329;
    --green:     #1b9e54;
    --green2:    #22873f;
    --amber:     #e5a720;
    --amber2:    #c47a0a;
    --blue:      #1a3a6b;
    --border:    rgba(15, 14, 11, 0.1);
    --border2:   rgba(15, 14, 11, 0.05);
    
    /* Glassmorphism */
    --glass-bg:  rgba(255, 255, 255, 0.6);
    --glass-bg-hover: rgba(255, 255, 255, 0.8);
    --glass-border: rgba(255, 255, 255, 0.4);
    --glass-dark: rgba(15, 14, 11, 0.85);
    --glass-blur: 24px;
    
    /* 3D Shadows */
    --shadow-subtle: 0 4px 12px rgba(15, 14, 11, 0.05);
    --shadow-moderate: 0 8px 24px rgba(15, 14, 11, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.6);
    --shadow-bold: 0 20px 40px rgba(200, 40, 30, 0.12), 0 0 0 1px rgba(200, 40, 30, 0.08);
    --shadow-float: 0 25px 50px -12px rgba(15, 14, 11, 0.25);
    --shadow-glow-red: 0 0 40px rgba(200, 40, 30, 0.3), 0 0 80px rgba(200, 40, 30, 0.15);
    --shadow-glow-green: 0 0 40px rgba(27, 158, 84, 0.3), 0 0 80px rgba(27, 158, 84, 0.15);
    --shadow-glow-amber: 0 0 40px rgba(229, 167, 32, 0.3), 0 0 80px rgba(229, 167, 32, 0.15);
    
    /* Gradients */
    --gradient-warm: linear-gradient(135deg, rgba(200, 40, 30, 0.08) 0%, rgba(245, 242, 237, 0) 70%);
    --gradient-hero: linear-gradient(180deg, rgba(245, 242, 237, 0) 0%, rgba(200, 40, 30, 0.05) 100%);
    --gradient-glass: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.4) 100%);
    --gradient-dark: linear-gradient(135deg, rgba(15, 14, 11, 0.95) 0%, rgba(42, 40, 32, 0.9) 100%);
    
    /* Typography */
    --fd:        'Playfair Display', 'Cabinet Grotesk', serif;
    --fc:        'Barlow Condensed', 'Cabinet Grotesk', sans-serif;
    --fb:        'IBM Plex Sans', 'Barlow', sans-serif;
    --fm:        'JetBrains Mono', monospace;
    
    /* Radius */
    --r-sm:      8px;
    --r-md:      16px;
    --r-lg:      24px;
    --r-xl:      32px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, .stApp {
    background: var(--bg) !important;
    background-image: var(--gradient-warm) !important;
    color: var(--ink) !important;
    font-family: var(--fb) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { 
    padding: 0 2.5rem 4rem !important; 
    max-width: 1900px !important; 
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg2); border-radius: 3px; }
::-webkit-scrollbar-thumb { 
    background: linear-gradient(180deg, var(--ink3), var(--ink4)); 
    border-radius: 3px; 
}

/* ── ANIMATED BACKGROUND ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 40%, rgba(200, 40, 30, 0.04) 0%, transparent 50%),
        radial-gradient(ellipse 60% 40% at 80% 60%, rgba(27, 158, 84, 0.03) 0%, transparent 50%),
        url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23noise)' opacity='0.025'/%3E%3C/svg%3E\");
    pointer-events: none;
    z-index: 0;
    animation: gradientShift 20s ease-in-out infinite alternate;
}

@keyframes gradientShift {
    0% { opacity: 1; }
    50% { opacity: 0.8; }
    100% { opacity: 1; }
}

/* ── GLASSMORPHIC MASTHEAD ── */
.masthead {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: var(--r-lg);
    padding: 24px 32px;
    margin: 1rem 0 1.5rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-moderate);
    transform: translateZ(0);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    animation: slideDown 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

.masthead:hover {
    box-shadow: var(--shadow-bold);
    transform: translateY(-2px);
}

.masthead::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, var(--red), var(--red2), var(--amber));
    border-radius: var(--r-lg) var(--r-lg) 0 0;
}

.masthead::after {
    content: '';
    position: absolute;
    top: 0; right: 0; width: 200px; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(200, 40, 30, 0.03));
    pointer-events: none;
}

.masthead-left { display: flex; flex-direction: column; gap: 4px; z-index: 1; }

.masthead-flag {
    font-family: var(--fc);
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: .3em;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 6px;
    text-shadow: 0 1px 2px rgba(200, 40, 30, 0.1);
}

.masthead-title {
    font-family: var(--fd);
    font-size: 2.8rem;
    font-weight: 900;
    color: var(--ink);
    line-height: 1;
    letter-spacing: -.03em;
    text-shadow: 0 2px 4px rgba(15, 14, 11, 0.1);
}

.masthead-title em {
    font-style: italic;
    background: linear-gradient(135deg, var(--red), var(--red2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.masthead-sub {
    font-family: var(--fc);
    font-size: .72rem;
    font-weight: 400;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
}

.masthead-right {
    text-align: right;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 10px;
    z-index: 1;
}

.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, var(--red), var(--red2));
    color: #fff;
    font-family: var(--fc);
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    padding: 8px 18px;
    border-radius: var(--r-md);
    box-shadow: var(--shadow-glow-red);
    animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: var(--shadow-glow-red); }
    50% { box-shadow: 0 0 60px rgba(200, 40, 30, 0.5), 0 0 100px rgba(200, 40, 30, 0.25); }
}

.live-dot-r {
    width: 8px; height: 8px;
    background: #fff;
    border-radius: 50%;
    animation: blink 1s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(255, 255, 255, 0.8);
}

@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:.3;} }

.masthead-meta {
    font-family: var(--fc);
    font-size: .64rem;
    font-weight: 500;
    letter-spacing: .1em;
    color: var(--ink3);
}

/* ── 3D NAV BAR ── */
.nav-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: var(--r-md);
    margin: 0 0 2rem;
    padding: 8px;
    box-shadow: var(--shadow-subtle);
    animation: slideUp 0.5s cubic-bezier(0.4, 0, 0.2, 1) 0.1s both;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── RULE LABEL ── */
.rule-label {
    font-family: var(--fc);
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .25em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
    animation: fadeIn 0.4s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.rule-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── 3D SIGNAL CARD ── */
.signal-board {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: var(--r-lg);
    padding: 28px 24px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    animation: cardFloat 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes cardFloat {
    from { opacity: 0; transform: translateY(20px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

.signal-board::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 5px;
    border-radius: var(--r-lg) var(--r-lg) 0 0;
}

.signal-board.green {
    box-shadow: var(--shadow-moderate), var(--shadow-glow-green);
}
.signal-board.green::before { 
    background: linear-gradient(90deg, var(--green), var(--green2)); 
}

.signal-board.amber {
    box-shadow: var(--shadow-moderate), var(--shadow-glow-amber);
}
.signal-board.amber::before { 
    background: linear-gradient(90deg, var(--amber), var(--amber2)); 
}

.signal-board.red-phase {
    box-shadow: var(--shadow-moderate), var(--shadow-glow-red);
}
.signal-board.red-phase::before { 
    background: linear-gradient(90deg, var(--red), var(--red2)); 
}

.signal-board:hover {
    transform: translateY(-4px) scale(1.01);
}

.signal-pair-label {
    font-family: var(--fc);
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .25em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-bottom: 12px;
}

.signal-verdict-text {
    font-family: var(--fd);
    font-size: 3.5rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -.03em;
    margin-bottom: 6px;
    text-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.signal-verdict-text.green { 
    background: linear-gradient(135deg, var(--green), var(--green2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.signal-verdict-text.amber { 
    background: linear-gradient(135deg, var(--amber), var(--amber2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.signal-verdict-text.red { 
    background: linear-gradient(135deg, var(--red), var(--red2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.signal-pair-dirs {
    font-family: var(--fc);
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--ink2);
    margin-bottom: 8px;
}

.signal-msg {
    font-family: var(--fb);
    font-size: .78rem;
    color: var(--ink3);
    margin-bottom: 20px;
    line-height: 1.6;
}

.timer-display {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-bottom: 14px;
}

.timer-big {
    font-family: var(--fm);
    font-size: 4.5rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -.05em;
    color: var(--ink);
    text-shadow: 0 4px 12px rgba(15, 14, 11, 0.15);
}

.timer-unit {
    font-family: var(--fc);
    font-size: .85rem;
    font-weight: 600;
    letter-spacing: .12em;
    color: var(--ink3);
    text-transform: uppercase;
}

.timer-track {
    height: 6px;
    background: rgba(15, 14, 11, 0.08);
    border-radius: 3px;
    overflow: hidden;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
}

.timer-fill {
    height: 100%;
    border-radius: 3px;
    transition: width .5s linear;
    box-shadow: 0 0 10px currentColor;
}

.timer-fill.green { 
    background: linear-gradient(90deg, var(--green), var(--green2)); 
}
.timer-fill.amber { 
    background: linear-gradient(90deg, var(--amber), var(--amber2)); 
}
.timer-fill.red { 
    background: linear-gradient(90deg, var(--red), var(--red2)); 
}

/* ── 3D INTERSECTION MAP ── */
.intersection-map {
    display: grid;
    grid-template-columns: 1fr 90px 1fr;
    grid-template-rows: 1fr 90px 1fr;
    gap: 6px;
    margin-bottom: 18px;
    height: 260px;
    perspective: 1000px;
}

.int-cell {
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: var(--r-md);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 12px 6px;
    position: relative;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow-subtle);
    transform-style: preserve-3d;
}

.int-cell:hover {
    transform: translateY(-6px) rotateX(5deg);
    box-shadow: var(--shadow-float);
}

.int-cell.active {
    background: linear-gradient(135deg, rgba(27, 158, 84, 0.15), rgba(34, 135, 63, 0.1));
    border-color: var(--green);
    box-shadow: var(--shadow-moderate), var(--shadow-glow-green);
    animation: activeGlow 2s ease-in-out infinite;
}

@keyframes activeGlow {
    0%, 100% { box-shadow: var(--shadow-moderate), var(--shadow-glow-green); }
    50% { box-shadow: var(--shadow-moderate), 0 0 50px rgba(27, 158, 84, 0.4); }
}

.int-cell.waiting {
    background: rgba(245, 242, 237, 0.6);
    border-color: var(--border);
}

.int-cell.yellow-phase {
    background: linear-gradient(135deg, rgba(229, 167, 32, 0.15), rgba(196, 122, 10, 0.1));
    border-color: var(--amber);
    box-shadow: var(--shadow-moderate), var(--shadow-glow-amber);
}

.int-dir {
    font-family: var(--fc);
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-bottom: 6px;
}

.int-count {
    font-family: var(--fm);
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    color: var(--ink);
}

.int-den {
    font-family: var(--fc);
    font-size: .6rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-top: 4px;
    padding: 2px 8px;
    border-radius: var(--r-sm);
}

.int-den.Low { 
    color: var(--green); 
    background: rgba(27, 158, 84, 0.1);
}
.int-den.Medium { 
    color: var(--amber); 
    background: rgba(229, 167, 32, 0.1);
}
.int-den.High { 
    color: var(--red); 
    background: rgba(200, 40, 30, 0.1);
}

.int-signal {
    position: absolute;
    bottom: 8px;
    font-size: .72rem;
    font-weight: 700;
    font-family: var(--fc);
    letter-spacing: .08em;
    padding: 2px 8px;
    border-radius: var(--r-sm);
}

.int-signal.green { 
    color: #fff;
    background: var(--green);
    box-shadow: 0 0 12px rgba(27, 158, 84, 0.4);
}
.int-signal.red { 
    color: #fff;
    background: var(--red);
    box-shadow: 0 0 12px rgba(200, 40, 30, 0.4);
}
.int-signal.amber { 
    color: var(--ink);
    background: var(--amber);
    box-shadow: 0 0 12px rgba(229, 167, 32, 0.4);
}

.int-center {
    background: var(--gradient-dark);
    border-radius: var(--r-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    box-shadow: var(--shadow-bold);
    border: 2px solid rgba(255,255,255,0.1);
}

.int-road-h, .int-road-v {
    background: linear-gradient(135deg, var(--ink2), var(--ink));
    border-radius: var(--r-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
}

/* ── 3D STAT ROW ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0;
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: var(--r-lg);
    overflow: hidden;
    margin-bottom: 18px;
    box-shadow: var(--shadow-moderate);
    animation: cardFloat 0.6s cubic-bezier(0.4, 0, 0.2, 1) 0.2s both;
}

.stat-cell {
    padding: 18px 14px;
    border-right: 1px solid var(--border2);
    text-align: center;
    position: relative;
    transition: all 0.3s ease;
}

.stat-cell:last-child { border-right: none; }

.stat-cell:hover {
    background: var(--glass-bg-hover);
}

.stat-cell::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
}

.stat-cell:nth-child(1)::before { background: linear-gradient(90deg, var(--red), var(--red2)); }
.stat-cell:nth-child(2)::before { background: linear-gradient(90deg, #0EA5E9, #06B6D4); }
.stat-cell:nth-child(3)::before { background: linear-gradient(90deg, var(--green), var(--green2)); }
.stat-cell:nth-child(4)::before { background: linear-gradient(90deg, var(--amber), var(--amber2)); }
.stat-cell:nth-child(5)::before { background: linear-gradient(90deg, #8B5CF6, #A78BFA); }

.stat-num {
    font-family: var(--fm);
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    color: var(--ink);
    letter-spacing: -.03em;
}

.stat-lbl {
    font-family: var(--fc);
    font-size: .62rem;
    font-weight: 600;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-top: 6px;
}

/* ── 3D FEED PANEL ── */
.feed-wrap {
    background: var(--gradient-dark);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 2px solid rgba(255, 255, 255, 0.1);
    border-radius: var(--r-lg);
    overflow: hidden;
    box-shadow: var(--shadow-bold);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    animation: cardFloat 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

.feed-wrap:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-float);
}

.feed-header {
    background: linear-gradient(135deg, rgba(200, 40, 30, 0.15), transparent);
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.feed-hl {
    font-family: var(--fc);
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 10px;
}

.feed-tag {
    font-family: var(--fc);
    font-size: .6rem;
    color: rgba(255,255,255,.5);
    letter-spacing: .1em;
}

/* ── 3D VEHICLE BREAKDOWN ── */
.vt-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 18px;
}

.vt-cell {
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: var(--r-md);
    padding: 14px 10px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow-subtle);
    cursor: default;
}

.vt-cell:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: var(--shadow-bold);
    background: var(--glass-bg-hover);
}

.vt-icon { 
    font-size: 1.5rem; 
    display: block; 
    margin-bottom: 6px;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
}

.vt-num {
    font-family: var(--fm);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1;
}

.vt-name {
    font-family: var(--fc);
    font-size: .58rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-top: 4px;
}

/* ── 3D ALERT ── */
.alert-strip {
    background: linear-gradient(135deg, var(--red), var(--red2));
    color: #fff;
    padding: 16px 24px;
    margin-bottom: 18px;
    border-radius: var(--r-md);
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: var(--shadow-glow-red);
    animation: alertPulse 1.5s ease-in-out infinite;
}

@keyframes alertPulse {
    0%, 100% { 
        box-shadow: var(--shadow-glow-red);
        transform: scale(1);
    }
    50% { 
        box-shadow: 0 0 60px rgba(200, 40, 30, 0.5), 0 0 100px rgba(200, 40, 30, 0.3);
        transform: scale(1.01);
    }
}

.alert-text {
    font-family: var(--fc);
    font-size: .85rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
}

/* ── 3D SUMMARY ── */
.summary-wrap {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: var(--r-lg);
    overflow: hidden;
    margin-bottom: 22px;
    box-shadow: var(--shadow-bold);
    animation: cardFloat 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.summary-head {
    background: var(--gradient-dark);
    padding: 14px 24px;
    font-family: var(--fc);
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: #fff;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.summary-body {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
}

.summary-item {
    padding: 22px 18px;
    border-right: 1px solid var(--border2);
    text-align: center;
    transition: background 0.3s ease;
}

.summary-item:last-child { border-right: none; }

.summary-item:hover {
    background: var(--glass-bg-hover);
}

.summary-num {
    font-family: var(--fm);
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -.03em;
}

.summary-lbl {
    font-family: var(--fc);
    font-size: .62rem;
    font-weight: 600;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-top: 6px;
}

/* ── 3D ANALYTICS CARDS ── */
.ov-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}

.ov-cell {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: var(--r-lg);
    padding: 24px 20px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow-moderate);
    position: relative;
    overflow: hidden;
}

.ov-cell::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 4px; height: 100%;
    border-radius: var(--r-lg) 0 0 var(--r-lg);
}

.ov-cell:nth-child(1)::before { background: linear-gradient(180deg, var(--red), var(--red2)); }
.ov-cell:nth-child(2)::before { background: linear-gradient(180deg, var(--green), var(--green2)); }
.ov-cell:nth-child(3)::before { background: linear-gradient(180deg, var(--amber), var(--amber2)); }
.ov-cell:nth-child(4)::before { background: linear-gradient(180deg, #8B5CF6, #A78BFA); }

.ov-cell:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: var(--shadow-bold);
}

.ov-icon { 
    font-size: 1.6rem; 
    margin-bottom: 10px; 
    display: block;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
}

.ov-num {
    font-family: var(--fm);
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1;
    letter-spacing: -.03em;
}

.ov-lbl {
    font-family: var(--fc);
    font-size: .62rem;
    font-weight: 600;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-top: 8px;
}

/* ── 3D DB STATS ── */
.db-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 22px;
}

.db-cell {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: var(--r-md);
    padding: 18px 16px;
    text-align: center;
    box-shadow: var(--shadow-subtle);
    transition: all 0.3s ease;
}

.db-cell:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-moderate);
}

.db-num {
    font-family: var(--fm);
    font-size: 2rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1;
}

.db-lbl {
    font-family: var(--fc);
    font-size: .6rem;
    font-weight: 600;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-top: 6px;
}

/* ── 3D ABOUT PAGE ── */
.about-masthead {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: var(--r-lg);
    padding: 36px;
    margin-bottom: 32px;
    box-shadow: var(--shadow-bold);
    position: relative;
    overflow: hidden;
    animation: cardFloat 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.about-masthead::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 6px; height: 100%;
    background: linear-gradient(180deg, var(--red), var(--amber), var(--green));
}

.about-vol {
    font-family: var(--fc);
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: .3em;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 12px;
}

.about-hed {
    font-family: var(--fd);
    font-size: 3.5rem;
    font-weight: 900;
    color: var(--ink);
    line-height: 1.05;
    letter-spacing: -.03em;
    margin-bottom: 18px;
}

.about-hed em { 
    font-style: italic; 
    background: linear-gradient(135deg, var(--red), var(--red2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.about-dek {
    font-family: var(--fb);
    font-size: .95rem;
    color: var(--ink2);
    line-height: 1.9;
    max-width: 680px;
    border-left: 3px solid var(--red);
    padding-left: 20px;
    margin-left: 20px;
}

.info-3col {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 26px;
}

.info-col {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: var(--r-lg);
    padding: 24px 20px;
    box-shadow: var(--shadow-moderate);
    transition: all 0.3s ease;
}

.info-col:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-bold);
}

.info-col-title {
    font-family: var(--fc);
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: .25em;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}

.info-kv {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border2);
    transition: background 0.2s ease;
}

.info-kv:last-child { border-bottom: none; }

.info-kv:hover {
    background: rgba(200, 40, 30, 0.03);
    margin: 0 -10px;
    padding: 8px 10px;
    border-radius: var(--r-sm);
}

.info-k {
    font-family: var(--fc);
    font-size: .72rem;
    font-weight: 400;
    color: var(--ink3);
    letter-spacing: .05em;
}

.info-v {
    font-family: var(--fm);
    font-size: .72rem;
    font-weight: 600;
    color: var(--ink);
    text-align: right;
}

.tech-8 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}

.tech-item {
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: var(--r-md);
    padding: 16px 12px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow-subtle);
    cursor: default;
}

.tech-item:hover {
    transform: translateY(-6px) rotateX(5deg);
    box-shadow: var(--shadow-bold);
    background: var(--glass-bg-hover);
}

.tech-ico { 
    font-size: 1.6rem; 
    display: block; 
    margin-bottom: 8px;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
}

.tech-nm {
    font-family: var(--fc);
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--ink2);
}

.dev-strip {
    display: flex;
    align-items: center;
    gap: 28px;
    background: var(--gradient-dark);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: var(--r-lg);
    padding: 28px 32px;
    margin-top: 26px;
    box-shadow: var(--shadow-bold);
    transition: all 0.4s ease;
}

.dev-strip:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-float);
}

.dev-av {
    width: 72px; height: 72px;
    background: linear-gradient(135deg, var(--red), var(--red2));
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    flex-shrink: 0;
    box-shadow: var(--shadow-glow-red);
}

.dev-name {
    font-family: var(--fd);
    font-size: 1.5rem;
    font-weight: 900;
    color: #fff;
    margin-bottom: 6px;
}

.dev-role { 
    font-family: var(--fc); 
    font-size: .75rem; 
    color: rgba(255,255,255,.6); 
    margin-bottom: 12px; 
    letter-spacing:.08em;
}

.dev-tag {
    display: inline-block;
    font-family: var(--fc);
    font-size: .6rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    padding: 5px 12px;
    border: 1px solid rgba(255,255,255,.2);
    border-radius: var(--r-sm);
    color: rgba(255,255,255,.8);
    margin-right: 8px;
    margin-bottom: 6px;
    background: rgba(255,255,255,0.05);
    transition: all 0.3s ease;
}

.dev-tag:hover {
    background: rgba(255,255,255,0.1);
    border-color: rgba(255,255,255,.4);
}

/* ── 3D UPLOAD / IDLE ── */
.idle-box {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 2px dashed var(--border);
    border-radius: var(--r-lg);
    padding: 80px 50px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow-subtle);
}

.idle-box:hover { 
    border-color: var(--red);
    transform: translateY(-4px);
    box-shadow: var(--shadow-moderate);
}

.idle-icon { 
    font-size: 3rem; 
    margin-bottom: 16px; 
    opacity: .5; 
    display: block;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));
}

.idle-title {
    font-family: var(--fd);
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--ink2);
    margin-bottom: 10px;
}

.idle-sub { 
    font-size: .8rem; 
    color: var(--ink3); 
    line-height: 1.9; 
}

.idle-sub b { 
    background: linear-gradient(135deg, var(--red), var(--red2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── ENHANCED TABLE ── */
.stDataFrame { border: none !important; }
.stDataFrame table { 
    font-family: var(--fb) !important; 
    font-size: .78rem !important;
    border-radius: var(--r-md) !important;
    overflow: hidden !important;
}
.stDataFrame thead th {
    background: var(--gradient-dark) !important;
    color: #fff !important;
    font-size: .62rem !important;
    letter-spacing: .14em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    padding: 14px 16px !important;
    font-family: var(--fc) !important;
}
.stDataFrame tbody td {
    padding: 10px 16px !important;
    border-bottom: 1px solid var(--border2) !important;
    color: var(--ink2) !important;
    background: var(--glass-bg) !important;
    font-family: var(--fm) !important;
}
.stDataFrame tbody tr:hover td { 
    background: var(--glass-bg-hover) !important; 
}

/* ── 3D SIDEBAR ── */
section[data-testid=\"stSidebar\"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 22rem !important;
    min-width: 22rem !important;
    max-width: 22rem !important;
    overflow: visible !important;
    transform: none !important;
    background: var(--glass-bg) !important;
    backdrop-filter: blur(var(--glass-blur)) !important;
    -webkit-backdrop-filter: blur(var(--glass-blur)) !important;
    border-right: 1px solid var(--glass-border) !important;
    box-shadow: var(--shadow-moderate) !important;
}

[data-testid=\"stSidebar\"] .block-container { padding: 2rem 1.5rem !important; }

.sb-masthead {
    text-align: center;
    padding: 0 0 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
    position: relative;
}

.sb-masthead::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, var(--red), var(--red2));
    border-radius: 2px;
}

.sb-flag {
    font-family: var(--fc);
    font-size: .6rem;
    font-weight: 700;
    letter-spacing: .25em;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 6px;
}

.sb-name {
    font-family: var(--fd);
    font-size: 1.2rem;
    font-weight: 900;
    color: var(--ink);
    letter-spacing: -.02em;
}

.sb-sub {
    font-family: var(--fc);
    font-size: .6rem;
    letter-spacing: .12em;
    color: var(--ink3);
    margin-top: 4px;
}

.sb-sec {
    font-family: var(--fc);
    font-size: .6rem;
    font-weight: 700;
    letter-spacing: .25em;
    text-transform: uppercase;
    color: var(--ink3);
    margin: 18px 0 10px;
    display: flex;
    align-items: center;
    gap: 10px;
    padding-top: 14px;
    border-top: 1px solid var(--border);
}

.sb-info {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: var(--r-md);
    padding: 16px 18px;
    margin-top: 16px;
    font-size: .65rem;
    color: var(--ink3);
    line-height: 2;
    font-family: var(--fb);
    box-shadow: var(--shadow-subtle);
}

.sb-info b { color: var(--ink2); }

.sb-dev {
    margin-top: 14px;
    text-align: center;
    font-size: .6rem;
    color: var(--ink4);
    font-family: var(--fc);
    letter-spacing: .08em;
    line-height: 1.9;
}

/* ── ENHANCED WIDGETS ── */
.stSlider>label, .stNumberInput>label, .stCheckbox>label,
.stSelectbox>label, .stRadio>label {
    font-family: var(--fb) !important;
    font-size: .78rem !important;
    color: var(--ink2) !important;
    font-weight: 500 !important;
}

.stButton button {
    font-family: var(--fc) !important;
    font-weight: 700 !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    border-radius: var(--r-md) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: var(--shadow-subtle) !important;
}

.stButton button[kind=\"primary\"] {
    background: linear-gradient(135deg, var(--ink), var(--ink2)) !important;
    color: #fff !important;
    border: none !important;
}

.stButton button[kind=\"primary\"]:hover {
    background: linear-gradient(135deg, var(--red), var(--red2)) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-bold) !important;
}

.stButton button[kind=\"secondary\"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--ink) !important;
}

.stButton button[kind=\"secondary\"]:hover {
    background: var(--glass-bg-hover) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-moderate) !important;
}

.stTextInput input {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--r-md) !important;
    color: var(--ink) !important;
    font-family: var(--fb) !important;
    padding: 12px 16px !important;
    transition: all 0.3s ease !important;
}

.stTextInput input:focus {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 3px rgba(200, 40, 30, 0.1) !important;
}

/* ── 3D FOOTER ── */
.footer-bar {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: var(--r-lg);
    margin-top: 3.5rem;
    padding: 20px 28px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    box-shadow: var(--shadow-moderate);
}

.footer-l { 
    font-family: var(--fc); 
    font-size: .65rem; 
    color: var(--ink3); 
    line-height: 1.9; 
}

.footer-l b { color: var(--ink); }

.footer-r { 
    font-family: var(--fc); 
    font-size: .62rem; 
    color: var(--ink4); 
    text-align: right; 
    line-height: 1.9; 
}

.footer-r span { 
    background: linear-gradient(135deg, var(--red), var(--red2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700; 
}

/* ── MOBILE RESPONSIVE ── */
@media (max-width: 768px) {
    .stat-row { grid-template-columns: repeat(3, 1fr); }
    .ov-row   { grid-template-columns: repeat(2, 1fr); }
    .info-3col { grid-template-columns: 1fr; }
    .tech-8   { grid-template-columns: repeat(2, 1fr); }
    .summary-body { grid-template-columns: 1fr; }
    .masthead { flex-direction: column; align-items: flex-start; gap: 14px; }
    .intersection-map { height: 200px; }
    .dev-strip { flex-direction: column; text-align: center; }
    .db-row   { grid-template-columns: repeat(2, 1fr); }
    .masthead-title { font-size: 2rem; }
    .about-hed { font-size: 2.5rem; }
}

/* ── LOGIN PAGE SPECIFIC ── */
.login-container {
    max-width: 480px;
    margin: 60px auto 0;
    padding: 0 24px;
    animation: fadeInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.login-card {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: var(--r-xl);
    padding: 48px 40px;
    box-shadow: var(--shadow-bold);
    position: relative;
    overflow: hidden;
}

.login-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 5px;
    background: linear-gradient(90deg, var(--red), var(--amber), var(--green));
}

.login-brand {
    text-align: center;
    margin-bottom: 36px;
}

.login-flag {
    font-family: var(--fc);
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .3em;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 8px;
}

.login-title {
    font-family: var(--fd);
    font-size: 2.8rem;
    font-weight: 900;
    color: var(--ink);
    line-height: 1.1;
    margin-bottom: 8px;
}

.login-title em {
    font-style: italic;
    background: linear-gradient(135deg, var(--red), var(--red2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.login-sub {
    font-family: var(--fb);
    font-size: .85rem;
    color: var(--ink3);
    line-height: 1.7;
    border-left: 3px solid var(--red);
    padding-left: 14px;
    margin: 20px auto 0;
    max-width: 320px;
    text-align: left;
}

.login-hint {
    margin-top: 20px;
    font-family: var(--fc);
    font-size: .68rem;
    color: var(--ink4);
    letter-spacing: .08em;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    text-align: center;
}

/* Staggered entrance animations */
.stagger-1 { animation-delay: 0.1s; }
.stagger-2 { animation-delay: 0.2s; }
.stagger-3 { animation-delay: 0.3s; }
.stagger-4 { animation-delay: 0.4s; }
.stagger-5 { animation-delay: 0.5s; }
</style>
\"\"\", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PART 7 — LOGIN PAGE
# ════════════════════════════════════════════════════════════════════

def show_login():
    # IMPORTANT: Never use display:none on sidebar — it persists after login
    # Use width:0 + overflow:hidden instead which resets on rerun
    st.markdown(\"\"\"
    <style>
    section[data-testid=\"stSidebar\"]{
        width:0px !important;min-width:0px !important;overflow:hidden !important;
    }
    </style>
    \"\"\", unsafe_allow_html=True)

    st.markdown(\"\"\"
    <div class=\"login-container\">
        <div class=\"login-card\">
            <div class=\"login-brand\">
                <div class=\"login-flag\">K.R. Mangalam University</div>
                <div class=\"login-title\">Traffic<br><em>Vision</em></div>
                <div class=\"login-sub\">
                    4-Way Intersection Intelligence System.<br>
                    Real-time congestion analysis for Indian roads.
                </div>
            </div>
        </div>
    </div>
    \"\"\", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        with st.form(\"lf\"):
            username = st.text_input(\"Username\", placeholder=\"Enter username\")
            password = st.text_input(\"Password\", placeholder=\"Enter password\",
                                     type=\"password\")
            st.markdown(\"<br>\", unsafe_allow_html=True)
            sub = st.form_submit_button(\"Sign In\",
                                        use_container_width=True,
                                        type=\"primary\")
            if sub:
                role = check_login(username.strip(), password)
                if role:
                    st.session_state.logged_in = True
                    st.session_state.username  = username.strip()
                    st.session_state.role      = role
                    st.rerun()
                else:
                    st.error(\"Invalid credentials\")

        st.markdown(\"\"\"
        <div class=\"login-hint\">
            admin / krmu2025 &nbsp;&middot;&nbsp; sarthak / pass123
        </div>
        \"\"\", unsafe_allow_html=True)

if not st.session_state.logged_in:
    show_login()
    st.stop()


# ════════════════════════════════════════════════════════════════════
# PART 8 — MASTHEAD
# ════════════════════════════════════════════════════════════════════

st.markdown(f\"\"\"
<div class=\"masthead\">
    <div class=\"masthead-left\">
        <div class=\"masthead-flag\">K.R. Mangalam University &nbsp;&middot;&nbsp; School of Engineering & Technology</div>
        <div class=\"masthead-title\">IoT <em>Traffic</em> Vision</div>
        <div class=\"masthead-sub\">
            4-Way Intersection Intelligence &nbsp;&middot;&nbsp; Dynamic Signal Control
            &nbsp;&middot;&nbsp; YOLOv8 Neural Detection &nbsp;&middot;&nbsp; Indian Road System
        </div>
    </div>
    <div class=\"masthead-right\">
        <div class=\"live-badge\"><div class=\"live-dot-r\"></div>Live System</div>
        <div class=\"masthead-meta\">
            {st.session_state.username} ({st.session_state.role})
            &nbsp;&middot;&nbsp; B.Tech CSE &nbsp;&middot;&nbsp; 2025-26
        </div>
    </div>
</div>
\"\"\", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PART 9 — NAV
# ════════════════════════════════════════════════════════════════════

tabs_list = [\"🎯  Live Monitor\", \"📊  Analytics\", \"🗄️  Database\", \"ℹ️  About\"]
st.markdown('<div class=\"nav-bar\">', unsafe_allow_html=True)
nc = st.columns(len(tabs_list))
for i, tab in enumerate(tabs_list):
    with nc[i]:
        is_a = st.session_state.active_tab == tab
        if st.button(tab, key=f\"t{i}\", use_container_width=True,
                     type=\"primary\" if is_a else \"secondary\"):
            st.session_state.active_tab = tab
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
active = st.session_state.active_tab


# ════════════════════════════════════════════════════════════════════
# PART 10 — SIDEBAR
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(\"\"\"
    <div class=\"sb-masthead\">
        <div class=\"sb-flag\">TrafficVision</div>
        <div class=\"sb-name\">Control Panel</div>
        <div class=\"sb-sub\">IoT Intelligence Dashboard</div>
    </div>
    \"\"\", unsafe_allow_html=True)

    st.markdown('<div class=\"sb-sec\">Input Mode</div>', unsafe_allow_html=True)
    input_mode = st.radio(\"Source\", [\"📁 Upload Video\", \"📸 Camera Capture\"],
                          label_visibility=\"collapsed\")

    default_video = \"traffic.mp4\"
    uploaded_file = None
    camera_photo  = None

    if input_mode == \"📁 Upload Video\":
        uploaded_file = st.file_uploader(\"Upload Traffic Video\",
                                          type=[\"mp4\", \"avi\", \"mov\"])
        if os.path.exists(default_video) and uploaded_file is None:
            st.success(\"✅ traffic.mp4 detected\")
        video_ready = uploaded_file is not None or os.path.exists(default_video)
        if not video_ready:
            st.warning(\"Upload a video to begin\")
    else:
        st.info(\"📱 Back camera · Works on phone + deployed\")
        st.markdown(\"\"\"<script>
        function bCam(){const vs=document.querySelectorAll('video');
        vs.forEach(v=>{if(v.srcObject)v.srcObject.getTracks().forEach(t=>t.stop());});
        navigator.mediaDevices.getUserMedia({video:{facingMode:{exact:\"environment\"}})
        .then(s=>{document.querySelectorAll('video').forEach(v=>v.srcObject=s);})
        .catch(()=>navigator.mediaDevices.getUserMedia({video:{facingMode:\"environment\"})
        .then(s=>{document.querySelectorAll('video').forEach(v=>v.srcObject=s);}));}
        setTimeout(bCam,1000);setTimeout(bCam,2500);
        </script>\"\"\", unsafe_allow_html=True)
        camera_photo = st.camera_input(\"📷 Capture intersection\")
        video_ready  = camera_photo is not None

    st.markdown('<div class=\"sb-sec\">Detection Settings</div>', unsafe_allow_html=True)
    conf_val    = st.slider(\"Confidence Threshold\", 0.1, 0.9, 0.4, 0.05)
    pub_every   = st.slider(\"Save to DB every N frames\", 1, 30, 10)
    skip_frames = st.slider(\"Frame skip (speed)\", 1, 5, 2)

    st.markdown('<div class=\"sb-sec\">Controls</div>', unsafe_allow_html=True)
    if input_mode == \"📸 Camera Capture\":
        st.info(\"Capture above to analyse\")
        start = False
        stop  = st.button(\"🔄 Reset Results\", use_container_width=True)
    else:
        start = st.button(\"▶ Start Analysis\", use_container_width=True,
                          type=\"primary\", disabled=not video_ready)
        stop  = st.button(\"⏹ Stop\",          use_container_width=True)

    st.markdown('<div class=\"sb-sec\">Session</div>', unsafe_allow_html=True)
    if st.button(\"🚪 Logout\", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown(\"\"\"
    <div class=\"sb-info\">
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
    <div class=\"sb-dev\">
        Developed by<br>
        <b style=\"color:#c8281e;font-size:.72rem;\">Sarthak Mishra</b><br>
        B.Tech CSE · KRM University<br>
        <span style=\"color:#d4d0c8;\">uid·2301010232</span>
    </div>
    \"\"\", unsafe_allow_html=True)

# Controls
if start:
    st.session_state.running          = True
    st.session_state.show_summary     = False
    st.session_state.session_start    = datetime.now().strftime(\"%H:%M:%S\")
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

if active == \"🎯  Live Monitor\":

    alert_ph = st.empty()

    # Session summary
    if st.session_state.show_summary and st.session_state.total_frames > 0:
        st.markdown(f\"\"\"
        <div class=\"summary-wrap\">
            <div class=\"summary-head\">Session Complete — Summary Report</div>
            <div class=\"summary-body\">
                <div class=\"summary-item\">
                    <div class=\"summary-num\">{st.session_state.total_frames}</div>
                    <div class=\"summary-lbl\">Frames Processed</div>
                </div>
                <div class=\"summary-item\">
                    <div class=\"summary-num\">{st.session_state.session_vehicles}</div>
                    <div class=\"summary-lbl\">Vehicles Detected</div>
                </div>
                <div class=\"summary-item\">
                    <div class=\"summary-num\">{st.session_state.session_high}</div>
                    <div class=\"summary-lbl\">High Density Events</div>
                </div>
                <div class=\"summary-item\">
                    <div class=\"summary-num\">{get_total_records()}</div>
                    <div class=\"summary-lbl\">DB Records Saved</div>
                </div>
                <div class=\"summary-item\">
                    <div class=\"summary-num\" style=\"font-size:1rem;\">{st.session_state.session_start or '—'}</div>
                    <div class=\"summary-lbl\">Started</div>
                </div>
                <div class=\"summary-item\">
                    <div class=\"summary-num\" style=\"font-size:1rem;\">{datetime.now().strftime(\"%H:%M:%S\")}</div>
                    <div class=\"summary-lbl\">Ended</div>
                </div>
            </div>
        </div>
        \"\"\", unsafe_allow_html=True)

    feed_col, ctrl_col = st.columns([3, 2], gap=\"large\")

    with feed_col:
        st.markdown('<div class=\"rule-label\">Live Detection Feed — 4-Way View</div>',
                    unsafe_allow_html=True)
        st.markdown(\"\"\"
        <div class=\"feed-wrap\">
            <div class=\"feed-header\">
                <div class=\"feed-hl\">
                    <div class=\"live-dot-r\" style=\"background:#ef4444;\"></div>
                    Live · 4-Way Intersection Processing
                </div>
                <div class=\"feed-tag\">YOLOv8n · N / S / E / W Detection</div>
            </div>
        </div>
        \"\"\", unsafe_allow_html=True)
        frame_ph = st.empty()
        sc1, sc2 = st.columns([1, 3])
        with sc1:
            snap_btn = st.button(\"📸 Snapshot\", use_container_width=True)
        snap_msg = st.empty()

    with ctrl_col:
        st.markdown('<div class=\"rule-label\">Signal Verdict</div>',
                    unsafe_allow_html=True)
        verdict_ph = st.empty()

        st.markdown('<div class=\"rule-label\">Intersection Map</div>',
                    unsafe_allow_html=True)
        map_ph = st.empty()

        st.markdown('<div class=\"rule-label\">Vehicle Breakdown</div>',
                    unsafe_allow_html=True)
        vtype_ph = st.empty()

    st.markdown('<div class=\"rule-label\">Session Statistics</div>',
                unsafe_allow_html=True)
    stats_ph = st.empty()

    st.markdown(\"---\")
    st.markdown('<div class=\"rule-label\">Live Charts</div>',
                unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)
    with ch1:
        st.caption(\"Total Vehicles — Last 40 Readings\")
        chart1_ph = st.empty()
    with ch2:
        st.caption(\"Overall Density — Last 40 Readings\")
        chart2_ph = st.empty()

    # ── SIGNAL TIMER LOGIC ────────────────────────────────────────────
    def update_timer(new_gp, new_dur):
        now     = time.time()
        elapsed = now - st.session_state.sig_last_t

        if st.session_state.sig_phase == \"YELLOW\":
            # Counting down yellow (5s)
            rem = st.session_state.sig_remaining - elapsed
            if rem <= 0:
                # Switch to other pair GREEN
                other = \"EW\" if st.session_state.sig_verdict == \"NS\" else \"NS\"
                st.session_state.sig_verdict   = other
                st.session_state.sig_phase     = \"GREEN\"
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
            st.session_state.sig_phase     = \"YELLOW\"
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

        if phase == \"YELLOW\":
            v_cls = \"amber\"
            v_txt = \"YELLOW\"
            dirs  = \"↕ N·S &nbsp;&middot;&nbsp; ↔ E·W\"
            msg   = \"Signal changing — prepare to stop\"
            icon  = \"🟡\"
        elif gp == \"NS\":
            v_cls = \"green\"
            v_txt = \"GREEN\"
            dirs  = \"↕ NORTH · SOUTH\"
            msg   = \"North–South corridor open · East–West waiting\"
            icon  = \"🟢\"
        else:
            v_cls = \"green\"
            v_txt = \"GREEN\"
            dirs  = \"↔ EAST · WEST\"
            msg   = \"East–West corridor open · North–South waiting\"
            icon  = \"🟢\"

        pct = (rem / max(dur, 1)) * 100

        verdict_ph.markdown(f\"\"\"
        <div class=\"signal-board {v_cls}\">
            <div class=\"signal-pair-label\">Active Signal Pair</div>
            <div class=\"signal-verdict-text {v_cls}\">{v_txt}</div>
            <div class=\"signal-pair-dirs\">{dirs}</div>
            <div class=\"signal-msg\">{msg}</div>
            <div class=\"timer-display\">
                <div class=\"timer-big\">{rem:02d}</div>
                <div class=\"timer-unit\">sec</div>
            </div>
            <div class=\"timer-track\">
                <div class=\"timer-fill {v_cls}\" style=\"width:{pct:.1f}%\"></div>
            </div>
        </div>
        \"\"\", unsafe_allow_html=True)

    def render_map(nc, sc, ec, wc, gp, phase):
        nd = get_density(nc)
        sd = get_density(sc)
        ed = get_density(ec)
        wd = get_density(wc)

        def sig_label(direction):
            if phase == \"YELLOW\":
                return '<span class=\"int-signal amber\">●YLW</span>'
            if (direction in [\"N\",\"S\"] and gp == \"NS\") or \
               (direction in [\"E\",\"W\"] and gp == \"EW\"):
                return '<span class=\"int-signal green\">●GRN</span>'
            return '<span class=\"int-signal red\">●RED</span>'

        def active_cls(direction):
            if phase == \"YELLOW\": return \"yellow-phase\"
            if (direction in [\"N\",\"S\"] and gp == \"NS\") or \
               (direction in [\"E\",\"W\"] and gp == \"EW\"):
                return \"active\"
            return \"waiting\"

        map_ph.markdown(f\"\"\"
        <div class=\"intersection-map\">
            <div class=\"int-cell {active_cls('N')}\">
                <div class=\"int-dir\">North</div>
                <div class=\"int-count\">{nc}</div>
                <div class=\"int-den {nd}\">{nd}</div>
                {sig_label('N')}
            </div>
            <div class=\"int-road-v\"></div>
            <div class=\"int-cell {active_cls('N')}\">
                <div class=\"int-dir\">North ↓</div>
                <div class=\"int-count\">{nc}</div>
                <div class=\"int-den {nd}\">{nd}</div>
            </div>
            <div class=\"int-road-h\"></div>
            <div class=\"int-center\">🚦</div>
            <div class=\"int-road-h\"></div>
            <div class=\"int-cell {active_cls('W')}\">
                <div class=\"int-dir\">West</div>
                <div class=\"int-count\">{wc}</div>
                <div class=\"int-den {wd}\">{wd}</div>
                {sig_label('W')}
            </div>
            <div class=\"int-road-v\"></div>
            <div class=\"int-cell {active_cls('E')}\">
                <div class=\"int-dir\">East</div>
                <div class=\"int-count\">{ec}</div>
                <div class=\"int-den {ed}\">{ed}</div>
                {sig_label('E')}
            </div>
            <div class=\"int-cell {active_cls('S')}\">
                <div class=\"int-dir\">South ↑</div>
                <div class=\"int-count\">{sc}</div>
                <div class=\"int-den {sd}\">{sd}</div>
            </div>
            <div class=\"int-road-v\"></div>
            <div class=\"int-cell {active_cls('S')}\">
                <div class=\"int-dir\">South</div>
                <div class=\"int-count\">{sc}</div>
                <div class=\"int-den {sd}\">{sd}</div>
                {sig_label('S')}
            </div>
        </div>
        \"\"\", unsafe_allow_html=True)

    def render_vtypes(vtypes):
        icons = {\"Car\": \"🚗\", \"Motorcycle\": \"🏍️\", \"Bus\": \"🚌\", \"Truck\": \"🚛\"}
        html  = '<div class=\"vt-row\">'
        for vt, icon in icons.items():
            html += f\"\"\"
            <div class=\"vt-cell\">
                <span class=\"vt-icon\">{icon}</span>
                <div class=\"vt-num\">{vtypes.get(vt, 0)}</div>
                <div class=\"vt-name\">{vt}</div>
            </div>\"\"\"
        html += \"</div>\"
        vtype_ph.markdown(html, unsafe_allow_html=True)

    def render_stats(nc, sc, ec, wc, total, den):
        stats_ph.markdown(f\"\"\"
        <div class=\"stat-row\">
            <div class=\"stat-cell\">
                <div class=\"stat-num\">{total}</div>
                <div class=\"stat-lbl\">Total Vehicles</div>
            </div>
            <div class=\"stat-cell\">
                <div class=\"stat-num\">{nc}</div>
                <div class=\"stat-lbl\">North</div>
            </div>
            <div class=\"stat-cell\">
                <div class=\"stat-num\">{sc}</div>
                <div class=\"stat-lbl\">South</div>
            </div>
            <div class=\"stat-cell\">
                <div class=\"stat-num\">{ec}</div>
                <div class=\"stat-lbl\">East</div>
            </div>
            <div class=\"stat-cell\">
                <div class=\"stat-num\">{wc}</div>
                <div class=\"stat-lbl\">West</div>
            </div>
        </div>
        \"\"\", unsafe_allow_html=True)

    def render_charts():
        if len(st.session_state.total_hist) < 2:
            return
        df_c = pd.DataFrame({
            \"Frame\": list(range(len(st.session_state.total_hist))),
            \"Total\": list(st.session_state.total_hist),
        })
        chart1_ph.line_chart(df_c.set_index(\"Frame\"),
                             color=[\"#c8281e\"], height=150,
                             use_container_width=True)
        df_d = pd.DataFrame({
            \"Frame\": list(range(len(st.session_state.den_hist))),
            \"Density\": [DENSITY_NUM[d] for d in st.session_state.den_hist],
        })
        chart2_ph.line_chart(df_d.set_index(\"Frame\"),
                             color=[\"#1b9e54\"], height=150,
                             use_container_width=True)

    def show_alert(den):
        if den == \"High\":
            alert_ph.markdown(\"\"\"
            <div class=\"alert-strip\">
                <span style=\"font-size:1.4rem;\">🚨</span>
                <div>
                    <div class=\"alert-text\">High Congestion — Intersection Critical</div>
                    <div style=\"font-size:.72rem;opacity:.8;margin-top:3px;\">
                        All directions heavily congested · Signal duration extended
                    </div>
                </div>
            </div>\"\"\", unsafe_allow_html=True)
        else:
            alert_ph.empty()

    # ── VIDEO LOOP ────────────────────────────────────────────────────
    if st.session_state.running and input_mode == \"📁 Upload Video\":

        if uploaded_file is not None:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=\".mp4\")
            tf.write(uploaded_file.read())
            vpath = tf.name
        else:
            vpath = default_video

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            st.error(\"❌ Cannot open video. Please re-upload.\")
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
                if oden == \"High\":
                    st.session_state.session_high += 1

                st.session_state.total_hist.append(total)
                st.session_state.den_hist.append(oden)

                # Update timer
                update_timer(gp, dur)

                # Render
                frame_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                               channels=\"RGB\", use_container_width=True)
                if snap_btn and st.session_state.last_frame is not None:
                    fname = f\"snapshot_{datetime.now().strftime('%H%M%S')}.jpg\"
                    cv2.imwrite(fname, st.session_state.last_frame)
                    snap_msg.success(f\"📸 Saved: {fname}\")

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
    elif input_mode == \"📸 Camera Capture\":
        if camera_photo is not None:
            img   = Image.open(io.BytesIO(camera_photo.getvalue()))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            (ann, nc, sc, ec, wc, total,
             gp, gden, dur, oden,
             avg_conf, vtypes) = detect_4way(frame, conf_val)

            frame_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                           channels=\"RGB\", use_container_width=True)
            update_timer(gp, dur)
            show_alert(oden)
            render_verdict()
            render_map(nc, sc, ec, wc, st.session_state.sig_verdict,
                       st.session_state.sig_phase)
            render_vtypes(vtypes)
            render_stats(nc, sc, ec, wc, total, oden)
            publish_iot(nc, sc, ec, wc, total, gp,
                        st.session_state.sig_phase, dur, oden)
            st.success(\"✅ Analysed! Capture again to update.\")
        else:
            frame_ph.markdown(\"\"\"
            <div class=\"idle-box\">
                <span class=\"idle-icon\">📸</span>
                <div class=\"idle-title\">Camera Capture Mode</div>
                <div class=\"idle-sub\">
                    Click camera above to capture traffic<br>
                    Works on <b>phone browser</b> and deployed link
                </div>
            </div>\"\"\", unsafe_allow_html=True)
            render_verdict()
            render_map(st.session_state.last_nc, st.session_state.last_sc,
                       st.session_state.last_ec, st.session_state.last_wc,
                       st.session_state.last_gp, st.session_state.sig_phase)
            render_vtypes(st.session_state.last_vtypes)
    else:
        frame_ph.markdown(\"\"\"
        <div class=\"idle-box\">
            <span class=\"idle-icon\">🎬</span>
            <div class=\"idle-title\">Select Input Mode</div>
            <div class=\"idle-sub\">
                Choose <b>Upload Video</b> or <b>Camera Capture</b><br>
                Free videos → <b>pexels.com</b> → search \"highway traffic\"
            </div>
        </div>\"\"\", unsafe_allow_html=True)
        render_verdict()
        render_map(0, 0, 0, 0, \"NS\", \"GREEN\")
        render_vtypes({\"Car\":0,\"Motorcycle\":0,\"Bus\":0,\"Truck\":0})
        render_stats(0, 0, 0, 0, 0, \"Low\")
        render_charts()


# ════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ════════════════════════════════════════════════════════════════════

elif active == \"📊  Analytics\":
    st.markdown('<div class=\"rule-label\">Traffic Analytics</div>',
                unsafe_allow_html=True)
    df_all = get_db_stats()

    if df_all.empty:
        st.info(\"No data yet. Run Live Monitor first.\")
    else:
        total  = len(df_all)
        avg_v  = round(df_all[\"total_count\"].mean(), 1) if \"total_count\" in df_all else 0
        high_c = len(df_all[df_all[\"overall_density\"]==\"High\"]) if \"overall_density\" in df_all else 0
        green  = len(df_all[df_all[\"signal_phase\"]==\"GREEN\"]) if \"signal_phase\" in df_all else 0

        st.markdown(f\"\"\"
        <div class=\"ov-row\">
            <div class=\"ov-cell\">
                <span class=\"ov-icon\">🗄️</span>
                <div class=\"ov-num\">{total}</div>
                <div class=\"ov-lbl\">Total Records</div>
            </div>
            <div class=\"ov-cell\">
                <span class=\"ov-icon\">🚗</span>
                <div class=\"ov-num\">{avg_v}</div>
                <div class=\"ov-lbl\">Avg Vehicles</div>
            </div>
            <div class=\"ov-cell\">
                <span class=\"ov-icon\">🚨</span>
                <div class=\"ov-num\">{high_c}</div>
                <div class=\"ov-lbl\">High Density</div>
            </div>
            <div class=\"ov-cell\">
                <span class=\"ov-icon\">🟢</span>
                <div class=\"ov-num\">{green}</div>
                <div class=\"ov-lbl\">GREEN Phases</div>
            </div>
        </div>
        \"\"\", unsafe_allow_html=True)

        if \"total_count\" in df_all.columns:
            st.markdown('<div class=\"rule-label\">Total Vehicle Count Over Time</div>',
                        unsafe_allow_html=True)
            st.line_chart(df_all[[\"id\",\"total_count\"]].set_index(\"id\"),
                          color=[\"#c8281e\"], height=200, use_container_width=True)

        cols = [\"north_count\",\"south_count\",\"east_count\",\"west_count\"]
        if all(c in df_all.columns for c in cols):
            st.markdown('<div class=\"rule-label\">Per Direction — N / S / E / W</div>',
                        unsafe_allow_html=True)
            df_d = df_all[[\"id\"]+cols].set_index(\"id\")
            df_d.columns = [\"North\",\"South\",\"East\",\"West\"]
            st.line_chart(df_d, color=[\"#c8281e\",\"#1b9e54\",\"#e5a720\",\"#1a3a6b\"],
                          height=200, use_container_width=True)

        if \"green_pair\" in df_all.columns:
            st.markdown('<div class=\"rule-label\">Signal Pair Distribution</div>',
                        unsafe_allow_html=True)
            a1, a2 = st.columns(2, gap=\"medium\")
            with a1:
                st.caption(\"Green Pair (NS vs EW)\")
                st.bar_chart(df_all[\"green_pair\"].value_counts(),
                             color=\"#1b9e54\", height=180, use_container_width=True)
            with a2:
                st.caption(\"Overall Density Distribution\")
                if \"overall_density\" in df_all.columns:
                    st.bar_chart(df_all[\"overall_density\"].value_counts(),
                                 color=\"#e5a720\", height=180, use_container_width=True)

        if \"timestamp\" in df_all.columns:
            st.markdown('<div class=\"rule-label\">Peak Hour Analysis</div>',
                        unsafe_allow_html=True)
            df_all[\"hour\"] = pd.to_datetime(
                df_all[\"timestamp\"], errors=\"coerce\").dt.hour
            df_hr = df_all.groupby(\"hour\")[\"total_count\"].mean().round(1)
            if not df_hr.empty:
                st.bar_chart(df_hr, color=\"#1a3a6b\",
                             height=180, use_container_width=True)
                pk = df_hr.idxmax()
                st.caption(f\"🔴 Peak hour: **{pk}:00 – {pk+1}:00**\")


# ════════════════════════════════════════════════════════════════════
# PAGE 3 — DATABASE
# ════════════════════════════════════════════════════════════════════

elif active == \"🗄️  Database\":
    st.markdown('<div class=\"rule-label\">SQLite Database — traffic_data.db</div>',
                unsafe_allow_html=True)
    df = read_db(limit=100)

    if df.empty:
        st.info(\"Database is empty. Run Live Monitor to populate.\")
    else:
        total  = get_total_records()
        avg_v  = round(df[\"total_count\"].mean(), 1) if \"total_count\" in df else 0
        green  = len(df[df[\"signal_phase\"]==\"GREEN\"]) if \"signal_phase\" in df else 0
        latest = df.iloc[0][\"timestamp\"][:16] if not df.empty else \"—\"

        st.markdown(f\"\"\"
        <div class=\"db-row\">
            <div class=\"db-cell\">
                <div class=\"db-num\">{total}</div>
                <div class=\"db-lbl\">Total Records</div>
            </div>
            <div class=\"db-cell\">
                <div class=\"db-num\">{avg_v}</div>
                <div class=\"db-lbl\">Avg Vehicles</div>
            </div>
            <div class=\"db-cell\">
                <div class=\"db-num\">{green}</div>
                <div class=\"db-lbl\">GREEN Phases</div>
            </div>
            <div class=\"db-cell\">
                <div class=\"db-num\" style=\"font-size:1rem;\">{latest}</div>
                <div class=\"db-lbl\">Latest Record</div>
            </div>
        </div>
        \"\"\", unsafe_allow_html=True)

        limit_opt = st.selectbox(\"Show records\", [10, 25, 50, 100], index=0)
        df_show   = read_db(limit=limit_opt)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        csv = df_show.to_csv(index=False).encode(\"utf-8\")
        st.download_button(\"⬇️ Download CSV\", data=csv,
                           file_name=f\"traffic_{datetime.now().strftime('%Y%m%d_%H%M')}.csv\",
                           mime=\"text/csv\")

        if st.session_state.role == \"Admin\":
            st.markdown('<div class=\"rule-label\" style=\"margin-top:20px\">Users Table — Admin View</div>',
                        unsafe_allow_html=True)
            conn = sqlite3.connect(DB_FILE)
            u_df = pd.read_sql(\"SELECT id,username,role FROM users\", conn)
            conn.close()
            st.dataframe(u_df, use_container_width=True, hide_index=True)

        st.caption(\"View raw database → sqlitebrowser.org → Open traffic_data.db\")


# ════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════════════════════════════════

elif active == \"ℹ️  About\":
    st.markdown(\"\"\"
    <div class=\"about-masthead\">
        <div class=\"about-vol\">K.R. Mangalam University · B.Tech CSE · Pre-Final Year · 2025–26</div>
        <div class=\"about-hed\">IoT <em>Intelligent</em><br>Traffic Vision</div>
        <div class=\"about-dek\">
            A real-time 4-way intersection traffic monitoring and congestion prediction system.
            Modelled on Indian road logic — North–South and East–West signal pairs run together,
            with dynamic timers based on live congestion analysis. Built with YOLOv8 AI,
            SQLite database, secure authentication, and deployed on Streamlit Cloud.
        </div>
    </div>
    \"\"\", unsafe_allow_html=True)

    st.markdown('<div class=\"rule-label\">Project Information</div>',
                unsafe_allow_html=True)
    st.markdown(\"\"\"
    <div class=\"info-3col\">
        <div class=\"info-col\">
            <div class=\"info-col-title\">Project Details</div>
            <div class=\"info-kv\"><span class=\"info-k\">Title</span><span class=\"info-v\">IoT Traffic Monitoring</span></div>
            <div class=\"info-kv\"><span class=\"info-k\">Type</span><span class=\"info-v\">IoT Based</span></div>
            <div class=\"info-kv\"><span class=\"info-k\">Model</span><span class=\"info-v\">4-Way Intersection</span></div>
            <div class=\"info-kv\"><span class=\"info-k\">Signal Logic</span><span class=\"info-v\">NS ↔ EW Pair System</span></div>
            <div class=\"info-kv\"><span class=\"info-k\">Status</span><span class=\"info-v\" style=\"color:#1b9e54;\">✅ Approved</span></div>
        </div>
        <div class=\"info-col\">
            <div class=\"info-col-title\">Academic Details</div>
            <div class=\"info-kv\"><span class=\"info-k\">University</span><span class=\"info-v\">K.R. Mangalam</span></div>
            <div class=\"info-kv\"><span class=\"info-k\">School</span><span class=\"info-v\">Engg & Technology</span></div>
            <div class=\"info-kv\"><span class=\"info-k\">Program</span><span class=\"info-v\">B.Tech CSE</span></div>
            <div class=\"info-kv\"><span class=\"info-k\">Semester</span><span class=\"info-v\">Pre-Final Year</span></div>
            <div class=\"info-kv\"><span class=\"info-k\">Year</span><span class=\"info-v\">2025–2026</span></div>
        </div>
        <div class=\"info-col\">
            <div class=\"info-col-title\">System Specs</div>
            <div class=\"info-kv\"><span class=\"info-k\">AI Model</span><span class=\"info-v\">YOLOv8 Nano</span></div>
            <div class=\"info-kv\"><span class=\"info-k\">Dataset</span><span class=\"info-v\">COCO 80 classes</span></div>
            <div class=\"info-kv\"><span class=\"info-k\">Signal Timer</span><span class=\"info-v\">15s / 30s / 45s</span></div>
            <div class=\"info-kv\"><span class=\"info-k\">Yellow Phase</span><span class=\"info-v\">5s transition</span></div>
            <div class=\"info-kv\"><span class=\"info-k\">Auth</span><span class=\"info-v\">SHA-256 Hashed</span></div>
        </div>
    </div>
    \"\"\", unsafe_allow_html=True)

    st.markdown('<div class=\"rule-label\">Technology Stack</div>',
                unsafe_allow_html=True)
    st.markdown(\"\"\"
    <div class=\"tech-8\">
        <div class=\"tech-item\"><span class=\"tech-ico\">🐍</span><div class=\"tech-nm\">Python 3.11</div></div>
        <div class=\"tech-item\"><span class=\"tech-ico\">🤖</span><div class=\"tech-nm\">YOLOv8</div></div>
        <div class=\"tech-item\"><span class=\"tech-ico\">📷</span><div class=\"tech-nm\">OpenCV</div></div>
        <div class=\"tech-item\"><span class=\"tech-ico\">🌐</span><div class=\"tech-nm\">Streamlit</div></div>
        <div class=\"tech-item\"><span class=\"tech-ico\">🗄️</span><div class=\"tech-nm\">SQLite</div></div>
        <div class=\"tech-item\"><span class=\"tech-ico\">📡</span><div class=\"tech-nm\">MQTT</div></div>
        <div class=\"tech-item\"><span class=\"tech-ico\">🐼</span><div class=\"tech-nm\">Pandas</div></div>
        <div class=\"tech-item\"><span class=\"tech-ico\">☁️</span><div class=\"tech-nm\">Streamlit Cloud</div></div>
    </div>
    \"\"\", unsafe_allow_html=True)

    st.markdown(\"\"\"
    <div class=\"dev-strip\">
        <div class=\"dev-av\">👨‍💻</div>
        <div>
            <div class=\"dev-name\">Sarthak Mishra</div>
            <div class=\"dev-role\">B.Tech Computer Science & Engineering · K.R. Mangalam University</div>
            <div>
                <span class=\"dev-tag\">IoT Systems</span>
                <span class=\"dev-tag\">Computer Vision</span>
                <span class=\"dev-tag\">Python</span>
                <span class=\"dev-tag\">4-Way Intersection Logic</span>
                <span class=\"dev-tag\">Machine Learning</span>
            </div>
        </div>
    </div>
    \"\"\", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════

st.markdown(\"\"\"
<div class=\"footer-bar\">
    <div class=\"footer-l\">
        <b>IoT-Based Intelligent Traffic Monitoring & Congestion Prediction System</b><br>
        K.R. Mangalam University · School of Engineering & Technology · B.Tech CSE · 2025–26
    </div>
    <div class=\"footer-r\">
        Designed & Developed by<br>
        <span>Sarthak Mishra</span> · B.Tech CSE<br>
        K.R. Mangalam University
    </div>
</div>
\"\"\", unsafe_allow_html=True)
"
Observation: Create successful: /app/combined.py
