"""
Smart Waste Segregation System using YOLOv8
Streamlit Application

Two-level classification:
  1. Bin Type: Wet Bin / Dry Bin / Metal Bin
  2. Degradability: Biodegradable / Non-Biodegradable

Features: Upload, Webcam, DroidCam snapshot, DroidCam live video,
          Pie charts, Green Credit Points, Analytics dashboard.
"""

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import time
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from waste_classifier import load_model, detect_and_classify, BIN_LABELS

st.set_page_config(
    page_title="Smart Waste Segregation System",
    page_icon="♻️",
    layout="wide",
)

# ────────────────── GREEN CREDIT POINTS SYSTEM ──────────────────
# Points awarded per item detected:
#   Biodegradable + Wet bin   → +10  (best: composting organic waste)
#   Biodegradable + Dry bin   → +7   (paper/wood recycled properly)
#   Non-Biodeg + Metal bin    → +5   (metal recycling is valuable)
#   Non-Biodeg + Dry bin      → +2   (plastic/glass — at least sorted)
POINTS_MAP = {
    ("Biodegradable", "Wet"):   10,
    ("Biodegradable", "Dry"):    7,
    ("Non-Biodegradable", "Metal"): 5,
    ("Non-Biodegradable", "Dry"):   2,
    ("Non-Biodegradable", "Wet"):   0,
    ("Biodegradable", "Metal"):     3,
}

ECO_LEVELS = [
    (0,    "Beginner",        "🌱"),
    (50,   "Eco Starter",     "🌿"),
    (150,  "Green Warrior",   "🌳"),
    (300,  "Eco Champion",    "🏆"),
    (500,  "Planet Saver",    "🌍"),
    (1000, "Eco Legend",      "👑"),
]


def calc_points(degradability, bin_type):
    return POINTS_MAP.get((degradability, bin_type), 1)


def get_eco_level(total_points):
    level_name, emoji = "Beginner", "🌱"
    for threshold, name, em in ECO_LEVELS:
        if total_points >= threshold:
            level_name, emoji = name, em
    # Next level
    next_threshold = None
    for threshold, name, _ in ECO_LEVELS:
        if total_points < threshold:
            next_threshold = threshold
            break
    return level_name, emoji, next_threshold


# ────────────────── CSS ──────────────────
st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem 0; }
    .wet-card {
        background-color: #d4edda; border-left: 5px solid #28a745;
        padding: 12px; border-radius: 5px; margin: 5px 0;
    }
    .dry-card {
        background-color: #cce5ff; border-left: 5px solid #007bff;
        padding: 12px; border-radius: 5px; margin: 5px 0;
    }
    .metal-card {
        background-color: #e2e3e5; border-left: 5px solid #6c757d;
        padding: 12px; border-radius: 5px; margin: 5px 0;
    }
    .bio-tag {
        display: inline-block; background: #28a745; color: white;
        padding: 2px 8px; border-radius: 10px; font-size: 12px;
    }
    .nonbio-tag {
        display: inline-block; background: #dc3545; color: white;
        padding: 2px 8px; border-radius: 10px; font-size: 12px;
    }
    .live-badge {
        display: inline-block; background: #dc3545; color: white;
        padding: 4px 12px; border-radius: 15px; font-weight: bold;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .points-card {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white; padding: 20px; border-radius: 15px;
        text-align: center; margin: 10px 0;
    }
    .points-card h2 { margin: 0; font-size: 2.5em; }
    .points-card p { margin: 5px 0 0 0; font-size: 1.1em; }
    .points-earned {
        display: inline-block; background: #ffc107; color: #333;
        padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────── SESSION STATE ──────────────────
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []
if "green_points" not in st.session_state:
    st.session_state.green_points = 0
if "live_running" not in st.session_state:
    st.session_state.live_running = False
if "live_last_save" not in st.session_state:
    st.session_state.live_last_save = 0


@st.cache_resource
def get_model():
    return load_model("n")


def add_to_history(detections, source):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scan_points = 0
    for d in detections:
        pts = calc_points(d["degradability"], d["bin_type"])
        scan_points += pts
        st.session_state.detection_history.append({
            "Timestamp": timestamp,
            "Source": source,
            "Item": d["item"],
            "Bin Type": BIN_LABELS[d["bin_type"]],
            "Degradability": d["degradability"],
            "Confidence": f"{d['confidence']:.1%}",
            "Points": pts,
        })
    st.session_state.green_points += scan_points
    return scan_points


def grab_droidcam_frame(ip_url):
    """Grab a single frame — tries snapshot first, then video stream."""
    snapshot_url = get_droidcam_snapshot_url(ip_url)

    # Method 1: HTTP snapshot (fastest, most reliable)
    try:
        import urllib.request
        resp = urllib.request.urlopen(snapshot_url, timeout=5)
        img_array = np.frombuffer(resp.read(), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    # Method 2: OpenCV VideoCapture on snapshot URL
    cap = cv2.VideoCapture(snapshot_url)
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Method 3: OpenCV VideoCapture on video stream
    video_url = get_droidcam_video_url(ip_url)
    cap = cv2.VideoCapture(video_url)
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return None


def normalize_droidcam_url(ip_url):
    """Strip trailing /video or /shot.jpg to get base URL like http://IP:PORT."""
    base = ip_url.strip().rstrip("/")
    for suffix in ["/video", "/shot.jpg", "/shot"]:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    return base


def get_droidcam_video_url(ip_url):
    return normalize_droidcam_url(ip_url) + "/video"


def get_droidcam_snapshot_url(ip_url):
    return normalize_droidcam_url(ip_url) + "/shot.jpg"


def render_detection_cards(detections):
    wet = [d for d in detections if d["bin_type"] == "Wet"]
    dry = [d for d in detections if d["bin_type"] == "Dry"]
    metal = [d for d in detections if d["bin_type"] == "Metal"]

    for bin_label, items, css_class, emoji in [
        ("Wet Bin (Green) — Organic / Food Waste", wet, "wet-card", "🟢"),
        ("Dry Bin (Blue) — Plastic, Paper, Glass, Electronics", dry, "dry-card", "🔵"),
        ("Metal Bin (Grey) — Metallic Items", metal, "metal-card", "⚙️"),
    ]:
        if items:
            st.markdown(f"#### {emoji} {bin_label}")
            for d in items:
                deg_tag = "bio-tag" if d["degradability"] == "Biodegradable" else "nonbio-tag"
                pts = calc_points(d["degradability"], d["bin_type"])
                st.markdown(
                    f"<div class='{css_class}'>"
                    f"<b>{d['item']}</b> &nbsp; "
                    f"<span class='{deg_tag}'>{d['degradability']}</span> &nbsp; "
                    f"<span class='points-earned'>+{pts} pts</span><br>"
                    f"<small>Confidence: {d['confidence']:.0%}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def render_pie_charts(detections):
    """Render pie charts for current detection."""
    col_pie1, col_pie2 = st.columns(2)

    # Bin type pie
    bin_counts = {}
    for d in detections:
        label = BIN_LABELS[d["bin_type"]]
        bin_counts[label] = bin_counts.get(label, 0) + 1

    with col_pie1:
        fig1 = px.pie(
            names=list(bin_counts.keys()),
            values=list(bin_counts.values()),
            title="Bin Type Distribution",
            color=list(bin_counts.keys()),
            color_discrete_map={
                "Wet Bin (Green)": "#28a745",
                "Dry Bin (Blue)": "#007bff",
                "Metal Bin (Grey)": "#6c757d",
            },
            hole=0.4,
        )
        fig1.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig1, use_container_width=True)

    # Degradability pie
    deg_counts = {}
    for d in detections:
        deg_counts[d["degradability"]] = deg_counts.get(d["degradability"], 0) + 1

    with col_pie2:
        fig2 = px.pie(
            names=list(deg_counts.keys()),
            values=list(deg_counts.values()),
            title="Biodegradable vs Non-Biodegradable",
            color=list(deg_counts.keys()),
            color_discrete_map={
                "Biodegradable": "#28a745",
                "Non-Biodegradable": "#dc3545",
            },
            hole=0.4,
        )
        fig2.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)


def render_green_points_card():
    """Render the Green Credits card in sidebar."""
    total = st.session_state.green_points
    level_name, emoji, next_thresh = get_eco_level(total)

    st.markdown(
        f"<div class='points-card'>"
        f"<h2>{total}</h2>"
        f"<p>Green Credits {emoji}</p>"
        f"<p><b>{level_name}</b></p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if next_thresh:
        progress = total / next_thresh
        st.progress(min(progress, 1.0), text=f"{total}/{next_thresh} to next level")
    else:
        st.progress(1.0, text="MAX LEVEL REACHED!")

    st.caption(
        "**How points work:**\n"
        "- Wet bin (organic): **+10 pts**\n"
        "- Dry bin (paper/wood): **+7 pts**\n"
        "- Metal bin (recycling): **+5 pts**\n"
        "- Dry bin (plastic/glass): **+2 pts**"
    )


def render_analytics_dashboard(history_df):
    """Render the analytics section with charts."""
    st.subheader("Analytics Dashboard")

    # Row 1: Pie charts from history
    ch1, ch2 = st.columns(2)

    with ch1:
        bin_counts = history_df["Bin Type"].value_counts()
        fig = px.pie(
            names=bin_counts.index,
            values=bin_counts.values,
            title="All-Time Bin Distribution",
            color=bin_counts.index,
            color_discrete_map={
                "Wet Bin (Green)": "#28a745",
                "Dry Bin (Blue)": "#007bff",
                "Metal Bin (Grey)": "#6c757d",
            },
            hole=0.4,
        )
        fig.update_layout(height=350, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        deg_counts = history_df["Degradability"].value_counts()
        fig = px.pie(
            names=deg_counts.index,
            values=deg_counts.values,
            title="All-Time Degradability Split",
            color=deg_counts.index,
            color_discrete_map={
                "Biodegradable": "#28a745",
                "Non-Biodegradable": "#dc3545",
            },
            hole=0.4,
        )
        fig.update_layout(height=350, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Top detected items bar chart
    top_items = history_df["Item"].value_counts().head(10)
    if len(top_items) > 0:
        fig_bar = px.bar(
            x=top_items.values,
            y=top_items.index,
            orientation="h",
            title="Top 10 Most Detected Items",
            labels={"x": "Count", "y": "Item"},
            color=top_items.values,
            color_continuous_scale="Greens",
        )
        fig_bar.update_layout(
            height=400,
            margin=dict(t=40, b=10, l=10, r=10),
            showlegend=False,
            yaxis=dict(autorange="reversed"),
        )
        fig_bar.update_coloraxes(showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Row 3: Points earned over time
    if "Points" in history_df.columns:
        points_over_time = history_df.groupby("Timestamp")["Points"].sum().reset_index()
        points_over_time["Cumulative Points"] = points_over_time["Points"].cumsum()
        fig_line = px.area(
            points_over_time,
            x="Timestamp",
            y="Cumulative Points",
            title="Green Credits Over Time",
            color_discrete_sequence=["#28a745"],
        )
        fig_line.update_layout(
            height=300,
            margin=dict(t=40, b=10, l=10, r=10),
            xaxis_title="",
            yaxis_title="Cumulative Points",
        )
        st.plotly_chart(fig_line, use_container_width=True)


def render_live_sidebar_detections(placeholder, detections):
    with placeholder.container():
        if not detections:
            st.caption("No items detected in current frame.")
            return

        wet_count = sum(1 for d in detections if d["bin_type"] == "Wet")
        dry_count = sum(1 for d in detections if d["bin_type"] == "Dry")
        metal_count = sum(1 for d in detections if d["bin_type"] == "Metal")

        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Wet", wet_count)
        c2.metric("🔵 Dry", dry_count)
        c3.metric("⚙️ Metal", metal_count)

        st.markdown("---")
        for d in detections:
            deg_tag = "bio-tag" if d["degradability"] == "Biodegradable" else "nonbio-tag"
            pts = calc_points(d["degradability"], d["bin_type"])
            bin_name = BIN_LABELS[d["bin_type"]]
            st.markdown(
                f"**{d['item']}** — {bin_name} "
                f"<span class='{deg_tag}'>{d['degradability']}</span> "
                f"<span class='points-earned'>+{pts}</span> "
                f"({d['confidence']:.0%})",
                unsafe_allow_html=True,
            )


def grab_webcam_frame(cam_index=0):
    """Grab a single frame from local webcam via OpenCV."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        cap.release()
        return None
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def run_live_feed(model, source, confidence, save_interval, source_label=""):
    """
    Live feed — keeps camera open in a loop for smooth video.
    source: int (webcam index) or str (DroidCam IP).
    """
    import urllib.request

    if not source_label:
        source_label = f"Webcam {source}" if isinstance(source, int) else normalize_droidcam_url(source)

    st.markdown(
        "<span class='live-badge'>LIVE</span> &nbsp; "
        f"Streaming from <b>{source_label}</b>",
        unsafe_allow_html=True,
    )

    col_video, col_stats = st.columns([3, 2])
    with col_video:
        video_placeholder = st.empty()
        info_placeholder = st.empty()
    with col_stats:
        st.subheader("Live Detections")
        stats_placeholder = st.empty()
    table_placeholder = st.empty()
    points_placeholder = st.empty()

    # Open camera / stream once and keep it open
    cap = None
    use_snapshot = False

    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            video_placeholder.error(
                "Cannot access webcam.\n\n"
                "**Troubleshooting:**\n"
                "- Run Streamlit from **Terminal.app** (not Claude)\n"
                "- Allow camera: System Settings > Privacy > Camera > Terminal\n"
                "- Close other apps using the camera"
            )
            st.session_state.live_running = False
            return
    else:
        base_url = normalize_droidcam_url(source)
        # Try video stream first
        cap = cv2.VideoCapture(base_url + "/video")
        if not cap.isOpened():
            cap.release()
            cap = None
            use_snapshot = True
            # Test snapshot works
            try:
                resp = urllib.request.urlopen(base_url + "/shot.jpg", timeout=5)
                _ = resp.read()
            except Exception:
                video_placeholder.error(
                    f"Cannot connect to DroidCam at `{base_url}`.\n\n"
                    "**Troubleshooting:**\n"
                    "- DroidCam app is open on your phone\n"
                    "- Phone and computer on **same WiFi**"
                )
                st.session_state.live_running = False
                return

    frame_count = 0
    last_fps_time = time.time()
    last_save_time = time.time()
    fps = 0.0
    session_points = 0
    session_items = 0
    unique_items = set()

    try:
        while st.session_state.live_running:
            frame_rgb = None

            if use_snapshot:
                base_url = normalize_droidcam_url(source)
                try:
                    resp = urllib.request.urlopen(base_url + "/shot.jpg", timeout=3)
                    img_array = np.frombuffer(resp.read(), dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    pass
            else:
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_rgb is None:
                time.sleep(0.1)
                continue

            # Run detection
            annotated, detections = detect_and_classify(
                model, frame_rgb, confidence_threshold=confidence
            )

            # FPS counter
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            video_placeholder.image(annotated, channels="RGB", use_container_width=True)
            info_placeholder.caption(f"FPS: {fps:.1f} | Objects: {len(detections)} | {source_label}")
            render_live_sidebar_detections(stats_placeholder, detections)

            if detections:
                table_data = [{
                    "Item": d["item"],
                    "Bin": BIN_LABELS[d["bin_type"]],
                    "Degradability": d["degradability"],
                    "Conf": f"{d['confidence']:.0%}",
                } for d in detections]
                table_placeholder.table(table_data)

                # Save to history + award credits every save_interval seconds
                if (now - last_save_time) >= save_interval:
                    scan_points = add_to_history(detections, f"Live: {source_label}")
                    last_save_time = now
                    session_points += scan_points
                    session_items += len(detections)
                    for d in detections:
                        unique_items.add(d["item"])

                # Always show running session total
                points_placeholder.markdown(
                    f"<div style='background:#d4edda; padding:12px; border-radius:10px; text-align:center;'>"
                    f"<b>Session:</b> {session_points} Green Credits | "
                    f"{session_items} items saved | "
                    f"{len(unique_items)} unique items | "
                    f"Total: {st.session_state.green_points} credits"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                table_placeholder.empty()

            time.sleep(0.03)

    finally:
        if cap is not None:
            cap.release()
        st.session_state.live_running = False


# ────────────────── MAIN ──────────────────
def main():
    st.markdown("<h1 class='main-header'>Smart Waste Segregation System</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-size:18px;'>"
        "AI-powered waste detection using <b>YOLOv8</b><br>"
        "Classifies into <b style='color:#28a745;'>Wet</b> / "
        "<b style='color:#007bff;'>Dry</b> / "
        "<b style='color:#6c757d;'>Metal</b> bins &amp; "
        "<b style='color:#28a745;'>Biodegradable</b> / "
        "<b style='color:#dc3545;'>Non-Biodegradable</b>"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Sidebar ──
    with st.sidebar:
        # Green Credits Card
        st.header("Green Credits")
        render_green_points_card()

        st.divider()
        st.header("Settings")
        confidence = st.slider(
            "Detection Confidence",
            min_value=0.1, max_value=0.9, value=0.35, step=0.05,
        )

        st.divider()
        st.header("Live Feed Settings")
        cam_index = st.selectbox("Webcam Camera Index", [0, 1, 2], index=0)
        save_interval = st.slider(
            "Save history every (sec)",
            min_value=1, max_value=30, value=3, step=1,
        )

        st.divider()
        st.header("DroidCam Setup")
        droidcam_ip = st.text_input(
            "DroidCam IP Address",
            placeholder="http://192.168.1.100:4747",
        )

        st.divider()
        st.header("Bin Guide")
        st.markdown(
            "🟢 **Wet Bin (Green):** Food, fruits, vegetables, plants\n\n"
            "🔵 **Dry Bin (Blue):** Plastic, paper, glass, electronics\n\n"
            "⚙️ **Metal Bin (Grey):** Metal utensils, keys, cans"
        )

        st.divider()
        st.markdown(
            "**Project:** Smart Waste Segregation System\n\n"
            "**By:** Gauri Chopra & Shivaansh Kaushik\n\n"
            "**Tech:** Python, YOLOv8, OpenCV, Streamlit"
        )

    # Load model
    with st.spinner("Loading YOLOv8 model..."):
        model = get_model()

    # ── Input Tabs ──
    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Take Photo (Webcam)", "Webcam Live Feed", "DroidCam Snapshot", "DroidCam Live Feed"],
        horizontal=True,
    )

    image = None
    source_label = ""

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an image of waste items",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            source_label = f"Upload: {uploaded_file.name}"

    elif input_method == "Take Photo (Webcam)":
        camera_input = st.camera_input("Take a photo of waste items")
        if camera_input is not None:
            image = Image.open(camera_input).convert("RGB")
            source_label = "Webcam"

    elif input_method == "Webcam Live Feed":
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Start Live Feed", type="primary", disabled=st.session_state.live_running):
                st.session_state.live_running = True
                st.session_state.live_source = "webcam"
                st.rerun()
        with btn_col2:
            if st.button("Stop Live Feed", type="secondary", disabled=not st.session_state.live_running):
                st.session_state.live_running = False
                st.rerun()

        if st.session_state.live_running and st.session_state.get("live_source") == "webcam":
            run_live_feed(model, cam_index, confidence, save_interval, source_label=f"Webcam {cam_index}")
            return

    elif input_method == "DroidCam Snapshot":
        if not droidcam_ip:
            st.info(
                "**Setup:** Enter DroidCam IP in the sidebar.\n\n"
                "1. Install **DroidCam** on phone\n"
                "2. Same WiFi for phone & computer\n"
                "3. Enter IP (e.g. `http://192.168.1.100:4747`)"
            )
        else:
            st.success(f"DroidCam: **{droidcam_ip}**")
        if st.button("Capture from DroidCam", type="primary", disabled=not droidcam_ip):
            with st.spinner("Connecting to DroidCam..."):
                frame = grab_droidcam_frame(droidcam_ip)
            if frame is not None:
                image = Image.fromarray(frame)
                source_label = f"DroidCam: {droidcam_ip}"
            else:
                st.error("Could not connect. Check WiFi, DroidCam app, and IP/port.")

    elif input_method == "DroidCam Live Feed":
        if not droidcam_ip:
            st.info("**Enter DroidCam IP in the sidebar first**, then click Start.")
        else:
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Start DroidCam Live", type="primary", disabled=st.session_state.live_running):
                    st.session_state.live_running = True
                    st.session_state.live_source = "droidcam"
                    st.rerun()
            with btn_col2:
                if st.button("Stop DroidCam Live", type="secondary", disabled=not st.session_state.live_running):
                    st.session_state.live_running = False
                    st.rerun()
            if st.session_state.live_running and st.session_state.get("live_source") == "droidcam":
                run_live_feed(model, droidcam_ip, confidence, save_interval)
                return

    # ── Static Detection ──
    if image is not None:
        img_array = np.array(image)

        with st.spinner("Detecting and classifying waste..."):
            annotated_image, detections = detect_and_classify(
                model, img_array, confidence_threshold=confidence
            )

        scan_points = 0
        if detections:
            scan_points = add_to_history(detections, source_label)

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Detection Results")
            st.image(annotated_image, channels="RGB", use_container_width=True)

        with col2:
            st.subheader("Classification Summary")

            if not detections:
                st.warning(
                    "No waste items detected. Try:\n"
                    "- A clearer image\n"
                    "- Lowering the confidence threshold\n"
                    "- Better lighting"
                )
            else:
                # Points earned this scan
                st.markdown(
                    f"<div style='text-align:center; padding:10px; background:#d4edda; "
                    f"border-radius:10px; margin-bottom:15px;'>"
                    f"<b style='font-size:1.3em;'>+{scan_points} Green Credits earned!</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Bin type metrics
                wet_count = sum(1 for d in detections if d["bin_type"] == "Wet")
                dry_count = sum(1 for d in detections if d["bin_type"] == "Dry")
                metal_count = sum(1 for d in detections if d["bin_type"] == "Metal")
                bio_count = sum(1 for d in detections if d["degradability"] == "Biodegradable")
                nonbio_count = sum(1 for d in detections if d["degradability"] == "Non-Biodegradable")

                st.markdown("**By Bin Type:**")
                b1, b2, b3 = st.columns(3)
                with b1:
                    st.metric("🟢 Wet", wet_count)
                with b2:
                    st.metric("🔵 Dry", dry_count)
                with b3:
                    st.metric("⚙️ Metal", metal_count)

                st.markdown("**By Degradability:**")
                d1, d2 = st.columns(2)
                with d1:
                    st.metric("Biodegradable", bio_count)
                with d2:
                    st.metric("Non-Biodegradable", nonbio_count)

                st.divider()
                render_detection_cards(detections)

        # Pie charts for current detection
        if detections:
            st.divider()
            st.subheader("Current Scan Breakdown")
            render_pie_charts(detections)

        # Current detection table
        if detections:
            st.divider()
            st.subheader("Current Detection Table")
            table_data = []
            for d in detections:
                pts = calc_points(d["degradability"], d["bin_type"])
                table_data.append({
                    "Item": d["item"],
                    "Bin Type": BIN_LABELS[d["bin_type"]],
                    "Degradability": d["degradability"],
                    "Confidence": f"{d['confidence']:.1%}",
                    "Points": f"+{pts}",
                })
            st.table(table_data)

    # ── Detection History + Analytics ──
    st.divider()

    history_tab, analytics_tab = st.tabs(["Detection History", "Analytics Dashboard"])

    with history_tab:
        if not st.session_state.detection_history:
            st.info("No detections yet. Upload an image or capture from camera to start.")
        else:
            history_df = pd.DataFrame(st.session_state.detection_history)

            total_scans = history_df["Timestamp"].nunique()
            total_items = len(history_df)
            h_bio = len(history_df[history_df["Degradability"] == "Biodegradable"])
            h_nonbio = len(history_df[history_df["Degradability"] == "Non-Biodegradable"])
            total_pts = history_df["Points"].sum() if "Points" in history_df.columns else 0

            hc1, hc2, hc3, hc4, hc5 = st.columns(5)
            with hc1:
                st.metric("Total Scans", total_scans)
            with hc2:
                st.metric("Total Items", total_items)
            with hc3:
                st.metric("Biodegradable", h_bio)
            with hc4:
                st.metric("Non-Biodegradable", h_nonbio)
            with hc5:
                st.metric("Total Credits", total_pts)

            f1, f2, f3 = st.columns(3)
            with f1:
                cat_filter = st.selectbox("Filter by Bin", ["All", "Wet Bin (Green)", "Dry Bin (Blue)", "Metal Bin (Grey)"])
            with f2:
                deg_filter = st.selectbox("Filter by Degradability", ["All", "Biodegradable", "Non-Biodegradable"])
            with f3:
                src_filter = st.selectbox("Filter by Source", ["All"] + list(history_df["Source"].unique()))

            filtered = history_df.copy()
            if cat_filter != "All":
                filtered = filtered[filtered["Bin Type"] == cat_filter]
            if deg_filter != "All":
                filtered = filtered[filtered["Degradability"] == deg_filter]
            if src_filter != "All":
                filtered = filtered[filtered["Source"] == src_filter]

            st.dataframe(
                filtered.iloc[::-1].reset_index(drop=True),
                use_container_width=True,
                height=400,
            )

            btn1, btn2 = st.columns(2)
            with btn1:
                csv_data = history_df.to_csv(index=False)
                st.download_button(
                    "Download History as CSV",
                    data=csv_data,
                    file_name=f"waste_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            with btn2:
                if st.button("Clear History", type="secondary"):
                    st.session_state.detection_history = []
                    st.session_state.green_points = 0
                    st.rerun()

    with analytics_tab:
        if not st.session_state.detection_history:
            st.info("No data yet. Start scanning waste to see analytics.")
        else:
            history_df = pd.DataFrame(st.session_state.detection_history)
            render_analytics_dashboard(history_df)


if __name__ == "__main__":
    main()
