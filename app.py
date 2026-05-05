import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="Hand 3D Visualizer", layout="wide")

# Joint & Color Definitions (from your v2 script)
FINGER_PREFIXES = {
    "Thumb": "TH",
    "Pinky": "F1",
    "Ring": "F2",
    "Middle": "F3",
    "Index": "F4",
}
FINGER_JOINTS = ["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"]

FINGER_COLORS = {
    "Thumb": "#e63946",
    "Index": "#457b9d",
    "Middle": "#e9c46a",
    "Ring": "#f4a261",
    "Pinky": "#2a9d8f",
}
PALM_COLOR = "#a8dadc"
BG_COLOR = "#0d1117"

# --- LOGIC FUNCTIONS ---
def build_skeleton():
    bones = []
    for finger in FINGER_PREFIXES:
        color = FINGER_COLORS[finger]
        chain = [f"{finger}_{s}" for s in FINGER_JOINTS]
        knu1b, knu1a, knu2a, knu3a = chain
        if finger == "Thumb":
            bones.append(("Palm", knu1a, color, True))
            bones.append((knu1a, knu2a, color, False))
            bones.append((knu2a, knu3a, color, False))
        else:
            bones.append(("Palm", knu1b, color, True))
            bones.append((knu1b, knu1a, color, False))
            bones.append((knu1a, knu2a, color, False))
            bones.append((knu2a, knu3a, color, False))
    return bones

BONES = build_skeleton()
HIDDEN_JOINTS = {f"{f}_KNU1_B" for f in FINGER_PREFIXES if f == "Thumb"}

def extract_joints(row):
    joints = {"Palm": (row["PALM_POSITION_X"], row["PALM_POSITION_Y"], row["PALM_POSITION_Z"])}
    for finger, prefix in FINGER_PREFIXES.items():
        for suffix in FINGER_JOINTS:
            joints[f"{finger}_{suffix}"] = (
                row[f"{prefix}_{suffix}_X"],
                row[f"{prefix}_{suffix}_Y"],
                row[f"{prefix}_{suffix}_Z"]
            )
    return joints

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Make sure this filename matches your CSV exactly
    return pd.read_csv("data/normalised_hand_data_DATA18REMOVED.csv")

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not find CSV file. Error: {e}")
    st.stop()

# --- STREAMLIT STATE & NAVIGATION ---
if 'idx' not in st.session_state:
    st.session_state.idx = 0

st.sidebar.title("Controls")

# Search Functionality
search_vid = st.sidebar.text_input("Video ID (e.g. 4)")
search_frm = st.sidebar.text_input("Frame ID (e.g. 948)")

if st.sidebar.button("Go to Search"):
    vid_full = f"data_{search_vid}" if search_vid.isdigit() else search_vid
    frm_full = f"{search_frm}_joints" if search_frm.isdigit() else search_frm
    mask = pd.Series([True] * len(df))
    if search_vid: mask &= (df["video_id"] == vid_full)
    if search_frm: mask &= (df["frame_id"] == frm_full)
    matches = df[mask]
    if not matches.empty:
        st.session_state.idx = int(matches.index[0])
    else:
        st.sidebar.warning("No matches found.")

# Prev/Next Buttons
col1, col2 = st.sidebar.columns(2)
if col1.button("◀ Prev"):
    st.session_state.idx = max(0, st.session_state.idx - 1)
if col2.button("Next ▶"):
    st.session_state.idx = min(len(df) - 1, st.session_state.idx + 1)

# Slider
st.session_state.idx = st.sidebar.slider("Row Index", 0, len(df)-1, st.session_state.idx)

# --- PLOTTING ---
row = df.iloc[st.session_state.idx]
joints = extract_joints(row)

fig = go.Figure()

# Draw Bones
for a, b, color, dotted in BONES:
    fig.add_trace(go.Scatter3d(
        x=[joints[a][0], joints[b][0]],
        y=[joints[a][1], joints[b][1]],
        z=[joints[a][2], joints[b][2]],
        mode='lines',
        line=dict(color=color, width=6, dash='dash' if dotted else 'solid'),
        showlegend=False, hoverinfo='none'
    ))

# Draw Joints
for name, pos in joints.items():
    if name in HIDDEN_JOINTS: continue
    is_palm = (name == "Palm")
    is_tip = "KNU3_A" in name
    
    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers+text' if is_tip else 'markers',
        marker=dict(
            color=PALM_COLOR if is_palm else FINGER_COLORS.get(name.split('_')[0]),
            size=10 if is_palm else 6,
            line=dict(color='white', width=2)
        ),
        text=name.split('_')[0] if is_tip else None,
        textposition="top center",
        name=name
    ))

# View Formatting
fig.update_layout(
    title=f"Video: {row['video_id']} | Frame: {row['frame_id']} | Row: {st.session_state.idx}",
    template="plotly_dark",
    height=800,
    scene=dict(
        aspectmode='data',
        xaxis=dict(gridcolor='gray', showbackground=False),
        yaxis=dict(gridcolor='gray', showbackground=False),
        zaxis=dict(gridcolor='gray', showbackground=False),
    ),
    uirevision='constant' # Keeps camera fixed during Next/Prev
)

st.plotly_chart(fig, use_container_width=True)