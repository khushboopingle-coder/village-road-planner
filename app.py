import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from skimage.graph import route_through_array
from streamlit_folium import st_folium


# =========================================================
# TERRAPATH — VILLAGE ROAD ALIGNMENT PLANNER
# + Slope-coloured route and alternatives
# =========================================================

st.set_page_config(
    page_title="TerraPath | Road Alignment Planner",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA = Path(__file__).parent / "data" / "terrain_layers.npz"

LC_COLORS = {
    10: (0, 100, 0),
    20: (255, 187, 34),
    30: (255, 255, 76),
    40: (240, 150, 255),
    50: (250, 0, 0),
    60: (180, 180, 180),
    80: (0, 100, 200),
    90: (0, 150, 160),
}

LAND_COVER_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    80: "Water",
    90: "Herbaceous wetland",
}

# Slope classes used to colour the road: (label, colour)
SLOPE_BINS = [5, 10, 15, 25]
SLOPE_CLASSES = [
    ("Gentle  (0–5°)", "#2E9E4F"),
    ("Moderate  (5–10°)", "#F2C94C"),
    ("Steep  (10–15°)", "#F2994A"),
    ("Very steep  (15–25°)", "#E5484D"),
    ("Extreme  (> 25°)", "#7B1E3A"),
]

LANDSCAPE_URL = (
    "https://images.unsplash.com/photo-1470770841072-f978cf4d019e"
    "?auto=format&fit=crop&w=2200&q=85"
)

# ---------------------- STYLING ---------------------------

def inject_styles(night=False):
    if night:
        bg = "#101C19"
        surface = "#192823"
        surface2 = "#21352D"
        text = "#F0F2E8"
        muted = "#A7B9AC"
        border = "#30483D"
        accent = "#B9D9B2"
        hero_overlay = "rgba(10, 22, 18, .64)"
    else:
        bg = "#F5F4EC"
        surface = "#FFFFFF"
        surface2 = "#EAF0E6"
        text = "#243B2E"
        muted = "#718073"
        border = "#DFE5D9"
        accent = "#416B4D"
        hero_overlay = "rgba(20, 44, 31, .42)"

    st.markdown(
        f"""
        <style>
        :root {{
            --page: {bg};
            --surface: {surface};
            --surface2: {surface2};
            --text: {text};
            --muted: {muted};
            --border: {border};
            --accent: {accent};
        }}

        html, body, [data-testid="stAppViewContainer"] {{
            background: var(--page);
            color: var(--text);
            font-family: Inter, -apple-system, BlinkMacSystemFont,
                         "Segoe UI", sans-serif;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stMainBlockContainer"] {{
            padding-top: 1.3rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }}

        [data-testid="stSidebar"] {{
            background: var(--surface);
            border-right: 1px solid var(--border);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--text);
        }}

        h1, h2, h3, h4 {{
            color: var(--text) !important;
            letter-spacing: -0.035em;
        }}

        p, li, label {{
            color: var(--text);
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 11px;
            margin: 0 0 1.3rem 0;
        }}

        .brand-mark {{
            width: 43px;
            height: 43px;
            border-radius: 15px;
            background: #DDE9D8;
            color: #31563B;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }}

        .brand-name {{
            font-size: 22px;
            font-weight: 750;
            letter-spacing: -0.8px;
            color: var(--text);
            line-height: 1.15;
        }}

        .brand-sub {{
            font-size: 9px;
            letter-spacing: 2px;
            color: var(--muted);
            margin-top: 4px;
            font-weight: 700;
        }}

        .hero {{
            position: relative;
            overflow: hidden;
            min-height: 275px;
            border-radius: 25px;
            padding: 34px 38px;
            margin-bottom: 26px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            background-image:
                linear-gradient(
                    90deg,
                    {hero_overlay} 0%,
                    rgba(15, 35, 24, .25) 100%
                ),
                url("{LANDSCAPE_URL}");
            background-size: cover;
            background-position: center 52%;
            box-shadow: 0 14px 38px rgba(28, 49, 34, .10);
        }}

        .hero-kicker {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #E3F0DB;
            font-size: 10px;
            font-weight: 750;
            letter-spacing: 2px;
            margin-bottom: 15px;
        }}

        .hero h1 {{
            color: #FFFFFF !important;
            font-size: clamp(30px, 4vw, 47px);
            line-height: 1.07;
            font-weight: 750;
            max-width: 650px;
            margin: 0 0 13px 0;
        }}

        .hero p {{
            color: #F0F4E9;
            font-size: 14px;
            line-height: 1.8;
            max-width: 540px;
            margin: 0;
        }}

        .hero-chip {{
            display: inline-block;
            margin-top: 22px;
            padding: 9px 13px;
            border: 1px solid rgba(255,255,255,.36);
            background: rgba(255,255,255,.12);
            color: white;
            border-radius: 30px;
            font-size: 11px;
            letter-spacing: .2px;
            width: fit-content;
        }}

        .section-label {{
            font-size: 10px;
            letter-spacing: 2px;
            font-weight: 800;
            color: var(--muted);
            margin-bottom: 8px;
            text-transform: uppercase;
        }}

        .section-title {{
            font-size: 25px;
            line-height: 1.25;
            font-weight: 750;
            color: var(--text);
            letter-spacing: -.8px;
            margin-bottom: 5px;
        }}

        .section-description {{
            color: var(--muted);
            font-size: 13px;
            line-height: 1.65;
            margin-bottom: 18px;
        }}

        .soft-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 19px;
            padding: 19px 20px;
            margin-bottom: 13px;
        }}

        .metric-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 17px;
            padding: 17px 17px 15px 17px;
            min-height: 116px;
            margin-bottom: 8px;
        }}

        .metric-label {{
            font-size: 11px;
            font-weight: 650;
            color: var(--muted);
            margin-bottom: 12px;
        }}

        .metric-value {{
            font-size: clamp(23px, 2.5vw, 30px);
            font-weight: 760;
            letter-spacing: -1px;
            color: var(--text);
            line-height: 1.2;
        }}

        .metric-foot {{
            font-size: 11px;
            color: var(--muted);
            margin-top: 7px;
        }}

        .status-pill {{
            display: inline-block;
            border-radius: 30px;
            padding: 7px 11px;
            background: var(--surface2);
            color: var(--accent);
            font-size: 11px;
            font-weight: 750;
        }}

        .step-number {{
            width: 27px;
            height: 27px;
            border-radius: 10px;
            background: var(--surface2);
            color: var(--accent);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 800;
            flex-shrink: 0;
        }}

        .slope-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 10px 0;
            font-size: 12px;
            color: var(--text);
        }}
        .slope-swatch {{
            width: 14px;
            height: 14px;
            border-radius: 4px;
            flex-shrink: 0;
        }}
        .slope-name {{
            width: 150px;
            flex-shrink: 0;
            font-weight: 600;
        }}
        .slope-bar-bg {{
            flex: 1;
            height: 10px;
            background: var(--surface2);
            border-radius: 6px;
            overflow: hidden;
        }}
        .slope-bar {{
            height: 100%;
            border-radius: 6px;
        }}
        .slope-val {{
            width: 110px;
            text-align: right;
            color: var(--muted);
            flex-shrink: 0;
        }}

        div.stButton > button,
        div.stDownloadButton > button {{
            border-radius: 12px;
            min-height: 43px;
            font-weight: 700;
            border: 1px solid var(--border);
            transition: transform .18s ease, box-shadow .18s ease;
        }}

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 15px rgba(45, 78, 50, .10);
            border-color: var(--accent);
        }}

        div.stButton > button[kind="primary"] {{
            background: #355D41;
            color: white;
            border: none;
        }}

        div.stDownloadButton > button {{
            background: var(--surface2);
            color: var(--text);
            width: 100%;
        }}

        [data-testid="stMetric"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 15px;
            padding: 15px;
        }}

        [data-testid="stMetricLabel"] {{
            color: var(--muted);
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: 13px;
            overflow: hidden;
        }}

        [data-testid="stAlert"] {{
            border-radius: 13px;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 18px;
        }}

        hr {{
            border-color: var(--border);
        }}

        .footer {{
            text-align: center;
            color: var(--muted);
            font-size: 11px;
            line-height: 1.8;
            margin-top: 30px;
            padding-top: 18px;
            border-top: 1px solid var(--border);
        }}

        @media (max-width: 700px) {{
            .hero {{
                min-height: 250px;
                padding: 24px;
                border-radius: 19px;
            }}
            .hero h1 {{
                font-size: 31px;
            }}
            .soft-card {{
                padding: 15px;
            }}
            .slope-name {{ width: 105px; }}
            .slope-val {{ width: 80px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------- TERRAIN DATA ----------------------

@st.cache_resource
def load_layers():
    with np.load(DATA) as z:
        t = z["transform"]
        L = {
            k: z[k]
            for k in ["slope", "landcover", "river_mask", "road_mask"]
        }

    L["a"] = float(t[0])
    L["c"] = float(t[2])
    L["e"] = float(t[4])
    L["f"] = float(t[5])

    H, W = L["slope"].shape
    L["H"], L["W"] = H, W

    L["lon0"], L["lon1"] = L["c"], L["c"] + L["a"] * W
    L["lat1"], L["lat0"] = L["f"], L["f"] + L["e"] * H

    lat_mid = (L["lat0"] + L["lat1"]) / 2
    L["dy"] = abs(L["e"]) * 111120
    L["dx"] = abs(L["a"]) * 111120 * np.cos(np.radians(lat_mid))

    s = L["slope"]
    L["slope_cost"] = np.select(
        [s < 5, s < 10, s < 15, s < 25, s < 35],
        [1, 2, 5, 15, 60],
        default=500,
    ).astype("float32")

    lc = L["landcover"][::2, ::2]
    rgba = np.zeros(lc.shape + (4,), dtype="uint8")

    for code, col in LC_COLORS.items():
        mask = lc == code
        rgba[mask, :3] = col
        rgba[mask, 3] = 255

    L["overlay"] = rgba
    return L


def to_rc(L, lat, lon):
    r = int((L["f"] - lat) / -L["e"])
    c = int((lon - L["c"]) / L["a"])
    return (
        min(max(r, 0), L["H"] - 1),
        min(max(c, 0), L["W"] - 1),
    )


def to_latlon(L, rows, cols):
    lat = L["f"] + L["e"] * (np.asarray(rows) + 0.5)
    lon = L["c"] + L["a"] * (np.asarray(cols) + 0.5)
    return lat, lon


def inside(L, lat, lon):
    return (
        L["lat0"] <= lat <= L["lat1"]
        and L["lon0"] <= lon <= L["lon1"]
    )


# ---------------------- COST SURFACES ---------------------

def make_cost(L, w_forest, w_crop, w_built, w_river, use_roads):
    """Balanced cost: slope + land cover + rivers (+ existing roads)."""
    add = np.zeros(L["slope_cost"].shape, dtype="float32")

    weights = {
        10: w_forest,
        20: 1,
        30: 0,
        40: w_crop,
        50: w_built,
        60: 0,
        80: 200,
        90: 10,
    }

    for code, weight in weights.items():
        add[L["landcover"] == code] = weight

    cost = L["slope_cost"] + add
    cost[L["river_mask"]] += w_river

    if use_roads:
        cost[L["road_mask"]] = np.minimum(
            cost[L["road_mask"]], 1.5
        )

    return cost


def make_flat_cost(L, w_river):
    """Slope-dominated cost: finds the gentlest possible alignment."""
    s = np.nan_to_num(L["slope"].astype("float32"), nan=30.0)
    cost = (1.0 + (s / 4.0) ** 2).astype("float32")
    cost[L["landcover"] == 80] += 500
    cost[L["river_mask"]] += w_river
    return cost


def make_short_cost(L):
    """Distance-dominated cost: shortest path that still avoids open water."""
    cost = np.ones(L["slope"].shape, dtype="float32")
    cost[L["landcover"] == 80] = 200
    return cost


# ---------------------- ROUTE ANALYSIS --------------------

def slope_class(values):
    return np.digitize(values, SLOPE_BINS)


def smooth_slope(L, rows, cols, window=5):
    s = np.nan_to_num(L["slope"][rows, cols].astype("float64"), nan=0.0)
    if len(s) < window:
        return s
    kernel = np.ones(window) / window
    padded = np.pad(s, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def route_geometry(L, path):
    """Smoothed slope per point, step lengths (m) and cumulative km."""
    rows, cols = path[:, 0], path[:, 1]
    sm = smooth_slope(L, rows, cols)
    step = np.hypot(
        np.diff(rows) * L["dy"],
        np.diff(cols) * L["dx"],
    )
    cum_km = np.concatenate([[0.0], np.cumsum(step)]) / 1000
    return sm, step, cum_km


def slope_breakdown(sm, step):
    """Kilometres of road in each slope class."""
    seg_slope = (sm[:-1] + sm[1:]) / 2
    cls = slope_class(seg_slope)
    km = np.bincount(cls, weights=step, minlength=len(SLOPE_CLASSES)) / 1000
    return km


def slope_runs(cls):
    """Return (start, end) index pairs of consecutive same-class points."""
    n = len(cls)
    breaks = np.flatnonzero(np.diff(cls)) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [n - 1]])
    return list(zip(starts.tolist(), ends.tolist()))


def find_hotspots(sm, cum_km, threshold=15.0, top=6):
    """Peak of every stretch steeper than threshold; returns the worst few."""
    mask = sm >= threshold
    if not mask.any():
        return []

    edges = np.diff(mask.astype(int))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]

    spots = []
    for a, b in zip(starts, ends):
        k = a + int(np.argmax(sm[a:b]))
        spots.append((k, float(sm[k]), float(cum_km[k])))

    spots.sort(key=lambda x: -x[1])
    return spots[:top]


def metrics(L, rows, cols):
    s = L["slope"][rows, cols]
    lc = L["landcover"][rows, cols]
    rv = L["river_mask"][rows, cols]

    length = (
        np.sum(
            np.hypot(
                np.diff(rows) * L["dy"],
                np.diff(cols) * L["dx"],
            )
        ) / 1000
    )

    return {
        "Length (km)": round(float(length), 2),
        "Average slope (deg)": round(float(s.mean()), 1),
        "Steepest section (deg)": round(float(s.max()), 1),
        "% steeper than 15 deg": round(float((s > 15).mean() * 100), 1),
        "% through forest": round(float((lc == 10).mean() * 100), 1),
        "% through farmland": round(float((lc == 40).mean() * 100), 1),
        "Built-up pixels crossed": int((lc == 50).sum()),
        "Water pixels crossed": int((lc == 80).sum()),
        "River crossings": int(
            np.sum(np.diff(rv.astype(int)) == 1) + int(rv[0])
        ),
        "% on existing roads": round(
            float(L["road_mask"][rows, cols].mean() * 100), 1
        ),
    }


def metric_card(label, value, foot=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------- SESSION STATE ---------------------

defaults = {
    "pts": [],
    "last_click": None,
    "result": None,
    "theme": "Daylight",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------- SIDEBAR ----------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">⌁</div>
            <div>
                <div class="brand-name">TerraPath</div>
                <div class="brand-sub">ROAD ALIGNMENT PLANNER</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Workspace")
    st.caption("Terrain-aware rural road planning")

    theme = st.radio(
        "Appearance",
        ["Daylight", "Night"],
        horizontal=True,
        index=0 if st.session_state.theme == "Daylight" else 1,
    )
    st.session_state.theme = theme
    inject_styles(night=(theme == "Night"))

    st.divider()

    st.markdown("### 01 · Planning scenario")
    scenario = st.radio(
        "Choose your approach",
        [
            "New build (ignore existing roads)",
            "Upgrade existing roads",
        ],
        help=(
            "New build does not give existing roads a lower cost. "
            "Upgrade mode favours existing roads."
        ),
    )

    st.markdown("### 02 · Terrain preferences")
    w_forest = st.slider("Forest penalty", 0, 30, 6)
    w_crop = st.slider("Farmland penalty", 0, 20, 2)
    w_built = st.slider("Built-up area penalty", 0, 60, 15)
    w_river = st.slider("River crossing penalty", 0, 100, 30)

    st.markdown("### 03 · Map layers")
    show_lc = st.checkbox("Land-cover overlay", True)
    show_slope = st.checkbox("Colour road by slope", True)
    show_alt = st.checkbox("Alternative routes", True)
    show_hot = st.checkbox("Steep-section markers", True)

    st.caption(
        "Land-cover colours: forest, farmland, built-up areas, "
        "water and other classes. Road colours show slope steepness."
    )

    st.divider()

    if st.button("↺  Reset points and route", use_container_width=True):
        st.session_state.pts = []
        st.session_state.result = None
        st.session_state.last_click = None
        st.rerun()

    st.markdown(
        """
        <div style="font-size:11px;line-height:1.8;color:var(--muted)">
        <b>Planning note</b><br>
        This tool is an exploratory planning aid. Cost weights are
        assumptions, not engineering specifications.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------- DATA VALIDATION --------------------

if not DATA.exists():
    st.markdown(
        """
        <div class="soft-card">
            <h3>Terrain data not found</h3>
            <p>
                The planner needs <code>data/terrain_layers.npz</code>.
                Place your existing terrain file inside a folder named
                <code>data</code> beside <code>app.py</code>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    L = load_layers()
except Exception as exc:
    st.error(f"Could not load the terrain data: {exc}")
    st.stop()


# ---------------------- HERO -------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">
            <span>✳</span>
            TERRAIN INTELLIGENCE · RURAL CONNECTIVITY
        </div>
        <h1>Find a better path<br>through the landscape.</h1>
        <p>
            Explore terrain, compare road alignments, and understand
            how slope, land cover, rivers, and existing roads influence
            your proposed route.
        </p>
        <div class="hero-chip">🌿 &nbsp; Thoughtful planning for better connections</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------- INTRODUCTION -----------------------

pts = st.session_state.pts

left_intro, right_intro = st.columns([1.6, 1], gap="large")

with left_intro:
    st.markdown(
        '<div class="section-label">YOUR WORKSPACE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">Road alignment studio</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-description">'
        'Select two locations on the terrain map to compare a balanced '
        'route, the shortest route, the flattest route and a straight line. '
        'The recommended road is coloured by slope so steep sections stand out.'
        '</div>',
        unsafe_allow_html=True,
    )

with right_intro:
    if len(pts) == 0:
        pill = "● Awaiting Point A"
    elif len(pts) == 1:
        pill = "● Select Point B"
    else:
        pill = "✓ Points selected"
    st.markdown(
        f'<div style="text-align:right;padding-top:12px">'
        f'<span class="status-pill">{pill}</span></div>',
        unsafe_allow_html=True,
    )


# ---------------------- PROGRESS ---------------------------

p1, p2, p3 = st.columns(3)

for col, number, heading, detail in [
    (p1, "01", "Choose origin", "Click Point A on the map"),
    (p2, "02", "Choose destination", "Click Point B on the map"),
    (p3, "03", "Calculate route", "Explore the routes and metrics"),
]:
    with col:
        st.markdown(
            f"""
            <div class="soft-card">
                <div style="display:flex;gap:12px;align-items:center">
                    <div class="step-number">{number}</div>
                    <div>
                        <div style="font-weight:750;font-size:13px;color:var(--text)">
                            {heading}
                        </div>
                        <div style="font-size:11px;color:var(--muted);margin-top:4px">
                            {detail}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------- ROUTE SEARCH -----------------------

pts = st.session_state.pts

if len(pts) == 2:
    st.markdown("### Ready to calculate?")
    st.caption(
        "Three alignments are calculated: balanced (recommended), "
        "shortest distance and flattest slope."
    )

    if st.button("✦  Find best road alignment", type="primary", use_container_width=True):
        (r1, c1), (r2, c2) = to_rc(L, *pts[0]), to_rc(L, *pts[1])

        if (r1, c1) == (r2, c2):
            st.warning(
                "Point A and Point B fall in the same terrain cell. "
                "Please choose two different locations."
            )
        else:
            try:
                with st.spinner("Analysing terrain and searching for routes..."):
                    cost_sets = [
                        (
                            "Recommended (balanced)",
                            make_cost(
                                L, w_forest, w_crop, w_built, w_river,
                                use_roads=scenario.startswith("Upgrade"),
                            ),
                            "#D95B4B",
                        ),
                        (
                            "Shortest distance",
                            make_short_cost(L),
                            "#2F6FDE",
                        ),
                        (
                            "Flattest slope",
                            make_flat_cost(L, w_river),
                            "#8E44AD",
                        ),
                    ]

                    routes = []
                    for name, cost, color in cost_sets:
                        path, _ = route_through_array(
                            cost,
                            (r1, c1),
                            (r2, c2),
                            fully_connected=True,
                            geometric=True,
                        )
                        path = np.asarray(path)
                        sm, step, cum_km = route_geometry(L, path)
                        routes.append(
                            {
                                "name": name,
                                "color": color,
                                "path": path,
                                "sm": sm,
                                "cum_km": cum_km,
                                "breakdown": slope_breakdown(sm, step),
                                "metrics": metrics(L, path[:, 0], path[:, 1]),
                            }
                        )

                    n = 500
                    sl_rows = np.linspace(r1, r2, n).astype(int)
                    sl_cols = np.linspace(c1, c2, n).astype(int)

                    st.session_state.result = {
                        "routes": routes,
                        "straight": metrics(L, sl_rows, sl_cols),
                        "scenario": scenario,
                    }
                st.rerun()

            except Exception as exc:
                st.error(f"Route calculation failed: {exc}")


# ---------------------- MAP -------------------------------

st.markdown("### Explore the terrain")
st.markdown(
    '<div class="section-description">'
    'Click inside the outlined study area. Your first click sets Point A; '
    'your second click sets Point B. Use the layer control (top right of the '
    'map) to switch routes on and off.'
    '</div>',
    unsafe_allow_html=True,
)

center = [
    (L["lat0"] + L["lat1"]) / 2,
    (L["lon0"] + L["lon1"]) / 2,
]

m = folium.Map(
    location=center,
    zoom_start=10,
    tiles="OpenStreetMap",
    control_scale=True,
)

bounds = [[L["lat0"], L["lon0"]], [L["lat1"], L["lon1"]]]

folium.Rectangle(
    bounds,
    color="#426B4C",
    weight=2,
    fill=True,
    fill_color="#A7C5A0",
    fill_opacity=0.04,
    tooltip="Study area",
).add_to(m)

if show_lc:
    folium.raster_layers.ImageOverlay(
        image=L["overlay"],
        bounds=bounds,
        opacity=0.45,
        name="Land cover",
    ).add_to(m)

res = st.session_state.result

if res is not None:
    main = res["routes"][0]
    alts = res["routes"][1:]

    # ---- alternative routes (drawn first so the main road sits on top)
    if show_alt:
        for alt in alts:
            g = folium.FeatureGroup(name=f'{alt["name"]} route', show=True)
            lat_a, lon_a = to_latlon(L, alt["path"][:, 0], alt["path"][:, 1])
            folium.PolyLine(
                list(zip(lat_a.tolist(), lon_a.tolist())),
                color=alt["color"],
                weight=4,
                opacity=0.9,
                tooltip=(
                    f'{alt["name"]}: {alt["metrics"]["Length (km)"]} km, '
                    f'avg slope {alt["metrics"]["Average slope (deg)"]}°, '
                    f'max {alt["metrics"]["Steepest section (deg)"]}°'
                ),
            ).add_to(g)
            g.add_to(m)

    # ---- straight-line alternative
    if len(pts) == 2:
        g = folium.FeatureGroup(name="Straight line", show=True)
        folium.PolyLine(
            [pts[0], pts[1]],
            color="#243B2E",
            weight=2,
            dash_array="7",
            opacity=0.8,
            tooltip=f'Straight line: {res["straight"]["Length (km)"]} km',
        ).add_to(g)
        g.add_to(m)

    # ---- recommended route
    lat_m, lon_m = to_latlon(L, main["path"][:, 0], main["path"][:, 1])
    lat_list, lon_list = lat_m.tolist(), lon_m.tolist()
    full_coords = list(zip(lat_list, lon_list))

    g_main = folium.FeatureGroup(name="Recommended road", show=True)

    if show_slope:
        # white casing so the coloured road stays visible on any background
        folium.PolyLine(
            full_coords, color="#FFFFFF", weight=10, opacity=0.9
        ).add_to(g_main)

        cls_pts = slope_class(main["sm"])
        for a, b in slope_runs(cls_pts):
            if b <= a:
                continue
            label, colour = SLOPE_CLASSES[int(cls_pts[a])]
            seg_avg = float(main["sm"][a:b + 1].mean())
            folium.PolyLine(
                full_coords[a:b + 1],
                color=colour,
                weight=6,
                opacity=1.0,
                tooltip=f"{label.replace('  ', ' ')} · avg {seg_avg:.1f}°",
            ).add_to(g_main)
    else:
        folium.PolyLine(
            full_coords,
            color=main["color"],
            weight=5,
            opacity=0.95,
            tooltip="Recommended road",
        ).add_to(g_main)

    g_main.add_to(m)

    # ---- steep hotspots
    if show_hot:
        g_hot = folium.FeatureGroup(name="Steep sections", show=True)
        for k, slope_val, km_val in find_hotspots(main["sm"], main["cum_km"]):
            folium.CircleMarker(
                (lat_list[k], lon_list[k]),
                radius=8,
                color="#FFFFFF",
                weight=2,
                fill=True,
                fill_color="#7B1E3A",
                fill_opacity=1,
                tooltip=f"Steep section: {slope_val:.1f}° at {km_val:.2f} km",
            ).add_to(g_hot)
        g_hot.add_to(m)

    # ---- river crossings
    rv = L["river_mask"][main["path"][:, 0], main["path"][:, 1]].astype(int)
    cross_idx = np.flatnonzero(np.diff(rv) == 1) + 1
    if len(cross_idx):
        g_riv = folium.FeatureGroup(name="River crossings", show=True)
        for k in cross_idx:
            folium.CircleMarker(
                (lat_list[k], lon_list[k]),
                radius=6,
                color="#FFFFFF",
                weight=2,
                fill=True,
                fill_color="#1E63D6",
                fill_opacity=1,
                tooltip="River crossing",
            ).add_to(g_riv)
        g_riv.add_to(m)

    # ---- legend
    swatches = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
        f'<span style="width:22px;height:5px;border-radius:3px;background:{c};display:inline-block"></span>'
        f'{lab.replace("  ", " ")}</div>'
        for lab, c in SLOPE_CLASSES
    )
    legend = f"""
    <div style="position:fixed;bottom:34px;right:12px;z-index:9999;
                background:rgba(255,255,255,.95);padding:10px 13px;
                border-radius:10px;border:1px solid #ccc;font-size:11px;
                font-family:sans-serif;color:#243B2E;line-height:1.3;
                box-shadow:0 2px 8px rgba(0,0,0,.18)">
        <b>Road slope</b>
        {swatches}
        <hr style="margin:6px 0;border:none;border-top:1px solid #ddd">
        <div style="display:flex;align-items:center;gap:8px;margin:3px 0">
            <span style="width:22px;height:4px;background:#2F6FDE;display:inline-block"></span>Shortest route</div>
        <div style="display:flex;align-items:center;gap:8px;margin:3px 0">
            <span style="width:22px;height:4px;background:#8E44AD;display:inline-block"></span>Flattest route</div>
        <div style="display:flex;align-items:center;gap:8px;margin:3px 0">
            <span style="width:22px;border-top:2px dashed #243B2E;display:inline-block"></span>Straight line</div>
        <div style="display:flex;align-items:center;gap:8px;margin:3px 0">
            <span style="width:10px;height:10px;border-radius:50%;background:#7B1E3A;display:inline-block"></span>Steepest spots</div>
        <div style="display:flex;align-items:center;gap:8px;margin:3px 0">
            <span style="width:10px;height:10px;border-radius:50%;background:#1E63D6;display:inline-block"></span>River crossing</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(" ".join(legend.split())))

for i, point in enumerate(pts):
    folium.Marker(
        point,
        tooltip=f"Point {'AB'[i]}",
        popup=f"Point {'AB'[i]}",
        icon=folium.Icon(
            color="green" if i == 0 else "red",
            icon="flag",
        ),
    ).add_to(m)

folium.LayerControl(collapsed=True).add_to(m)

map_output = st_folium(
    m,
    key="terrain_map",
    height=600,
    use_container_width=True,
    returned_objects=["last_clicked"],
)

click = map_output.get("last_clicked") if map_output else None

if click:
    point = (round(click["lat"], 6), round(click["lng"], 6))

    if point != st.session_state.last_click:
        st.session_state.last_click = point

        if not inside(L, *point):
            st.warning(
                "That location is outside the study area. "
                "Please click inside the outlined rectangle."
            )
        else:
            if len(st.session_state.pts) >= 2:
                st.session_state.pts = [point]
            else:
                st.session_state.pts = st.session_state.pts + [point]

            st.session_state.result = None
            st.rerun()


# ---------------------- RESULTS ----------------------------

res = st.session_state.result

if res is not None:
    st.divider()

    main = res["routes"][0]
    route = main["metrics"]
    straight = res["straight"]

    st.markdown(
        '<div class="section-label">ROUTE ANALYSIS</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">Your alignment at a glance</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="section-description">Scenario: {res["scenario"]}. '
        'Figures below describe the recommended (balanced) route and compare '
        'it with the straight-line alternative.'
        '</div>',
        unsafe_allow_html=True,
    )

    steep_km = float(main["breakdown"][3:].sum())

    a, b, c, d = st.columns(4, gap="medium")

    with a:
        metric_card(
            "PLANNED ROAD LENGTH",
            f'{route["Length (km)"]:.2f} km',
            f'{route["Length (km)"] - straight["Length (km)"]:+.2f} km vs straight',
        )
    with b:
        metric_card(
            "AVERAGE SLOPE",
            f'{route["Average slope (deg)"]:.1f}°',
            f'{route["Average slope (deg)"] - straight["Average slope (deg)"]:+.1f}° vs straight',
        )
    with c:
        metric_card(
            "STEEPEST SECTION",
            f'{route["Steepest section (deg)"]:.1f}°',
            f'{route["Steepest section (deg)"] - straight["Steepest section (deg)"]:+.1f}° vs straight',
        )
    with d:
        metric_card(
            "RIVER CROSSINGS",
            str(route["River crossings"]),
            f'{route["River crossings"] - straight["River crossings"]:+d} vs straight',
        )

    e, f, g, h = st.columns(4, gap="medium")

    with e:
        metric_card(
            "ROAD STEEPER THAN 15°",
            f"{steep_km:.2f} km",
            f'{route["% steeper than 15 deg"]:.1f}% of the route',
        )
    with f:
        metric_card(
            "THROUGH FOREST",
            f'{route["% through forest"]:.1f}%',
            f'{route["% through forest"] - straight["% through forest"]:+.1f}% vs straight',
        )
    with g:
        metric_card(
            "ON EXISTING ROADS",
            f'{route["% on existing roads"]:.1f}%',
            "Higher in upgrade mode",
        )
    with h:
        metric_card(
            "STEEP HOTSPOTS",
            str(len(find_hotspots(main["sm"], main["cum_km"], top=99))),
            "Stretches above 15°",
        )

    # ---- slope breakdown bars
    st.markdown("### Slope along the recommended road")
    total_km = float(main["breakdown"].sum()) or 1.0

    rows_html = ""
    for (label, colour), km in zip(SLOPE_CLASSES, main["breakdown"]):
        pct = km / total_km * 100
        name = label.replace("  ", " ")
        rows_html += (
            '<div class="slope-row">'
            f'<span class="slope-swatch" style="background:{colour}"></span>'
            f'<span class="slope-name">{name}</span>'
            '<div class="slope-bar-bg">'
            f'<div class="slope-bar" style="width:{pct:.1f}%;background:{colour}"></div>'
            '</div>'
            f'<span class="slope-val">{km:.2f} km · {pct:.0f}%</span>'
            '</div>'
        )
    st.markdown(f'<div class="soft-card">{rows_html}</div>', unsafe_allow_html=True)

    # ---- slope profile
    st.markdown("### Slope profile")
    st.caption("Slope (degrees) at every point along each route, from Point A to Point B.")

    profile = pd.DataFrame(
        {
            r["name"]: pd.Series(
                r["sm"], index=np.round(r["cum_km"], 3)
            )
            for r in res["routes"]
        }
    )
    profile = profile.groupby(level=0).mean().sort_index().interpolate()
    profile.index.name = "Distance along route (km)"
    st.line_chart(
        profile,
        color=[r["color"] for r in res["routes"]],
        height=260,
    )

    # ---- comparison table
    st.markdown("### Detailed comparison")

    comparison = pd.DataFrame({"Metric": list(route.keys())})
    for r in res["routes"]:
        comparison[r["name"]] = list(r["metrics"].values())
    comparison["Straight line"] = list(straight.values())

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )

    # ---- export
    features = []
    for r in res["routes"]:
        la, lo = to_latlon(L, r["path"][:, 0], r["path"][:, 1])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": r["name"],
                    "scenario": res["scenario"],
                    "length_km": r["metrics"]["Length (km)"],
                    "average_slope_deg": r["metrics"]["Average slope (deg)"],
                    "max_slope_deg": r["metrics"]["Steepest section (deg)"],
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(x), float(y)] for y, x in zip(la, lo)
                    ],
                },
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}

    st.markdown("### Export your routes")
    st.caption(
        "Download all three alignments as a GeoJSON FeatureCollection "
        "for compatible GIS applications."
    )

    st.download_button(
        "↓  Download routes (.geojson)",
        data=json.dumps(geojson, indent=2),
        file_name="terrapath_routes.geojson",
        mime="application/geo+json",
        use_container_width=True,
    )


    # ---- readable report downloads
    st.markdown("### Download a readable report")
    st.caption(
        "The report opens in any browser (print it to save as PDF). "
        "The CSV table opens in Excel."
    )

    breakdown_df = pd.DataFrame(
        {
            "Slope class": [lab.replace("  ", " ") for lab, _ in SLOPE_CLASSES],
            "Length (km)": [round(float(k), 2) for k in main["breakdown"]],
            "Share of road (%)": [
                round(float(k) / (float(main["breakdown"].sum()) or 1) * 100, 1)
                for k in main["breakdown"]
            ],
        }
    )

    hot_rows = "".join(
        f"<tr><td>{km_val:.2f} km</td><td>{sv:.1f}&deg;</td></tr>"
        for _, sv, km_val in find_hotspots(main["sm"], main["cum_km"], top=10)
    ) or "<tr><td colspan='2'>No stretches steeper than 15&deg;</td></tr>"

    report_html = (
        "<html><head><meta charset='utf-8'><title>TerraPath route report</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;"
        "margin:30px auto;color:#243B2E;line-height:1.5}"
        "h1{margin-bottom:0}h2{margin-top:32px;border-bottom:2px solid #DFE5D9;"
        "padding-bottom:6px}table{border-collapse:collapse;width:100%;font-size:14px}"
        "th,td{border:1px solid #DFE5D9;padding:7px 10px;text-align:left}"
        "th{background:#EAF0E6}.muted{color:#718073;font-size:13px}</style></head><body>"
        "<h1>TerraPath route report</h1>"
        f"<p class='muted'>Scenario: {res['scenario']}</p>"
        f"<p>Point A: {pts[0][0]:.5f}, {pts[0][1]:.5f}<br>"
        f"Point B: {pts[1][0]:.5f}, {pts[1][1]:.5f}</p>"
        "<h2>Recommended road at a glance</h2>"
        f"<p>Length: <b>{route['Length (km)']} km</b> &middot; "
        f"Average slope: <b>{route['Average slope (deg)']}&deg;</b> &middot; "
        f"Steepest section: <b>{route['Steepest section (deg)']}&deg;</b> &middot; "
        f"River crossings: <b>{route['River crossings']}</b></p>"
        "<h2>Slope along the recommended road</h2>"
        + breakdown_df.to_html(index=False)
        + "<h2>Steepest spots</h2><table><tr><th>Distance from Point A</th>"
        "<th>Slope</th></tr>" + hot_rows + "</table>"
        "<h2>Route comparison</h2>"
        + comparison.to_html(index=False)
        + "<p class='muted'>Exploratory planning only. Validate alignments with "
        "qualified engineering, environmental and field assessments.</p>"
        "</body></html>"
    )

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "↓  Download report (.html)",
            data=report_html,
            file_name="terrapath_report.html",
            mime="text/html",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "↓  Download comparison table (.csv)",
            data=comparison.to_csv(index=False).encode("utf-8-sig"),
            file_name="terrapath_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ---------------------- FOOTER -----------------------------

st.markdown(
    """
    <div class="footer">
        <b>TerraPath</b> · Village Road Alignment Planner<br>
        Terrain data: Copernicus GLO-30 DEM, ESA WorldCover 2021,
        and OpenStreetMap, as supplied by the project.<br>
        Exploratory planning only. Validate alignments with qualified
        engineering, environmental, and field assessments.
    </div>
    """,
    unsafe_allow_html=True,
)