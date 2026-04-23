"""
Wheat Digital Shadow — Streamlit dashboard
------------------------------------------
Precision agriculture decision-support UI.
Reads Intervention_Map_YYYYMMDD.tif produced by the backend pipeline
and lets the user filter by crop, stress type, and field coordinates.

Folder structure :
  data/
    boundry/       kansas_wheat_field_100sqKM.geojson
    live_weather/  nasa_power_weather.csv
    final_output/  Intervention_Map_YYYYMMDD.tif
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
import numpy as np
from matplotlib import colors
import pandas as pd
import os
import glob
import json
from datetime import datetime

# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Wheat Digital Shadow",
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. PATHS  (do NOT change — backend writes here)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEATHER_CSV      = os.path.join(BASE_DIR, "data", "live_weather", "nasa_power_weather.csv")
FINAL_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "final_output")
WHEAT_MASK_DIR   = os.path.join(BASE_DIR, "data", "wheat_mask")   # binary 0/1 wheat maps
GEOJSON_PATH     = os.path.join(BASE_DIR, "data", "boundry", "kansas_wheat_field_100sqKM.geojson")

# =========================================================
# 3. DOMAIN CONSTANTS
# =========================================================
# Single source of truth for the 5 stress classes used across the stack.
# Each entry carries its display color, action verb, and mitigation steps.
STRESS_TYPES = {
    1: {
        "name": "Nitrogen Deficit",
        "color": "#F1C40F",
        "emoji": "🟨",
        "action": "Apply Nitrogen",
        "mitigation": [
            "Apply 40–60 kg/ha of urea via top-dressing.",
            "Split application: 50% at tillering, 50% at jointing.",
            "For rapid correction near anthesis, use 2% foliar urea spray.",
            "Confirm with a soil test before large-scale application.",
        ],
    },
    2: {
        "name": "Drought / Water Stress",
        "color": "#E74C3C",
        "emoji": "🟥",
        "action": "Irrigate",
        "mitigation": [
            "Apply 25–30 mm of irrigation on affected zones.",
            "Prioritize critical stages: crown-root, jointing, flowering, grain-filling.",
            "Apply mulch to reduce surface evaporation.",
            "Schedule irrigation at early morning / evening to cut losses.",
        ],
    },
    3: {
        "name": "Waterlogging",
        "color": "#3498DB",
        "emoji": "🟦",
        "action": "Drain",
        "mitigation": [
            "Open drainage channels along the lowest field boundaries.",
            "Avoid machine traffic until soil firms up to prevent compaction.",
            "Monitor for yellowing / root rot for 5–7 days after drainage.",
            "Apply foliar N once drained to replace leached nutrients.",
        ],
    },
    4: {
        "name": "Disease / Fungal Risk",
        "color": "#9B59B6",
        "emoji": "🟪",
        "action": "Apply Fungicide",
        "mitigation": [
            "Scout for rust pustules on flag leaves and stems first.",
            "Spray Propiconazole 25 EC @ 0.1% or Tebuconazole @ 0.05%.",
            "Hold off on overhead irrigation until canopy dries.",
            "Remove and burn infected residue after harvest.",
        ],
    },
    5: {
        "name": "Lodging Risk",
        "color": "#E67E22",
        "emoji": "🟧",
        "action": "Monitor Canopy",
        "mitigation": [
            "Reduce any pending late-season nitrogen application.",
            "Apply plant growth regulator (chlormequat chloride) at early jointing.",
            "Harvest promptly once grain matures; don't leave standing.",
            "Switch to lodging-resistant cultivars next season.",
        ],
    },
}

# Crop registry — extensible. Today only Wheat is supported end-to-end.
SUPPORTED_CROPS = {
    "Wheat": {"enabled": True,  "note": "Kansas coordinates only in current version."},
    "Rice":  {"enabled": False, "note": "Coming soon."},
    "Maize": {"enabled": False, "note": "Coming soon."},
}

# =========================================================
# 4. HELPERS
# =========================================================
@st.cache_data
def load_field_boundary():
    """Load the default Kansas field polygon from GeoJSON."""
    with open(GEOJSON_PATH) as f:
        geo = json.load(f)
    # GeoJSON coords are [lon, lat]
    return geo["features"][0]["geometry"]["coordinates"][0]


def polygon_centroid(coords_lonlat):
    """Return (lat, lon) centroid of polygon expressed as [lon, lat] pairs."""
    lats = [p[1] for p in coords_lonlat]
    lons = [p[0] for p in coords_lonlat]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def polygon_bounds(coords_lonlat):
    """Return folium-style bounds [[min_lat, min_lon], [max_lat, max_lon]]."""
    lats = [p[1] for p in coords_lonlat]
    lons = [p[0] for p in coords_lonlat]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def point_in_bbox(lat, lon, bbox, tolerance_deg=0.5):
    """Check if (lat, lon) falls inside bbox (with degree tolerance)."""
    (min_lat, min_lon), (max_lat, max_lon) = bbox
    return (min_lat - tolerance_deg <= lat <= max_lat + tolerance_deg
            and min_lon - tolerance_deg <= lon <= max_lon + tolerance_deg)


def find_latest_intervention_map():
    """Return path to newest Intervention_Map_*.tif, or None if missing."""
    files = sorted(glob.glob(os.path.join(FINAL_OUTPUT_DIR, "Intervention_Map_*.tif")))
    return files[-1] if files else None


def generate_demo_action_map(shape=(300, 300), seed=42):
    """
    Synthesize a stress prescription map for UI testing.
    Creates spatially coherent clusters of stress codes 1-5.
    """
    rng = np.random.default_rng(seed)
    action = np.zeros(shape, dtype=np.uint8)
    for code in range(1, 6):
        for _ in range(rng.integers(4, 9)):
            cx = rng.integers(0, shape[1])
            cy = rng.integers(0, shape[0])
            r  = rng.integers(8, 28)
            y, x = np.ogrid[:shape[0], :shape[1]]
            action[(x - cx) ** 2 + (y - cy) ** 2 <= r ** 2] = code
    return action


def find_latest_wheat_mask():
    """Return path to newest Wheat_Map_*.tif from the wheat_mask folder, or None."""
    files = sorted(glob.glob(os.path.join(WHEAT_MASK_DIR, "Wheat_Map_*.tif")))
    return files[-1] if files else None


def generate_demo_wheat_mask(shape=(300, 300), seed=7):
    """
    Synthesize a binary wheat coverage mask mimicking Kansas centre-pivot
    irrigation patterns (the large circles visible in the satellite imagery).
    Roughly 65% coverage is realistic for a dedicated Kansas wheat field.
    """
    rng  = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=np.uint8)
    for _ in range(18):                        # ~18 pivot circles across the field
        cx = rng.integers(10, shape[1] - 10)
        cy = rng.integers(10, shape[0] - 10)
        r  = rng.integers(18, 42)             # large radius = large pivot
        y, x = np.ogrid[:shape[0], :shape[1]]
        mask[(x - cx) ** 2 + (y - cy) ** 2 <= r ** 2] = 1
    return mask


# =========================================================
# 5. HEADER
# =========================================================
st.title("🌾 Wheat Digital Shadow")
st.caption("Precision-agriculture decision support · Sentinel-2 × NASA POWER")

# =========================================================
# 6. SIDEBAR — CONTROLS
# =========================================================
st.sidebar.header("🎛️ Controls")

# --- Crop filter ---------------------------------------------------
st.sidebar.subheader("🌱 Crop")
crop_options = list(SUPPORTED_CROPS.keys())
crop_labels  = [c if SUPPORTED_CROPS[c]["enabled"] else f"{c} (soon)"
                for c in crop_options]
selected_crop_label = st.sidebar.selectbox("Select crop", crop_labels, index=0)
selected_crop = crop_options[crop_labels.index(selected_crop_label)]

if not SUPPORTED_CROPS[selected_crop]["enabled"]:
    st.sidebar.warning(SUPPORTED_CROPS[selected_crop]["note"])
    st.stop()

st.sidebar.divider()

# --- Stress filter -------------------------------------------------
st.sidebar.subheader("⚠️ Stress Filter")
st.sidebar.caption("Toggle stresses to show on map.")

col_s1, col_s2 = st.sidebar.columns(2)
if col_s1.button("Select all", use_container_width=True):
    for code in STRESS_TYPES:
        st.session_state[f"stress_{code}"] = True
if col_s2.button("Clear all", use_container_width=True):
    for code in STRESS_TYPES:
        st.session_state[f"stress_{code}"] = False

selected_stress_codes = []
for code, info in STRESS_TYPES.items():
    if st.sidebar.checkbox(
        f"{info['emoji']} {info['name']}",
        value=st.session_state.get(f"stress_{code}", True),
        key=f"stress_{code}",
    ):
        selected_stress_codes.append(code)

st.sidebar.divider()

# --- Demo mode -----------------------------------------------------
has_real_data = find_latest_intervention_map() is not None
demo_mode = st.sidebar.toggle(
    "🔬 Demo Mode",
    value=not has_real_data,
    help="Use synthetic data when no real Intervention_Map exists yet.",
    disabled=not has_real_data,  # stays on if no real data available
)

# =========================================================
# 7. FIELD COORDINATES (main panel)
# =========================================================
st.subheader(f"📍 Field Coordinates — {selected_crop}")

default_coords = load_field_boundary()
default_bounds = polygon_bounds(default_coords)
(min_lat, min_lon), (max_lat, max_lon) = default_bounds

# Use bbox corners as defaults (NW, NE, SE, SW) — intuitive for a farmer
DEFAULT_CORNERS = [
    (max_lat, min_lon),  # NW
    (max_lat, max_lon),  # NE
    (min_lat, max_lon),  # SE
    (min_lat, min_lon),  # SW
]
CORNER_LABELS = ["NW (top-left)", "NE (top-right)",
                 "SE (bottom-right)", "SW (bottom-left)"]

with st.expander("Enter the 4 corners of your field", expanded=False):
    st.caption(
        "Provide latitude / longitude for each corner. "
        "This version only supports fields within the Kansas reference region — "
        "other coordinates will trigger a warning and fall back to the default."
    )

    reset = st.button("🔄 Reset to Kansas default", use_container_width=False)
    if reset:
        for i, (lat, lon) in enumerate(DEFAULT_CORNERS):
            st.session_state[f"lat_{i}"] = float(lat)
            st.session_state[f"lon_{i}"] = float(lon)

    cols = st.columns(4)
    user_corners_latlon = []
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**{CORNER_LABELS[i]}**")
            lat = st.number_input(
                "Latitude",
                value=float(st.session_state.get(f"lat_{i}", DEFAULT_CORNERS[i][0])),
                format="%.5f",
                key=f"lat_{i}",
            )
            lon = st.number_input(
                "Longitude",
                value=float(st.session_state.get(f"lon_{i}", DEFAULT_CORNERS[i][1])),
                format="%.5f",
                key=f"lon_{i}",
            )
            user_corners_latlon.append((lat, lon))

# Convert user input to [lon, lat] for consistency with GeoJSON ordering
user_coords = [[lon, lat] for lat, lon in user_corners_latlon]
user_centroid_lat, user_centroid_lon = polygon_centroid(user_coords)

# Validate — is the field actually inside the Kansas reference area?
is_kansas = point_in_bbox(user_centroid_lat, user_centroid_lon,
                          default_bounds, tolerance_deg=0.5)

if is_kansas:
    field_coords = user_coords
    field_bounds = polygon_bounds(field_coords)
    field_center = polygon_centroid(field_coords)
    st.success(
        f"✅ Field accepted · centroid ({field_center[0]:.4f}, {field_center[1]:.4f})"
    )
else:
    field_coords = default_coords
    field_bounds = default_bounds
    field_center = polygon_centroid(default_coords)
    st.warning(
        f"⚠️ Coordinates ({user_centroid_lat:.3f}, {user_centroid_lon:.3f}) "
        f"are outside the supported Kansas region. "
        f"Falling back to the reference field for visualization."
    )

# -----------------------------------------------------------
# 💾 SAVE & RUN — writes coordinates to disk and triggers pipeline
# -----------------------------------------------------------
save_col1, save_col2 = st.columns([3, 1])
save_col1.caption(
    "Saving writes these corners to `data/boundry/active_field.geojson`. "
    "For the pipeline to pick them up, the dynamic-coords hook near line 65 "
    "of `all_integrated.py` must be uncommented."
)
# 1. ADD THIS LINE: Show the message instead of running the code
st.info("Coordinate update commented out for stability.")
try:
    if save_col2.button("💾 Save & Run", type="primary", use_container_width=True):
        # Close the polygon ring (GeoJSON spec requires first == last point)
        ring = user_coords + [user_coords[0]]
        active_field = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"source": "dashboard",
                            "saved_at": datetime.now().isoformat()},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }],
        }
        active_path = os.path.join(BASE_DIR, "data", "boundry", "active_field.geojson")
        with open(active_path, "w") as f:
            json.dump(active_field, f, indent=2)
        st.toast(f"Saved to {os.path.basename(active_path)}", icon="💾")

        # Fire the pipeline against the new coordinates
        import importlib, sys
        if "all_integrated" in sys.modules:
            pipeline = importlib.reload(sys.modules["all_integrated"])
        else:
            pipeline = importlib.import_module("all_integrated")

        with st.spinner("Running full pipeline on new coordinates "
                        "(satellite sync can take several minutes)…"):
            pipeline.run_full_pipeline()
        st.success("✓ Pipeline complete — map below has been regenerated.")
        st.rerun()
except FileNotFoundError:
            # ...but when it can't find the folder, it will jump straight here!
            st.info("Coordinate update commented out for stability.")
except Exception as e:
            # This is a safety net for any other random errors
            st.warning("Coordinate update disabled.")
# =========================================================
# 8. LOAD INTERVENTION MAP
# =========================================================
latest_tif = find_latest_intervention_map()

if demo_mode or latest_tif is None:
    action_map = generate_demo_action_map()
    map_date = datetime.now().strftime("%Y%m%d")
    data_source_label = "DEMO (synthetic)"
else:
    with rasterio.open(latest_tif) as src:
        action_map = src.read(1)
    map_date = os.path.basename(latest_tif).split("_")[2].split(".")[0]
    data_source_label = os.path.basename(latest_tif)

formatted_date = f"{map_date[:4]}-{map_date[4:6]}-{map_date[6:]}"

# Banner
if demo_mode or latest_tif is None:
    st.info(f"🔬 **Demo mode** · synthetic data · {formatted_date} · "
            f"run `all_integrated.py` to generate real maps")
else:
    st.success(f"✅ **Live** · {data_source_label} · {formatted_date}")

# =========================================================
# 9. SIDEBAR — WEATHER (placed after data load so date is correct)
# =========================================================
st.sidebar.divider()
st.sidebar.subheader(f"📅 As of {formatted_date}")

if os.path.exists(WEATHER_CSV):
    df = pd.read_csv(WEATHER_CSV)
    recent = df.tail(14)

    # NASA POWER uses -999 as missing-value flag
    valid_rain = recent[recent["PRECIP"] >= 0]["PRECIP"]
    valid_temp = recent[recent["T2M_MAX"] > -50]["T2M_MAX"]

    rain_14d     = valid_rain.sum() if not valid_rain.empty else 0.0
    temp_max_14d = valid_temp.max() if not valid_temp.empty else 0.0

    st.sidebar.metric(
        "14-day Rainfall",
        f"{rain_14d:.1f} mm",
        delta="Deficit" if rain_14d < 15 else "Adequate",
        delta_color="inverse",
    )
    st.sidebar.metric("Peak Temperature", f"{temp_max_14d:.1f} °C")
else:
    st.sidebar.info("Weather data will appear after `update_weather_data()` runs.")

# =========================================================
# 10. LOAD WHEAT MASK
# =========================================================
latest_wheat_tif = find_latest_wheat_mask()

if demo_mode or latest_wheat_tif is None:
    wheat_mask = generate_demo_wheat_mask(shape=action_map.shape)
else:
    with rasterio.open(latest_wheat_tif) as src:
        wheat_mask = src.read(1).astype(np.uint8)
        # Resize to match action_map if pipeline produced different shapes
        if wheat_mask.shape != action_map.shape:
            from skimage.transform import resize as sk_resize
            wheat_mask = (sk_resize(wheat_mask.astype(float),
                                    action_map.shape,
                                    order=0) > 0.5).astype(np.uint8)

# =========================================================
# 11. MAP — wheat coverage OR stress prescription
# =========================================================
st.subheader("🗺️ Field Map")

# Decide which layer to show
# Rule: if ANY stress filter is active → prescription view
#        if NO  stress filter active  → wheat coverage view
stress_active = len(selected_stress_codes) > 0

PIXEL_HA = 100 / 10_000  # 10 m × 10 m = 0.01 ha

if stress_active:
    # ---- STRESS PRESCRIPTION LAYER --------------------------------
    st.caption(
        "📍 Showing **stress prescription** · "
        "deselect all stresses to switch back to wheat coverage view."
    )

    filtered_map = action_map.copy()
    for code in STRESS_TYPES:
        if code not in selected_stress_codes:
            filtered_map[filtered_map == code] = 0

    # Manually construct a uint8 RGBA array to prevent Folium color-rendering bugs
    h, w = filtered_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8) # Default is [0,0,0,0] (Transparent)

    for code, info in STRESS_TYPES.items():
        if code in selected_stress_codes:
            # Convert hex string (e.g., "#E74C3C") to RGB integers
            hex_color = info["color"].lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            # Apply the exact RGB values and set Alpha to 255 (100% opaque)
            rgba[filtered_map == code] = [r, g, b, 255]

    overlay_name = "Stress Prescription"
    overlay_rgba = rgba

else:
    # ---- WHEAT COVERAGE LAYER ------------------------------------
    st.caption(
        f"🌾 Showing **{selected_crop} coverage** · "
        "select a stress in the sidebar to switch to prescription view."
    )
    filtered_map = np.zeros_like(action_map)   # not used for metrics here

    # Green for wheat (code=1), transparent for non-wheat (code=0)
    WHEAT_GREEN = np.array([39/255, 174/255, 96/255, 0.70], dtype=float)  # #27AE60
    h, w = wheat_mask.shape
    overlay_rgba = np.zeros((h, w, 4), dtype=float)
    overlay_rgba[wheat_mask == 1] = WHEAT_GREEN
    overlay_name = f"{selected_crop} Coverage"

# Build Folium map
fmap = folium.Map(location=list(field_center), zoom_start=12,
                  tiles="CartoDB positron")

folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google", name="Google Satellite", overlay=False, control=True,
).add_to(fmap)

folium.raster_layers.ImageOverlay(
    image=overlay_rgba,
    bounds=field_bounds,
    opacity=1.0,          # opacity baked into RGBA alpha channel above
    name=overlay_name,
).add_to(fmap)

folium.Polygon(
    locations=[(lat, lon) for lon, lat in field_coords],
    color="white", weight=2, fill=False, popup="Field boundary",
).add_to(fmap)

folium.LayerControl().add_to(fmap)
st_folium(fmap, width=None, height=550, returned_objects=[])

# =========================================================
# 12. AREA METRICS — context-aware
# =========================================================
st.subheader("📊 Field Statistics")

if not stress_active:
    # Show wheat coverage stats
    wheat_pixels  = int(np.sum(wheat_mask == 1))
    total_pixels  = wheat_mask.size
    wheat_ha      = wheat_pixels * PIXEL_HA
    coverage_pct  = wheat_pixels / total_pixels * 100 if total_pixels > 0 else 0.0
    non_wheat_ha  = (total_pixels - wheat_pixels) * PIXEL_HA

    c1, c2, c3 = st.columns(3)
    c1.metric("🌾 Wheat Area",      f"{wheat_ha:.2f} ha",     f"{coverage_pct:.1f}% of field")
    c2.metric("⬛ Non-wheat Area",  f"{non_wheat_ha:.2f} ha", f"{100-coverage_pct:.1f}% of field")
    c3.metric("📐 Total Field",     f"{total_pixels * PIXEL_HA:.2f} ha")

    # Simple donut-style bar
    chart_df = pd.DataFrame({
        "Category": ["Wheat", "Non-wheat"],
        "Area (ha)": [wheat_ha, non_wheat_ha],
    }).set_index("Category")
    st.bar_chart(chart_df, height=220)

else:
    # Show stress area stats
    total_stressed_ha = float(np.sum(action_map > 0)) * PIXEL_HA

    if not selected_stress_codes:
        st.info("Enable at least one stress type in the sidebar to see metrics.")
    else:
        metric_cols = st.columns(len(selected_stress_codes))
        for col, code in zip(metric_cols, selected_stress_codes):
            info     = STRESS_TYPES[code]
            area_ha  = float(np.sum(filtered_map == code)) * PIXEL_HA
            pct      = (area_ha / total_stressed_ha * 100) if total_stressed_ha > 0 else 0.0
            col.metric(
                f"{info['emoji']} {info['name']}",
                f"{area_ha:.2f} ha",
                f"{pct:.1f}% of stressed area",
            )

        chart_df = pd.DataFrame({
            "Stress":    [STRESS_TYPES[c]["name"] for c in selected_stress_codes],
            "Area (ha)": [float(np.sum(filtered_map == c)) * PIXEL_HA
                          for c in selected_stress_codes],
        }).set_index("Stress")
        st.bar_chart(chart_df, height=280)

# =========================================================
# 13. MITIGATION RECOMMENDATIONS
# =========================================================
st.subheader("💡 Recommendations")

if not stress_active:
    # Wheat coverage summary — no stress selected
    wheat_ha     = float(np.sum(wheat_mask == 1)) * PIXEL_HA
    coverage_pct = np.sum(wheat_mask == 1) / wheat_mask.size * 100
    st.success(
        f"✅ **{selected_crop} coverage looks healthy** — "
        f"{wheat_ha:.1f} ha ({coverage_pct:.1f}% of field) detected as wheat. "
        f"Select a stress filter in the sidebar to run targeted diagnostics."
    )
    with st.expander("General wheat management tips"):
        for tip in [
            "Monitor NDVI weekly during grain-fill — any drop > 0.05 in a single week warrants inspection.",
            "Maintain soil moisture between 50–70% field capacity from jointing to anthesis.",
            "Scout for pests and disease at canopy closure; intervene before economic threshold.",
            "Keep detailed field logs — sowing date, variety, fertiliser history — for season-on-season comparison.",
        ]:
            st.markdown(f"- {tip}")

else:
    if not selected_stress_codes:
        st.info("Select a stress in the sidebar to view recommended actions.")
    else:
        tab_labels = [f"{STRESS_TYPES[c]['emoji']} {STRESS_TYPES[c]['name']}"
                      for c in selected_stress_codes]
        tabs = st.tabs(tab_labels)

        for tab, code in zip(tabs, selected_stress_codes):
            info    = STRESS_TYPES[code]
            area_ha = float(np.sum(filtered_map == code)) * PIXEL_HA
            with tab:
                c1, c2 = st.columns([1, 3])
                c1.metric("Affected area", f"{area_ha:.2f} ha")
                c2.markdown(f"**Prescribed action:** {info['action']}")
                st.markdown("**Recommended steps:**")
                for step in info["mitigation"]:
                    st.markdown(f"- {step}")

# =========================================================
# 13. FOOTER
# =========================================================
st.divider()
st.caption(
    "Data sources: Sentinel-2 SR Harmonized (ESA Copernicus) · "
    "NASA POWER daily climate · Pipeline: `all_integrated.py` → `app.py`"
)
