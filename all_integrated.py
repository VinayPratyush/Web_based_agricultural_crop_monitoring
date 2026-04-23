

import os
import json
import glob
import argparse
import datetime as dt
import traceback
import warnings
from typing import Callable

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import rasterio
from rasterio.windows import Window

try:
    import ee
    import geemap
    _EE_AVAILABLE = True
except Exception:
    _EE_AVAILABLE = False


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")

LIVE_DIR           = os.path.join(DATA_DIR, "live")
WEATHER_FOLDER     = os.path.join(DATA_DIR, "live_weather")
WHEAT_MASK_FOLDER  = os.path.join(DATA_DIR, "wheat_mask")
OUTPUT_FOLDER      = os.path.join(DATA_DIR, "final_output")
HISTORY_FOLDER     = os.path.join(DATA_DIR, "history")
STATE_FOLDER       = os.path.join(DATA_DIR, "state")

GEOJSON_PATH  = os.path.join(DATA_DIR, "boundry", "kansas_wheat_field_100sqKM.geojson")

# -----------------------------------------------------------
# DYNAMIC COORDINATES HOOK  
# If the dashboard has saved an active field, prefer it over the default.
# -----------------------------------------------------------
# _active_geojson = os.path.join(DATA_DIR, "boundry", "active_field.geojson")
# if os.path.exists(_active_geojson):
#     print(f"[coords] Using dashboard-supplied field: {_active_geojson}")
#     GEOJSON_PATH = _active_geojson

WEATHER_CSV   = os.path.join(WEATHER_FOLDER, "nasa_power_weather.csv")
DOWNLOAD_LOG  = os.path.join(LIVE_DIR, "download_log.txt")
RUN_LOG_CSV   = os.path.join(HISTORY_FOLDER, "run_log.csv")
STATE_JSON    = os.path.join(STATE_FOLDER, "field_state.json")

# Trained model + normalization
MODEL_PATH = os.path.join(BASE_DIR, "wheat_model.pth")
MEAN_PATH  = os.path.join(BASE_DIR, "mean.npy")
STD_PATH   = os.path.join(BASE_DIR, "std.npy")

NUM_FEATURES  = 11     # B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 + NDVI
NUM_TIMESTEPS = 12     # monthly
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EE_PROJECT = "moonlit-caster-468904-i8"

# =========================================================
# STRESS ENGINE THRESHOLDS
# Z-score anomaly detection — field-relative, not absolute.
# Rules fire when a pixel's index deviates by this many standard
# deviations from the field's own wheat-pixel mean.
# =========================================================
STRESS_THRESHOLDS = {
    # Z-score anomaly triggers (how far from field baseline)
    "z_ccci_low":       -1.0,   # nitrogen deficit
    "z_ccci_high":       2.0,   # lodging co-trigger (over-fertilized)
    "z_lswi_low":       -1.5,   # drought (dehydrated canopy)
    "z_lswi_high":       1.5,   # waterlogging
    "z_sipi_high":       1.5,   # disease (pigment imbalance spike)
    "z_evi_low":        -1.0,   # waterlog co-trigger (biomass crash)
    "z_evi_high":        2.0,   # lodging co-trigger (excess biomass)

    # Weather gates — prevent false positives during genuinely dry/wet weeks
    "precip_drought":    15.0,  # mm / 14 days → only flag drought if dry
    "precip_waterlog":   30.0,  # mm / 14 days → only flag waterlog if wet
}

for folder in (LIVE_DIR, WEATHER_FOLDER, WHEAT_MASK_FOLDER,
               OUTPUT_FOLDER, HISTORY_FOLDER, STATE_FOLDER):
    os.makedirs(folder, exist_ok=True)

os.environ["CPL_LOG"] = "OFF"


# =========================================================
# 2. MODEL DEFINITION
# =========================================================
class WheatLSTM(nn.Module):
    def __init__(self, input_size: int = NUM_FEATURES):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc1  = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(self.fc1(out[:, -1, :]))
        return self.fc2(out)


# =========================================================
# 3. SHADOW STATE HELPERS
# =========================================================
def append_run_log(job: str, status: str, duration_s: float, notes: str = ""):
    row = {
        "timestamp":   dt.datetime.now().isoformat(timespec="seconds"),
        "job":         job,
        "status":      status,
        "duration_s":  round(duration_s, 2),
        "notes":       notes,
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(RUN_LOG_CSV):
        df_row.to_csv(RUN_LOG_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(RUN_LOG_CSV, index=False)


def read_state() -> dict:
    if os.path.exists(STATE_JSON):
        try:
            with open(STATE_JSON) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def write_state(new_values: dict):
    state = read_state()
    state.update(new_values)
    state["last_updated"] = dt.datetime.now().isoformat(timespec="seconds")
    with open(STATE_JSON, "w") as f:
        json.dump(state, f, indent=2)


def write_provenance(map_path: str, metadata: dict):
    sidecar = os.path.splitext(map_path)[0] + ".json"
    metadata["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    with open(sidecar, "w") as f:
        json.dump(metadata, f, indent=2)


def timed_stage(job_name: str):
    def wrap(func: Callable):
        def inner(*args, **kwargs):
            start = dt.datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (dt.datetime.now() - start).total_seconds()
                append_run_log(job_name, "ok", duration, "")
                print(f"  [✓] {job_name}  ({duration:.1f}s)")
                return result
            except Exception as e:
                duration = (dt.datetime.now() - start).total_seconds()
                note = f"{type(e).__name__}: {e}"
                append_run_log(job_name, "error", duration, note)
                print(f"  [✗] {job_name}  ({duration:.1f}s)  {note}")
                traceback.print_exc()
                return None
        return inner
    return wrap


# =========================================================
# 4. STAGE 1 — SATELLITE SYNC
# =========================================================
@timed_stage("sync_satellite_data")
def sync_satellite_data():
    if not _EE_AVAILABLE:
        raise RuntimeError("earthengine-api / geemap not installed")

    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)
    
    # When switching to a different field, clear old tiles — they're from
# a different location and will pollute the monthly composite.
if os.path.exists(GEOJSON_PATH):
    import hashlib
    with open(GEOJSON_PATH, "rb") as f:
        current_field_hash = hashlib.md5(f.read()).hexdigest()[:8]

    hash_marker = os.path.join(LIVE_DIR, ".field_hash")
    previous_hash = None
    if os.path.exists(hash_marker):
        with open(hash_marker) as f:
            previous_hash = f.read().strip()

    if previous_hash and previous_hash != current_field_hash:
        print(f"[coords] Field changed ({previous_hash} → {current_field_hash}) — clearing old tiles")
        for old in glob.glob(os.path.join(LIVE_DIR, "S2_*.tif")):
            os.remove(old)
        if os.path.exists(DOWNLOAD_LOG):
            os.remove(DOWNLOAD_LOG)

    with open(hash_marker, "w") as f:
        f.write(current_field_hash)

    region = geemap.geojson_to_ee(GEOJSON_PATH)

    already_downloaded = set()
    if os.path.exists(DOWNLOAD_LOG):
        with open(DOWNLOAD_LOG) as f:
            already_downloaded = set(f.read().splitlines())

    today = dt.date.today()
    start = today - dt.timedelta(days=1095)

    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(region)
                    .filterDate(start.isoformat(), today.isoformat())
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
                    .sort("system:time_start", False)
                    .limit(150))

    img_list = collection.toList(collection.size())
    count    = img_list.size().getInfo()
    bands    = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    new_files = 0
    for i in range(count):
        img       = ee.Image(img_list.get(i))
        date_str  = ee.Date(img.get("system:time_start")).format("YYYYMMdd").getInfo()
        if date_str in already_downloaded:
            continue

        out_path = os.path.join(LIVE_DIR, f"S2_{date_str}.tif")
        try:
            geemap.ee_export_image(
                img.select(bands).clip(region.geometry()),
                filename=out_path, scale=10, region=region.geometry(),
            )
            with open(DOWNLOAD_LOG, "a") as f:
                f.write(date_str + "\n")
            new_files += 1
        except Exception as e:
            print(f"    skip {date_str}: {e}")

    write_state({"new_satellite_tiles": new_files,
                 "total_satellite_tiles": count})
    return new_files


# =========================================================
# 5. STAGE 2 — WEATHER UPDATE
# =========================================================
@timed_stage("update_weather_data")
def update_weather_data():
    with open(GEOJSON_PATH) as f:
        geo = json.load(f)
    coords = geo["features"][0]["geometry"]["coordinates"][0]
    centroid_lat = sum(p[1] for p in coords) / len(coords)
    centroid_lon = sum(p[0] for p in coords) / len(coords)

    start_date = dt.date.today() - dt.timedelta(days=30)
    if os.path.exists(WEATHER_CSV):
        df_ex = pd.read_csv(WEATHER_CSV)
        if not df_ex.empty:
            last_date = pd.to_datetime(df_ex["DATE"], format="mixed").max().date()
            start_date = last_date + dt.timedelta(days=1)

    today = dt.date.today()
    if start_date >= today:
        return 0

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?parameters=T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M"
        "&community=AG"
        f"&longitude={centroid_lon}&latitude={centroid_lat}"
        f"&start={start_date.strftime('%Y%m%d')}&end={today.strftime('%Y%m%d')}"
        "&format=JSON"
    )
    payload = requests.get(url, timeout=30).json()["properties"]["parameter"]

    records = []
    for d in payload["T2M_MAX"].keys():
        records.append({
            "DATE":      d,
            "T2M_MAX":   payload["T2M_MAX"][d],
            "T2M_MIN":   payload["T2M_MIN"][d],
            "PRECIP":    payload["PRECTOTCORR"][d],
            "SOLAR_RAD": payload["ALLSKY_SFC_SW_DWN"][d],
            "RH":        payload.get("RH2M", {}).get(d, -999),
        })

    if not records:
        return 0

    df_new = pd.DataFrame(records)
    df_new["DATE"] = pd.to_datetime(df_new["DATE"], format="%Y%m%d")
    df_new.replace(-999, np.nan, inplace=True)

    if os.path.exists(WEATHER_CSV):
        df_final = (pd.concat([pd.read_csv(WEATHER_CSV), df_new])
                      .assign(DATE=lambda d: pd.to_datetime(d["DATE"], format="mixed"))
                      .drop_duplicates(subset=["DATE"])
                      .sort_values("DATE"))
    else:
        df_final = df_new.sort_values("DATE")

    df_final.to_csv(WEATHER_CSV, index=False)
    write_state({"weather_days_total": int(len(df_final))})
    return len(df_new)


# =========================================================
# 6. STAGE 3 — MONTHLY COMPOSITES
# =========================================================
def _parse_date_from_filename(fname: str) -> dt.date | None:
    try:
        stem = os.path.basename(fname).replace("S2_", "").replace(".tif", "")
        return dt.date(int(stem[:4]), int(stem[4:6]), int(stem[6:8]))
    except Exception:
        return None

def build_monthly_composite_stack(window: Window | None = None):
    """
    Aggregate the last 12 months of Sentinel-2 tiles.
    Filters out NoData/Empty tiles and falls back to previous valid picture.
    """
    tifs = sorted(glob.glob(os.path.join(LIVE_DIR, "S2_*.tif")))
    if len(tifs) == 0:
        raise RuntimeError("No Sentinel-2 tiles in data/live — run sync_satellite_data first.")

    today = dt.date.today()
    month_keys = []
    for i in range(12):
        m = today.month - i
        y = today.year
        while m <= 0:
            m += 12; y -= 1
        month_keys.append((y, m))
    month_keys = list(reversed(month_keys))

    buckets: dict[tuple[int, int], list[str]] = {k: [] for k in month_keys}
    for tif in tifs:
        d = _parse_date_from_filename(tif)
        if d is None: continue
        key = (d.year, d.month)
        if key in buckets: buckets[key].append(tif)

    with rasterio.open(tifs[-1]) as src:
        win  = window or Window(0, 0, src.width, src.height)
        h, w = win.height, win.width

    monthly = np.full((h, w, 12, 11), np.nan, dtype=np.float32)

    for t, key in enumerate(month_keys):
        paths = buckets[key]
        if not paths:
            continue

        valid_stacks = []
        for p in paths:
            arr = rasterio.open(p).read(window=win).astype(np.float32) / 10_000.0
            arr = np.clip(arr, 0, 1)

            # FIX 1: Earth Engine uses exactly 0.0 for NoData/Cloud Masked pixels.
            # Convert 0.0 to NaN so the median function ignores the bad pixels.
            arr[arr == 0] = np.nan

            # Only keep the tile if it isn't 100% empty
            if not np.all(np.isnan(arr)):
                valid_stacks.append(arr)

        if not valid_stacks:
            continue

        stacked = np.stack(valid_stacks)

        # Suppress "All-NaN slice encountered" warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(stacked, axis=0)

        med = np.transpose(med, (1, 2, 0))
        monthly[:, :, t, :10] = med

        # Calculate NDVI safely
        ndvi = (med[..., 6] - med[..., 2]) / (med[..., 6] + med[..., 2] + 1e-6)
        monthly[:, :, t, 10] = ndvi

    # FIX 2: Last Observation Carried Forward (Fallback to previous picture)
    for t in range(12):
        mask = np.isnan(monthly[:, :, t, 0])
        if mask.any():
            if t > 0:
                monthly[mask, t, :] = monthly[mask, t - 1, :]
            else:
                fallback = np.nanmedian(monthly, axis=2)
                monthly[mask, t, :] = fallback[mask]

    np.nan_to_num(monthly, copy=False)
    return monthly


# =========================================================
# 7. STAGE 4 — WHEAT INFERENCE
# =========================================================
@timed_stage("run_wheat_inference")
def run_wheat_inference(threshold: float = 0.5) -> str:
    mean = np.load(MEAN_PATH).astype(np.float32)
    std  = np.load(STD_PATH).astype(np.float32)

    print(f"    [diag] train mean (first 3 bands): {mean.flatten()[:3]}")
    print(f"    [diag] train std  (first 3 bands): {std.flatten()[:3]}")

    model = WheatLSTM(input_size=NUM_FEATURES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,
                                     weights_only=True))
    model.eval()

    tifs = sorted(glob.glob(os.path.join(LIVE_DIR, "S2_*.tif")))
    if not tifs:
        raise RuntimeError("No Sentinel-2 tiles found.")

    with rasterio.open(tifs[-1]) as src:
        meta = src.meta.copy()
        h, w = src.height, src.width
        meta.update(count=1, dtype="uint8", compress="lzw")

    out_mask  = np.zeros((h, w), dtype=np.uint8)
    prob_sum  = 0.0
    prob_n    = 0
    first_sample_logged = False

    CHUNK = 256
    for y0 in range(0, h, CHUNK):
        for x0 in range(0, w, CHUNK):
            win = Window(x0, y0, min(CHUNK, w - x0), min(CHUNK, h - y0))
            monthly = build_monthly_composite_stack(win)
            wh, ww  = monthly.shape[:2]
            flat    = monthly.reshape(-1, 12, 11)

            if not first_sample_logged and (x0 > w // 2) and (y0 > h // 2):
                sample = flat[flat.shape[0] // 2]
                print(f"    [diag] sample pixel raw band ranges: "
                      f"min={sample[:, :10].min():.3f} "
                      f"max={sample[:, :10].max():.3f} "
                      f"mean={sample[:, :10].mean():.3f}")
                print(f"    [diag] sample pixel NDVI range: "
                      f"{sample[:, 10].min():.3f} → {sample[:, 10].max():.3f}")
                first_sample_logged = True

            flat    = (flat - mean) / (std + 1e-6)

            with torch.no_grad():
                batch = torch.tensor(flat, dtype=torch.float32).to(DEVICE)
                probs = torch.sigmoid(model(batch).squeeze(-1)).cpu().numpy()

            prob_sum += float(probs.sum())
            prob_n   += len(probs)
            preds = (probs > threshold).astype(np.uint8).reshape(wh, ww)
            out_mask[y0:y0 + wh, x0:x0 + ww] = preds

    mean_prob = prob_sum / max(prob_n, 1)
    print(f"    [diag] mean predicted P(wheat) across field: {mean_prob:.4f}")
    print(f"    [diag] pixels classified as wheat: {int((out_mask == 1).sum())} / {out_mask.size}")

    date_str = dt.date.today().strftime("%Y%m%d")
    out_path = os.path.join(WHEAT_MASK_FOLDER, f"Wheat_Map_{date_str}.tif")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(out_mask, 1)

    wheat_px    = int((out_mask == 1).sum())
    total_px    = int(out_mask.size)
    coverage    = wheat_px / total_px if total_px else 0.0

    write_state({
        "latest_wheat_mask":  out_path,
        "wheat_coverage_pct": round(coverage * 100, 2),
    })
    write_provenance(out_path, {
        "stage":     "wheat_inference",
        "model":     os.path.basename(MODEL_PATH),
        "threshold": threshold,
        "input_tiles": len(tifs),
        "coverage_pct": round(coverage * 100, 2),
    })
    return out_path


# =========================================================
# 8. STAGE 5 — STRESS ENGINE
# =========================================================
def _growth_stage_from_gdd(gdd: float) -> str:
    if gdd < 500:   return "Emergence/Tillering"
    if gdd < 1000:  return "Jointing/Booting"
    if gdd < 1500:  return "Heading/Anthesis"
    return "Grain-fill/Senescence"


@timed_stage("run_stress_engine")
def run_stress_engine() -> str:
    wheat_tifs = sorted(glob.glob(os.path.join(WHEAT_MASK_FOLDER, "Wheat_Map_*.tif")))
    if not wheat_tifs:
        raise RuntimeError("No wheat mask found — run_wheat_inference first.")
    with rasterio.open(wheat_tifs[-1]) as src:
        wheat_mask = src.read(1)
        meta = src.meta.copy()
    meta.update(count=1, dtype="uint8", compress="lzw")

    monthly = build_monthly_composite_stack()
    h, w = wheat_mask.shape[:2]
    if monthly.shape[:2] != (h, w):
        mh, mw = monthly.shape[:2]
        ch, cw = min(h, mh), min(w, mw)
        m2 = np.zeros((h, w, 12, 11), dtype=np.float32)
        m2[:ch, :cw] = monthly[:ch, :cw]
        monthly = m2

    # =========================================================
    # VEGETATION INDICES (computed on the most recent monthly composite)
    # =========================================================
    # Band positions in the 11-feature vector:
    #   idx 0=B2, 1=B3, 2=B4, 3=B5, 4=B6, 5=B7, 6=B8, 7=B8A, 8=B11, 9=B12,
    #   idx 10 = NDVI (pre-computed monthly median)
    B2  = monthly[..., -1, 0]          # Blue
    B4  = monthly[..., -1, 2]          # Red
    B5  = monthly[..., -1, 3]          # Red-edge 1
    B8  = monthly[..., -1, 6]          # NIR (10 m)
    B11 = monthly[..., -1, 8]          # SWIR 1
    ndvi_now = monthly[..., -1, 10]    # already stored

    eps = 1e-6

    # ---- NDRE — intermediate index for CCCI ---------------------------------
    # Red-edge variant of NDVI; penetrates deeper into canopy so it's still
    # sensitive to chlorophyll when NDVI has saturated.
    ndre = (B8 - B5) / (B8 + B5 + eps)

    # ---- CCCI — Canopy Chlorophyll Content Index ----------------------------
    # CCCI = NDRE / NDVI. Cammarano et al. 2014. Dividing by NDVI removes
    # biomass noise, isolating the true nitrogen signal.
    ccci = ndre / (ndvi_now + eps)

    # ---- LSWI — Land Surface Water Index ------------------------------------
    # Bajgain et al. 2015. SWIR (B11) absorbs liquid water directly, so LSWI
    # tracks canopy+soil moisture independently of greenness.
    lswi = (B8 - B11) / (B8 + B11 + eps)

    # ---- SIPI — Structure-Intensive Pigment Index ---------------------------
    # (B8 − B2) / (B8 − B4). Scientific Reports PMC5809472. Carotenoid-to-
    # chlorophyll ratio — spikes during pathogenic attacks.
    sipi = (B8 - B2) / (B8 - B4 + eps)

    # ---- EVI — Enhanced Vegetation Index ------------------------------------
    # 2.5·(B8−B4)/(B8+6·B4−7.5·B2+1). Corrects for soil background and
    # atmospheric scattering. Unusually high EVI at late stages = lodging risk.
    evi = 2.5 * ((B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0 + eps))

    # =========================================================
    # Z-SCORE NORMALIZATION (field-relative anomaly detection)
    # =========================================================
    # Absolute thresholds fail when the whole field is in the same condition
    # (e.g. everyone looks "drought" during a dry month). Z-scoring finds
    # pixels that deviate from the field's own baseline — so we always catch
    # local hot-spots regardless of global conditions.
    W = (wheat_mask == 1)
    valid_baseline = W & (ndvi_now > 0.15)   # exclude no-data pixels from stats

    def zscore(index_array: np.ndarray) -> np.ndarray:
        """Standardize against the mean/std of wheat pixels in this field."""
        if not valid_baseline.any():
            return np.zeros_like(index_array)
        vals = index_array[valid_baseline]
        mu   = float(np.nanmean(vals))
        sd   = float(np.nanstd(vals))
        return (index_array - mu) / (sd + eps)

    z_ccci = zscore(ccci)
    z_lswi = zscore(lswi)
    z_sipi = zscore(sipi)
    z_evi  = zscore(evi)

    # =========================================================
    # WEATHER CONTEXT (field-wide, from NASA POWER 14-day window)
    # =========================================================
    recent_precip = 0.0
    recent_tmax   = 0.0
    growth_stage  = "Unknown"
    cum_gdd       = 0.0

    if os.path.exists(WEATHER_CSV):
        wx = pd.read_csv(WEATHER_CSV)
        wx["DATE"] = pd.to_datetime(wx["DATE"], format="mixed")
        wx["GDD"]  = ((wx["T2M_MAX"] + wx["T2M_MIN"]) / 2).clip(lower=0)
        cum_gdd      = float(wx["GDD"].sum())
        growth_stage = _growth_stage_from_gdd(cum_gdd)
        last14       = wx.tail(14)
        recent_precip = float(last14["PRECIP"].clip(lower=0).sum())
        tmax_series   = last14["T2M_MAX"].dropna()
        recent_tmax   = float(tmax_series.max()) if not tmax_series.empty else 0.0

    # =========================================================
    # PER-PIXEL ANOMALY RULES (Z-score based)
    # =========================================================
    stress  = np.zeros_like(wheat_mask, dtype=np.uint8)
    n_wheat = int(W.sum())
    n_valid = int(valid_baseline.sum())

    print(f"    [diag] wheat pixels: {n_wheat}  (valid baseline: {n_valid})")
    if n_valid > 0:
        v = valid_baseline
        print(f"    [diag] raw  — CCCI mean={ccci[v].mean():.3f}  "
              f"LSWI mean={lswi[v].mean():.3f}  "
              f"SIPI mean={sipi[v].mean():.3f}  "
              f"EVI mean={evi[v].mean():.3f}")
        print(f"    [diag] weather: precip14={recent_precip:.1f}mm  "
              f"tmax14={recent_tmax:.1f}°C  stage={growth_stage}")

    # --- CODE 1: Nitrogen deficit ----------------------------------------
    # Chlorophyll significantly low (Z_CCCI < -1.0) BUT water is okay —
    # otherwise a dry pixel would look nitrogen-deficient.
    nitrogen = valid_baseline & (z_ccci < -1.0) & (z_lswi > -0.5)

    # --- CODE 2: Drought / irrigation needed -----------------------------
    # Canopy water severely below field baseline AND recent rain is low.
    drought = valid_baseline & (z_lswi < -1.5) & (recent_precip < 15.0)

    # --- CODE 3: Waterlogging / halt irrigation --------------------------
    # Hyper-saturated (high LSWI) AND biomass crashing (low EVI) AND the
    # sky actually dumped water. All three needed to avoid false positives.
    waterlog = valid_baseline & (z_lswi > 1.5) & (z_evi < -1.0) & (recent_precip > 30.0)

    # --- CODE 4: Disease / fungicide -------------------------------------
    # SIPI spike not explained by drought (water is still OK).
    disease = valid_baseline & (z_sipi > 1.5) & (z_lswi > -1.0)

    # --- CODE 5: Lodging risk (over-fertilized canopy) -------------------
    # Extreme nitrogen AND extreme biomass together → stalks can't hold the
    # head. Just one high signal isn't enough — must be both.
    lodging = valid_baseline & (z_ccci > 2.0) & (z_evi > 2.0)

    # Assign in severity priority
    stress[drought]                     = 2
    stress[waterlog  & (stress == 0)]   = 3
    stress[disease   & (stress == 0)]   = 4
    stress[nitrogen  & (stress == 0)]   = 1
    stress[lodging   & (stress == 0)]   = 5

    print(f"    [diag] rule firings — "
          f"nitrogen={int(nitrogen.sum())}  "
          f"drought={int(drought.sum())}  "
          f"waterlog={int(waterlog.sum())}  "
          f"disease={int(disease.sum())}  "
          f"lodging={int(lodging.sum())}")

    date_str = dt.date.today().strftime("%Y%m%d")
    out_path = os.path.join(OUTPUT_FOLDER, f"Intervention_Map_{date_str}.tif")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(stress, 1)

    hist_path = os.path.join(HISTORY_FOLDER, f"Intervention_Map_{date_str}.tif")
    if not os.path.exists(hist_path):
        import shutil
        shutil.copy2(out_path, hist_path)

    counts = {int(k): int(v) for k, v in zip(*np.unique(stress, return_counts=True))}
    dominant = max((c for c in counts if c > 0), default=0,
                   key=lambda c: counts.get(c, 0))
    write_state({
        "latest_intervention_map": out_path,
        "growth_stage":            growth_stage,
        "cumulative_gdd":          round(cum_gdd, 1),
        "precip_14d_mm":           round(recent_precip, 1),
        "tmax_14d_c":              round(recent_tmax, 1),
        "stress_counts":           counts,
        "dominant_stress_code":    dominant,
    })

    write_provenance(out_path, {
        "stage":      "stress_engine",
        "thresholds": T,
        "weather": {
            "precip_14d": recent_precip,
            "tmax_14d":   recent_tmax,
            "cum_gdd":    cum_gdd,
            "growth_stage": growth_stage,
        },
        "stress_counts": counts,
    })
    return out_path


# =========================================================
# 9. STAGE 6 — DECISION SUPPORT
# =========================================================
@timed_stage("run_decision_support")
def run_decision_support():
    state = read_state()
    lines = []
    lines.append("=" * 50)
    lines.append("DECISION SUPPORT")
    lines.append("=" * 50)
    lines.append(f"Growth stage : {state.get('growth_stage', 'n/a')}")
    lines.append(f"Cumulative GDD: {state.get('cumulative_gdd', 'n/a')}")
    lines.append(f"14-day precip : {state.get('precip_14d_mm', 'n/a')} mm")
    lines.append(f"14-day tmax   : {state.get('tmax_14d_c', 'n/a')} °C")
    lines.append(f"Wheat coverage: {state.get('wheat_coverage_pct', 'n/a')} %")

    # FIX: Convert JSON string keys back to integers so we don't get 0 values
    raw_counts = state.get("stress_counts", {})
    counts = {int(k): v for k, v in raw_counts.items()}

    if counts:
        lines.append("\nStress pixel counts:")
        names = {1: "Nitrogen", 2: "Drought", 3: "Waterlog",
                 4: "Disease",  5: "Lodging"}
        for code in (1, 2, 3, 4, 5):
            lines.append(f"  {names[code]:>9}: {counts.get(code, 0)}")
    print("\n".join(lines))


# =========================================================
# 10. FULL PIPELINE + SCHEDULE / CLI
# =========================================================
def run_full_pipeline():
    print(f"\n{'#' * 50}\n# Shadow pipeline · {dt.datetime.now():%Y-%m-%d %H:%M}\n{'#' * 50}")
    sync_satellite_data()
    update_weather_data()
    run_wheat_inference()
    run_stress_engine()
    run_decision_support()
    print("\nPipeline complete.\n")


STAGES = {
    "satellite":    sync_satellite_data,
    "weather":      update_weather_data,
    "inference":    run_wheat_inference,
    "stress":       run_stress_engine,
    "decision":     run_decision_support,
}


def start_scheduler():
    from apscheduler.schedulers.blocking import BlockingScheduler

    sched = BlockingScheduler(timezone="UTC")

    sched.add_job(update_weather_data, "interval", hours=1,
                  next_run_time=dt.datetime.now() + dt.timedelta(seconds=5),
                  id="weather")
    sched.add_job(sync_satellite_data,  "interval", hours=6, id="satellite")
    sched.add_job(run_wheat_inference,  "cron", hour=2, minute=15, id="inference")
    sched.add_job(run_stress_engine,    "cron", hour=2, minute=25, id="stress")
    sched.add_job(run_decision_support, "cron", hour=2, minute=30, id="decision")

    print("Scheduler started. Ctrl-C to stop.")
    for j in sched.get_jobs():
        print(f"  • {j.id:<10} → {j.trigger}")
    sched.start()


# =========================================================
# 11. CLI
# =========================================================
def _parse_args():
    p = argparse.ArgumentParser(description="Shadow pipeline orchestrator")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--now",      action="store_true",
                     help="Run the full pipeline once and exit.")
    grp.add_argument("--schedule", action="store_true",
                     help="Start APScheduler and stay resident.")
    grp.add_argument("--step",     choices=STAGES.keys(),
                     help="Run a single stage.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.schedule:
        start_scheduler()
    elif args.step:
        STAGES[args.step]()
    else:
        run_full_pipeline()
