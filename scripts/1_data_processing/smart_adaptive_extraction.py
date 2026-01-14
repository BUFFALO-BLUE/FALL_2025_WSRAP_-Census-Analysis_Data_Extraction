import os
import cv2
import json
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "data/from_jeremy/images_aligned_to_first"
OUTPUT_DIR = "data/processed/smart_adaptive_extraction_batch_v13_1p1_safe_snap"

NUM_ROWS = 40
FIRST_ROW_Y_PRIOR = 1263

# Your measurement priors (keep stable!)
TABLE_HEIGHT_PX = 3160
TABLE_WIDTH_PX = 6150

TABLE_X_MARGIN = 140
ROI_TOP_PAD = 260
ROI_BOTTOM_PAD = 420

FIRST_LINE_ROI_UP = 340
FIRST_LINE_ROI_DOWN = 740

# Enhancement knobs
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
BLACKHAT_KSIZE = 35

# Projection / peak knobs
PROJ_SMOOTH_K = 31
PEAK_MIN_REL = 0.26
PEAK_MERGE_DIST = 14

# Snapping knobs
SNAP_BAND = 16
RR_THRESH_PCT = 82
CONTINUITY_MIN_FRAC = 0.55
INK_WEIGHT = 1.75
RR_WEIGHT = 1.0
CONTINUITY_WEIGHT = 0.65
LOCAL_SMOOTH_HALF = 2

# NEW: hard whitespace lock
HARD_INK_MAX = 0.12   # if ink density > this, separator is forbidden at that y
HARD_INK_PAD = 3      # require neighborhood around y to be low ink too

# NEW: keep bottom stable, but allow tiny clamp to avoid footer
BOTTOM_ADJUST_MAX = 160  # only allow bottom to move ±160px from prior bottom

TOP_CANDIDATES_MAX = 12

# IMPORTANT: start with slope OFF to avoid drift
USE_SLOPE = False

# Optional slope (only enable later)
SLOPE_NUM_WINDOWS = 10
SLOPE_WIN_OVERLAP = 0.35
SLOPE_MIN_REL = 0.20
SLOPE_SEARCH_BAND = 14

RECT_ROW_H = 78  # not used if you stop saving rows/cells

COLUMNS = {
    "street": (629, 718),
    "house_number": (718, 836),
    "rented": (914, 954),
    "owned":  (954, 994),
    "price_rent": (996, 1143),
    "head": (1889, 2204),
    "gender": (2204, 2285),
    "race": (2285, 2388),
    "marital_status": (2491, 2574),
    "hours_worked": (4939, 5092),
    "wages": (6433, 6588),
}

# Save only what you asked (fast)
SAVE_VIZ = True
SAVE_DEBUG = True
SAVE_CELLS = False  # keep False for speed


# ============================================================
# FS helpers
# ============================================================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def list_images(folder: str):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    )

def read_gray(path: str):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    k = int(max(3, k))
    if k % 2 == 0:
        k += 1
    return np.convolve(x.astype(np.float32), np.ones(k, dtype=np.float32) / k, mode="same")


# ============================================================
# Enhancement: CLAHE + Blackhat -> rule_response
# ============================================================

def enhance_faint_rules(gray: np.ndarray):
    clahe = cv2.createCLAHE(clipLimit=float(CLAHE_CLIP), tileGridSize=tuple(CLAHE_GRID))
    g = clahe.apply(gray)

    k = int(BLACKHAT_KSIZE)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
    rule_response = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return rule_response

def robust_ink_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )
    ink = 255 - bw
    ink = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1
    )
    return ink


# ============================================================
# Projections + peaks
# ============================================================

def row_energy(rr: np.ndarray) -> np.ndarray:
    return np.sum(rr.astype(np.float32), axis=1)

def find_peaks_1d(signal: np.ndarray, min_rel: float, merge_dist: int) -> list:
    if signal.size == 0:
        return []
    s = signal.astype(np.float32).copy()
    s -= float(np.min(s))
    mx = float(np.max(s)) if float(np.max(s)) > 0 else 1.0
    s /= mx

    peaks = []
    for i in range(1, len(s) - 1):
        if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] >= float(min_rel):
            peaks.append(i)

    merged = []
    for p in peaks:
        if not merged or abs(p - merged[-1]) > int(merge_dist):
            merged.append(p)
        else:
            if signal[p] > signal[merged[-1]]:
                merged[-1] = p
    return merged


# ============================================================
# Whitespace-safe snapping (HARD)
# ============================================================

def continuity_fraction(rr_row: np.ndarray, thr: float) -> float:
    return float(np.mean(rr_row.astype(np.float32) >= thr))

def ink_density(ink: np.ndarray, y: int) -> float:
    h = ink.shape[0]
    y0 = max(0, y - LOCAL_SMOOTH_HALF)
    y1 = min(h, y + LOCAL_SMOOTH_HALF + 1)
    band = ink[y0:y1, :]
    return float(np.mean(band.astype(np.float32) / 255.0))  # 0..1

def hard_ink_ok(ink: np.ndarray, y: int) -> bool:
    h = ink.shape[0]
    lo = max(0, y - HARD_INK_PAD)
    hi = min(h - 1, y + HARD_INK_PAD)
    # require every row in neighborhood to be reasonably low ink
    for yy in range(lo, hi + 1):
        if ink_density(ink, yy) > HARD_INK_MAX:
            return False
    return True

def separator_score(rr: np.ndarray, ink: np.ndarray, y: int, rr_thr: float) -> float:
    h = rr.shape[0]
    if y < 1 or y >= h - 1:
        return -1e9

    # HARD constraint: don't cut through writing
    if not hard_ink_ok(ink, y):
        return -1e9

    rr_row = rr[y, :].astype(np.float32)
    rr_e = float(np.mean(rr_row / 255.0))
    cont = continuity_fraction(rr_row, rr_thr)
    ink_d = ink_density(ink, y)

    cont_penalty = 0.0
    if cont < CONTINUITY_MIN_FRAC:
        cont_penalty = (CONTINUITY_MIN_FRAC - cont) * 0.9

    score = (RR_WEIGHT * rr_e) + (CONTINUITY_WEIGHT * cont) - (INK_WEIGHT * ink_d) - cont_penalty
    return float(score)

def snap_separator(rr: np.ndarray, ink: np.ndarray, y_expected: int, band: int) -> (int, dict):
    h = rr.shape[0]
    lo = max(0, int(y_expected - band))
    hi = min(h - 1, int(y_expected + band))

    rr_thr = float(np.percentile(rr, RR_THRESH_PCT))

    best_y = None
    best_s = -1e9
    for y in range(lo, hi + 1):
        s = separator_score(rr, ink, y, rr_thr)
        if s > best_s:
            best_s = s
            best_y = y

    # If everything was forbidden by hard-ink, fall back to soft scoring (still better than crashing)
    if best_y is None:
        # soft fallback: remove hard constraint just for this snap
        best_y = int(np.clip(y_expected, 0, h - 1))
        best_s = -1e9
        for y in range(lo, hi + 1):
            rr_row = rr[y, :].astype(np.float32)
            rr_e = float(np.mean(rr_row / 255.0))
            cont = continuity_fraction(rr_row, rr_thr)
            ink_d = ink_density(ink, y)
            s = (RR_WEIGHT * rr_e) + (CONTINUITY_WEIGHT * cont) - (INK_WEIGHT * ink_d)
            if s > best_s:
                best_s = s
                best_y = y

        dbg = {"fallback": "soft_only", "y_expected": int(y_expected), "y_picked": int(best_y), "score": float(best_s)}
        return int(best_y), dbg

    dbg = {"fallback": None, "y_expected": int(y_expected), "y_picked": int(best_y), "score": float(best_s)}
    return int(best_y), dbg


# ============================================================
# Grid-aware table top selection (kept from v13.1)
# ============================================================

def choose_table_top_gridaware(rr: np.ndarray, ink: np.ndarray, peaks: list, first_prior_in_crop: int) -> (int, dict):
    if not peaks:
        return int(first_prior_in_crop), {"picked_from": "prior_fallback_no_peaks"}

    roi_lo = int(first_prior_in_crop - FIRST_LINE_ROI_UP)
    roi_hi = int(first_prior_in_crop + FIRST_LINE_ROI_DOWN)

    roi_peaks = [p for p in peaks if roi_lo <= p <= roi_hi]
    if not roi_peaks:
        nearest = min(peaks, key=lambda p: abs(p - first_prior_in_crop))
        return int(nearest), {"picked_from": "nearest_peak_fallback", "roi_lo": roi_lo, "roi_hi": roi_hi}

    energy = row_energy(rr)
    roi_peaks = sorted(roi_peaks, key=lambda p: energy[p], reverse=True)[:TOP_CANDIDATES_MAX]

    best_top = int(roi_peaks[0])
    best_score = -1e9
    best_detail = None

    step = float(TABLE_HEIGHT_PX) / float(NUM_ROWS)

    for cand_top in roi_peaks:
        total = 0.0
        snaps = 0
        dbg_snaps = []
        for i in range(NUM_ROWS + 1):
            y_exp = int(round(cand_top + i * step))
            if y_exp < 0 or y_exp >= rr.shape[0]:
                continue
            y_pick, dbg = snap_separator(rr, ink, y_exp, band=SNAP_BAND)
            total += dbg["score"]
            snaps += 1
            if i in (0, 1, 2, 20, 39, 40):
                dbg_snaps.append(dbg)

        avg = total / max(1, snaps)
        dist_pen = 0.0009 * float(abs(cand_top - first_prior_in_crop))
        score = avg - dist_pen
        if score > best_score:
            best_score = score
            best_top = int(cand_top)
            best_detail = {"candidate_top": int(cand_top), "avg_score": float(avg), "dist_pen": float(dist_pen),
                           "score": float(score), "sample_snaps": dbg_snaps}

    return best_top, {"picked_from": "gridaware_top_scoring", "roi_lo": roi_lo, "roi_hi": roi_hi,
                      "best": best_detail, "candidates_considered": [int(p) for p in roi_peaks]}


# ============================================================
# Visualization (draw ONLY inside crop x-band to protect street later)
# ============================================================

def draw_overlay(gray: np.ndarray, columns: dict, y_seps_full: list, xL: int, xR: int, out_path: str, title: str):
    viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape

    if title:
        cv2.putText(viz, title, (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 0), 2)

    # vertical guides (current fixed columns)
    for _, (a, b) in columns.items():
        cv2.line(viz, (a, 0), (a, h), (255, 0, 0), 2)
        cv2.line(viz, (b, 0), (b, h), (255, 0, 0), 2)

    # draw row lines only across [xL..xR]
    for y in y_seps_full:
        cv2.line(viz, (xL, y), (xR, y), (0, 0, 255), 2)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, viz)


# ============================================================
# Per-image
# ============================================================

def process_one_image(img_path: str) -> None:
    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n=== Processing: {name} ===")

    gray = read_gray(img_path)
    if gray is None:
        print("⚠️ Could not read image. Skipping.")
        return

    h, w = gray.shape
    out_dir = os.path.join(OUTPUT_DIR, name)
    ensure_dir(out_dir)

    # X band from known columns
    min_x1 = min(v[0] for v in COLUMNS.values())
    xL = int(min_x1 - TABLE_X_MARGIN)
    xR = int(xL + TABLE_WIDTH_PX + 2 * TABLE_X_MARGIN)
    xL = max(0, min(w - 2, xL))
    xR = max(xL + 1, min(w - 1, xR))

    # Crop around expected table band
    y0 = max(0, FIRST_ROW_Y_PRIOR - ROI_TOP_PAD)
    y1 = min(h, FIRST_ROW_Y_PRIOR + TABLE_HEIGHT_PX + ROI_BOTTOM_PAD)

    crop_gray = gray[y0:y1, xL:xR]
    rr = enhance_faint_rules(crop_gray)
    ink = robust_ink_mask(crop_gray)

    # Peaks as candidates
    e = row_energy(rr)
    e_s = smooth_1d(e, PROJ_SMOOTH_K)
    peaks = find_peaks_1d(e_s, min_rel=PEAK_MIN_REL, merge_dist=PEAK_MERGE_DIST)

    first_prior_in_crop = int(FIRST_ROW_Y_PRIOR - y0)
    top_c, top_dbg = choose_table_top_gridaware(rr, ink, peaks, first_prior_in_crop)

    # STABLE bottom from measured height
    bottom_prior = int(min(rr.shape[0] - 1, top_c + TABLE_HEIGHT_PX))

    # Tiny bottom clamp only (prevents 1 extra row leaking into footer)
    # We just choose the best whitespace-safe separator near bottom prior.
    bottom_c, bottom_dbg = snap_separator(rr, ink, bottom_prior, band=BOTTOM_ADJUST_MAX)

    # Build exactly 41 separators by expected spacing between (top..bottom)
    span = max(2400, int(bottom_c - top_c))
    step = float(span) / float(NUM_ROWS)

    y_seps_crop = []
    snap_debug = []
    for i in range(NUM_ROWS + 1):
        y_exp = int(round(top_c + i * step))
        y_pick, dbg = snap_separator(rr, ink, y_exp, band=SNAP_BAND)
        y_seps_crop.append(int(y_pick))
        if i in (0, 1, 2, 20, 39, 40):
            snap_debug.append(dbg)

    # Convert crop separators to full image y
    y_seps_full = [int(y + y0) for y in y_seps_crop]

    if SAVE_VIZ:
        viz_path = os.path.join(out_dir, "grid_overlay.png")
        title = f"{name} | peaks={len(peaks)} | top={top_c+y0} bottom={bottom_c+y0}"
        draw_overlay(gray, COLUMNS, y_seps_full, xL, xR, viz_path, title)

    if SAVE_DEBUG:
        # Keep debug minimal for speed
        cv2.imwrite(os.path.join(out_dir, "debug_rule_response_crop.png"), rr)
        cv2.imwrite(os.path.join(out_dir, "debug_ink_crop.png"), ink)

    report = {
        "image_name": name,
        "source_path": img_path,
        "x_band": {"xL": int(xL), "xR": int(xR)},
        "crop": {"y0": int(y0), "y1": int(y1)},
        "top_c_full": int(top_c + y0),
        "bottom_c_full": int(bottom_c + y0),
        "bottom_prior_full": int(bottom_prior + y0),
        "bottom_snap": bottom_dbg,
        "rows_expected": int(NUM_ROWS),
        "separators_count": int(len(y_seps_full)),
        "separators_full": [int(y) for y in y_seps_full],
        "peaks_count": int(len(peaks)),
        "top_selection": top_dbg,
        "snap_debug_samples": snap_debug,
        "hard_ink_max": float(HARD_INK_MAX),
        "note": "v13.1p1: stable 3160px height + hard ink rejection snap. Draws separators only inside table x-band.",
    }

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"top={top_c+y0} bottom={bottom_c+y0} peaks={len(peaks)} sep={len(y_seps_full)}")
    print(f"✅ Saved: {out_dir}")


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)
    imgs = list_images(INPUT_DIR)
    if not imgs:
        print(f"❌ No images found in: {INPUT_DIR}")
        return

    print("=== SMART ADAPTIVE EXTRACTION v13.1p1 (HARD WHITESPACE SNAP + STABLE HEIGHT) ===")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images found: {len(imgs)}")
    print(f"HARD_INK_MAX={HARD_INK_MAX} | USE_SLOPE={USE_SLOPE}")

    for i, p in enumerate(imgs, start=1):
        print(f"\n[{i}/{len(imgs)}]")
        process_one_image(p)

    print("\n🎯 DONE")

if __name__ == "__main__":
    main()
