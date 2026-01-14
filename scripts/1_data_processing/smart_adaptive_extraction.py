import os
import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "data/from_jeremy/images_aligned_to_first"
OUTPUT_DIR = "data/processed/smart_adaptive_extraction_batch_v18_band_anchor"

# Table priors
NUM_ROWS = 40
FIRST_ROW_Y_PRIOR = 1263
TABLE_HEIGHT_PX = 3160
TABLE_WIDTH_PX = 6150

# ROI pads and x-band
TABLE_X_MARGIN = 140
ROI_TOP_PAD = 260
ROI_BOTTOM_PAD = 420

# Enhancement knobs
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
BLACKHAT_KSIZE = 35
BLACKHAT_MIX = 0.85

# Horizontal mask aggressiveness (auto fallback)
PEAK_MIN_REL = 0.42
PROJ_SMOOTH_K = 19
PEAK_MERGE_DIST = 14
PEAK_TARGET_GOOD = 28

# ========= NEW (band-based anchoring) =========
# Coverage threshold for "this y is a horizontal rule"
# Higher => stricter (less text). Too high can break faint lines.
COVER_THRESH = 0.18

# How many pixels a line-band can be thick before we compress it to a single center
# (purely for stability)
BAND_MIN_THICK = 1
BAND_MAX_THICK = 16

# Snapping tolerances (in pixels)
SNAP_ACCEPT_PX = 6
PEAK_SEARCH_BAND = 16

# Table-top search window around prior
FIRST_LINE_ROI_UP = 420
FIRST_LINE_ROI_DOWN = 900
TOP_CAND_LIMIT = 30

# Scoring
GRID_SCORE_W_ERR = 0.35
TOP_PRIOR_PENALTY_W = 0.020  # keep anchor near prior (stronger than before)

# Output (only these two)
SAVE_VIZ = True
SAVE_RULE_RESPONSE_CROP = True

# Columns (only used for x-band + viz)
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


# ============================================================
# Helpers
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

def enhance_faint_rules(gray: np.ndarray):
    clahe = cv2.createCLAHE(clipLimit=float(CLAHE_CLIP), tileGridSize=tuple(CLAHE_GRID))
    g = clahe.apply(gray)

    k = int(BLACKHAT_KSIZE)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
    rule_response = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    enhanced_gray = cv2.addWeighted(g, 1.0, rule_response, float(BLACKHAT_MIX), 0)
    return enhanced_gray, rule_response

def row_energy(mask: np.ndarray) -> np.ndarray:
    return np.sum(mask.astype(np.float32), axis=1)

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

def snap_to_nearest(y: int, ys: list, search_band: int, accept_band: int) -> int:
    """Fail-closed snapping to nearest candidate y in ys."""
    if not ys:
        return int(y)
    lo, hi = int(y - search_band), int(y + search_band)
    cands = [p for p in ys if lo <= p <= hi]
    if not cands:
        return int(y)
    best = int(min(cands, key=lambda p: abs(p - y)))
    return best if abs(best - y) <= int(accept_band) else int(y)

def bands_from_hmask(hmask: np.ndarray, y0_full: int) -> list:
    """
    Convert hmask -> coverage(y) -> contiguous bands -> band center y in FULL coords.
    This is more reliable than peaks when there are many strong row lines.
    """
    h, w = hmask.shape
    cov = np.sum(hmask > 0, axis=1).astype(np.float32) / float(max(1, w))
    cov_s = smooth_1d(cov, 9)

    on = cov_s >= float(COVER_THRESH)
    ys = np.where(on)[0]
    if ys.size == 0:
        return []

    bands = []
    start = int(ys[0])
    prev = int(ys[0])
    for y in ys[1:]:
        y = int(y)
        if y == prev + 1:
            prev = y
        else:
            bands.append((start, prev))
            start = y
            prev = y
    bands.append((start, prev))

    centers_full = []
    for a, b in bands:
        thick = (b - a + 1)
        if thick < BAND_MIN_THICK:
            continue
        # If a band is very thick, still compress to its center (works well for scan thickness)
        if thick > BAND_MAX_THICK:
            c = int(round(0.5 * (a + b)))
        else:
            c = int(round(0.5 * (a + b)))
        centers_full.append(int(c + y0_full))

    return sorted(set(centers_full))


# ============================================================
# Auto-aggressive horizontal mask (kills text, keeps rules)
# ============================================================

def make_hmask_aggressive(rr_crop: np.ndarray):
    h, w = rr_crop.shape
    trials = [
        {"pctl": 88, "kfrac": 0.30, "dilate": 1},
        {"pctl": 86, "kfrac": 0.28, "dilate": 1},
        {"pctl": 84, "kfrac": 0.26, "dilate": 1},
        {"pctl": 82, "kfrac": 0.24, "dilate": 1},
        {"pctl": 80, "kfrac": 0.22, "dilate": 1},
        {"pctl": 78, "kfrac": 0.20, "dilate": 1},
    ]

    best = None

    for t in trials:
        thr = int(np.percentile(rr_crop, t["pctl"]))
        thr = max(10, min(240, thr))
        bw = (rr_crop >= thr).astype(np.uint8) * 255

        klen = int(max(180, w * float(t["kfrac"])))
        if klen % 2 == 0:
            klen += 1

        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)

        if int(t["dilate"]) > 0:
            dk = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            opened = cv2.dilate(opened, dk, iterations=int(t["dilate"]))

        # keep the old peak metric just to decide "good enough"
        energy = row_energy(opened)
        energy_s = smooth_1d(energy, PROJ_SMOOTH_K)
        peaks = find_peaks_1d(energy_s, min_rel=PEAK_MIN_REL, merge_dist=PEAK_MERGE_DIST)
        peak_count = len(peaks)

        dbg = f"pctl={t['pctl']} kfrac={t['kfrac']:.2f} klen={klen} thr={thr} peaks={peak_count}"

        if best is None or peak_count > best["peak_count"]:
            best = {"hmask": opened, "peak_count": peak_count, "dbg": dbg}

        if peak_count >= int(PEAK_TARGET_GOOD):
            return opened, dbg

    return best["hmask"], f"(fallback best) {best['dbg']}"


# ============================================================
# Table anchor selection using BANDS (not peaks)
# ============================================================

def choose_table_top_by_band_grid(band_lines_full: list, first_y_prior: int):
    """
    Choose table_top among band lines by:
      - how well a uniform 40-row grid snaps to band lines (41 lines)
      - soft penalty away from prior
    This avoids starting mid-table.
    """
    if not band_lines_full:
        return int(first_y_prior), "(no bands; prior fallback)"

    roi_lo = int(first_y_prior - FIRST_LINE_ROI_UP)
    roi_hi = int(first_y_prior + FIRST_LINE_ROI_DOWN)

    cands = [y for y in band_lines_full if roi_lo <= y <= roi_hi]
    if not cands:
        nearest = int(min(band_lines_full, key=lambda y: abs(y - first_y_prior)))
        return nearest, "(no top bands in ROI; nearest fallback)"

    # try tops closest to prior (don’t let it drift)
    cands = sorted(cands, key=lambda y: abs(y - first_y_prior))[:int(TOP_CAND_LIMIT)]
    step = float(TABLE_HEIGHT_PX) / float(NUM_ROWS)

    best_score = None
    best_top = None

    for top in cands:
        good = 0
        errs = []

        for i in range(NUM_ROWS + 1):
            ye = float(top + i * step)
            ys = snap_to_nearest(int(round(ye)), band_lines_full,
                                 search_band=PEAK_SEARCH_BAND,
                                 accept_band=SNAP_ACCEPT_PX)
            err = abs(float(ys) - ye)
            if err <= float(SNAP_ACCEPT_PX):
                good += 1
                errs.append(err)

        avg_err = float(np.mean(errs)) if errs else 999.0
        score = float(good) - float(GRID_SCORE_W_ERR) * avg_err

        # strong-ish prior penalty: prevents mid-table anchoring
        score -= float(TOP_PRIOR_PENALTY_W) * abs(float(top) - float(first_y_prior))

        if best_score is None or score > best_score:
            best_score = score
            best_top = int(top)

    dbg = f"(band grid) top={best_top} score={best_score:.2f} cands={len(cands)} bands={len(band_lines_full)}"
    return int(best_top), dbg


# ============================================================
# Visualization
# ============================================================

def draw_overlay(gray: np.ndarray, columns: dict, lines_y: list,
                 table_top: int, table_bottom: int,
                 out_path: str, title: str, xL: int, xR: int):
    viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape

    if title:
        cv2.putText(viz, title, (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 0), 2)

    for _, (a, b) in columns.items():
        cv2.line(viz, (a, 0), (a, h), (255, 0, 0), 2)
        cv2.line(viz, (b, 0), (b, h), (255, 0, 0), 2)

    cv2.line(viz, (xL, 0), (xL, h), (150, 150, 0), 2)
    cv2.line(viz, (xR, 0), (xR, h), (150, 150, 0), 2)

    cv2.line(viz, (0, table_top), (w, table_top), (255, 255, 0), 2)
    cv2.line(viz, (0, table_bottom), (w, table_bottom), (0, 255, 255), 3)

    for y in lines_y:
        y = int(np.clip(y, 0, h - 1))
        cv2.line(viz, (xL, y), (xR, y), (0, 0, 255), 2)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, viz)


# ============================================================
# Main per-image
# ============================================================

def process_one_image(img_path: str) -> None:
    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n=== Processing: {name} ===")

    gray = read_gray(img_path)
    if gray is None:
        print("⚠️ Could not read image. Skipping.")
        return

    h, w = gray.shape
    img_out = os.path.join(OUTPUT_DIR, name)
    ensure_dir(img_out)

    # X band from known columns
    min_x1 = min(v[0] for v in COLUMNS.values())
    xL = int(min_x1 - TABLE_X_MARGIN)
    xR = int(xL + TABLE_WIDTH_PX + 2 * TABLE_X_MARGIN)
    xL = max(0, min(w - 2, xL))
    xR = max(xL + 1, min(w - 1, xR))

    # Crop around expected table band
    y0 = max(0, FIRST_ROW_Y_PRIOR - ROI_TOP_PAD)
    y1 = min(h, FIRST_ROW_Y_PRIOR + TABLE_HEIGHT_PX + ROI_BOTTOM_PAD)
    crop = gray[y0:y1, xL:xR]

    # Rule response (save)
    _, rr_crop = enhance_faint_rules(crop)

    # Horizontal mask (aggressive, auto fallback)
    hmask, dbg_h = make_hmask_aggressive(rr_crop)

    # Convert hmask -> band lines (FULL coords)
    band_lines_full = bands_from_hmask(hmask, y0_full=y0)

    # Choose table_top using band-grid scoring
    table_top, dbg_top = choose_table_top_by_band_grid(band_lines_full, FIRST_ROW_Y_PRIOR)
    table_bottom = int(table_top + TABLE_HEIGHT_PX)

    # Build 41 separators using bands as the snapping targets
    step = float(TABLE_HEIGHT_PX) / float(NUM_ROWS)
    lines_y = []
    for i in range(NUM_ROWS + 1):
        y_expect = int(round(table_top + i * step))
        y_snap = snap_to_nearest(y_expect, band_lines_full,
                                 search_band=PEAK_SEARCH_BAND,
                                 accept_band=SNAP_ACCEPT_PX)
        lines_y.append(int(y_snap))

    # Save ONLY requested outputs
    if SAVE_RULE_RESPONSE_CROP:
        cv2.imwrite(os.path.join(img_out, "debug_rule_response_crop.png"), rr_crop)

    if SAVE_VIZ:
        viz_path = os.path.join(img_out, "grid_overlay.png")
        title = f"{name} | {dbg_h} | {dbg_top}"
        draw_overlay(gray, COLUMNS, lines_y, table_top, table_bottom, viz_path, title, xL, xR)

    print(dbg_h)
    print(dbg_top)
    print(f"table_top={table_top} bottom={table_bottom} bands={len(band_lines_full)}")
    print(f"✅ Saved: {img_out}")


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    imgs = list_images(INPUT_DIR)
    if not imgs:
        print(f"❌ No images found in: {INPUT_DIR}")
        return

    print("=== SMART ADAPTIVE EXTRACTION v18 (HMASK + BAND ANCHOR) ===")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images found: {len(imgs)}")

    for i, p in enumerate(imgs, start=1):
        print(f"\n[{i}/{len(imgs)}]")
        process_one_image(p)

    print("\n🎯 DONE")

if __name__ == "__main__":
    main()
