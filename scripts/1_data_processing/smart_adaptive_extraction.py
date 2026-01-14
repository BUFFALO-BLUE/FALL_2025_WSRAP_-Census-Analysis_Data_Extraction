import os
import cv2
import json
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "data/from_jeremy/images_aligned_to_first"
OUTPUT_DIR = "data/processed/smart_adaptive_extraction_batch_v13_2_sloped_41lines"

NUM_ROWS = 40
FIRST_ROW_Y_PRIOR = 1263

# Priors (used as guidance, NOT absolute truth)
TABLE_HEIGHT_PX_PRIOR = 3160
TABLE_WIDTH_PX_PRIOR = 6150

# Crop padding (keeps header/footer mostly out, but still safe)
ROI_TOP_PAD = 320
ROI_BOTTOM_PAD = 520

# Strict top-line ROI around FIRST_ROW_Y_PRIOR (in CROP coords)
FIRST_LINE_ROI_UP = 360
FIRST_LINE_ROI_DOWN = 820

# “Don’t use margins” for row finding
# We build an X-band around the table interior, then detect rows only inside this band.
DETECT_X_INSET = 80  # pixels inset from table band edges, avoids margins & side junk

# Enhancement knobs (faint rules)
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
BLACKHAT_KSIZE = 35

# Projection / peak knobs
PROJ_SMOOTH_K = 31
PEAK_MIN_REL = 0.26
PEAK_MERGE_DIST = 14
TOP_CANDIDATES_MAX = 12

# Whitespace-safe snapping (core behavior)
SNAP_BAND = 16
RR_THRESH_PCT = 82
CONTINUITY_MIN_FRAC = 0.55
INK_WEIGHT = 1.75
RR_WEIGHT = 1.0
CONTINUITY_WEIGHT = 0.65
LOCAL_SMOOTH_HALF = 2

# Sloped separators (windowed points -> line fit)
USE_SLOPE = True
SLOPE_NUM_WINDOWS = 10
SLOPE_WIN_OVERLAP = 0.35
SLOPE_SEARCH_BAND = 14  # additional snapping band used in windows

# Bottom clamp via vertical rule density
BOTTOM_SEARCH_PAD = 700     # search around bottom prior
VERT_KERNEL_H = 220         # vertical morphology kernel height (tune if needed)
VERT_DENS_SMOOTH = 31
VERT_DROP_REL = 0.30        # relative threshold vs “typical” vertical density

# Columns (fixed for now; we’ll align later)
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

# Output controls
SAVE_VIZ = True
SAVE_DEBUG_IMAGES = False  # keep False for speed; enable only when debugging a bad case


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
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
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
# Whitespace-safe snapping (core fix)
# ============================================================

def continuity_fraction(rr_row: np.ndarray, thr: float) -> float:
    return float(np.mean(rr_row.astype(np.float32) >= thr))

def ink_density(ink: np.ndarray, y: int) -> float:
    h = ink.shape[0]
    y0 = max(0, y - LOCAL_SMOOTH_HALF)
    y1 = min(h, y + LOCAL_SMOOTH_HALF + 1)
    band = ink[y0:y1, :]
    return float(np.mean(band.astype(np.float32) / 255.0))

def separator_score(rr: np.ndarray, ink: np.ndarray, y: int, rr_thr: float) -> float:
    h = rr.shape[0]
    if y < 1 or y >= h - 1:
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

    best_y = int(np.clip(y_expected, 0, h - 1))
    best_s = -1e9
    for y in range(lo, hi + 1):
        s = separator_score(rr, ink, y, rr_thr)
        if s > best_s:
            best_s = s
            best_y = y

    dbg = {
        "y_expected": int(y_expected),
        "y_picked": int(best_y),
        "score": float(best_s),
        "band": int(band),
        "rr_thr": float(rr_thr),
    }
    return int(best_y), dbg


# ============================================================
# Table top selection (strict ROI + grid-aware scoring)
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

    step = float(TABLE_HEIGHT_PX_PRIOR) / float(NUM_ROWS)

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

        # Additional safety: top line must be "line-like" more than not
        rr_thr = float(np.percentile(rr, RR_THRESH_PCT))
        cont0 = continuity_fraction(rr[int(np.clip(cand_top, 0, rr.shape[0]-1)), :], rr_thr)
        cont_pen = 0.15 * max(0.0, (0.45 - cont0))  # soft penalty if top isn't line-like

        score = avg - dist_pen - cont_pen
        if score > best_score:
            best_score = score
            best_top = int(cand_top)
            best_detail = {
                "candidate_top": int(cand_top),
                "avg_score": float(avg),
                "dist_pen": float(dist_pen),
                "cont0": float(cont0),
                "cont_pen": float(cont_pen),
                "score": float(score),
                "sample_snaps": dbg_snaps,
            }

    return best_top, {
        "picked_from": "gridaware_top_scoring",
        "roi_lo": roi_lo,
        "roi_hi": roi_hi,
        "best": best_detail,
        "candidates_considered": [int(p) for p in roi_peaks],
    }


# ============================================================
# Bottom clamp via vertical-rule density (prevents footer leak)
# ============================================================

def vertical_rule_density(rr: np.ndarray) -> np.ndarray:
    # rr bright where rules are strong; isolate vertical-ish structures
    hk = int(max(60, VERT_KERNEL_H))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, hk))
    vert = cv2.morphologyEx(rr, cv2.MORPH_OPEN, kernel, iterations=1)
    dens = np.sum(vert.astype(np.float32), axis=1)  # per-row vertical presence
    dens = smooth_1d(dens, VERT_DENS_SMOOTH)
    return dens, vert

def clamp_table_bottom(rr: np.ndarray, top_c: int, bottom_prior_c: int) -> (int, dict, np.ndarray):
    dens, vert_mask = vertical_rule_density(rr)
    h = rr.shape[0]

    # Search only near the bottom prior (avoid being tricked by header verticals)
    lo = int(max(0, bottom_prior_c - BOTTOM_SEARCH_PAD))
    hi = int(min(h - 1, bottom_prior_c + BOTTOM_SEARCH_PAD))
    band = dens[lo:hi+1]
    if band.size == 0:
        return int(min(h - 1, bottom_prior_c)), {"picked_from": "bottom_prior_fallback_empty_band"}, vert_mask

    # Typical density in table band should be high; footer should drop sharply.
    typical = float(np.percentile(band, 75))
    thr = float(max(1.0, typical * VERT_DROP_REL))

    # Find the last y (within search band) where density still looks "table-like"
    good = np.where(band >= thr)[0]
    if good.size == 0:
        # fallback: use prior
        return int(min(h - 1, bottom_prior_c)), {
            "picked_from": "bottom_prior_fallback_no_good",
            "lo": lo, "hi": hi, "typical": typical, "thr": thr
        }, vert_mask

    last_good = int(lo + int(good[-1]))

    # Safety: bottom must be below top and must allow 40 rows at least ~50px each
    min_bottom = int(top_c + 40 * 50)
    if last_good < min_bottom:
        last_good = int(min(h - 1, bottom_prior_c))

    return int(last_good), {
        "picked_from": "vertical_rule_density_clamp",
        "lo": lo, "hi": hi,
        "typical": typical, "thr": thr,
        "last_good": int(last_good),
        "bottom_prior": int(bottom_prior_c),
    }, vert_mask


# ============================================================
# Sloped separators (windowed snapping within interior band)
# ============================================================

def slope_lines_from_windows(rr: np.ndarray, ink: np.ndarray, table_top: int, table_bottom: int, x_offset_in_crop: int = 0):
    """
    Fit each separator line y = m x + b in CROP coords, using x in CROP coords.
    rr/ink provided are the DETECTION BAND (subset of crop width).
    x_offset_in_crop tells where this band begins inside the crop.
    """
    h, w_det = rr.shape
    span = max(1, int(table_bottom - table_top))
    step = float(span) / float(NUM_ROWS)

    nW = int(max(2, SLOPE_NUM_WINDOWS))
    win_w = int(max(240, w_det / nW))
    stride = int(max(90, win_w * (1.0 - float(SLOPE_WIN_OVERLAP))))

    windows = []
    x0 = 0
    while x0 < w_det:
        x1 = min(w_det, x0 + win_w)
        if x1 - x0 >= 180:
            windows.append((x0, x1))
        if x1 == w_det:
            break
        x0 += stride

    # Points per separator
    sep_pts = [[] for _ in range(NUM_ROWS + 1)]

    for (a, b) in windows:
        x_center_det = 0.5 * (a + b)
        x_center_crop = float(x_offset_in_crop) + float(x_center_det)

        # window rr/ink
        rrw = rr[:, a:b]
        inkw = ink[:, a:b]

        for i in range(NUM_ROWS + 1):
            y_exp = int(round(table_top + i * step))
            y_pick, _ = snap_separator(rrw, inkw, y_exp, band=int(SLOPE_SEARCH_BAND))
            sep_pts[i].append((x_center_crop, float(y_pick)))

    lines = []
    for i in range(NUM_ROWS + 1):
        pts = sep_pts[i]
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)

        med = float(np.median(ys))
        mad = float(np.median(np.abs(ys - med))) + 1e-6
        keep = np.abs(ys - med) <= (2.6 * mad + 6.0)
        xs2, ys2 = xs[keep], ys[keep]

        ok = bool(len(xs2) >= max(4, int(0.55 * len(xs))))
        if not ok:
            lines.append((0.0, float(med), False, int(len(xs2))))
            continue

        A = np.column_stack([xs2, np.ones_like(xs2)])
        m, b0 = np.linalg.lstsq(A, ys2, rcond=None)[0]
        lines.append((float(m), float(b0), True, int(len(xs2))))

    return lines, {"windows_det": windows, "x_offset_in_crop": int(x_offset_in_crop)}


# ============================================================
# Visualization
# ============================================================

def draw_overlay(gray: np.ndarray, columns: dict, lines_full: list,
                 xL: int, xR: int, out_path: str, title: str):
    viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape

    if title:
        cv2.putText(viz, title, (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 0), 2)

    # draw vertical column guides (current fixed coords)
    for _, (a, b) in columns.items():
        cv2.line(viz, (a, 0), (a, h), (255, 0, 0), 2)
        cv2.line(viz, (b, 0), (b, h), (255, 0, 0), 2)

    # draw table crop band
    cv2.line(viz, (xL, 0), (xL, h), (150, 150, 0), 2)
    cv2.line(viz, (xR, 0), (xR, h), (150, 150, 0), 2)

    # draw sloped horizontal separators only across [xL..xR] (not page margins)
    xs = np.arange(xL, xR, dtype=np.int32)
    for (m, b, ok, _) in lines_full:
        color = (0, 0, 255) if ok else (0, 128, 255)
        ys = (m * xs + b).astype(np.int32)
        pts = np.column_stack([xs, np.clip(ys, 0, h - 1)])
        cv2.polylines(viz, [pts], isClosed=False, color=color, thickness=2)

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

    # Table X band from your known columns (we do NOT use page margins anymore)
    x_min = min(v[0] for v in COLUMNS.values())
    x_max = max(v[1] for v in COLUMNS.values())

    # Expand to approximate table width, but keep within image
    xL = int(max(0, x_min - 140))
    xR = int(min(w, xL + TABLE_WIDTH_PX_PRIOR + 280))
    if xR <= xL + 10:
        xL, xR = 0, w

    # Crop around expected table band
    y0 = max(0, FIRST_ROW_Y_PRIOR - ROI_TOP_PAD)
    y1 = min(h, FIRST_ROW_Y_PRIOR + TABLE_HEIGHT_PX_PRIOR + ROI_BOTTOM_PAD)

    crop_gray = gray[y0:y1, xL:xR]
    rr_full = enhance_faint_rules(crop_gray)
    ink_full = robust_ink_mask(crop_gray)

    # Detection band INSIDE crop to avoid margins dominating
    det_x0 = int(np.clip(DETECT_X_INSET, 0, max(0, rr_full.shape[1] - 2)))
    det_x1 = int(np.clip(rr_full.shape[1] - DETECT_X_INSET, det_x0 + 1, rr_full.shape[1]))
    rr = rr_full[:, det_x0:det_x1]
    ink = ink_full[:, det_x0:det_x1]

    # Peak candidates from detection band only
    e = row_energy(rr)
    e_s = smooth_1d(e, PROJ_SMOOTH_K)
    peaks = find_peaks_1d(e_s, min_rel=PEAK_MIN_REL, merge_dist=PEAK_MERGE_DIST)

    first_prior_in_crop = int(FIRST_ROW_Y_PRIOR - y0)
    top_c, top_dbg = choose_table_top_gridaware(rr, ink, peaks, first_prior_in_crop)

    # Bottom: prior then clamp using vertical-rule density (still in detection band)
    bottom_prior_c = int(min(rr.shape[0] - 1, top_c + TABLE_HEIGHT_PX_PRIOR))
    bottom_c, bottom_dbg, vert_mask = clamp_table_bottom(rr, top_c, bottom_prior_c)

    # Force a usable span (avoid degeneracy)
    span = int(max(2200, bottom_c - top_c))
    bottom_c = int(min(rr.shape[0] - 1, top_c + span))

    # Now the KEY: distribute exactly 41 expected lines across [top..bottom]
    step = float(bottom_c - top_c) / float(NUM_ROWS)

    # Build sloped or flat separators (in CROP coords)
    if USE_SLOPE:
        # slope lines are in CROP coords x within rr band, but we fit with x in crop coords using x_offset
        lines_crop, slope_dbg = slope_lines_from_windows(rr, ink, top_c, bottom_c, x_offset_in_crop=int(det_x0))
    else:
        # flat: snap each expected y using rr band, then make y = const lines (still in crop coords)
        lines_crop = []
        slope_dbg = {"windows_det": [], "x_offset_in_crop": int(det_x0)}
        snap_samples = []
        for i in range(NUM_ROWS + 1):
            y_exp = int(round(top_c + i * step))
            y_pick, dbg = snap_separator(rr, ink, y_exp, band=SNAP_BAND)
            lines_crop.append((0.0, float(y_pick), True, 0))
            if i in (0, 1, 2, 20, 39, 40):
                snap_samples.append(dbg)
        slope_dbg["flat_snap_debug_samples"] = snap_samples

    # Convert lines_crop (crop coords) to FULL image coords: y_full = m*x_full + b_full
    # Here, lines_crop already use x in CROP coords (because we fit with x_offset_in_crop).
    # Crop mapping: x_crop = x_full - xL, y_full = y_crop + y0
    lines_full = []
    for (m, b0, ok, npts) in lines_crop:
        # y_crop = m*(x_crop) + b0 => y_full = m*(x_full - xL) + b0 + y0 = m*x_full + (b0 + y0 - m*xL)
        b_full = float(b0 + y0 - m * xL)
        lines_full.append((float(m), float(b_full), bool(ok), int(npts)))

    # Save overlay + report
    if SAVE_VIZ:
        viz_path = os.path.join(out_dir, "grid_overlay.png")
        title = f"{name} | slope={USE_SLOPE} | peaks={len(peaks)} | top={top_c+y0} bottom={bottom_c+y0}"
        draw_overlay(gray, COLUMNS, lines_full, xL, xR, viz_path, title)

    if SAVE_DEBUG_IMAGES:
        cv2.imwrite(os.path.join(out_dir, "debug_rule_response_crop_full.png"), rr_full)
        cv2.imwrite(os.path.join(out_dir, "debug_rule_response_crop_detectband.png"), rr)
        cv2.imwrite(os.path.join(out_dir, "debug_ink_crop_detectband.png"), ink)
        cv2.imwrite(os.path.join(out_dir, "debug_vertical_mask_detectband.png"), vert_mask)

    report = {
        "image_name": name,
        "source_path": img_path,
        "full_shape": {"h": int(h), "w": int(w)},
        "crop": {"xL": int(xL), "xR": int(xR), "y0": int(y0), "y1": int(y1)},
        "detect_band_in_crop": {"det_x0": int(det_x0), "det_x1": int(det_x1)},
        "top_full": int(top_c + y0),
        "bottom_full": int(bottom_c + y0),
        "span_crop": int(bottom_c - top_c),
        "step_crop": float(step),
        "peaks_count": int(len(peaks)),
        "top_selection": top_dbg,
        "bottom_selection": bottom_dbg,
        "use_slope": bool(USE_SLOPE),
        "slope_debug": slope_dbg,
        "lines_full": [
            {"i": i, "m": float(m), "b": float(b), "ok": bool(ok), "npts": int(npts)}
            for i, (m, b, ok, npts) in enumerate(lines_full)
        ],
        "note": "This version saves only grid_overlay.png + report.json for speed. 41 separators are always produced (sloped or flat).",
    }

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"top={top_c+y0} bottom={bottom_c+y0} peaks={len(peaks)}")
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

    print("=== SMART ADAPTIVE EXTRACTION v13.2 (SLOPED 41 LINES + INTERIOR BAND + BOTTOM CLAMP) ===")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images found: {len(imgs)}")
    print(f"USE_SLOPE={USE_SLOPE} | SAVE_DEBUG_IMAGES={SAVE_DEBUG_IMAGES}")

    for i, p in enumerate(imgs, start=1):
        print(f"\n[{i}/{len(imgs)}]")
        process_one_image(p)

    print("\n🎯 DONE")

if __name__ == "__main__":
    main()
