import os
import cv2
import json
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "data/from_jeremy/images_aligned_to_first"
OUTPUT_DIR = "data/processed/smart_adaptive_extraction_batch_v13_proj"

# Table priors (your measurements / priors)
NUM_ROWS = 40
FIRST_ROW_Y_PRIOR = 1263
EXPECTED_ROW_HEIGHT = 78
TABLE_HEIGHT_PX = 3160
TABLE_WIDTH_PX = 6150

# ROI pads and x-band (keep footer out of the party)
TABLE_X_MARGIN = 140
ROI_TOP_PAD = 260
ROI_BOTTOM_PAD = 420

# Enhancement knobs (faint rules)
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
BLACKHAT_KSIZE = 35
BLACKHAT_MIX = 0.85

# Projection + peak knobs
PROJ_SMOOTH_K = 19
PEAK_MIN_REL = 0.22
PEAK_MERGE_DIST = 12
PEAK_SEARCH_BAND = 10   # when snapping to grid, search ± this many px

# Table-top first peak search (relative to prior)
FIRST_LINE_ROI_UP = 340
FIRST_LINE_ROI_DOWN = 740

# Optional slope fit with windowed projections
USE_SLOPE = True
SLOPE_NUM_WINDOWS = 10
SLOPE_WIN_OVERLAP = 0.35
SLOPE_MIN_REL = 0.18
SLOPE_SEARCH_BAND = 12

# Row rectification output height (for later column cutouts)
RECT_ROW_H = EXPECTED_ROW_HEIGHT

# Columns (fixed for now; column alignment later)
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

SAVE_VIZ = True
SAVE_CELLS = True
SAVE_DEBUG = True


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
# Enhancement: CLAHE + Blackhat
# ============================================================

def enhance_faint_rules(gray: np.ndarray,
                        clahe_clip=CLAHE_CLIP,
                        clahe_grid=CLAHE_GRID,
                        blackhat_ksize=BLACKHAT_KSIZE,
                        mix=BLACKHAT_MIX):
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_grid))
    g = clahe.apply(gray)

    k = int(blackhat_ksize)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)

    rule_response = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    # This is just for visualization; projections use rule_response directly.
    enhanced_gray = cv2.addWeighted(g, 1.0, rule_response, float(mix), 0)
    return enhanced_gray, rule_response


# ============================================================
# Projections + peaks
# ============================================================

def row_energy_from_rule_response(rr: np.ndarray) -> np.ndarray:
    # Sum across x. rr is uint8, but cast to float for stability.
    return np.sum(rr.astype(np.float32), axis=1)

def find_peaks_1d(signal: np.ndarray, min_rel: float, merge_dist: int) -> list:
    if signal.size == 0:
        return []
    s = signal.copy().astype(np.float32)
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

def pick_table_top_peak(peaks_full_y: list, first_y_prior: int) -> (int, dict):
    if not peaks_full_y:
        return int(first_y_prior), {"picked_from": "prior_fallback_no_peaks"}

    roi_lo = int(first_y_prior - FIRST_LINE_ROI_UP)
    roi_hi = int(first_y_prior + FIRST_LINE_ROI_DOWN)
    cands = [y for y in peaks_full_y if roi_lo <= y <= roi_hi]

    if cands:
        return int(min(cands)), {"picked_from": "roi_earliest_peak", "roi_lo": roi_lo, "roi_hi": roi_hi}

    nearest = min(peaks_full_y, key=lambda y: abs(y - first_y_prior))
    return int(nearest), {"picked_from": "nearest_peak_fallback", "roi_lo": roi_lo, "roi_hi": roi_hi}

def snap_to_nearest_peak(y: int, peaks: list, search_band: int) -> int:
    if not peaks:
        return int(y)
    lo = int(y - search_band)
    hi = int(y + search_band)
    cands = [p for p in peaks if lo <= p <= hi]
    if not cands:
        return int(y)
    # choose nearest
    return int(min(cands, key=lambda p: abs(p - y)))


# ============================================================
# Windowed slope fitting from projections
# ============================================================

def window_slope_lines(rr_crop: np.ndarray, x_offsets: list, table_top_in_crop: int, table_bottom_in_crop: int):
    """
    For each x-window, compute row-energy peaks and snap to expected grid positions,
    yielding points (x_center, y) for each separator index. Then fit y = m x + b.
    Returns lines_mb: list of (m,b,ok,points_used_count)
    """
    h, w = rr_crop.shape
    span = max(1, int(table_bottom_in_crop - table_top_in_crop))
    step = float(span) / float(NUM_ROWS)

    # Prepare windows
    nW = int(max(2, SLOPE_NUM_WINDOWS))
    win_w = int(max(220, w / nW))
    overlap = float(SLOPE_WIN_OVERLAP)
    stride = int(max(80, win_w * (1.0 - overlap)))

    windows = []
    x0 = 0
    while x0 < w:
        x1 = min(w, x0 + win_w)
        if x1 - x0 >= 160:
            windows.append((x0, x1))
        if x1 == w:
            break
        x0 += stride

    # For each window, find peaks in that window's projection
    per_win_peaks = []
    for (a, b) in windows:
        rrw = rr_crop[:, a:b]
        energy = row_energy_from_rule_response(rrw)
        energy_s = smooth_1d(energy, PROJ_SMOOTH_K)
        peaks = find_peaks_1d(energy_s, min_rel=SLOPE_MIN_REL, merge_dist=PEAK_MERGE_DIST)
        per_win_peaks.append(peaks)

    # For each separator i, collect (x_center, y_in_crop) points from windows
    sep_points = [[] for _ in range(NUM_ROWS + 1)]
    for wi, (a, b) in enumerate(windows):
        peaks = per_win_peaks[wi]
        x_center = 0.5 * (a + b)
        for i in range(NUM_ROWS + 1):
            y_expect = int(round(table_top_in_crop + i * step))
            y_snap = snap_to_nearest_peak(y_expect, peaks, search_band=SLOPE_SEARCH_BAND)
            sep_points[i].append((x_center, y_snap))

    # Fit y = m x + b for each separator using robust least squares (median prune)
    lines_mb = []
    for i in range(NUM_ROWS + 1):
        pts = sep_points[i]
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)

        # remove outliers by median absolute deviation
        med = float(np.median(ys))
        mad = float(np.median(np.abs(ys - med))) + 1e-6
        keep = np.abs(ys - med) <= (2.8 * mad + 6.0)
        xs2, ys2 = xs[keep], ys[keep]

        ok = bool(len(xs2) >= max(4, int(0.5 * len(xs))))
        if not ok:
            m, b0 = 0.0, float(med)
            lines_mb.append((float(m), float(b0), False, int(len(xs2))))
            continue

        A = np.column_stack([xs2, np.ones_like(xs2)])
        m, b0 = np.linalg.lstsq(A, ys2, rcond=None)[0]
        lines_mb.append((float(m), float(b0), True, int(len(xs2))))

    return lines_mb, {"windows": windows}


# ============================================================
# Rectify row strip between two lines
# ============================================================

def rectify_row_strip(gray_ds: np.ndarray, m1: float, b1: float, m2: float, b2: float,
                      xL_full: int, xR_full: int, out_h: int):
    h, w = gray_ds.shape
    xL = int(np.clip(xL_full, 0, w - 2))
    xR = int(np.clip(xR_full, xL + 1, w - 1))
    out_w = int(xR - xL)

    xs = np.arange(xL, xR, dtype=np.float32)
    y_top = (m1 * (xs - xL) + b1).astype(np.float32)  # note: m,b in crop coords unless adjusted
    y_bot = (m2 * (xs - xL) + b2).astype(np.float32)

    y_top = np.clip(y_top, 0, h - 1)
    y_bot = np.clip(y_bot, 0, h - 1)
    y_min = np.minimum(y_top, y_bot)
    y_max = np.maximum(y_top, y_bot)
    y_top, y_bot = y_min, y_max

    t = np.linspace(0.0, 1.0, out_h, dtype=np.float32)[:, None]
    map_x = np.tile(xs[None, :], (out_h, 1))
    map_y = y_top[None, :] + t * (y_bot[None, :] - y_top[None, :])

    strip = cv2.remap(gray_ds, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return strip


# ============================================================
# Visualization helpers
# ============================================================

def draw_overlay_sloped(gray: np.ndarray, columns: dict, lines_mb_full: list,
                        table_top: int, table_bottom: int, out_path: str, title: str,
                        xL: int, xR: int):
    viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape

    if title:
        cv2.putText(viz, title, (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 0), 2)

    for col_name, (a, b) in columns.items():
        cv2.line(viz, (a, 0), (a, h), (255, 0, 0), 2)
        cv2.line(viz, (b, 0), (b, h), (255, 0, 0), 2)

    cv2.line(viz, (xL, 0), (xL, h), (150, 150, 0), 2)
    cv2.line(viz, (xR, 0), (xR, h), (150, 150, 0), 2)

    cv2.line(viz, (0, table_top), (w, table_top), (255, 255, 0), 2)
    cv2.line(viz, (0, table_bottom), (w, table_bottom), (0, 255, 255), 3)

    xs = np.arange(xL, xR, dtype=np.int32)
    for i, (m, b, ok, _) in enumerate(lines_mb_full):
        color = (0, 0, 255) if ok else (0, 128, 255)
        ys = (m * xs + b).astype(np.int32)
        pts = np.column_stack([xs, np.clip(ys, 0, h - 1)])
        cv2.polylines(viz, [pts], isClosed=False, color=color, thickness=2)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, viz)

def save_signal_debug(signal: np.ndarray, out_path: str, marks: list = None):
    s = signal.astype(np.float32).copy()
    s -= float(np.min(s))
    mx = float(np.max(s)) if float(np.max(s)) > 0 else 1.0
    s = (s / mx) * 255.0
    img = s.astype(np.uint8).reshape(-1, 1)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if marks:
        for y, color in marks:
            if 0 <= y < img.shape[0]:
                img[y, 0] = color

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, img)


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

    # Crop around expected table band (keeps footer out)
    y0 = max(0, FIRST_ROW_Y_PRIOR - ROI_TOP_PAD)
    y1 = min(h, FIRST_ROW_Y_PRIOR + TABLE_HEIGHT_PX + ROI_BOTTOM_PAD)
    crop = gray[y0:y1, xL:xR]

    # Enhance + rule response
    enh_crop, rr_crop = enhance_faint_rules(crop)

    # Row energy + peaks in crop coords
    energy = row_energy_from_rule_response(rr_crop)
    energy_s = smooth_1d(energy, PROJ_SMOOTH_K)
    peaks = find_peaks_1d(energy_s, min_rel=PEAK_MIN_REL, merge_dist=PEAK_MERGE_DIST)
    peaks_full_y = [int(p + y0) for p in peaks]

    # Pick table top peak
    table_top, top_dbg = pick_table_top_peak(peaks_full_y, FIRST_ROW_Y_PRIOR)
    table_bottom = int(table_top + TABLE_HEIGHT_PX)  # fixed height prior (no footer leakage)

    # Convert table top/bottom into crop coords
    table_top_c = int(table_top - y0)
    table_bottom_c = int(min(rr_crop.shape[0] - 1, table_top_c + TABLE_HEIGHT_PX))

    # Build separators: either sloped (windowed) or flat (global snap)
    if USE_SLOPE:
        lines_mb_crop, slope_dbg = window_slope_lines(rr_crop, [], table_top_c, table_bottom_c)
        # Convert crop lines into FULL-image y = m x + b (in full coords)
        # In window_slope_lines, x is in crop coords [0..w_crop). We'll translate to full x by adding xL.
        # That means: y_crop = m*(x_crop) + b  => y_full = (m*(x_full - xL)) + b + y0
        lines_mb_full = []
        for (m, b0, ok, npts) in lines_mb_crop:
            b_full = float(b0 + y0 - m * xL)
            lines_mb_full.append((float(m), float(b_full), bool(ok), int(npts)))
        method = "rule_response_projection_windowed_slope"
    else:
        # Flat grid: uniform + snap to peaks
        span = float(TABLE_HEIGHT_PX)
        step = span / float(NUM_ROWS)
        y_centers = [int(round(table_top + i * step)) for i in range(NUM_ROWS + 1)]
        # snap to peaks within band
        lines_mb_full = []
        for y in y_centers:
            y_snap = snap_to_nearest_peak(y, peaks_full_y, search_band=PEAK_SEARCH_BAND)
            lines_mb_full.append((0.0, float(y_snap), True, 0))
        slope_dbg = {"windows": []}
        method = "rule_response_projection_flat"

    # Rectify rows and save cells (optional)
    head_dir = os.path.join(img_out, "rows_rectified")
    ensure_dir(head_dir)

    rows_found = 0
    for i in range(NUM_ROWS):
        m1, b1, ok1, _ = lines_mb_full[i]
        m2, b2, ok2, _ = lines_mb_full[i + 1]

        # stop if row is beyond table bottom (midpoint)
        midx = 0.5 * (xL + xR)
        y_mid_bot = m2 * midx + b2
        if y_mid_bot > table_bottom + 50:
            break

        # rectify full-image strip within xL..xR
        row_strip = rectify_row_strip(gray, m1, b1, m2, b2, xL_full=xL, xR_full=xR, out_h=RECT_ROW_H)
        cv2.imwrite(os.path.join(head_dir, f"row{i:02d}.png"), row_strip)
        rows_found += 1

        if SAVE_CELLS:
            # also cut columns in rectified row
            for col_name, (cx1, cx2) in COLUMNS.items():
                cell = row_strip[:, cx1:cx2]
                if cell.size:
                    cv2.imwrite(os.path.join(img_out, f"row{i:02d}_{col_name}.png"), cell)

    # Overlay
    if SAVE_VIZ:
        viz_path = os.path.join(img_out, "grid_overlay.png")
        title = f"{name} | method={method} | peaks={len(peaks)} | rows={rows_found}"
        draw_overlay_sloped(gray, COLUMNS, lines_mb_full, table_top, table_bottom, viz_path, title, xL, xR)

    # Debug artifacts
    if SAVE_DEBUG:
        cv2.imwrite(os.path.join(img_out, "debug_rule_response_crop.png"), rr_crop)
        cv2.imwrite(os.path.join(img_out, "debug_enhanced_crop.png"), enh_crop)
        save_signal_debug(energy_s, os.path.join(img_out, "debug_row_energy.png"),
                          marks=[(int(table_top - y0), (0, 255, 0)), (int(table_bottom - y0), (0, 0, 255))])
        # mark peaks too
        peak_marks = [(p, (0, 255, 255)) for p in peaks]
        save_signal_debug(energy_s, os.path.join(img_out, "debug_row_energy_peaks.png"), marks=peak_marks)

    report = {
        "image_name": name,
        "source_path": img_path,
        "x_band": {"xL": int(xL), "xR": int(xR)},
        "crop": {"y0": int(y0), "y1": int(y1)},
        "method": method,
        "table_top": int(table_top),
        "table_bottom": int(table_bottom),
        "table_top_debug": top_dbg,
        "peaks_count": int(len(peaks)),
        "rows_found": int(rows_found),
        "slope_debug": slope_dbg,
        "lines_mb_full": [{"i": i, "m": float(m), "b": float(b), "ok": bool(ok), "npts": int(npts)}
                          for i, (m, b, ok, npts) in enumerate(lines_mb_full)]
    }
    with open(os.path.join(img_out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"table_top={table_top} bottom={table_bottom} peaks={len(peaks)} rows={rows_found}")
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

    print("=== SMART ADAPTIVE EXTRACTION v13 (RULE_RESPONSE + PROJECTIONS) ===")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images found: {len(imgs)}")
    print(f"USE_SLOPE={USE_SLOPE} windows={SLOPE_NUM_WINDOWS}")

    for i, p in enumerate(imgs, start=1):
        print(f"\n[{i}/{len(imgs)}]")
        process_one_image(p)

    print("\n🎯 DONE")

if __name__ == "__main__":
    main()
