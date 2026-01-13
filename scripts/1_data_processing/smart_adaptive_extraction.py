import os
import cv2
import json
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "data/from_jeremy/images_aligned_to_first"
OUTPUT_DIR = "data/processed/smart_adaptive_extraction_batch_v12_rule_response_rows"

NUM_ROWS = 40
FIRST_ROW_Y_PRIOR = 1263
EXPECTED_ROW_HEIGHT = 78

# Your measured priors
TABLE_HEIGHT_PX = 3160
TABLE_WIDTH_PX = 6150

TABLE_X_MARGIN = 140
ROI_TOP_PAD = 260
ROI_BOTTOM_PAD = 420

FIRST_LINE_ROI_UP = 340
FIRST_LINE_ROI_DOWN = 740

# Deskew (table ROI) via minAreaRect on rule-like pixels
DESKEW_MIN_POINTS = 2500

# Enhancement knobs (faint rules)
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
BLACKHAT_KSIZE = 35
BLACKHAT_MIX = 0.85

# Rule-response line mask from blackhat response
RULEMASK_BLUR_K = 3            # small blur before threshold
RULEMASK_THRESH_PCT = 88       # percentile threshold on rule response (higher = stricter)
RULEMASK_HOPEN_KERNEL_DIV = 18 # bigger = smaller kernel; tune for your scans
RULEMASK_HOPEN_ITERS = 1
RULEMASK_DILATE_W = 35

# Peak detection on row strength from rule mask
PEAK_SMOOTH_K = 11
PEAK_MIN_FRAC = 0.16           # threshold relative to max strength
PEAK_MERGE_DIST = 10

# Fit band around each detected separator y (in full coords)
FIT_BAND_HALFHEIGHT = 10
FIT_MIN_POINTS = 1400

# Rectified row strip height (pixels)
RECT_ROW_H = EXPECTED_ROW_HEIGHT

# Head detection knobs (kept; used after rectification)
INK_PAD = 12
MIN_INK_RATIO = 0.010
MIN_CC_AREA = 60

# Columns (fixed for now)
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
# FS + Basic helpers
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

def robust_binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )

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
    enhanced_gray = cv2.addWeighted(g, 1.0, rule_response, float(mix), 0)
    return enhanced_gray, rule_response


# ============================================================
# Build horizontal rule mask DIRECTLY from rule_response
# ============================================================

def rulemask_from_rule_response(rule_response: np.ndarray) -> np.ndarray:
    """
    Produces a binary mask emphasizing long horizontal rules using rule_response (blackhat).
    """
    rr = rule_response.copy()
    if RULEMASK_BLUR_K and RULEMASK_BLUR_K >= 3:
        rr = cv2.GaussianBlur(rr, (RULEMASK_BLUR_K, RULEMASK_BLUR_K), 0)

    # percentile threshold (more stable than Otsu across scans)
    thr = np.percentile(rr, RULEMASK_THRESH_PCT)
    _, bw = cv2.threshold(rr, thr, 255, cv2.THRESH_BINARY)

    h, w = bw.shape
    hk = max(60, w // RULEMASK_HOPEN_KERNEL_DIV)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))

    # keep long horizontal structures
    hm = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel, iterations=RULEMASK_HOPEN_ITERS)

    # strengthen continuity
    hm = cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_RECT, (RULEMASK_DILATE_W, 1)), iterations=1)
    return hm


# ============================================================
# Deskew using minAreaRect on rulemask pixels (table ROI)
# ============================================================

def rotate_image(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.05:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def estimate_skew_angle_from_rulemask(gray_roi: np.ndarray) -> float:
    _, rr = enhance_faint_rules(gray_roi)
    rm = rulemask_from_rule_response(rr)

    ys, xs = np.where(rm > 0)
    if len(xs) < DESKEW_MIN_POINTS:
        return 0.0

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    rect = cv2.minAreaRect(pts)
    angle = float(rect[-1])
    rw, rh = rect[1]
    if rw < rh:
        angle += 90.0
    if angle > 20 or angle < -20:
        return 0.0
    return angle

def deskew_using_table_roi(gray: np.ndarray, xL: int, xR: int, y_top_prior: int):
    h, w = gray.shape
    y0 = max(0, y_top_prior - ROI_TOP_PAD)
    y1 = min(h, y_top_prior + TABLE_HEIGHT_PX + ROI_BOTTOM_PAD)

    xL = int(np.clip(xL, 0, w - 2))
    xR = int(np.clip(xR, xL + 1, w - 1))
    roi = gray[y0:y1, xL:xR]

    angle = estimate_skew_angle_from_rulemask(roi)
    gray_ds = rotate_image(gray, -angle)

    dbg = {"method": "table_roi_minAreaRect_rulemask", "roi": {"xL": xL, "xR": xR, "y0": y0, "y1": y1}, "angle_deg": float(angle)}
    return gray_ds, float(angle), dbg


# ============================================================
# Table top detection from rulemask row strength (rule_response-based)
# ============================================================

def find_peaks_from_strength(strength: np.ndarray, min_frac: float, merge_dist: int):
    if strength.size == 0 or float(strength.max()) <= 0:
        return []
    s = strength / float(strength.max())
    peaks = []
    for y in range(1, len(s) - 1):
        if s[y] > s[y - 1] and s[y] > s[y + 1] and s[y] >= float(min_frac):
            peaks.append(y)

    merged = []
    for p in peaks:
        if not merged or abs(p - merged[-1]) > merge_dist:
            merged.append(p)
        else:
            if strength[p] > strength[merged[-1]]:
                merged[-1] = p
    return merged

def pick_table_top_from_peaks(peaks_full_y: list, first_y_prior: int):
    if not peaks_full_y:
        return int(first_y_prior), {"picked_from": "prior_fallback_no_peaks"}
    roi_lo = int(first_y_prior - FIRST_LINE_ROI_UP)
    roi_hi = int(first_y_prior + FIRST_LINE_ROI_DOWN)
    cands = [y for y in peaks_full_y if roi_lo <= y <= roi_hi]
    if cands:
        # earliest strong boundary in ROI
        return int(min(cands)), {"picked_from": "roi_earliest_peak", "roi_lo": roi_lo, "roi_hi": roi_hi}
    nearest = min(peaks_full_y, key=lambda y: abs(y - first_y_prior))
    return int(nearest), {"picked_from": "nearest_peak_fallback", "roi_lo": roi_lo, "roi_hi": roi_hi}


# ============================================================
# Bottom detection (rulemask-based density drop)
# ============================================================

def detect_table_bottom_from_rulemask(rulemask_xcrop: np.ndarray, table_top_y: int):
    """
    rulemask_xcrop: binary mask (255=rule) built from rule_response on x-cropped full-height table area.
    """
    h, w = rulemask_xcrop.shape
    dens = np.sum(rulemask_xcrop > 0, axis=1).astype(np.float32)
    dens_s = smooth_1d(dens, 51)

    expected_bottom = int(table_top_y + TABLE_HEIGHT_PX)
    search_start = int(max(0, expected_bottom - 380))
    search_end = int(min(h - 1, expected_bottom + 950))

    mid0 = int(max(0, table_top_y + 400))
    mid1 = int(min(h - 1, table_top_y + 2200))
    typ = float(np.median(dens_s[mid0:mid1])) if mid1 > mid0 else float(np.median(dens_s))

    thr = typ * 0.18  # if rule density collapses below this, table likely ended
    window = 95
    bottom_y = expected_bottom

    for y in range(search_start, max(search_start, search_end - window)):
        seg = dens_s[y:y + window]
        if seg.size == window and float(np.max(seg)) < thr:
            bottom_y = y
            break

    # Guardrail: don't allow bottom to go way too high
    min_ok = int(table_top_y + 0.85 * TABLE_HEIGHT_PX)
    if bottom_y < min_ok:
        bottom_y = expected_bottom

    debug = {
        "method": "rulemask_density_drop_xcrop",
        "expected_bottom": expected_bottom,
        "search_start": search_start,
        "search_end": search_end,
        "typ": float(typ),
        "thr": float(thr),
        "bottom_y": int(bottom_y),
        "window": int(window),
        "guardrail_min_ok": int(min_ok)
    }
    return int(bottom_y), debug, dens_s


# ============================================================
# Fit sloped line y = m x + b using RULEMASK pixels
# ============================================================

def fit_sloped_rule_line_from_rulemask(rulemask_full_x: np.ndarray, xL: int, xR: int, y_center: int):
    """
    Fit y = m x + b for the rule near y_center using rulemask pixels in a narrow band.
    rulemask_full_x is a binary mask in FULL IMAGE coords (not x-cropped),
    but we fit only x in [xL, xR).
    """
    h, w = rulemask_full_x.shape
    y0 = max(0, int(y_center - FIT_BAND_HALFHEIGHT))
    y1 = min(h, int(y_center + FIT_BAND_HALFHEIGHT + 1))

    xL2 = max(0, min(w - 2, int(xL)))
    xR2 = max(xL2 + 1, min(w - 1, int(xR)))

    band = rulemask_full_x[y0:y1, xL2:xR2]
    ys, xs = np.where(band > 0)
    n = int(len(xs))
    if n < FIT_MIN_POINTS:
        return 0.0, float(y_center), False, n

    xs = xs.astype(np.float32) + float(xL2)
    ys = ys.astype(np.float32) + float(y0)

    pts = np.column_stack([xs, ys]).astype(np.float32)

    vx, vy, x0, y0p = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx = float(vx); vy = float(vy); x0 = float(x0); y0p = float(y0p)
    if abs(vx) < 1e-6:
        return 0.0, float(y_center), False, n

    m = vy / vx
    b = y0p - m * x0
    return float(m), float(b), True, n


# ============================================================
# Row rectification: remap between two sloped lines
# ============================================================

def rectify_row_strip(gray: np.ndarray, m1: float, b1: float, m2: float, b2: float,
                      xL: int, xR: int, out_h: int):
    h, w = gray.shape
    xL = int(np.clip(xL, 0, w - 2))
    xR = int(np.clip(xR, xL + 1, w - 1))
    out_w = int(xR - xL)

    xs = np.arange(xL, xR, dtype=np.float32)
    y_top = (m1 * xs + b1).astype(np.float32)
    y_bot = (m2 * xs + b2).astype(np.float32)

    y_top = np.clip(y_top, 0, h - 1)
    y_bot = np.clip(y_bot, 0, h - 1)

    y_min = np.minimum(y_top, y_bot)
    y_max = np.maximum(y_top, y_bot)
    y_top, y_bot = y_min, y_max

    t = np.linspace(0.0, 1.0, out_h, dtype=np.float32)[:, None]
    map_x = np.tile(xs[None, :], (out_h, 1))
    map_y = y_top[None, :] + t * (y_bot[None, :] - y_top[None, :])

    strip = cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return strip


# ============================================================
# Head detection on rectified strip (rented/owned)
# ============================================================

def remove_table_lines(ink_mask: np.ndarray) -> np.ndarray:
    h, w = ink_mask.shape
    hk = max(20, w // 14)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    horiz = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

    vk = max(25, h // 2)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    vert = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    lines = cv2.bitwise_or(horiz, vert)
    return cv2.bitwise_and(ink_mask, cv2.bitwise_not(lines))

def cell_has_ink(cell_gray: np.ndarray) -> bool:
    if cell_gray is None or cell_gray.size == 0:
        return False
    h, w = cell_gray.shape
    if h <= 2 * INK_PAD or w <= 2 * INK_PAD:
        return False

    roi = cell_gray[INK_PAD:h - INK_PAD, INK_PAD:w - INK_PAD]
    bin_img = robust_binarize(roi)
    ink = 255 - bin_img

    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    ink = remove_table_lines(ink)

    ink_pixels = int(np.count_nonzero(ink > 0))
    total = int(ink.size)
    if total == 0:
        return False
    if (ink_pixels / total) < MIN_INK_RATIO:
        return False

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return False
    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest_area >= MIN_CC_AREA

def detect_head_row_from_tenure_cols_rectified(row_strip: np.ndarray, columns: dict):
    rented_x1, rented_x2 = columns["rented"]
    owned_x1, owned_x2 = columns["owned"]
    rented_cell = row_strip[:, rented_x1:rented_x2]
    owned_cell = row_strip[:, owned_x1:owned_x2]

    is_rented = cell_has_ink(rented_cell)
    is_owned = cell_has_ink(owned_cell)

    is_head = bool(is_rented or is_owned)
    if is_rented and not is_owned:
        tenure = "RENTED"
    elif is_owned and not is_rented:
        tenure = "OWNED"
    elif is_rented and is_owned:
        tenure = "BOTH_UNCLEAR"
    else:
        tenure = "NONE"
    return is_head, tenure


# ============================================================
# Visualization
# ============================================================

def save_density_debug(arr: np.ndarray, out_path: str, mark_top: int, mark_bottom: int):
    s = arr.copy()
    if s.max() > 0:
        s = s / s.max()
    img = (s * 255).astype(np.uint8).reshape(-1, 1)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if 0 <= mark_top < img.shape[0]:
        img[mark_top, 0] = (0, 255, 0)
    if 0 <= mark_bottom < img.shape[0]:
        img[mark_bottom, 0] = (0, 0, 255)
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, img)

def draw_overlay_sloped(gray: np.ndarray, columns: dict, lines_mb: list,
                        table_top: int, table_bottom: int, head_rows: list,
                        out_path: str, title: str, xL: int, xR: int):
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
    for i, (m, b, ok, npts) in enumerate(lines_mb):
        color = (0, 255, 0) if i in head_rows else (0, 0, 255)
        if not ok:
            color = (0, 128, 255)
        ys = (m * xs + b).astype(np.int32)
        pts = np.column_stack([xs, np.clip(ys, 0, h - 1)])
        cv2.polylines(viz, [pts], isClosed=False, color=color, thickness=2)

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

    img_out = os.path.join(OUTPUT_DIR, name)
    ensure_dir(img_out)

    # Table x-range early (based on known columns)
    min_x1 = min(v[0] for v in COLUMNS.values())
    xL = int(min_x1 - TABLE_X_MARGIN)
    xR = int(xL + TABLE_WIDTH_PX + 2 * TABLE_X_MARGIN)

    h, w = gray.shape
    xL = max(0, min(w - 2, xL))
    xR = max(xL + 1, min(w - 1, xR))

    # Deskew using rulemask-in-ROI
    gray_ds, angle, deskew_dbg = deskew_using_table_roi(gray, xL, xR, FIRST_ROW_Y_PRIOR)

    # Build crop around expected table band
    y0 = max(0, FIRST_ROW_Y_PRIOR - ROI_TOP_PAD)
    y1 = min(gray_ds.shape[0], FIRST_ROW_Y_PRIOR + TABLE_HEIGHT_PX + ROI_BOTTOM_PAD)
    crop = gray_ds[y0:y1, xL:xR]

    # rule_response_crop and rulemask_crop
    enh_crop, rr_crop = enhance_faint_rules(crop)
    rm_crop = rulemask_from_rule_response(rr_crop)

    # Row strength on rulemask (more stable than failing binarization)
    strength = np.sum(rm_crop > 0, axis=1).astype(np.float32)
    strength_s = smooth_1d(strength, PEAK_SMOOTH_K)

    peaks = find_peaks_from_strength(strength_s, min_frac=PEAK_MIN_FRAC, merge_dist=PEAK_MERGE_DIST)
    peaks_full_y = [int(p + y0) for p in peaks]

    table_top, top_dbg = pick_table_top_from_peaks(peaks_full_y, FIRST_ROW_Y_PRIOR)

    # Now build FULL-height rulemask in table-x region for bottom + fitting
    table_x_crop = gray_ds[:, xL:xR]
    _, rr_full_x = enhance_faint_rules(table_x_crop)
    rm_full_x = rulemask_from_rule_response(rr_full_x)

    # Bottom detection from rulemask density
    table_bottom, bottom_dbg, dens_s = detect_table_bottom_from_rulemask(rm_full_x, table_top)

    # Convert rm_full_x (x-cropped) into FULL-image coords for fitting
    rm_full = np.zeros_like(gray_ds, dtype=np.uint8)
    rm_full[:, xL:xR] = rm_full_x

    # Build 41 separator y centers using uniform spacing between detected top/bottom
    # (We fit lines using pixels, so slope will adapt even if y centers are approximate.)
    table_span = max(1, int(table_bottom - table_top))
    step = float(table_span) / float(NUM_ROWS)
    y_centers = [int(round(table_top + i * step)) for i in range(NUM_ROWS + 1)]

    # Fit sloped line for each separator using RULEMASK pixels
    lines_mb = []
    fit_debug = []
    for yc in y_centers:
        m, b, ok, npts = fit_sloped_rule_line_from_rulemask(rm_full, xL, xR, yc)
        if not ok:
            m, b = 0.0, float(yc)
        lines_mb.append((float(m), float(b), bool(ok), int(npts)))
        fit_debug.append({"y_center": int(yc), "m": float(m), "b": float(b), "ok": bool(ok), "npts": int(npts)})

    # Rectify rows, then cut columns
    head_rows = []
    head_row_tenure = {}
    rows_found = 0

    head_dir = os.path.join(img_out, "head_rows")
    non_dir = os.path.join(img_out, "non_head_rows")
    ensure_dir(head_dir)
    ensure_dir(non_dir)

    for i in range(NUM_ROWS):
        m1, b1, ok1, _ = lines_mb[i]
        m2, b2, ok2, _ = lines_mb[i + 1]

        # stop if row would extend below table_bottom (using midpoint eval)
        midx = 0.5 * (xL + xR)
        y_mid_top = m1 * midx + b1
        y_mid_bot = m2 * midx + b2
        if y_mid_bot > table_bottom + 140:
            break

        row_strip = rectify_row_strip(gray_ds, m1, b1, m2, b2, xL=xL, xR=xR, out_h=RECT_ROW_H)

        is_head, tenure = detect_head_row_from_tenure_cols_rectified(row_strip, COLUMNS)
        if is_head:
            head_rows.append(i)
            head_row_tenure[i] = tenure

        out = head_dir if is_head else non_dir
        prefix = f"HEAD_{tenure}_" if is_head else ""

        for col_name, (cx1, cx2) in COLUMNS.items():
            cell = row_strip[:, cx1:cx2]
            if cell.size == 0:
                continue
            cv2.imwrite(os.path.join(out, f"{prefix}row{i:02d}_{col_name}.png"), cell)

        rows_found += 1

    # Overlay
    if SAVE_VIZ:
        viz_path = os.path.join(img_out, "grid_overlay.png")
        title = f"{name} | deskew={angle:.2f} | peaks={len(peaks)} | rows={rows_found} | head={len(head_rows)}"
        draw_overlay_sloped(gray_ds, COLUMNS, lines_mb, table_top, table_bottom, head_rows, viz_path, title, xL, xR)

    # Debug outputs
    if SAVE_DEBUG:
        cv2.imwrite(os.path.join(img_out, "debug_rule_response_crop.png"), rr_crop)
        cv2.imwrite(os.path.join(img_out, "debug_rulemask_crop.png"), rm_crop)
        cv2.imwrite(os.path.join(img_out, "debug_rulemask_full_xcrop.png"), rm_full_x)
        save_density_debug(dens_s, os.path.join(img_out, "debug_rulemask_density.png"), mark_top=table_top, mark_bottom=table_bottom)

        # Strength plot as an image
        s = strength_s.copy()
        if s.max() > 0:
            s = s / s.max()
        strength_img = (s * 255).astype(np.uint8).reshape(-1, 1)
        strength_img = cv2.cvtColor(strength_img, cv2.COLOR_GRAY2BGR)
        for p in peaks:
            if 0 <= p < strength_img.shape[0]:
                strength_img[p, 0] = (0, 255, 0)
        cv2.imwrite(os.path.join(img_out, "debug_row_strength_peaks.png"), strength_img)

    # Report
    report = {
        "image_name": name,
        "source_path": img_path,
        "xL_xR": {"xL": int(xL), "xR": int(xR)},
        "deskew": deskew_dbg,
        "deskew_angle_deg": float(angle),
        "table_top": int(table_top),
        "table_top_debug": top_dbg,
        "table_bottom": int(table_bottom),
        "bottom_detection": bottom_dbg,
        "peaks_in_crop": int(len(peaks)),
        "rows_found": int(rows_found),
        "row_separators_fit": fit_debug,
        "head_rows": [{"row_idx": int(i), "tenure": head_row_tenure.get(i, "NONE")} for i in head_rows],
        "head_rows_count": int(len(head_rows)),
    }
    with open(os.path.join(img_out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"deskew={angle:.3f} | top={table_top} | bottom={table_bottom} | rows={rows_found} | head={len(head_rows)}")
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

    print("=== SMART ADAPTIVE EXTRACTION v12 (ROW SEPARATORS FROM RULE_RESPONSE_CROP) ===")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images found: {len(imgs)}")

    for i, p in enumerate(imgs, start=1):
        print(f"\n[{i}/{len(imgs)}]")
        process_one_image(p)

    print("\n🎯 DONE")

if __name__ == "__main__":
    main()
