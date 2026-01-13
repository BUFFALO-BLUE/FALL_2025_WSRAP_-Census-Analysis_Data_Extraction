import os
import cv2
import json
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "data/from_jeremy/images_aligned_to_first"
OUTPUT_DIR = "data/processed/smart_adaptive_extraction_batch_v10_enhanced_rules"

NUM_ROWS = 40
FIRST_ROW_Y_PRIOR = 1263
EXPECTED_ROW_HEIGHT = 78

# Approx table height (top row line -> last row line)
TABLE_HEIGHT_PX = 3160

# X span prior (you mentioned ~6150px). This is used to restrict line fitting/drawing
TABLE_WIDTH_PX = 6150
TABLE_X_MARGIN = 80  # a bit wider than before to be safe

# ROI for deskew and processing around the table
ROI_TOP_PAD = 260
ROI_BOTTOM_PAD = 320

# Bottom search
BOTTOM_SEARCH_PAD = 600

# Dual-signal bottom detection thresholds
VERT_KERNEL_H_DIV = 18
VERT_SMOOTH_K = 41
VERT_MIN_DENSITY_FRAC = 0.18

HORIZ_SMOOTH_K = 41
HORIZ_MIN_DENSITY_FRAC = 0.22  # tune 0.15–0.30 if needed

# First-line pick ROI around expected FIRST_ROW_Y_PRIOR
FIRST_LINE_ROI_UP = 340
FIRST_LINE_ROI_DOWN = 740

# Enhancement knobs (faint rules)
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
BLACKHAT_KSIZE = 35
BLACKHAT_MIX = 0.85

# Horizontal mask morphology (depends on width)
HORIZ_KERNEL_DIV = 16  # kernel width ~ w/div (smaller div => larger kernel)
HORIZ_OPEN_ITERS = 2
HORIZ_DILATE_W = 35

# Hough parameters for horizontal rules
HOUGH_THRESHOLD = 90
HOUGH_MIN_LINE_LEN = 200
HOUGH_MAX_GAP = 60

# Merge double-lines for the same printed rule
RULE_Y_TOL_PX = 12

# Slanted separator detection controls
MAX_LINE_ANGLE_DEG = 10.0
CLUSTER_Y_TOL = 14
MIN_SEGMENTS_PER_CLUSTER = 2

# Head detection knobs (rented/owned ink)
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

# Save outputs
SAVE_VIZ = True
SAVE_CELLS = True
SAVE_DEBUG = True  # debug_rule_response, debug_horizontal_mask, densities

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

def robust_binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )

# ============================================================
# Enhancement: CLAHE + Blackhat
# ============================================================

def enhance_faint_rules(gray: np.ndarray,
                        clahe_clip=CLAHE_CLIP,
                        clahe_grid=CLAHE_GRID,
                        blackhat_ksize=BLACKHAT_KSIZE,
                        mix=BLACKHAT_MIX):
    """
    Boost faint dark rules on light background.
    Returns:
      enhanced_gray: boosted grayscale
      rule_response: rule-likelihood response (bright where rules exist)
    """
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

def horizontal_line_mask_from_enhanced(enhanced_gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)
    bin_img = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 9
    )
    inv = 255 - bin_img
    h, w = inv.shape

    hk = max(60, w // HORIZ_KERNEL_DIV)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=HORIZ_OPEN_ITERS)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (HORIZ_DILATE_W, 1)), iterations=1)
    return horiz

def detect_horizontal_segments_hough(horiz_mask: np.ndarray):
    edges = cv2.Canny(horiz_mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(HOUGH_THRESHOLD),
        minLineLength=int(HOUGH_MIN_LINE_LEN),
        maxLineGap=int(HOUGH_MAX_GAP)
    )
    if lines is None:
        return []
    return [tuple(map(int, l[0])) for l in lines]

def cluster_segments_into_rule_ys(segments, y_tol=RULE_Y_TOL_PX):
    """
    Merge double-lines (edges of the same printed rule).
    Returns sorted list of representative y-values (one per rule).
    """
    if not segments:
        return []
    items = []
    for x1, y1, x2, y2 in segments:
        ymid = 0.5 * (y1 + y2)
        length = abs(x2 - x1) + abs(y2 - y1)
        items.append((ymid, length))
    items.sort(key=lambda t: t[0])

    clusters = []
    cur = [items[0]]
    for it in items[1:]:
        if abs(it[0] - cur[-1][0]) <= y_tol:
            cur.append(it)
        else:
            clusters.append(cur)
            cur = [it]
    clusters.append(cur)

    rule_ys = []
    for cl in clusters:
        # take weighted avg by length
        ys = np.array([c[0] for c in cl], dtype=np.float32)
        ws = np.array([c[1] for c in cl], dtype=np.float32)
        y = float(np.sum(ys * ws) / max(1.0, float(np.sum(ws))))
        rule_ys.append(int(round(y)))

    rule_ys = sorted(list(set(rule_ys)))
    return rule_ys

# ============================================================
# ROI Deskew via minAreaRect on line pixels
# ============================================================

def extract_table_roi(gray: np.ndarray, first_row_y_prior: int):
    h = gray.shape[0]
    y0 = max(0, first_row_y_prior - ROI_TOP_PAD)
    y1 = min(h, first_row_y_prior + TABLE_HEIGHT_PX + ROI_BOTTOM_PAD)
    return gray[y0:y1, :], y0, y1

def estimate_skew_angle_minarearect(gray_roi: np.ndarray) -> float:
    # Enhance faint rules inside ROI for better angle estimation
    enh, _ = enhance_faint_rules(gray_roi)
    hmask = horizontal_line_mask_from_enhanced(enh)

    edges = cv2.Canny(hmask, 50, 150, apertureSize=3)
    ys, xs = np.where(edges > 0)
    if len(xs) < 2500:
        return 0.0

    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = float(rect[-1])
    rw, rh = rect[1]
    if rw < rh:
        angle += 90.0

    if angle > 20 or angle < -20:
        return 0.0
    return angle

def rotate_image(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.05:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def deskew_using_roi(gray: np.ndarray, first_row_y_prior: int):
    roi, y0, y1 = extract_table_roi(gray, first_row_y_prior)
    angle = estimate_skew_angle_minarearect(roi)
    gray_ds = rotate_image(gray, -angle)
    dbg = {"method": "roi_minAreaRect", "roi_y0": int(y0), "roi_y1": int(y1), "angle_deg": float(angle)}
    return gray_ds, float(angle), dbg

# ============================================================
# Table top: earliest rule near expected
# ============================================================

def pick_table_top_from_rule_ys(rule_ys, first_y_prior: int):
    if not rule_ys:
        return int(first_y_prior), {"picked_from": "prior_fallback_no_rules"}

    roi_lo = first_y_prior - FIRST_LINE_ROI_UP
    roi_hi = first_y_prior + FIRST_LINE_ROI_DOWN
    candidates = [y for y in rule_ys if roi_lo <= y <= roi_hi]
    if candidates:
        return int(min(candidates)), {"picked_from": "roi_earliest_rule", "roi_lo": int(roi_lo), "roi_hi": int(roi_hi)}
    # fallback: nearest rule to prior
    nearest = min(rule_ys, key=lambda y: abs(y - first_y_prior))
    return int(nearest), {"picked_from": "nearest_rule_fallback"}

# ============================================================
# Table bottom: dual-signal density (vertical + horizontal "table-ness")
# ============================================================

def vertical_lines_mask(gray: np.ndarray) -> np.ndarray:
    # use standard binarize (good enough), table bottom detection doesn’t need ultra faint detail
    bin_img = robust_binarize(gray)
    inv = 255 - bin_img
    h, w = inv.shape
    vk = max(55, h // VERT_KERNEL_H_DIV)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)), iterations=1)
    return vert

def detect_table_bottom_dual_signal(gray: np.ndarray, table_top_y: int):
    h, w = gray.shape

    vmask = vertical_lines_mask(gray)
    vdens = np.sum(vmask > 0, axis=1).astype(np.float32)
    vdens_s = smooth_1d(vdens, VERT_SMOOTH_K)

    # For horizontal density, use enhanced mask (more stable for table region)
    enh, _ = enhance_faint_rules(gray)
    hmask = horizontal_line_mask_from_enhanced(enh)
    hdens = np.sum(hmask > 0, axis=1).astype(np.float32)
    hdens_s = smooth_1d(hdens, HORIZ_SMOOTH_K)

    expected_bottom = int(table_top_y + TABLE_HEIGHT_PX)
    search_start = int(max(0, expected_bottom - 350))
    search_end = int(min(h - 1, expected_bottom + BOTTOM_SEARCH_PAD))

    mid0 = int(max(0, table_top_y + 400))
    mid1 = int(min(h - 1, table_top_y + 2200))

    v_typ = float(np.median(vdens_s[mid0:mid1])) if mid1 > mid0 else float(np.median(vdens_s))
    h_typ = float(np.median(hdens_s[mid0:mid1])) if mid1 > mid0 else float(np.median(hdens_s))

    v_thr = v_typ * float(VERT_MIN_DENSITY_FRAC)
    h_thr = h_typ * float(HORIZ_MIN_DENSITY_FRAC)

    window = 95
    bottom_y = expected_bottom

    for y in range(search_start, max(search_start, search_end - window)):
        vseg = vdens_s[y:y + window]
        hseg = hdens_s[y:y + window]
        if vseg.size == window and hseg.size == window:
            if float(np.max(vseg)) < v_thr and float(np.max(hseg)) < h_thr:
                bottom_y = y
                break

    debug = {
        "method": "dual_signal_vertical+horizontal_density",
        "expected_bottom": int(expected_bottom),
        "search_start": int(search_start),
        "search_end": int(search_end),
        "v_typ": float(v_typ),
        "h_typ": float(h_typ),
        "v_thr": float(v_thr),
        "h_thr": float(h_thr),
        "bottom_y": int(bottom_y),
        "window": int(window),
    }
    return int(bottom_y), debug, vdens_s, hdens_s

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

# ============================================================
# Slanted separators: fit y = m x + b per rule using Hough segments
# ============================================================

def fit_line_y_mx_b(points_xy: np.ndarray):
    xs = points_xy[:, 0].astype(np.float32)
    ys = points_xy[:, 1].astype(np.float32)
    if xs.size < 2:
        return 0.0, float(np.median(ys)) if ys.size else 0.0
    A = np.vstack([xs, np.ones_like(xs)]).T
    m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(m), float(b)

def segments_to_slanted_lines(segments, xL, xR):
    """
    Convert many horizontal-ish segments into a list of fitted slanted lines (m,b),
    clustered by y-at-mid, with near-duplicate merging handled by clustering.
    """
    if not segments:
        return [], {"status": "no_segments"}

    xmid = 0.5 * (xL + xR)
    max_tan = np.tan(np.deg2rad(MAX_LINE_ANGLE_DEG))

    items = []
    for x1, y1, x2, y2 in segments:
        dx = (x2 - x1)
        dy = (y2 - y1)
        if dx == 0:
            continue
        slope = dy / dx
        if abs(slope) > max_tan:
            continue
        m = slope
        b = y1 - m * x1
        y_at_mid = m * xmid + b
        length = abs(dx) + abs(dy)
        items.append((float(y_at_mid), float(length), (x1, y1, x2, y2)))

    if not items:
        return [], {"status": "no_near_horizontal_segments"}

    items.sort(key=lambda t: t[0])

    # cluster by y_at_mid
    clusters = []
    cur = [items[0]]
    for it in items[1:]:
        if abs(it[0] - cur[-1][0]) <= CLUSTER_Y_TOL:
            cur.append(it)
        else:
            clusters.append(cur)
            cur = [it]
    clusters.append(cur)

    fitted = []
    for cl in clusters:
        if len(cl) < MIN_SEGMENTS_PER_CLUSTER:
            continue
        pts = []
        for _, _, (x1, y1, x2, y2) in cl:
            pts.append((x1, y1))
            pts.append((x2, y2))
        pts = np.array(pts, dtype=np.float32)
        m, b = fit_line_y_mx_b(pts)
        yrep = m * xmid + b
        fitted.append((float(yrep), float(m), float(b), int(len(cl))))

    fitted.sort(key=lambda t: t[0])
    lines_mb = [(m, b) for (_, m, b, _) in fitted]
    dbg = {"status": "ok", "segments_in": int(len(segments)), "clusters_total": int(len(clusters)), "clusters_used": int(len(lines_mb))}
    return lines_mb, dbg

def merge_close_slanted_lines(lines_mb, xmid, merge_px=10):
    """
    Merge near-duplicate fitted lines (double-edge of same printed rule).
    """
    if not lines_mb:
        return []

    items = []
    for (m, b) in lines_mb:
        y = m * xmid + b
        items.append((float(y), float(m), float(b)))
    items.sort(key=lambda t: t[0])

    merged = []
    group = [items[0]]

    for it in items[1:]:
        if abs(it[0] - group[-1][0]) <= merge_px:
            group.append(it)
        else:
            ms = [g[1] for g in group]
            bs = [g[2] for g in group]
            merged.append((float(np.mean(ms)), float(np.mean(bs))))
            group = [it]

    ms = [g[1] for g in group]
    bs = [g[2] for g in group]
    merged.append((float(np.mean(ms)), float(np.mean(bs))))
    return merged

def select_41_separators(lines_mb, table_top, table_bottom, xL, xR):
    if not lines_mb:
        return []
    xmid = 0.5 * (xL + xR)

    ys = []
    for (m, b) in lines_mb:
        y = m * xmid + b
        if (table_top - 80) <= y <= (table_bottom + 30):
            ys.append((float(y), float(m), float(b)))
    ys.sort(key=lambda t: t[0])

    if len(ys) < NUM_ROWS + 1:
        return [(m, b) for (_, m, b) in ys]

    if len(ys) > NUM_ROWS + 1:
        idxs = np.linspace(0, len(ys) - 1, NUM_ROWS + 1).round().astype(int)
        chosen = [ys[i] for i in idxs]
        chosen.sort(key=lambda t: t[0])
        return [(m, b) for (_, m, b) in chosen]

    return [(m, b) for (_, m, b) in ys]

# ============================================================
# Fallback flat boundaries (clamped to bottom)
# ============================================================

def fallback_flat_boundaries(table_top: int, table_bottom: int):
    boundaries = [int(table_top)]
    cur = int(table_top)
    for _ in range(NUM_ROWS):
        nxt = cur + EXPECTED_ROW_HEIGHT
        if nxt >= table_bottom:
            nxt = table_bottom
        boundaries.append(int(nxt))
        cur = int(nxt)
        if cur >= table_bottom:
            break

    while len(boundaries) < NUM_ROWS + 1:
        boundaries.append(boundaries[-1] + 1)

    return boundaries[:NUM_ROWS + 1]

# ============================================================
# Row band warp (slanted)
# ============================================================

def warp_row_band(gray: np.ndarray, line_top, line_bot, xL: int, xR: int):
    h, w = gray.shape
    xL = int(np.clip(xL, 0, w - 1))
    xR = int(np.clip(xR, 0, w - 1))
    if xR <= xL + 10:
        return None

    m1, b1 = line_top
    m2, b2 = line_bot

    def y_on(m, b, x):
        return float(np.clip(m * x + b, 0, h - 1))

    y1L = y_on(m1, b1, xL)
    y1R = y_on(m1, b1, xR)
    y2R = y_on(m2, b2, xR)
    y2L = y_on(m2, b2, xL)

    src = np.array([[xL, y1L], [xR, y1R], [xR, y2R], [xL, y2L]], dtype=np.float32)

    out_w = int(xR - xL)
    out_h = int(max(20, 0.5 * ((y2L - y1L) + (y2R - y1R))))
    out_h = int(np.clip(out_h, 20, 220))

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(gray, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped

# ============================================================
# Head detection (rented/owned ink)
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

def detect_head_row_from_tenure_cols(row_img_gray: np.ndarray, rented_x1: int, rented_x2: int, owned_x1: int, owned_x2: int):
    rented_cell = row_img_gray[:, rented_x1:rented_x2]
    owned_cell = row_img_gray[:, owned_x1:owned_x2]

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

def draw_overlay(gray: np.ndarray, columns: dict,
                 mode: str, xL: int, xR: int,
                 table_top: int, table_bottom: int,
                 slanted_lines=None, flat_boundaries=None,
                 head_rows=None, head_row_tenure=None,
                 out_path: str = "", title: str = ""):
    slanted_lines = slanted_lines or []
    flat_boundaries = flat_boundaries or []
    head_rows = head_rows or []
    head_row_tenure = head_row_tenure or {}

    viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape

    if title:
        cv2.putText(viz, title, (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 0), 2)

    for col_name, (a, b) in columns.items():
        cv2.line(viz, (a, 0), (a, h), (255, 0, 0), 2)
        cv2.line(viz, (b, 0), (b, h), (255, 0, 0), 2)
        cv2.putText(viz, col_name, (a, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

    yt = int(np.clip(table_top, 0, h - 1))
    yb = int(np.clip(table_bottom, 0, h - 1))
    cv2.line(viz, (0, yt), (w, yt), (255, 255, 0), 2)
    cv2.putText(viz, "TABLE_TOP", (40, max(0, yt - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.line(viz, (0, yb), (w, yb), (0, 255, 255), 3)
    cv2.putText(viz, "TABLE_BOTTOM", (40, max(0, yb - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    xL = int(np.clip(xL, 0, w - 1))
    xR = int(np.clip(xR, 0, w - 1))

    if mode == "slanted":
        for i, (m, b) in enumerate(slanted_lines):
            yL = int(np.clip(m * xL + b, 0, h - 1))
            yR = int(np.clip(m * xR + b, 0, h - 1))
            is_head = i in head_rows
            color = (0, 255, 0) if is_head else (0, 0, 255)
            thick = 3 if is_head else 2
            cv2.line(viz, (xL, yL), (xR, yR), color, thick)
            if is_head:
                tenure = head_row_tenure.get(i, "HEAD")
                cv2.putText(viz, f"HEAD {i} [{tenure}]", (40, min(h - 10, max(10, yL + 18))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        for i, y in enumerate(flat_boundaries):
            yy = int(np.clip(y, 0, h - 1))
            is_head = i in head_rows
            color = (0, 255, 0) if is_head else (0, 0, 255)
            thick = 3 if is_head else 2
            cv2.line(viz, (0, yy), (w, yy), color, thick)
            if is_head:
                tenure = head_row_tenure.get(i, "HEAD")
                cv2.putText(viz, f"HEAD {i} [{tenure}]", (40, min(h - 10, max(10, yy + 18))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, viz)

# ============================================================
# Extraction
# ============================================================

def extract_cells_slanted(gray: np.ndarray, slanted_lines: list, columns: dict, xL: int, xR: int, out_dir: str):
    head_dir = os.path.join(out_dir, "head_rows")
    non_dir = os.path.join(out_dir, "non_head_rows")
    ensure_dir(head_dir); ensure_dir(non_dir)

    # columns relative to warp ROI
    col_warp = {k: (max(0, a - xL), max(0, b - xL)) for k, (a, b) in columns.items()}
    rented_x1, rented_x2 = col_warp["rented"]
    owned_x1, owned_x2 = col_warp["owned"]

    head_rows = []
    head_row_tenure = {}

    rows_found = min(NUM_ROWS, len(slanted_lines) - 1)
    for row_idx in range(rows_found):
        row_warp = warp_row_band(gray, slanted_lines[row_idx], slanted_lines[row_idx + 1], xL, xR)
        if row_warp is None or row_warp.size == 0:
            continue

        is_head, tenure = detect_head_row_from_tenure_cols(row_warp, rented_x1, rented_x2, owned_x1, owned_x2)
        if is_head:
            head_rows.append(row_idx)
            head_row_tenure[row_idx] = tenure

        out = head_dir if is_head else non_dir
        prefix = f"HEAD_{tenure}_" if is_head else ""

        for col_name, (a, b) in col_warp.items():
            a = int(np.clip(a, 0, row_warp.shape[1] - 1))
            b = int(np.clip(b, 0, row_warp.shape[1]))
            if b <= a:
                continue
            cell = row_warp[:, a:b]
            if cell.size == 0:
                continue
            cv2.imwrite(os.path.join(out, f"{prefix}row{row_idx:02d}_{col_name}.png"), cell)

    return head_rows, head_row_tenure, rows_found

def extract_cells_flat(gray: np.ndarray, boundaries: list, columns: dict, out_dir: str):
    head_dir = os.path.join(out_dir, "head_rows")
    non_dir = os.path.join(out_dir, "non_head_rows")
    ensure_dir(head_dir); ensure_dir(non_dir)

    rented_x1, rented_x2 = columns["rented"]
    owned_x1, owned_x2 = columns["owned"]

    head_rows = []
    head_row_tenure = {}

    rows_found = min(NUM_ROWS, len(boundaries) - 1)
    for row_idx in range(rows_found):
        y1, y2 = int(boundaries[row_idx]), int(boundaries[row_idx + 1])
        y1 = max(0, min(gray.shape[0]-1, y1))
        y2 = max(0, min(gray.shape[0], y2))
        if y2 <= y1:
            continue

        row_img = gray[y1:y2, :]
        is_head, tenure = detect_head_row_from_tenure_cols(row_img, rented_x1, rented_x2, owned_x1, owned_x2)
        if is_head:
            head_rows.append(row_idx)
            head_row_tenure[row_idx] = tenure

        out = head_dir if is_head else non_dir
        prefix = f"HEAD_{tenure}_" if is_head else ""

        for col_name, (x1, x2) in columns.items():
            cell = gray[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            cv2.imwrite(os.path.join(out, f"{prefix}row{row_idx:02d}_{col_name}.png"), cell)

    return head_rows, head_row_tenure, rows_found

# ============================================================
# Report
# ============================================================

def save_report_json(out_path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

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

    # 1) Deskew with ROI
    gray_ds, angle, deskew_dbg = deskew_using_roi(gray, FIRST_ROW_Y_PRIOR)

    # 2) Enhance + build horizontal mask + detect segments (for rule ys + slanted fit)
    enh, rule_resp = enhance_faint_rules(gray_ds)
    hmask = horizontal_line_mask_from_enhanced(enh)
    segs = detect_horizontal_segments_hough(hmask)
    rule_ys = cluster_segments_into_rule_ys(segs)

    # 3) Table top from earliest rule near expected
    table_top, top_dbg = pick_table_top_from_rule_ys(rule_ys, FIRST_ROW_Y_PRIOR)

    # 4) Table bottom dual signal
    table_bottom, bottom_dbg, vdens_s, hdens_s = detect_table_bottom_dual_signal(gray_ds, table_top)

    # 5) Define table x bounds
    min_x1 = min(v[0] for v in COLUMNS.values())
    xL = int(min_x1 - TABLE_X_MARGIN)
    xR = int(xL + TABLE_WIDTH_PX + 2 * TABLE_X_MARGIN)

    h, w = gray_ds.shape
    xL = max(0, min(w - 2, xL))
    xR = max(xL + 1, min(w - 1, xR))

    # 6) Build slanted separators from segments + merge close
    xmid = 0.5 * (xL + xR)
    lines_mb_all, slant_dbg = segments_to_slanted_lines(segs, xL, xR)
    lines_mb_all = merge_close_slanted_lines(lines_mb_all, xmid=xmid, merge_px=10)
    lines_mb = select_41_separators(lines_mb_all, table_top, table_bottom, xL, xR)

    slanted_mode = (len(lines_mb) >= NUM_ROWS + 1)

    # 7) Extract (always)
    mode_used = "slanted" if slanted_mode else "flat_fallback"
    head_rows, head_row_tenure, rows_found = [], {}, 0

    if SAVE_CELLS:
        if slanted_mode:
            head_rows, head_row_tenure, rows_found = extract_cells_slanted(gray_ds, lines_mb, COLUMNS, xL, xR, img_out)
        else:
            boundaries = fallback_flat_boundaries(table_top, table_bottom)
            head_rows, head_row_tenure, rows_found = extract_cells_flat(gray_ds, boundaries, COLUMNS, img_out)

    # 8) Overlay (always)
    if SAVE_VIZ:
        viz_path = os.path.join(img_out, "grid_overlay.png")
        title = f"{name} | mode={mode_used} | deskew={angle:.2f} | rules={len(rule_ys)} | rows={rows_found} | head={len(head_rows)}"
        if slanted_mode:
            draw_overlay(gray_ds, COLUMNS, "slanted", xL, xR, table_top, table_bottom,
                         slanted_lines=lines_mb, head_rows=head_rows, head_row_tenure=head_row_tenure,
                         out_path=viz_path, title=title)
        else:
            boundaries = fallback_flat_boundaries(table_top, table_bottom)
            draw_overlay(gray_ds, COLUMNS, "flat", xL, xR, table_top, table_bottom,
                         flat_boundaries=boundaries, head_rows=head_rows, head_row_tenure=head_row_tenure,
                         out_path=viz_path, title=title)

    # 9) Debug images
    if SAVE_DEBUG:
        cv2.imwrite(os.path.join(img_out, "debug_rule_response.png"), rule_resp)
        cv2.imwrite(os.path.join(img_out, "debug_horizontal_mask.png"), hmask)
        save_density_debug(vdens_s, os.path.join(img_out, "debug_vertical_density.png"), mark_top=table_top, mark_bottom=table_bottom)
        save_density_debug(hdens_s, os.path.join(img_out, "debug_horizontal_density.png"), mark_top=table_top, mark_bottom=table_bottom)

    # 10) Report always
    report = {
        "image_name": name,
        "source_path": img_path,
        "deskew": deskew_dbg,
        "table_top": int(table_top),
        "table_top_debug": top_dbg,
        "table_bottom": int(table_bottom),
        "bottom_detection": bottom_dbg,
        "table_xL": int(xL),
        "table_xR": int(xR),
        "mode_used": mode_used,
        "enhancement": {
            "clahe_clip": float(CLAHE_CLIP),
            "clahe_grid": list(CLAHE_GRID),
            "blackhat_ksize": int(BLACKHAT_KSIZE),
            "blackhat_mix": float(BLACKHAT_MIX),
        },
        "hough": {
            "threshold": int(HOUGH_THRESHOLD),
            "min_line_len": int(HOUGH_MIN_LINE_LEN),
            "max_gap": int(HOUGH_MAX_GAP),
        },
        "segments_detected": int(len(segs)),
        "rules_detected_y_count": int(len(rule_ys)),
        "slanted_debug": slant_dbg,
        "slanted_raw_count": int(len(lines_mb_all)),
        "slanted_selected_count": int(len(lines_mb)),
        "rows_found": int(rows_found),
        "head_rows": [{"row_idx": int(i), "tenure": head_row_tenure.get(i, "NONE")} for i in head_rows],
        "head_rows_count": int(len(head_rows)),
    }
    save_report_json(os.path.join(img_out, "report.json"), report)

    print(f"deskew={angle:.3f}deg | top={table_top} | bottom={table_bottom} | rules={len(rule_ys)} | slanted_sel={len(lines_mb)} | mode={mode_used} | rows={rows_found} | head={len(head_rows)}")
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

    print("=== SMART ADAPTIVE EXTRACTION v10 (ENHANCED FAINT RULES + INDIVIDUAL LINE DETECTION) ===")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images found: {len(imgs)}")

    for i, p in enumerate(imgs, start=1):
        print(f"\n[{i}/{len(imgs)}]")
        process_one_image(p)

    print("\n🎯 DONE")

if __name__ == "__main__":
    main()
