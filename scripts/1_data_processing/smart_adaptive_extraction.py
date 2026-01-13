import os
import cv2
import json
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "data/from_jeremy/images_aligned_to_first"
OUTPUT_DIR = "data/processed/smart_adaptive_extraction_batch_v9_slanted_dualbottom"

NUM_ROWS = 40
FIRST_ROW_Y_PRIOR = 1263
EXPECTED_ROW_HEIGHT = 78

# Prior table height (used for ROI + fallback)
TABLE_HEIGHT_PX = 3160
BOTTOM_SEARCH_PAD = 500

# ROI for deskew and line detection: focus on table region only
ROI_TOP_PAD = 240
ROI_BOTTOM_PAD = 300

# Table-bottom detection via dual-signal (vertical + horizontal "table-ness")
VERT_KERNEL_H_DIV = 18
VERT_SMOOTH_K = 41
VERT_MIN_DENSITY_FRAC = 0.18

HORIZ_SMOOTH_K = 41
HORIZ_MIN_DENSITY_FRAC = 0.22  # tune 0.15–0.30 if needed

# Table X span (you said ~6150px)
TABLE_WIDTH_PX = 6150
TABLE_X_MARGIN = 50

# Column coordinates (fixed for now; later we’ll do column alignment)
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
SAVE_VERTICAL_DENSITY_DEBUG = False
SAVE_HORIZONTAL_DENSITY_DEBUG = False

# Head/ink detection knobs
INK_PAD = 12
MIN_INK_RATIO = 0.010
MIN_CC_AREA = 60

# First-line search ROI (around expected first table row line)
FIRST_LINE_ROI_UP = 320
FIRST_LINE_ROI_DOWN = 720

# Top-line detection (earliest peak, not strongest)
TOP_PEAK_THR = 0.55
TOP_CROSS_THR = 0.40

# Slanted line detection (Hough -> cluster -> fit y = m x + b)
MAX_LINE_ANGLE_DEG = 10.0
HOUGH_THRESHOLD = 120
HOUGH_MIN_LINE_LEN = 260
HOUGH_MAX_GAP = 45
CLUSTER_Y_TOL = 14
MIN_SEGMENTS_PER_CLUSTER = 2


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
# ROI deskew (minAreaRect on table lines)
# ============================================================

def extract_table_roi(gray: np.ndarray, first_row_y_prior: int) -> (np.ndarray, int, int):
    h = gray.shape[0]
    y0 = max(0, first_row_y_prior - ROI_TOP_PAD)
    y1 = min(h, first_row_y_prior + TABLE_HEIGHT_PX + ROI_BOTTOM_PAD)
    return gray[y0:y1, :], y0, y1

def estimate_skew_angle_minarearect(gray_roi: np.ndarray) -> float:
    bin_img = robust_binarize(gray_roi)
    inv = 255 - bin_img

    h, w = inv.shape
    hk = max(45, w // 22)
    vk = max(45, h // VERT_KERNEL_H_DIV)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    mask = cv2.bitwise_or(horiz, vert)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    ys, xs = np.where(mask > 0)
    if len(xs) < 5000:
        return 0.0

    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = float(rect[-1])
    (rw, rh) = rect[1]
    if rw < rh:
        angle = angle + 90.0

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
# Line masks (horizontal + vertical)
# ============================================================

def horizontal_lines_mask(gray: np.ndarray) -> np.ndarray:
    bin_img = robust_binarize(gray)
    inv = 255 - bin_img
    h, w = inv.shape

    hk = max(55, w // 18)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=2)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1)), iterations=1)
    return horiz

def vertical_lines_mask(gray: np.ndarray) -> np.ndarray:
    bin_img = robust_binarize(gray)
    inv = 255 - bin_img

    h, w = inv.shape
    vk = max(55, h // VERT_KERNEL_H_DIV)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)), iterations=1)
    return vert

def horizontal_line_strength(gray: np.ndarray) -> np.ndarray:
    mask = horizontal_lines_mask(gray)
    strength = np.sum(mask > 0, axis=1).astype(np.float32)
    return smooth_1d(strength, 9)


# ============================================================
# TOP line detection (earliest strong peak in ROI)
# ============================================================

def pick_first_line_earliest_peak(gray: np.ndarray, first_y_prior: int) -> (int, dict):
    strength = horizontal_line_strength(gray)
    h = len(strength)

    roi_lo = max(0, first_y_prior - FIRST_LINE_ROI_UP)
    roi_hi = min(h - 1, first_y_prior + FIRST_LINE_ROI_DOWN)

    roi = strength[roi_lo:roi_hi + 1]
    if roi.size == 0 or float(roi.max()) <= 0:
        return int(first_y_prior), {"roi_lo": roi_lo, "roi_hi": roi_hi, "picked_from": "prior_fallback"}

    r = roi.astype(np.float32)
    r = r / float(r.max())

    # earliest local maxima above TOP_PEAK_THR
    candidates = []
    for i in range(1, len(r) - 1):
        if r[i] > r[i-1] and r[i] > r[i+1] and r[i] >= TOP_PEAK_THR:
            candidates.append(i)

    if candidates:
        first = int(roi_lo + candidates[0])
        return first, {"roi_lo": int(roi_lo), "roi_hi": int(roi_hi), "picked_from": "roi_first_peak", "thr": float(TOP_PEAK_THR)}

    # fallback: earliest crossing
    idx = np.where(r >= TOP_CROSS_THR)[0]
    if idx.size > 0:
        first = int(roi_lo + int(idx[0]))
        return first, {"roi_lo": int(roi_lo), "roi_hi": int(roi_hi), "picked_from": "roi_first_crossing", "thr": float(TOP_CROSS_THR)}

    return int(first_y_prior), {"roi_lo": int(roi_lo), "roi_hi": int(roi_hi), "picked_from": "prior_fallback"}


# ============================================================
# BOTTOM detection (dual-signal: vertical + horizontal density)
# ============================================================

def detect_table_bottom_dual_signal(gray: np.ndarray, table_top_y: int):
    h, w = gray.shape

    vmask = vertical_lines_mask(gray)
    vdens = np.sum(vmask > 0, axis=1).astype(np.float32)
    vdens_s = smooth_1d(vdens, VERT_SMOOTH_K)

    hmask = horizontal_lines_mask(gray)
    hdens = np.sum(hmask > 0, axis=1).astype(np.float32)
    hdens_s = smooth_1d(hdens, HORIZ_SMOOTH_K)

    expected_bottom = int(table_top_y + TABLE_HEIGHT_PX)
    search_start = int(max(0, expected_bottom - 300))
    search_end = int(min(h - 1, expected_bottom + BOTTOM_SEARCH_PAD))

    mid0 = int(max(0, table_top_y + 400))
    mid1 = int(min(h - 1, table_top_y + 2200))

    v_typ = float(np.median(vdens_s[mid0:mid1])) if mid1 > mid0 else float(np.median(vdens_s))
    h_typ = float(np.median(hdens_s[mid0:mid1])) if mid1 > mid0 else float(np.median(hdens_s))

    v_thr = v_typ * float(VERT_MIN_DENSITY_FRAC)
    h_thr = h_typ * float(HORIZ_MIN_DENSITY_FRAC)

    window = 90
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
# Slanted separator detection (Hough -> cluster -> fit y = m x + b)
# ============================================================

def fit_line_y_mx_b(points_xy: np.ndarray):
    xs = points_xy[:, 0].astype(np.float32)
    ys = points_xy[:, 1].astype(np.float32)
    if xs.size < 2:
        return 0.0, float(np.median(ys)) if ys.size else 0.0
    A = np.vstack([xs, np.ones_like(xs)]).T
    m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(m), float(b)

def detect_slanted_separators(gray: np.ndarray, table_top: int, table_bottom: int, xL: int, xR: int):
    h, w = gray.shape
    xL = int(max(0, min(w - 1, xL)))
    xR = int(max(0, min(w - 1, xR)))
    if xR <= xL + 10:
        return [], {"status": "bad_x_bounds"}

    y0 = int(max(0, table_top - 60))
    y1 = int(min(h - 1, table_bottom + 15))

    roi = gray[y0:y1, xL:xR]
    if roi.size == 0:
        return [], {"status": "empty_roi"}

    mask = horizontal_lines_mask(roi)
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_GAP
    )

    if lines is None:
        return [], {"status": "no_hough_lines", "y0": y0, "y1": y1, "xL": xL, "xR": xR}

    segs = []
    max_tan = np.tan(np.deg2rad(MAX_LINE_ANGLE_DEG))
    for (x1, yy1, x2, yy2) in lines[:, 0]:
        dx = (x2 - x1)
        dy = (yy2 - yy1)
        if dx == 0:
            continue
        slope = dy / dx
        if abs(slope) > max_tan:
            continue
        ax1, ay1 = int(x1 + xL), int(yy1 + y0)
        ax2, ay2 = int(x2 + xL), int(yy2 + y0)
        segs.append((ax1, ay1, ax2, ay2))

    if not segs:
        return [], {"status": "no_near_horizontal_segments"}

    xmid = 0.5 * (xL + xR)
    items = []
    for (x1, y1, x2, y2) in segs:
        if x2 == x1:
            continue
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        y_at_mid = m * xmid + b
        items.append((float(y_at_mid), (x1, y1, x2, y2)))

    items.sort(key=lambda t: t[0])

    clusters = []
    cur = []
    cur_y = None

    for ymid, seg in items:
        if cur_y is None:
            cur = [seg]
            cur_y = ymid
        elif abs(ymid - cur_y) <= CLUSTER_Y_TOL:
            cur.append(seg)
            cur_y = (cur_y * 0.7 + ymid * 0.3)
        else:
            clusters.append(cur)
            cur = [seg]
            cur_y = ymid
    if cur:
        clusters.append(cur)

    fitted = []
    for cl in clusters:
        if len(cl) < MIN_SEGMENTS_PER_CLUSTER:
            continue
        pts = []
        for (x1, y1, x2, y2) in cl:
            pts.append((x1, y1))
            pts.append((x2, y2))
        pts = np.array(pts, dtype=np.float32)
        m, b = fit_line_y_mx_b(pts)
        yrep = m * xmid + b
        fitted.append((float(yrep), float(m), float(b), int(len(cl))))

    fitted.sort(key=lambda t: t[0])

    debug = {
        "status": "ok",
        "segments_total": int(len(segs)),
        "clusters_total": int(len(clusters)),
        "clusters_used": int(len(fitted)),
        "y0": int(y0),
        "y1": int(y1),
        "xL": int(xL),
        "xR": int(xR),
    }
    return [(m, b) for (_, m, b, _) in fitted], debug

def select_41_separators(lines_mb, table_top, table_bottom, xL, xR):
    if not lines_mb:
        return []

    xmid = 0.5 * (xL + xR)
    ys = []
    for (m, b) in lines_mb:
        y = m * xmid + b
        if table_top - 60 <= y <= table_bottom + 25:
            ys.append((float(y), float(m), float(b)))
    ys.sort(key=lambda t: t[0])

    if len(ys) <= NUM_ROWS + 1:
        return [(m, b) for (_, m, b) in ys]

    target = NUM_ROWS + 1
    idxs = np.linspace(0, len(ys) - 1, target).round().astype(int)
    chosen = [ys[i] for i in idxs]
    chosen.sort(key=lambda t: t[0])
    return [(m, b) for (_, m, b) in chosen]


# ============================================================
# Warp each row band to rectangle + cut cells
# ============================================================

def y_on_line(m: float, b: float, x: float) -> float:
    return m * x + b

def warp_row_band(gray: np.ndarray, line_top, line_bot, xL: int, xR: int):
    h, w = gray.shape
    xL = int(np.clip(xL, 0, w - 1))
    xR = int(np.clip(xR, 0, w - 1))
    if xR <= xL + 10:
        return None, None

    m1, b1 = line_top
    m2, b2 = line_bot

    y1L = float(np.clip(y_on_line(m1, b1, xL), 0, h - 1))
    y1R = float(np.clip(y_on_line(m1, b1, xR), 0, h - 1))
    y2R = float(np.clip(y_on_line(m2, b2, xR), 0, h - 1))
    y2L = float(np.clip(y_on_line(m2, b2, xL), 0, h - 1))

    src = np.array([[xL, y1L], [xR, y1R], [xR, y2R], [xL, y2L]], dtype=np.float32)

    out_w = int(xR - xL)
    out_h = int(max(20, 0.5 * ((y2L - y1L) + (y2R - y1R))))
    out_h = int(np.clip(out_h, 20, 220))

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(gray, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    meta = {"src_quad": src.tolist(), "out_w": out_w, "out_h": out_h}
    return warped, meta


# ============================================================
# Head detection (rented/owned ink) on warped row
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

def detect_head_row_from_tenure_cols_warped(row_warp: np.ndarray, rented_x1: int, rented_x2: int, owned_x1: int, owned_x2: int):
    rented_cell = row_warp[:, rented_x1:rented_x2]
    owned_cell = row_warp[:, owned_x1:owned_x2]

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
# Visualization + extraction
# ============================================================

def draw_overlay_slanted(gray: np.ndarray, columns: dict, lines_mb: list, head_rows: list, head_row_tenure: dict,
                         xL: int, xR: int, table_bottom_y: int, out_path: str, title: str = ""):
    viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape

    if title:
        cv2.putText(viz, title, (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 0), 2)

    for col_name, (a, b) in columns.items():
        cv2.line(viz, (a, 0), (a, h), (255, 0, 0), 2)
        cv2.line(viz, (b, 0), (b, h), (255, 0, 0), 2)
        cv2.putText(viz, col_name, (a, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

    yb = int(np.clip(table_bottom_y, 0, h - 1))
    cv2.line(viz, (0, yb), (w, yb), (0, 255, 255), 3)
    cv2.putText(viz, "TABLE_BOTTOM", (40, yb - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    xL = int(np.clip(xL, 0, w - 1))
    xR = int(np.clip(xR, 0, w - 1))

    for i, (m, b) in enumerate(lines_mb):
        yL = int(np.clip(m * xL + b, 0, h - 1))
        yR = int(np.clip(m * xR + b, 0, h - 1))

        is_head = (i in head_rows)
        color = (0, 255, 0) if is_head else (0, 0, 255)
        thick = 3 if is_head else 2

        cv2.line(viz, (xL, yL), (xR, yR), color, thick)

        if i < len(lines_mb) - 1:
            if is_head:
                tenure = head_row_tenure.get(i, "HEAD")
                cv2.putText(viz, f"HEAD {i} [{tenure}]", (xL + 10, yL + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
            else:
                cv2.putText(viz, f"{i}", (xL + 10, yL + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, viz)

def extract_cells_warped_rows(gray: np.ndarray, lines_mb: list, columns: dict, xL: int, xR: int, out_dir: str):
    head_dir = os.path.join(out_dir, "head_rows")
    non_dir = os.path.join(out_dir, "non_head_rows")
    ensure_dir(head_dir)
    ensure_dir(non_dir)

    col_warp = {k: (max(0, a - xL), max(0, b - xL)) for k, (a, b) in columns.items()}

    rented_x1, rented_x2 = col_warp["rented"]
    owned_x1, owned_x2 = col_warp["owned"]

    head_rows = []
    head_row_tenure = {}

    rows_found = min(NUM_ROWS, len(lines_mb) - 1)
    for row_idx in range(rows_found):
        row_warp, meta = warp_row_band(gray, lines_mb[row_idx], lines_mb[row_idx + 1], xL, xR)
        if row_warp is None or row_warp.size == 0:
            continue

        is_head, tenure = detect_head_row_from_tenure_cols_warped(row_warp, rented_x1, rented_x2, owned_x1, owned_x2)
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
            fname = f"{prefix}row{row_idx:02d}_{col_name}.png"
            cv2.imwrite(os.path.join(out, fname), cell)

    return head_rows, head_row_tenure, rows_found

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

    # 1) ROI deskew
    gray_ds, angle, deskew_dbg = deskew_using_roi(gray, FIRST_ROW_Y_PRIOR)

    # 2) TOP line (earliest peak)
    table_top, top_dbg = pick_first_line_earliest_peak(gray_ds, FIRST_ROW_Y_PRIOR)

    # 3) BOTTOM line (dual-signal)
    table_bottom, bottom_dbg, vdens_s, hdens_s = detect_table_bottom_dual_signal(gray_ds, table_top)

    # 4) table x-bounds using your width prior (~6150px), anchored at min column x1
    min_x1 = min(v[0] for v in COLUMNS.values())
    xL = int(min_x1 - TABLE_X_MARGIN)
    xR = int(xL + TABLE_WIDTH_PX + 2 * TABLE_X_MARGIN)

    # 5) slanted separators
    lines_mb_all, slant_dbg = detect_slanted_separators(gray_ds, table_top, table_bottom, xL, xR)
    lines_mb = select_41_separators(lines_mb_all, table_top, table_bottom, xL, xR)

    print(f"deskew={angle:.3f}deg | top={table_top} | bottom={table_bottom} | slanted={len(lines_mb)} (raw={len(lines_mb_all)})")

    img_out = os.path.join(OUTPUT_DIR, name)
    ensure_dir(img_out)

    # If slanted detection fails, write report and skip
    if len(lines_mb) < NUM_ROWS + 1:
        print("⚠️ Not enough slanted separators detected for 40 rows. Skipping this image.")
        save_report_json(os.path.join(img_out, "report.json"), {
            "image_name": name,
            "source_path": img_path,
            "deskew": deskew_dbg,
            "table_top": int(table_top),
            "table_top_debug": top_dbg,
            "table_bottom": int(table_bottom),
            "bottom_detection": bottom_dbg,
            "slanted_debug": slant_dbg,
            "status": "failed_slanted_separator_detection",
            "slanted_count": int(len(lines_mb)),
            "slanted_raw_count": int(len(lines_mb_all)),
        })
        return

    # 6) Warp rows + extract cells
    head_rows, head_row_tenure, rows_found = ([], {}, 0)
    if SAVE_CELLS:
        head_rows, head_row_tenure, rows_found = extract_cells_warped_rows(gray_ds, lines_mb, COLUMNS, xL, xR, img_out)
        print(f"✅ Cells saved: {img_out}/head_rows and {img_out}/non_head_rows | rows={rows_found} | head={len(head_rows)}")

    # 7) Overlay
    if SAVE_VIZ:
        viz_path = os.path.join(img_out, "grid_overlay.png")
        title = f"{name} | deskew={angle:.2f}deg | rows={rows_found} | head={len(head_rows)}"
        draw_overlay_slanted(gray_ds, COLUMNS, lines_mb, head_rows, head_row_tenure, xL, xR, table_bottom, viz_path, title=title)
        print(f"✅ Grid visualization saved: {viz_path}")

    if SAVE_VERTICAL_DENSITY_DEBUG:
        vdbg_path = os.path.join(img_out, "vertical_density_debug.png")
        save_density_debug(vdens_s, vdbg_path, mark_top=table_top, mark_bottom=table_bottom)
        print(f"✅ Vertical density debug saved: {vdbg_path}")

    if SAVE_HORIZONTAL_DENSITY_DEBUG:
        hdbg_path = os.path.join(img_out, "horizontal_density_debug.png")
        save_density_debug(hdens_s, hdbg_path, mark_top=table_top, mark_bottom=table_bottom)
        print(f"✅ Horizontal density debug saved: {hdbg_path}")

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
        "slanted_debug": slant_dbg,
        "slanted_lines_selected": int(len(lines_mb)),
        "rows_found": int(rows_found),
        "head_rows": [{"row_idx": int(i), "tenure": head_row_tenure.get(i, "NONE")} for i in head_rows],
        "head_rows_count": int(len(head_rows)),
    }
    save_report_json(os.path.join(img_out, "report.json"), report)
    print(f"✅ Report saved: {os.path.join(img_out, 'report.json')}")


def main():
    ensure_dir(OUTPUT_DIR)

    imgs = list_images(INPUT_DIR)
    if not imgs:
        print(f"❌ No images found in: {INPUT_DIR}")
        return

    print("=== SMART ADAPTIVE EXTRACTION v9 (ROI DESKEW + TOP EARLIEST PEAK + DUAL-SIGNAL BOTTOM + SLOPED ROWS) ===")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images found: {len(imgs)}")

    for i, p in enumerate(imgs, start=1):
        print(f"\n[{i}/{len(imgs)}]")
        process_one_image(p)

    print("\n🎯 DONE")


if __name__ == "__main__":
    main()
