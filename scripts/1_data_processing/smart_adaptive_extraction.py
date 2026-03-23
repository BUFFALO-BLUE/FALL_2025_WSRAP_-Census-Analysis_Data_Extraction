import os
import json
import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "data/from_jeremy/Aligned_Test"
OUTPUT_DIR = "data/processed/v30_deskew_vfix_hfix"

NUM_ROWS = 40
FIRST_ROW_Y_PRIOR = 1263
TABLE_HEIGHT_PX = 3160

# Anchor vertical line (full-image x)
ANCHOR_X_PRIOR = 555
ANCHOR_SEARCH_BAND = 180

TABLE_WIDTH_PRIOR = 6100
OFFSET_MAX_ALLOW = int(TABLE_WIDTH_PRIOR + 500)

X_MARGIN = 220
ROI_TOP_PAD = 260
ROI_BOTTOM_PAD = 420

# Enhancement
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
BLACKHAT_KSIZE = 35
BLACKHAT_MIX = 0.85

# Offsets from anchor (cleaned)
OFFSETS_FROM_ANCHOR = [
    83,161,290,365,444,592,670,1331,1600,1650,1733,1835,1920,2025,2104,
    2181,2259,2594,2675,2750,3040,3303,3568,3649,3795,3898,3998,4102,4202,
    4330,4380,4533,4682,5064,5445,5523,5647,5752,5805,5884,6032,6133,6214
]

# ============================================================
# Manual per-image nudge
# dx > 0 moves verticals RIGHT
# dy > 0 moves horizontals UP (because we subtract dy)
# ============================================================

MANUAL_SHIFT = {
    "m-t0627-00538-00680": {"dx": 20, "dy": 45},
}

# ============================================================
# AUTO DESKEW
# ============================================================

AUTO_DESKEW = True

DESKEW_MIN_LINES = 6
DESKEW_MAX_ABS_DEG = 6.0
DESKEW_USE_MEDIAN = True

# ============================================================
# HORIZONTAL: bottom anchor + gated snapping
# ============================================================

HMASK_TRIALS = [
    {"pctl": 88, "kfrac": 0.30, "dilate": 1},
    {"pctl": 86, "kfrac": 0.28, "dilate": 1},
    {"pctl": 84, "kfrac": 0.26, "dilate": 1},
    {"pctl": 82, "kfrac": 0.24, "dilate": 1},
    {"pctl": 80, "kfrac": 0.22, "dilate": 1},
    {"pctl": 78, "kfrac": 0.20, "dilate": 1},
    {"pctl": 76, "kfrac": 0.18, "dilate": 1},
    {"pctl": 74, "kfrac": 0.16, "dilate": 1},
]
COVER_THRESH_START = 0.22
COVER_THRESH_MIN = 0.10
COVER_THRESH_STEP = 0.02

BOTTOM_EXPECT = FIRST_ROW_Y_PRIOR + TABLE_HEIGHT_PX
BOTTOM_SEARCH_BAND = 650
BOTTOM_PICK_MIN_COV = 0.12

SNAP_SEARCH_BAND_Y = 18
SNAP_ACCEPT_PX_Y = 8

H_GATE_STRONG_COV = 0.28
H_GATE_MAX_SHIFT = 5
H_REL_TO_BOTTOM = 0.55

# ============================================================
# VERTICAL: coverage-gated snapping
# ============================================================

ENABLE_VERTICAL_MICROSNAP = True
X_SNAP_BAND = 14
VDET_TOP_PAD = 40
VDET_BOT_PAD = 60

VLINE_THRESH_PCTL = 82
VLINE_K_FRAC = 0.22
VLINE_K_MIN = 420
VLINE_DILATE_ITERS = 1

VLINE_MIN_COVERAGE = 0.38
COVERAGE_X_SMOOTH_RADIUS = 2

# ============================================================
# OUTPUTS (GRID)
# ============================================================

SAVE_OVERLAY = True
SAVE_RULE_RESPONSE_CROP = True

# ============================================================
# DATASET EXTRACTION (HEAD ROWS)
# ============================================================

DATASET_DIR = "data/training/head_rows_AzureTest"
SAVE_ROW_IMG = True
SAVE_CELL_IMGS = True
SAVE_MASK_DEBUG = False  # set True to save mask images per trigger cell

# Trigger ONLY on these columns (must have ink in BOTH)
TRIGGER_COLS = ["rented_or_owned", "house_number"]

# Columns to save when trigger passes
SAVE_COLS = ["street", "house_number", "rented_or_owned", "price", "gender", "race"]

# Column mapping by column INDEX (1-based) -> line indices (0-based)
COLUMN_NUMBERS = {
    "street": 2,
    "house_number": 3,
    "rented_or_owned": 5,
    "price": 6,
    "gender": 11,
    "race": 12,
    "head": 9,
}

def colnum_to_line_range(col_num: int):
    return col_num - 1, col_num

COLUMN_LINE_RANGES = {k: colnum_to_line_range(v) for k, v in COLUMN_NUMBERS.items()}

# ============================================================
# BORDER-SAFE INK DETECTION (NO OCR)
# ============================================================

CELL_PAD_X = 6
CELL_PAD_Y = 4

# IMPORTANT: smaller inner crop ignores more border
INNER_FRAC = 0.70

INK_BLOCK = 31          # adaptive threshold block size (odd)
INK_C = 11              # higher -> fewer ink pixels
INK_MIN_RATIO = 0.021   # tune as needed

# --- NEW: anti-speck / anti-border / anti-line-fragment settings ---
BORDER_KILL_PX = 6      # kill pixels near the inner-crop border
CC_MIN_AREA = 30        # drop tiny connected components (specks)
LINE_AR = 12.0          # aspect ratio threshold to treat as "line-like"
LINE_THICK = 3          # max thickness of a line-like fragment

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

    rr = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = cv2.addWeighted(g, 1.0, rr, float(BLACKHAT_MIX), 0)
    return enhanced, rr

def rotate_image_keep_size(img: np.ndarray, angle_deg: float):
    h, w = img.shape[:2]
    center = (w * 0.5, h * 0.5)
    M = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, M

def apply_affine_to_points(M, pts_xy):
    pts = np.hstack([pts_xy.astype(np.float32), np.ones((len(pts_xy), 1), dtype=np.float32)])
    out = (pts @ M.T)
    return out[:, :2]

def invert_affine(M):
    return cv2.invertAffineTransform(M)

# ============================================================
# Cell cropping + ink detection
# ============================================================

def crop_cell(gray, x1, x2, y1, y2, pad_x=CELL_PAD_X, pad_y=CELL_PAD_Y):
    H, W = gray.shape
    x1 = int(max(0, min(W - 1, x1 + pad_x)))
    x2 = int(max(0, min(W,     x2 - pad_x)))
    y1 = int(max(0, min(H - 1, y1 + pad_y)))
    y2 = int(max(0, min(H,     y2 - pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None
    return gray[y1:y2, x1:x2]

def inner_crop(img, frac=INNER_FRAC):
    if img is None:
        return None
    h, w = img.shape
    fx = float(frac)
    fy = float(frac)
    dx = int((1.0 - fx) * w * 0.5)
    dy = int((1.0 - fy) * h * 0.5)
    x1, x2 = dx, w - dx
    y1, y2 = dy, h - dy
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2]

def binarize_for_ink(cell_gray):
    if cell_gray is None:
        return None
    g = cv2.GaussianBlur(cell_gray, (3, 3), 0)
    block = INK_BLOCK if INK_BLOCK % 2 == 1 else INK_BLOCK + 1
    bw = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # ink = white
        block, INK_C
    )
    bw = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return bw

def remove_straight_lines(bw):
    if bw is None:
        return None
    h, w = bw.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 2), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 2)))
    hlines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vlines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    lines = cv2.bitwise_or(hlines, vlines)
    cleaned = cv2.bitwise_and(bw, cv2.bitwise_not(lines))
    return cleaned

def drop_small_components(bw, min_area=25):
    """
    bw: uint8 mask, ink=255
    Removes connected components smaller than min_area.
    """
    if bw is None:
        return None
    num, labels, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return bw

    out = np.zeros_like(bw)
    for i in range(1, num):  # skip background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            out[labels == i] = 255
    return out

def kill_border_pixels(bw, margin=BORDER_KILL_PX):
    """Zero out a margin around the mask to ignore border fragments."""
    if bw is None:
        return None
    m = int(max(0, margin))
    if m <= 0:
        return bw
    out = bw.copy()
    out[:m, :] = 0
    out[-m:, :] = 0
    out[:, :m] = 0
    out[:, -m:] = 0
    return out

def remove_line_like_components(bw, ar=LINE_AR, thick=LINE_THICK):
    """
    Remove components that look like thin long lines (rule fragments).
    """
    if bw is None:
        return None
    num, labels, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return bw

    out = np.zeros_like(bw)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        w = int(w); h = int(h); area = int(area)
        if area <= 0:
            continue

        # classify as line-like
        if (w >= int(ar * max(1, h)) and h <= int(thick)) or (h >= int(ar * max(1, w)) and w <= int(thick)):
            continue

        out[labels == i] = 255
    return out

def ink_ratio(cell_gray):
    """
    Returns (ratio, mask) where mask is aggressively cleaned:
    - inner crop ignores outer border
    - adaptive threshold
    - remove straight line artifacts
    - kill border pixels inside the inner crop (prevents border leaks)
    - drop small components (specks)
    - remove line-like fragments
    """
    inner = inner_crop(cell_gray)
    bw = binarize_for_ink(inner)
    bw2 = remove_straight_lines(bw)
    bw2 = kill_border_pixels(bw2, margin=BORDER_KILL_PX)
    bw2 = drop_small_components(bw2, min_area=CC_MIN_AREA)
    bw2 = remove_line_like_components(bw2, ar=LINE_AR, thick=LINE_THICK)

    if bw2 is None or bw2.size == 0:
        return 0.0, None

    ink = float(np.count_nonzero(bw2))
    total = float(bw2.size)
    return (ink / max(1.0, total)), bw2

# ============================================================
# Column mapping validation + extraction
# ============================================================

def validate_column_ranges(v_lines_full):
    n = len(v_lines_full)
    for key, (li, ri) in COLUMN_LINE_RANGES.items():
        if li < 0 or ri >= n:
            raise ValueError(f"Column '{key}' needs v_lines_full[{li},{ri}] but only have {n} lines.")

def get_cell_from_grid(gray, v_lines_full, y1, y2, key):
    li, ri = COLUMN_LINE_RANGES[key]
    x1 = int(min(v_lines_full[li], v_lines_full[ri]))
    x2 = int(max(v_lines_full[li], v_lines_full[ri]))
    return crop_cell(gray, x1, x2, y1, y2)

def row_trigger_pass(gray, v_lines_full, y1, y2):
    ratios = {}
    masks = {}
    for k in TRIGGER_COLS:
        cell = get_cell_from_grid(gray, v_lines_full, y1, y2, k)
        r, m = ink_ratio(cell)
        ratios[k] = float(r)
        masks[k] = m
    ok = all(ratios[k] >= float(INK_MIN_RATIO) for k in TRIGGER_COLS)
    return ok, ratios, masks

def save_row_package(gray, name, row_idx, y1, y2, v_lines_full, ratios, masks):
    out_dir = os.path.join(DATASET_DIR, name, f"row_{row_idx:02d}")
    ensure_dir(out_dir)

    if SAVE_ROW_IMG:
        row_crop = crop_cell(gray, 0, gray.shape[1], y1, y2, pad_x=0, pad_y=0)
        if row_crop is not None:
            cv2.imwrite(os.path.join(out_dir, "row.png"), row_crop)

    if SAVE_CELL_IMGS:
        for key in SAVE_COLS:
            if key not in COLUMN_LINE_RANGES:
                continue
            cell = get_cell_from_grid(gray, v_lines_full, y1, y2, key)
            if cell is not None:
                cv2.imwrite(os.path.join(out_dir, f"{key}.png"), cell)

    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        for k, v in ratios.items():
            f.write(f"{k}_ink_ratio={v:.6f}\n")

    if SAVE_MASK_DEBUG:
        for k, m in masks.items():
            if m is not None:
                cv2.imwrite(os.path.join(out_dir, f"mask_{k}.png"), m)

# ============================================================
# NEW: Save street strip + rows.json (NO OCR)
# ============================================================

def save_street_strip_and_rows(gray, name, v_lines_full, h_lines):
    page_dir = os.path.join(DATASET_DIR, name)
    ensure_dir(page_dir)

    # --- crop full street column strip ---
    y_top = int(min(h_lines[0], h_lines[-1]))
    y_bot = int(max(h_lines[0], h_lines[-1]))

    strip = get_cell_from_grid(
        gray,
        v_lines_full,
        y_top,
        y_bot,
        key="street"
    )

    if strip is not None:
        cv2.imwrite(os.path.join(page_dir, "street_strip.png"), strip)

    # --- save row boundaries ---
    rows_meta = []
    for r in range(len(h_lines) - 1):
        y1 = int(min(h_lines[r], h_lines[r + 1]))
        y2 = int(max(h_lines[r], h_lines[r + 1]))
        rows_meta.append({
            "row_idx": r,
            "y1": y1,
            "y2": y2,
            "y_center": (y1 + y2) / 2.0
        })

    with open(os.path.join(page_dir, "rows.json"), "w", encoding="utf-8") as f:
        json.dump(rows_meta, f, indent=2)

# ============================================================
# Horizontal bands + bottom anchor
# ============================================================

def make_hmask(rr_crop: np.ndarray, pctl: int, kfrac: float, dilate: int):
    thr = int(np.percentile(rr_crop, pctl))
    thr = max(8, min(240, thr))
    bw = (rr_crop >= thr).astype(np.uint8) * 255

    h, w = rr_crop.shape
    klen = int(max(180, w * float(kfrac)))
    if klen % 2 == 0:
        klen += 1

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)

    if int(dilate) > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        opened = cv2.dilate(opened, dk, iterations=int(dilate))
    return opened

def bands_from_hmask(hmask: np.ndarray, y0_full: int, cover_thresh: float):
    h, w = hmask.shape
    cov = (np.sum(hmask > 0, axis=1).astype(np.float32) / float(max(1, w)))
    cov_s = smooth_1d(cov, 9)

    on = cov_s >= float(cover_thresh)
    ys = np.where(on)[0]
    if ys.size == 0:
        return []

    segs = []
    start = int(ys[0])
    prev = int(ys[0])
    for y in ys[1:]:
        y = int(y)
        if y == prev + 1:
            prev = y
        else:
            segs.append((start, prev))
            start = y
            prev = y
    segs.append((start, prev))

    bands = []
    for a, b in segs:
        c = int(round(0.5 * (a + b)))
        mean_cov = float(np.mean(cov_s[a:b+1]))
        bands.append({"center_full": int(c + y0_full), "mean_cov": mean_cov})

    seen = set()
    uniq = []
    for d in sorted(bands, key=lambda z: z["center_full"]):
        if d["center_full"] not in seen:
            uniq.append(d)
            seen.add(d["center_full"])
    return uniq

def build_bands_auto(rr_crop: np.ndarray, y0_full: int):
    best = None
    for t in HMASK_TRIALS:
        hmask = make_hmask(rr_crop, t["pctl"], t["kfrac"], t["dilate"])
        cover = float(COVER_THRESH_START)
        while cover >= float(COVER_THRESH_MIN):
            bands = bands_from_hmask(hmask, y0_full=y0_full, cover_thresh=cover)
            bottom_cands = [b for b in bands if abs(b["center_full"] - BOTTOM_EXPECT) <= BOTTOM_SEARCH_BAND]
            bottom_strength = max((b["mean_cov"] for b in bottom_cands), default=0.0)
            score = len(bands) + 15.0 * bottom_strength
            if (best is None) or (score > best["score"]):
                best = {"score": score, "bands": bands}
            if len(bands) >= 25:
                break
            cover -= float(COVER_THRESH_STEP)
    return [] if best is None else best["bands"]

def choose_bottom_anchor_with_strength(bands: list):
    if not bands:
        return int(BOTTOM_EXPECT), 0.0
    cands = [b for b in bands if abs(b["center_full"] - BOTTOM_EXPECT) <= BOTTOM_SEARCH_BAND]
    if not cands:
        best = max(bands, key=lambda b: b["mean_cov"])
        return int(best["center_full"]), float(best["mean_cov"])
    best = max(cands, key=lambda b: (b["mean_cov"], -abs(b["center_full"] - BOTTOM_EXPECT)))
    if best["mean_cov"] < float(BOTTOM_PICK_MIN_COV):
        return int(BOTTOM_EXPECT), float(best["mean_cov"])
    return int(best["center_full"]), float(best["mean_cov"])

def nearest_band(y_expect: int, bands: list, search_band: int):
    lo, hi = int(y_expect - search_band), int(y_expect + search_band)
    cands = [b for b in bands if lo <= b["center_full"] <= hi]
    if not cands:
        return None
    return min(cands, key=lambda b: abs(b["center_full"] - y_expect))

def gated_snap_y(y_expect: int, bands: list, bottom_cov: float):
    cand = nearest_band(y_expect, bands, SNAP_SEARCH_BAND_Y)
    if cand is None:
        return int(y_expect)

    shift = abs(int(cand["center_full"]) - int(y_expect))

    if cand["mean_cov"] < float(H_GATE_STRONG_COV):
        return int(y_expect)

    if bottom_cov > 1e-6 and cand["mean_cov"] < float(H_REL_TO_BOTTOM) * float(bottom_cov):
        return int(y_expect)

    if shift > int(H_GATE_MAX_SHIFT):
        return int(y_expect)
    if shift > int(SNAP_ACCEPT_PX_Y):
        return int(y_expect)

    return int(cand["center_full"])

# ============================================================
# Vertical: mask + coverage-gated snap + deskew angle estimate
# ============================================================

def vertical_rule_mask(rr_table: np.ndarray) -> np.ndarray:
    rr = rr_table.astype(np.uint8)
    thr = int(np.percentile(rr, VLINE_THRESH_PCTL))
    thr = max(8, min(240, thr))
    bw = (rr >= thr).astype(np.uint8) * 255

    h, w = bw.shape
    klen = int(max(VLINE_K_MIN, h * float(VLINE_K_FRAC)))
    if klen % 2 == 0:
        klen += 1

    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen))
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)

    if int(VLINE_DILATE_ITERS) > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        opened = cv2.dilate(opened, dk, iterations=int(VLINE_DILATE_ITERS))
    return opened

def estimate_deskew_angle_from_vmask(vmask: np.ndarray) -> float:
    edges = cv2.Canny(vmask, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=120,
        minLineLength=max(80, int(0.35 * vmask.shape[0])),
        maxLineGap=20
    )
    if lines is None or len(lines) < DESKEW_MIN_LINES:
        return 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dy) < 1e-3:
            continue
        ang = np.degrees(np.arctan2(dx, dy))  # relative to vertical
        if abs(ang) <= float(DESKEW_MAX_ABS_DEG):
            angles.append(ang)

    if len(angles) < DESKEW_MIN_LINES:
        return 0.0

    return float(np.median(angles)) if DESKEW_USE_MEDIAN else float(np.mean(angles))

def detect_anchor_x_by_coverage(vmask: np.ndarray, x_prior_crop: int, search_band: int) -> int:
    h, w = vmask.shape
    x_prior_crop = int(np.clip(x_prior_crop, 0, w - 1))
    lo = int(max(0, x_prior_crop - search_band))
    hi = int(min(w - 1, x_prior_crop + search_band))

    col_on = (vmask > 0).astype(np.uint8)
    col_sum = np.sum(col_on, axis=0).astype(np.float32)

    r = int(COVERAGE_X_SMOOTH_RADIUS)
    if r > 0:
        k = 2 * r + 1
        col_sum = np.convolve(col_sum, np.ones(k, dtype=np.float32), mode="same")

    denom = float(h * (2 * r + 1 if r > 0 else 1))
    cov = col_sum / max(1.0, denom)

    window = list(range(lo, hi + 1))
    best_x = max(window, key=lambda x: float(cov[x]))
    if float(cov[best_x]) < float(VLINE_MIN_COVERAGE):
        return int(x_prior_crop)
    return int(best_x)

def best_x_by_vertical_coverage(vmask: np.ndarray, x_expected: int, band: int) -> int:
    h, w = vmask.shape
    x_expected = int(np.clip(x_expected, 0, w - 1))
    lo = int(max(0, x_expected - band))
    hi = int(min(w - 1, x_expected + band))

    col_on = (vmask > 0).astype(np.uint8)
    col_sum = np.sum(col_on, axis=0).astype(np.float32)

    r = int(COVERAGE_X_SMOOTH_RADIUS)
    if r > 0:
        k = 2 * r + 1
        col_sum = np.convolve(col_sum, np.ones(k, dtype=np.float32), mode="same")

    denom = float(h * (2 * r + 1 if r > 0 else 1))
    cov = col_sum / max(1.0, denom)

    window = list(range(lo, hi + 1))
    best_x = max(window, key=lambda x: float(cov[x]))
    if float(cov[best_x]) < float(VLINE_MIN_COVERAGE):
        return int(x_expected)
    return int(best_x)

def microsnap_lines_by_coverage(lines_crop, vmask, band):
    out = [best_x_by_vertical_coverage(vmask, int(x), int(band)) for x in lines_crop]
    out2 = []
    last = -10**9
    for x in out:
        if x <= last:
            x = last + 1
        out2.append(int(x))
        last = x
    return out2

# ============================================================
# Overlay
# ============================================================

def draw_overlay(gray, h_lines_y, v_lines_x, table_top, table_bottom, out_path, highlight_cols=True):
    viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    H, W = gray.shape

    for y in h_lines_y:
        y = int(np.clip(y, 0, H - 1))
        cv2.line(viz, (0, y), (W, y), (0, 0, 255), 2)

    for x in v_lines_x:
        x = int(np.clip(x, 0, W - 1))
        cv2.line(viz, (x, table_top), (x, table_bottom), (0, 255, 0), 2)

    cv2.line(viz, (0, table_top), (W, table_top), (255, 255, 0), 2)
    cv2.line(viz, (0, table_bottom), (W, table_bottom), (0, 255, 255), 3)

    if highlight_cols:
        overlay = viz.copy()
        for key in SAVE_COLS:
            if key not in COLUMN_LINE_RANGES:
                continue
            li, ri = COLUMN_LINE_RANGES[key]
            if 0 <= li < len(v_lines_x) and 0 <= ri < len(v_lines_x):
                x1 = int(min(v_lines_x[li], v_lines_x[ri]))
                x2 = int(max(v_lines_x[li], v_lines_x[ri]))
                cv2.rectangle(overlay, (x1, table_top), (x2, table_bottom), (255, 0, 255), -1)
                cv2.rectangle(viz, (x1, table_top), (x2, table_bottom), (255, 0, 255), 2)
                cv2.putText(viz, key, (x1 + 6, table_top + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        viz = cv2.addWeighted(overlay, 0.18, viz, 0.82, 0)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, viz)

# ============================================================
# Main per-image
# ============================================================

def clean_offsets(offsets):
    good = [int(o) for o in offsets if 0 < int(o) <= OFFSET_MAX_ALLOW]
    return sorted(set(good))

OFFSETS_CLEAN = clean_offsets(OFFSETS_FROM_ANCHOR)

def process_one_image(img_path: str):
    name = os.path.splitext(os.path.basename(img_path))[0]
    gray = read_gray(img_path)
    if gray is None:
        print(f"⚠️ Could not read: {img_path}", flush=True)
        return

    H, W = gray.shape
    out_dir = os.path.join(OUTPUT_DIR, name)
    ensure_dir(out_dir)

    dx = int(MANUAL_SHIFT.get(name, {}).get("dx", 0))
    dy = int(MANUAL_SHIFT.get(name, {}).get("dy", 0))

    xL = int(ANCHOR_X_PRIOR - X_MARGIN)
    xR = int(ANCHOR_X_PRIOR + TABLE_WIDTH_PRIOR + X_MARGIN)
    xL = max(0, min(W - 2, xL))
    xR = max(xL + 1, min(W - 1, xR))

    y0 = max(0, FIRST_ROW_Y_PRIOR - ROI_TOP_PAD)
    y1 = min(H, FIRST_ROW_Y_PRIOR + TABLE_HEIGHT_PX + ROI_BOTTOM_PAD)

    crop = gray[y0:y1, xL:xR]
    _, rr_crop = enhance_faint_rules(crop)

    bands = build_bands_auto(rr_crop, y0_full=y0)
    table_bottom, bottom_cov = choose_bottom_anchor_with_strength(bands)
    table_top = int(table_bottom - TABLE_HEIGHT_PX)

    step_y = float(TABLE_HEIGHT_PX) / float(NUM_ROWS)
    h_lines = []
    for i in range(NUM_ROWS + 1):
        y_expect = int(round(table_top + i * step_y))
        y_snap = int(gated_snap_y(y_expect, bands, bottom_cov))
        h_lines.append(y_snap - dy)

    table_top_c = int(max(0, table_top - y0))
    table_bottom_c = int(min(rr_crop.shape[0] - 1, table_bottom - y0))
    rr_table = rr_crop[table_top_c + VDET_TOP_PAD : table_bottom_c - VDET_BOT_PAD, :]

    Mdeskew = None
    angle = 0.0

    vmask0 = vertical_rule_mask(rr_table)
    if AUTO_DESKEW:
        angle = estimate_deskew_angle_from_vmask(vmask0)
        if abs(angle) > 0.2:
            rr_table_rot, Mdeskew = rotate_image_keep_size(rr_table, -angle)
            vmask = vertical_rule_mask(rr_table_rot)
        else:
            vmask = vmask0
            Mdeskew = None
    else:
        vmask = vmask0

    anchor_prior_crop = int(ANCHOR_X_PRIOR - xL)
    anchor_x_crop = detect_anchor_x_by_coverage(vmask, anchor_prior_crop, search_band=ANCHOR_SEARCH_BAND)

    v_lines_crop = [anchor_x_crop] + [anchor_x_crop + o for o in OFFSETS_CLEAN]
    if ENABLE_VERTICAL_MICROSNAP:
        v_lines_crop = microsnap_lines_by_coverage(v_lines_crop, vmask, band=X_SNAP_BAND)

    v_lines_full = []
    if Mdeskew is None:
        for x in v_lines_crop:
            v_lines_full.append(int(xL + x + dx))
    else:
        Minv = invert_affine(Mdeskew)
        htab, _ = rr_table.shape[:2]
        yA = 5.0
        yB = float(htab - 6)
        for x in v_lines_crop:
            pts = np.array([[float(x), yA], [float(x), yB]], dtype=np.float32)
            pts_unrot = apply_affine_to_points(Minv, pts)
            x_unrot = float(np.mean(pts_unrot[:, 0]))
            v_lines_full.append(int(xL + x_unrot + dx))

    # Save street strip + row boundaries (NO OCR)
    save_street_strip_and_rows(gray, name, v_lines_full, h_lines)

    # ===== head-row extraction by cleaned ink trigger =====
    extracted = 0
    try:
        validate_column_ranges(v_lines_full)
        ensure_dir(DATASET_DIR)

        for r in range(len(h_lines) - 1):
            y1r = int(min(h_lines[r], h_lines[r + 1]))
            y2r = int(max(h_lines[r], h_lines[r + 1]))

            ok, ratios, masks = row_trigger_pass(gray, v_lines_full, y1r, y2r)
            if ok:
                save_row_package(gray, name, r, y1r, y2r, v_lines_full, ratios, masks)
                extracted += 1

    except Exception as e:
        print(f"⚠️ Extraction skipped for {name}: {e}", flush=True)

    if SAVE_RULE_RESPONSE_CROP:
        cv2.imwrite(os.path.join(out_dir, "debug_rule_response_crop.png"), rr_crop)

    if SAVE_OVERLAY:
        draw_overlay(
            gray, h_lines, v_lines_full,
            int(table_top - dy), int(table_bottom - dy),
            os.path.join(out_dir, "grid_overlay.png"),
            highlight_cols=True
        )

    print(
        f"{name}: extracted={extracted} manual(dx={dx},dy={dy}) "
        f"deskew_angle={angle:.3f}deg vlines={len(v_lines_full)}",
        flush=True
    )

# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)
    imgs = list_images(INPUT_DIR)
    if not imgs:
        print(f"❌ No images found in: {INPUT_DIR}", flush=True)
        return

    print("=== v30 (deskew + vertical microsnap + border-safe ink trigger) ===", flush=True)
    print(f"Images found: {len(imgs)}", flush=True)
    print(f"AUTO_DESKEW={AUTO_DESKEW}", flush=True)
    print(f"Offsets used: {len(OFFSETS_CLEAN)}", flush=True)
    print(f"MANUAL_SHIFT keys: {list(MANUAL_SHIFT.keys())}", flush=True)
    print(f"DATASET_DIR={DATASET_DIR}", flush=True)
    print(f"TRIGGER_COLS={TRIGGER_COLS} INK_MIN_RATIO={INK_MIN_RATIO}", flush=True)
    print(f"INNER_FRAC={INNER_FRAC} BORDER_KILL_PX={BORDER_KILL_PX} CC_MIN_AREA={CC_MIN_AREA} LINE_AR={LINE_AR} LINE_THICK={LINE_THICK}", flush=True)

    for i, p in enumerate(imgs, 1):
        base = os.path.splitext(os.path.basename(p))[0]
        print(f"[{i}/{len(imgs)}] {base}", flush=True)
        process_one_image(p)

    print("🎯 DONE", flush=True)

if __name__ == "__main__":
    main()
