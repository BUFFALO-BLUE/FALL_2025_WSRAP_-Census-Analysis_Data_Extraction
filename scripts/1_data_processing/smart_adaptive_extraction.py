import os
import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "data/from_jeremy/images_aligned_to_first"
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
    83,161,290,365,444,522,592,670,1331,1600,1650,1733,1835,1920,2025,2104,
    2181,2259,2594,2675,2750,3040,3303,3568,3649,3795,3898,3998,4102,4202,
    4330,4380,4533,4682,5064,5445,5523,5647,5752,5805,5884,6032,6133,6214
]

# ============================================================
# Manual per-image nudge (your request for 00680)
# dx > 0 moves verticals RIGHT
# dy > 0 moves horizontals UP (because we subtract dy)
# ============================================================

MANUAL_SHIFT = {
    "m-t0627-00538-00680": {"dx": 20, "dy": 45},
}

# ============================================================
# AUTO DESKEW (fix tilted scans like 00680/00682)
# ============================================================

AUTO_DESKEW = True

# Hough settings for angle estimation from vertical mask
DESKEW_MIN_LINES = 6
DESKEW_MAX_ABS_DEG = 6.0     # ignore insane angles
DESKEW_USE_MEDIAN = True     # robust

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
# OUTPUTS ONLY
# ============================================================

SAVE_OVERLAY = True
SAVE_RULE_RESPONSE_CROP = True


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
    # pts_xy: Nx2 float32
    pts = np.hstack([pts_xy.astype(np.float32), np.ones((len(pts_xy), 1), dtype=np.float32)])
    out = (pts @ M.T)
    return out[:, :2]

def invert_affine(M):
    Minv = cv2.invertAffineTransform(M)
    return Minv


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
    """
    Returns angle (deg) to rotate image so vertical lines become vertical.
    Uses HoughLinesP on vmask edges.
    """
    edges = cv2.Canny(vmask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=120,
                            minLineLength=max(80, int(0.35 * vmask.shape[0])),
                            maxLineGap=20)
    if lines is None or len(lines) < DESKEW_MIN_LINES:
        return 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dy) < 1e-3:
            continue
        # angle of the segment relative to vertical (in degrees):
        # perfect vertical -> 0
        ang = np.degrees(np.arctan2(dx, dy))
        # keep only near-vertical segments
        if abs(ang) <= float(DESKEW_MAX_ABS_DEG):
            angles.append(ang)

    if len(angles) < DESKEW_MIN_LINES:
        return 0.0

    if DESKEW_USE_MEDIAN:
        return float(np.median(angles))
    return float(np.mean(angles))

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

def draw_overlay(gray, h_lines_y, v_lines_x, table_top, table_bottom, out_path):
    viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    H, W = gray.shape

    # horizontals = red
    for y in h_lines_y:
        y = int(np.clip(y, 0, H - 1))
        cv2.line(viz, (0, y), (W, y), (0, 0, 255), 2)

    # verticals = green
    for x in v_lines_x:
        x = int(np.clip(x, 0, W - 1))
        cv2.line(viz, (x, table_top), (x, table_bottom), (0, 255, 0), 2)

    cv2.line(viz, (0, table_top), (W, table_top), (255, 255, 0), 2)
    cv2.line(viz, (0, table_bottom), (W, table_bottom), (0, 255, 255), 3)

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
        print(f"⚠️ Could not read: {img_path}")
        return

    H, W = gray.shape
    out_dir = os.path.join(OUTPUT_DIR, name)
    ensure_dir(out_dir)

    # manual shifts
    dx = int(MANUAL_SHIFT.get(name, {}).get("dx", 0))
    dy = int(MANUAL_SHIFT.get(name, {}).get("dy", 0))

    # Crop around expected table x-band
    xL = int(ANCHOR_X_PRIOR - X_MARGIN)
    xR = int(ANCHOR_X_PRIOR + TABLE_WIDTH_PRIOR + X_MARGIN)
    xL = max(0, min(W - 2, xL))
    xR = max(xL + 1, min(W - 1, xR))

    # Crop around expected table y-band
    y0 = max(0, FIRST_ROW_Y_PRIOR - ROI_TOP_PAD)
    y1 = min(H, FIRST_ROW_Y_PRIOR + TABLE_HEIGHT_PX + ROI_BOTTOM_PAD)

    crop = gray[y0:y1, xL:xR]
    _, rr_crop = enhance_faint_rules(crop)

    # Horizontal bottom anchor bands (in crop coords but y0_full used)
    bands = build_bands_auto(rr_crop, y0_full=y0)
    table_bottom, bottom_cov = choose_bottom_anchor_with_strength(bands)
    table_top = int(table_bottom - TABLE_HEIGHT_PX)

    # Build h-lines (flat in full coords)
    step_y = float(TABLE_HEIGHT_PX) / float(NUM_ROWS)
    h_lines = []
    for i in range(NUM_ROWS + 1):
        y_expect = int(round(table_top + i * step_y))
        y_snap = int(gated_snap_y(y_expect, bands, bottom_cov))
        h_lines.append(y_snap - dy)  # dy>0 moves UP

    # Prepare table band for vertical detection
    table_top_c = int(max(0, table_top - y0))
    table_bottom_c = int(min(rr_crop.shape[0] - 1, table_bottom - y0))
    rr_table = rr_crop[table_top_c + VDET_TOP_PAD : table_bottom_c - VDET_BOT_PAD, :]

    # Deskew in table band only (keeps everything consistent for vertical fitting)
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

    # Anchor prior in crop coords
    anchor_prior_crop = int(ANCHOR_X_PRIOR - xL)

    # If deskewed, anchor_prior should be in rotated table coords.
    # We approximate by keeping same x (rotation is small). Coverage search handles small mismatch.
    anchor_x_crop = detect_anchor_x_by_coverage(vmask, anchor_prior_crop, search_band=ANCHOR_SEARCH_BAND)

    # Build vertical lines in "deskewed table coords"
    v_lines_crop = [anchor_x_crop] + [anchor_x_crop + o for o in OFFSETS_CLEAN]

    if ENABLE_VERTICAL_MICROSNAP:
        v_lines_crop = microsnap_lines_by_coverage(v_lines_crop, vmask, band=X_SNAP_BAND)

    # Convert vertical lines back to full coords
    # If deskew was used, map x positions back approximately (we only need overlay).
    # We map a vertical line at x by sampling two y points and inverse-rotating them.
    v_lines_full = []

    if Mdeskew is None:
        for x in v_lines_crop:
            v_lines_full.append(int(xL + x + dx))
    else:
        Minv = invert_affine(Mdeskew)
        htab, wtab = rr_table.shape[:2]
        yA = 5.0
        yB = float(htab - 6)
        for x in v_lines_crop:
            pts = np.array([[float(x), yA], [float(x), yB]], dtype=np.float32)
            pts_unrot = apply_affine_to_points(Minv, pts)
            # take x from unrotated points (average)
            x_unrot = float(np.mean(pts_unrot[:, 0]))
            v_lines_full.append(int(xL + x_unrot + dx))

    # Save only requested outputs
    if SAVE_RULE_RESPONSE_CROP:
        cv2.imwrite(os.path.join(out_dir, "debug_rule_response_crop.png"), rr_crop)

    if SAVE_OVERLAY:
        draw_overlay(gray, h_lines, v_lines_full,
                     int(table_top - dy), int(table_bottom - dy),
                     os.path.join(out_dir, "grid_overlay.png"))

    print(f"{name}: manual(dx={dx},dy={dy}) deskew_angle={angle:.3f}deg vlines={len(v_lines_full)}")


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)
    imgs = list_images(INPUT_DIR)
    if not imgs:
        print(f"❌ No images found in: {INPUT_DIR}")
        return

    print("=== v30 (deskew + manual shift + green verticals) ===")
    print(f"Images found: {len(imgs)}")
    print(f"AUTO_DESKEW={AUTO_DESKEW}")
    print(f"Offsets used: {len(OFFSETS_CLEAN)} (outliers removed)")
    print(f"MANUAL_SHIFT keys: {list(MANUAL_SHIFT.keys())}")

    for i, p in enumerate(imgs, 1):
        base = os.path.splitext(os.path.basename(p))[0]
        print(f"[{i}/{len(imgs)}] {base}")
        process_one_image(p)

    print("🎯 DONE")

if __name__ == "__main__":
    main()
