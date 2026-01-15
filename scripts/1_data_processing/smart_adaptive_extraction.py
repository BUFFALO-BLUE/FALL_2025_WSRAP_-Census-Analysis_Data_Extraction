import os
import cv2
import json
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "data/from_jeremy/images_aligned_to_first"
OUTPUT_DIR = "data/processed/smart_adaptive_extraction_batch_v19_gated_snap"

NUM_ROWS = 40
FIRST_ROW_Y_PRIOR = 1263
TABLE_HEIGHT_PX = 3160
TABLE_WIDTH_PX = 6150

TABLE_X_MARGIN = 140
ROI_TOP_PAD = 260
ROI_BOTTOM_PAD = 420

# Enhancement knobs
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
BLACKHAT_KSIZE = 35
BLACKHAT_MIX = 0.85

# Horizontal mask trials (same spirit as your working v19/v18)
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

# Snap knobs
SNAP_SEARCH_BAND = 18
SNAP_ACCEPT_PX = 8

# Bottom anchor
BOTTOM_EXPECT = FIRST_ROW_Y_PRIOR + TABLE_HEIGHT_PX
BOTTOM_SEARCH_BAND = 650
BOTTOM_PICK_MIN_COV = 0.12

# ============================================================
# NEW: Confidence-gated snapping (THIS IS THE FIX)
# ============================================================

# Only snap to a band if it looks like a printed rule (not handwriting)
STRONG_RULE_COV = 0.22     # if too strict -> lower to 0.18; if still snapping into text -> raise to 0.26
MAX_ROW_SHIFT = 6          # max pixels we allow a snap to move from expected grid position

# If an image has many “unsafe snap candidates” we flag it into faulty_images.txt
FAULTY_MIN_UNSAFE_ROWS = 3

# ============================================================
# Columns (only used to set x band). Keep as your current values.
# ============================================================

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

# Outputs
SAVE_VIZ = True
SAVE_DEBUG_RULE_RESPONSE = True


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


# ============================================================
# Horizontal mask -> bands
# ============================================================

def make_hmask(rr_crop: np.ndarray, pctl: int, kfrac: float, dilate: int):
    h, w = rr_crop.shape
    thr = int(np.percentile(rr_crop, pctl))
    thr = max(8, min(240, thr))
    bw = (rr_crop >= thr).astype(np.uint8) * 255

    klen = int(max(180, w * float(kfrac)))
    if klen % 2 == 0:
        klen += 1

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)

    if int(dilate) > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        opened = cv2.dilate(opened, dk, iterations=int(dilate))

    return opened, thr, klen

def bands_from_hmask(hmask: np.ndarray, y0_full: int, cover_thresh: float):
    h, w = hmask.shape
    cov = (np.sum(hmask > 0, axis=1).astype(np.float32) / float(max(1, w)))
    cov_s = smooth_1d(cov, 9)

    on = cov_s >= float(cover_thresh)
    ys = np.where(on)[0]
    if ys.size == 0:
        return [], cov_s

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

    out = []
    for a, b in bands:
        thick = (b - a + 1)
        c = int(round(0.5 * (a + b)))
        mean_cov = float(np.mean(cov_s[a:b+1]))
        out.append({
            "center_full": int(c + y0_full),
            "thick": int(thick),
            "mean_cov": float(mean_cov),
        })

    # dedupe by center
    seen = set()
    uniq = []
    for d in sorted(out, key=lambda z: z["center_full"]):
        if d["center_full"] not in seen:
            uniq.append(d)
            seen.add(d["center_full"])
    return uniq, cov_s

def build_bands_auto(rr_crop: np.ndarray, y0_full: int):
    best = None

    for t in HMASK_TRIALS:
        hmask, thr, klen = make_hmask(rr_crop, t["pctl"], t["kfrac"], t["dilate"])
        cover = float(COVER_THRESH_START)

        while cover >= float(COVER_THRESH_MIN):
            bands, _ = bands_from_hmask(hmask, y0_full=y0_full, cover_thresh=cover)

            bottom_cands = [b for b in bands if abs(b["center_full"] - BOTTOM_EXPECT) <= BOTTOM_SEARCH_BAND]
            bottom_strength = max((b["mean_cov"] for b in bottom_cands), default=0.0)

            score = len(bands) + 15.0 * bottom_strength

            if (best is None) or (score > best["score"]):
                best = {
                    "score": score,
                    "bands": bands,
                    "dbg": f"(best) pctl={t['pctl']} kfrac={t['kfrac']:.2f} thr={thr} klen={klen} bands={len(bands)} cover={cover:.2f} bottom_str={bottom_strength:.3f}"
                }

            if len(bands) >= 25:
                break

            cover -= float(COVER_THRESH_STEP)

    if best is None:
        return [], [], "(no bands)"

    band_centers = [b["center_full"] for b in best["bands"]]
    return best["bands"], band_centers, best["dbg"]

def choose_bottom_anchor(bands: list):
    if not bands:
        return int(BOTTOM_EXPECT), "(no bands; bottom=prior)"

    cands = [b for b in bands if abs(b["center_full"] - BOTTOM_EXPECT) <= BOTTOM_SEARCH_BAND]
    if not cands:
        best = max(bands, key=lambda b: (b["mean_cov"], b["thick"]))
        return int(best["center_full"]), "(bottom fallback strongest)"

    best = max(cands, key=lambda b: (b["mean_cov"], b["thick"], -abs(b["center_full"] - BOTTOM_EXPECT)))
    if best["mean_cov"] < float(BOTTOM_PICK_MIN_COV):
        return int(BOTTOM_EXPECT), "(bottom too weak; prior)"

    return int(best["center_full"]), "(bottom anchored)"


# ============================================================
# NEW: confidence-gated snapping
# ============================================================

def nearest_band_candidate(y_expect: int, bands: list, search_band: int):
    """
    Return the nearest band dict within ±search_band, else None.
    """
    lo = int(y_expect - search_band)
    hi = int(y_expect + search_band)
    cands = [b for b in bands if lo <= b["center_full"] <= hi]
    if not cands:
        return None
    return min(cands, key=lambda b: abs(b["center_full"] - y_expect))

def gated_snap(y_expect: int, bands: list) -> (int, bool, str):
    """
    Snap only if:
      - nearest band exists
      - band mean_cov >= STRONG_RULE_COV
      - abs shift <= MAX_ROW_SHIFT
    Returns: (y_final, snapped_bool, reason)
    """
    cand = nearest_band_candidate(y_expect, bands, SNAP_SEARCH_BAND)
    if cand is None:
        return int(y_expect), False, "no_candidate"

    y_c = int(cand["center_full"])
    shift = abs(y_c - int(y_expect))

    if cand["mean_cov"] < float(STRONG_RULE_COV):
        return int(y_expect), False, f"weak_cov({cand['mean_cov']:.3f})"

    if shift > int(MAX_ROW_SHIFT):
        return int(y_expect), False, f"shift_too_big({shift})"

    if shift > int(SNAP_ACCEPT_PX):
        return int(y_expect), False, f"outside_accept({shift})"

    return int(y_c), True, "snapped_ok"


# ============================================================
# Visualization
# ============================================================

def draw_overlay(gray: np.ndarray, lines_y: list, table_top: int, table_bottom: int,
                 out_path: str, title: str, xL: int, xR: int):
    viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape

    if title:
        cv2.putText(viz, title, (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    # x band bounds
    cv2.line(viz, (xL, 0), (xL, h), (150, 150, 0), 2)
    cv2.line(viz, (xR, 0), (xR, h), (150, 150, 0), 2)

    # table top/bottom
    cv2.line(viz, (0, table_top), (w, table_top), (255, 255, 0), 2)
    cv2.line(viz, (0, table_bottom), (w, table_bottom), (0, 255, 255), 3)

    # horizontal lines (only across table x-band)
    for y in lines_y:
        y = int(np.clip(y, 0, h - 1))
        cv2.line(viz, (xL, y), (xR, y), (0, 0, 255), 2)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, viz)


# ============================================================
# Per-image
# ============================================================

def process_one_image(img_path: str):
    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n=== Processing: {name} ===")

    gray = read_gray(img_path)
    if gray is None:
        print("⚠️ Could not read image. Skipping.")
        return None

    H, W = gray.shape
    img_out = os.path.join(OUTPUT_DIR, name)
    ensure_dir(img_out)

    # X band from known columns
    min_x1 = min(v[0] for v in COLUMNS.values())
    xL = int(min_x1 - TABLE_X_MARGIN)
    xR = int(xL + TABLE_WIDTH_PX + 2 * TABLE_X_MARGIN)
    xL = max(0, min(W - 2, xL))
    xR = max(xL + 1, min(W - 1, xR))

    # Crop around expected table band
    y0 = max(0, FIRST_ROW_Y_PRIOR - ROI_TOP_PAD)
    y1 = min(H, FIRST_ROW_Y_PRIOR + TABLE_HEIGHT_PX + ROI_BOTTOM_PAD)
    crop = gray[y0:y1, xL:xR]

    # Rule response (debug)
    _, rr_crop = enhance_faint_rules(crop)

    # Bands + bottom anchor
    bands, band_centers, dbg_bands = build_bands_auto(rr_crop, y0_full=y0)
    table_bottom, dbg_bottom = choose_bottom_anchor(bands)
    table_top = int(table_bottom - TABLE_HEIGHT_PX)

    # Build expected grid and apply gated snapping
    step = float(TABLE_HEIGHT_PX) / float(NUM_ROWS)
    lines_y = []
    snapped_count = 0
    unsafe_count = 0
    reasons = {}

    for i in range(NUM_ROWS + 1):
        y_expect = int(round(table_top + i * step))
        y_final, snapped, reason = gated_snap(y_expect, bands)
        lines_y.append(int(y_final))
        if snapped:
            snapped_count += 1
        else:
            # count "unsafe" cases where we *would* have snapped but refused
            if reason.startswith("weak_cov") or reason.startswith("shift_too_big") or reason.startswith("outside_accept"):
                unsafe_count += 1
            reasons[reason] = reasons.get(reason, 0) + 1

    # Save requested outputs
    if SAVE_DEBUG_RULE_RESPONSE:
        cv2.imwrite(os.path.join(img_out, "debug_rule_response_crop.png"), rr_crop)

    if SAVE_VIZ:
        title = f"{name} | {dbg_bands} | {dbg_bottom} | snapped={snapped_count} unsafe_refused={unsafe_count}"
        draw_overlay(gray, lines_y, table_top, table_bottom,
                     os.path.join(img_out, "grid_overlay.png"),
                     title=title, xL=xL, xR=xR)

    print(f"{dbg_bands} {dbg_bottom}")
    print(f"table_top={table_top} bottom={table_bottom} snapped={snapped_count} unsafe_refused={unsafe_count}")
    return {
        "name": name,
        "unsafe_refused": unsafe_count,
        "reasons": reasons
    }


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    imgs = list_images(INPUT_DIR)
    if not imgs:
        print(f"❌ No images found in: {INPUT_DIR}")
        return

    print("=== SMART ADAPTIVE EXTRACTION v19 (GATED SNAP) ===")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images found: {len(imgs)}")
    print(f"STRONG_RULE_COV={STRONG_RULE_COV} MAX_ROW_SHIFT={MAX_ROW_SHIFT}")

    faulty = []
    stats = []

    for i, p in enumerate(imgs, start=1):
        print(f"\n[{i}/{len(imgs)}]")
        r = process_one_image(p)
        if r is None:
            continue
        stats.append(r)
        if r["unsafe_refused"] >= FAULTY_MIN_UNSAFE_ROWS:
            faulty.append(r["name"])

    # Write a clean list of the filenames to check
    faulty_path = os.path.join(OUTPUT_DIR, "faulty_images.txt")
    with open(faulty_path, "w", encoding="utf-8") as f:
        for n in faulty:
            f.write(n + "\n")

    print("\n🎯 DONE")
    print(f"Faulty (to check) count = {len(faulty)}")
    print(f"Wrote: {faulty_path}")

if __name__ == "__main__":
    main()
