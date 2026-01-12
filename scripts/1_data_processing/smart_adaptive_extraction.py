import os
import cv2
import json
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "data/from_jeremy/images_aligned_to_first"
OUTPUT_DIR = "data/processed/smart_adaptive_extraction_batch_v4"

NUM_ROWS = 40

# Priors (used to find first boundary + expected spacing for chaining)
FIRST_ROW_Y_PRIOR = 1263
EXPECTED_ROW_HEIGHT = 78

# Column coordinates (leave as-is for now; we’ll fix column alignment AFTER you verify rows)
COLUMNS = {
    "street": (629, 718),
    "house_number": (718, 836),

    # adjust if needed
    "rented": (914, 954),
    "owned":  (954, 994),

    "price_rent": (996, 1143),

    # kept as data, NOT used for head detection
    "head": (1889, 2204),

    "gender": (2204, 2285),
    "race": (2285, 2388),
    "marital_status": (2491, 2574),
    "hours_worked": (4939, 5092),
    "wages": (6433, 6588),
}

SAVE_VIZ = True
SAVE_CELLS = True

# Ink detection knobs (tune later if needed)
INK_PAD = 12
MIN_INK_RATIO = 0.010
MIN_CC_AREA = 60

# Row-line chaining knobs
PEAK_MIN_PROMINENCE = 0.15     # relative to max line strength
PEAK_MERGE_DIST = 6
FIRST_LINE_MAX_DELTA = 250
CHAIN_WINDOW = 28
CHAIN_MIN_STEP = 45
CHAIN_MAX_STEP = 120


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


# ============================================================
# Deskew
# ============================================================

def estimate_skew_angle_degrees(gray: np.ndarray) -> float:
    """
    Estimate skew using long horizontal table lines via HoughLinesP.
    Returns angle in degrees (near 0 if already straight).
    """
    bin_img = robust_binarize(gray)
    inv = 255 - bin_img

    h, w = inv.shape
    k_w = max(40, w // 25)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

    edges = cv2.Canny(horiz, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=max(250, w // 5),
        maxLineGap=25
    )

    if lines is None:
        return 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if -20 <= angle <= 20:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))

def rotate_image(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.05:
        return gray
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        gray,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


# ============================================================
# Row boundary detection: detect 41 REAL separator lines (peak chaining)
# ============================================================

def horizontal_line_strength(gray: np.ndarray) -> np.ndarray:
    """
    Returns strength[y] measuring how 'line-like' each y is.
    """
    bin_img = robust_binarize(gray)
    inv = 255 - bin_img

    h, w = inv.shape
    k_w = max(45, w // 22)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))

    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=2)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)), iterations=1)

    strength = np.sum(horiz > 0, axis=1).astype(np.float32)

    # smooth to merge broken lines
    k = 9
    strength = np.convolve(strength, np.ones(k, dtype=np.float32) / k, mode="same")
    return strength

def find_line_peaks(strength: np.ndarray, min_prominence: float = PEAK_MIN_PROMINENCE) -> list:
    """
    Find local maxima peaks in strength.
    min_prominence is relative to max strength.
    """
    if strength.size == 0 or float(strength.max()) <= 0:
        return []

    s = strength / float(strength.max())
    peaks = []
    for y in range(1, len(s) - 1):
        if s[y] > s[y - 1] and s[y] > s[y + 1] and s[y] >= min_prominence:
            peaks.append(y)

    # merge peaks too close, keep stronger
    merged = []
    for p in peaks:
        if not merged or abs(p - merged[-1]) > PEAK_MERGE_DIST:
            merged.append(p)
        else:
            if s[p] > s[merged[-1]]:
                merged[-1] = p
    return merged

def pick_first_line(peaks: list, strength: np.ndarray, first_y_prior: int, max_delta: int = FIRST_LINE_MAX_DELTA) -> int:
    """
    Pick the line peak closest to the prior start.
    If none within max_delta, pick strongest peak in a broad band around the prior.
    """
    if not peaks:
        return -1

    near = [p for p in peaks if abs(p - first_y_prior) <= max_delta]
    if near:
        return int(min(near, key=lambda p: abs(p - first_y_prior)))

    lo = max(0, first_y_prior - 400)
    hi = min(len(strength) - 1, first_y_prior + 400)
    band = [p for p in peaks if lo <= p <= hi]
    if band:
        return int(max(band, key=lambda p: strength[p]))

    return int(max(peaks, key=lambda p: strength[p]))

def pick_next_lines(peaks: list,
                    strength: np.ndarray,
                    first_line_y: int,
                    num_rows: int,
                    expected_row_height: int,
                    window: int = CHAIN_WINDOW,
                    min_step: int = CHAIN_MIN_STEP,
                    max_step: int = CHAIN_MAX_STEP) -> list:
    """
    After first boundary, pick next num_rows boundaries by searching around expected height each time.
    Chooses strongest peak inside the valid window.
    Returns boundaries list length num_rows+1.
    """
    if first_line_y < 0:
        return []

    boundaries = [int(first_line_y)]
    current = int(first_line_y)

    for _ in range(num_rows):
        target = current + expected_row_height

        lo = target - window
        hi = target + window

        lo = max(lo, current + min_step)
        hi = min(hi, current + max_step)

        candidates = [p for p in peaks if lo <= p <= hi]

        if not candidates:
            # fallback: broaden slightly
            lo2 = current + min_step
            hi2 = min(len(strength) - 1, current + max_step + 60)
            candidates = [p for p in peaks if lo2 <= p <= hi2]

        if candidates:
            nxt = int(max(candidates, key=lambda p: strength[p]))
        else:
            nxt = int(current + expected_row_height)

        boundaries.append(nxt)
        current = nxt

    # enforce strictly increasing
    fixed = [boundaries[0]]
    for b in boundaries[1:]:
        if b <= fixed[-1]:
            b = fixed[-1] + 1
        fixed.append(int(b))

    return fixed

def detect_41_boundaries(gray: np.ndarray, first_row_y_prior: int, num_rows: int, expected_row_height: int):
    strength = horizontal_line_strength(gray)
    peaks = find_line_peaks(strength, min_prominence=PEAK_MIN_PROMINENCE)
    first = pick_first_line(peaks, strength, first_row_y_prior, max_delta=FIRST_LINE_MAX_DELTA)
    boundaries = pick_next_lines(
        peaks, strength, first,
        num_rows=num_rows,
        expected_row_height=expected_row_height,
        window=CHAIN_WINDOW,
        min_step=CHAIN_MIN_STEP,
        max_step=CHAIN_MAX_STEP
    )

    debug = {
        "method": "peak_chain",
        "peaks_count": int(len(peaks)),
        "first_line": int(first),
        "expected_row_height": int(expected_row_height),
        "window": int(CHAIN_WINDOW),
        "min_step": int(CHAIN_MIN_STEP),
        "max_step": int(CHAIN_MAX_STEP),
    }
    return boundaries, debug


# ============================================================
# Head detection (rented/owned ink) — unchanged
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

def cell_has_ink(cell_gray: np.ndarray,
                 pad: int = INK_PAD,
                 min_ink_ratio: float = MIN_INK_RATIO,
                 min_cc_area: int = MIN_CC_AREA) -> bool:
    if cell_gray is None or cell_gray.size == 0:
        return False

    h, w = cell_gray.shape
    if h <= 2 * pad or w <= 2 * pad:
        return False

    roi = cell_gray[pad:h - pad, pad:w - pad]
    bin_img = robust_binarize(roi)
    ink = 255 - bin_img

    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1
    )

    ink = remove_table_lines(ink)

    ink_pixels = int(np.count_nonzero(ink > 0))
    total = int(ink.size)
    if total == 0:
        return False

    ink_ratio = ink_pixels / total
    if ink_ratio < min_ink_ratio:
        return False

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return False

    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest_area >= min_cc_area

def detect_head_row_from_tenure_cols(row_img_gray: np.ndarray,
                                     rented_x1: int, rented_x2: int,
                                     owned_x1: int, owned_x2: int):
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

    return is_head, tenure, {"is_rented": bool(is_rented), "is_owned": bool(is_owned)}


# ============================================================
# Visualization + extraction
# ============================================================

def draw_grid_overlay(gray: np.ndarray,
                      columns: dict,
                      row_boundaries: list,
                      head_rows: list,
                      head_row_tenure: dict,
                      out_path: str,
                      title: str = "") -> None:
    viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape

    if title:
        cv2.putText(viz, title, (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 0), 2)

    # Columns (fixed for now)
    for col_name, (x1, x2) in columns.items():
        cv2.line(viz, (x1, 0), (x1, h), (255, 0, 0), 2)
        cv2.line(viz, (x2, 0), (x2, h), (255, 0, 0), 2)
        cv2.putText(viz, col_name, (x1, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

    # Rows (from detected 41 separators)
    for i, y in enumerate(row_boundaries):
        is_head = i in head_rows
        color = (0, 255, 0) if is_head else (0, 0, 255)
        thick = 3 if is_head else 2
        cv2.line(viz, (0, y), (w, y), color, thick)

        if i < len(row_boundaries) - 1:
            rh = row_boundaries[i + 1] - y
            if is_head:
                tenure = head_row_tenure.get(i, "HEAD")
                label = f"HEAD {i} [{tenure}] ({rh}px)"
                cv2.putText(viz, label, (40, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
            else:
                cv2.putText(viz, f"Row {i} ({rh}px)", (40, y + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, viz)

def extract_cells_for_image(gray: np.ndarray,
                            columns: dict,
                            row_boundaries: list,
                            head_rows: list,
                            head_row_tenure: dict,
                            out_dir: str) -> None:
    head_dir = os.path.join(out_dir, "head_rows")
    non_dir = os.path.join(out_dir, "non_head_rows")
    ensure_dir(head_dir)
    ensure_dir(non_dir)

    rows_found = len(row_boundaries) - 1
    for row_idx in range(rows_found):
        y1, y2 = row_boundaries[row_idx], row_boundaries[row_idx + 1]
        is_head = row_idx in head_rows
        tenure = head_row_tenure.get(row_idx, "NONE")
        out = head_dir if is_head else non_dir

        for col_name, (x1, x2) in columns.items():
            cell = gray[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            prefix = f"HEAD_{tenure}_" if is_head else ""
            fname = f"{prefix}row{row_idx:02d}_{col_name}.png"
            cv2.imwrite(os.path.join(out, fname), cell)

def save_report_json(out_path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# Main
# ============================================================

def process_one_image(img_path: str) -> None:
    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n=== Processing: {name} ===")

    gray = read_gray(img_path)
    if gray is None:
        print("⚠️ Could not read image. Skipping.")
        return

    # 1) Deskew per image
    angle = estimate_skew_angle_degrees(gray)
    gray_ds = rotate_image(gray, -angle)

    # 2) Detect 41 row boundaries by chaining real horizontal separator peaks
    row_boundaries, debug_rows = detect_41_boundaries(
        gray_ds,
        first_row_y_prior=FIRST_ROW_Y_PRIOR,
        num_rows=NUM_ROWS,
        expected_row_height=EXPECTED_ROW_HEIGHT
    )

    if not row_boundaries or len(row_boundaries) < NUM_ROWS + 1:
        print("⚠️ Could not build 41 row boundaries. Skipping.")
        return

    rows_found = len(row_boundaries) - 1
    # Clamp within image
    row_boundaries = [max(0, min(gray_ds.shape[0] - 1, int(y))) for y in row_boundaries]

    print(f"deskew_angle={angle:.3f}deg | rows={rows_found} | peaks={debug_rows['peaks_count']} | first_line={debug_rows['first_line']}")

    # 3) Head detection
    rented_x1, rented_x2 = COLUMNS["rented"]
    owned_x1, owned_x2 = COLUMNS["owned"]

    head_rows = []
    head_row_tenure = {}

    for row_idx in range(rows_found):
        y1, y2 = row_boundaries[row_idx], row_boundaries[row_idx + 1]
        row_img = gray_ds[y1:y2, :]
        is_head, tenure, _dbg = detect_head_row_from_tenure_cols(row_img, rented_x1, rented_x2, owned_x1, owned_x2)
        if is_head:
            head_rows.append(row_idx)
            head_row_tenure[row_idx] = tenure

    # 4) Output
    img_out = os.path.join(OUTPUT_DIR, name)
    ensure_dir(img_out)

    if SAVE_VIZ:
        viz_path = os.path.join(img_out, "grid_overlay.png")
        title = f"{name} | deskew={angle:.2f}deg | rows={rows_found} | head={len(head_rows)} | peak_chain"
        draw_grid_overlay(gray_ds, COLUMNS, row_boundaries, head_rows, head_row_tenure, viz_path, title=title)
        print(f"✅ Grid visualization saved: {viz_path}")

    if SAVE_CELLS:
        extract_cells_for_image(gray_ds, COLUMNS, row_boundaries, head_rows, head_row_tenure, img_out)
        print(f"✅ Cells saved under: {img_out}/head_rows and {img_out}/non_head_rows")

    report = {
        "image_name": name,
        "source_path": img_path,
        "image_shape_original": {"h": int(gray.shape[0]), "w": int(gray.shape[1])},
        "image_shape_deskewed": {"h": int(gray_ds.shape[0]), "w": int(gray_ds.shape[1])},
        "deskew_angle_deg_estimated": float(angle),
        "row_detection": debug_rows,
        "rows_found": int(rows_found),
        "row_boundaries": [int(y) for y in row_boundaries],
        "columns": {k: {"x1": int(v[0]), "x2": int(v[1])} for k, v in COLUMNS.items()},
        "head_rows": [{"row_idx": int(i), "tenure": head_row_tenure.get(i, "NONE")} for i in head_rows],
        "head_rows_count": int(len(head_rows)),
        "ink_detection": {
            "pad": int(INK_PAD),
            "min_ink_ratio": float(MIN_INK_RATIO),
            "min_cc_area": int(MIN_CC_AREA),
        },
        "notes": [
            "Row boundaries are chosen by chaining real detected horizontal separator peaks (not uniform spacing).",
            "Head row count is not a quality metric; some pages can have 0 or 1 head rows.",
            "Column alignment is still fixed for now; we’ll fix per-image x-shift/scale after you verify rows."
        ],
    }
    save_report_json(os.path.join(img_out, "report.json"), report)
    print(f"✅ Report saved: {os.path.join(img_out, 'report.json')}")


def main():
    ensure_dir(OUTPUT_DIR)

    imgs = list_images(INPUT_DIR)
    if not imgs:
        print(f"❌ No images found in: {INPUT_DIR}")
        return

    print("=== SMART ADAPTIVE EXTRACTION v4 (DESKEW + CHAINED 41 ROW LINES) ===")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images found: {len(imgs)}")

    for i, p in enumerate(imgs, start=1):
        print(f"\n[{i}/{len(imgs)}]")
        process_one_image(p)

    print("\n🎯 DONE")


if __name__ == "__main__":
    main()
