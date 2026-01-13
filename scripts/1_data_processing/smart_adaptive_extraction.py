import os
import cv2
import json
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "data/from_jeremy/images_aligned_to_first"
OUTPUT_DIR = "data/processed/smart_adaptive_extraction_batch_v6"

NUM_ROWS = 40
FIRST_ROW_Y_PRIOR = 1263
EXPECTED_ROW_HEIGHT = 78

# 🚧 Hard fence so row detection can’t leak into the bottom “general info” section
# You measured ~3160px from the top table line to the bottom table line.
TABLE_HEIGHT_PX = 3160
TABLE_HEIGHT_MARGIN = 220  # allow scan variation

# Column coordinates (fixed for now; we’ll do column alignment later)
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
SAVE_ROW_STRENGTH_DEBUG = False  # set True if you want debug plots per image

# Head/ink detection knobs
INK_PAD = 12
MIN_INK_RATIO = 0.010
MIN_CC_AREA = 60

# Row-line peak detection knobs
# We'll adaptively relax prominence ONLY inside the first-line ROI if needed.
PEAK_MIN_PROMINENCE_START = 0.18
PEAK_MIN_PROMINENCE_MIN = 0.06
PEAK_MERGE_DIST = 6

# Critical: restrict first-line search to ROI band near expected top
FIRST_LINE_ROI_UP = 260     # pixels above FIRST_ROW_Y_PRIOR
FIRST_LINE_ROI_DOWN = 520   # pixels below FIRST_ROW_Y_PRIOR

# Chaining constraints (row-to-row)
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
# Deskew (OpenCV)
# ============================================================

def estimate_skew_angle_degrees(gray: np.ndarray) -> float:
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
# Row boundary detection: peak chaining with strict first-line ROI + bottom fence
# ============================================================

def horizontal_line_strength(gray: np.ndarray) -> np.ndarray:
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

def find_line_peaks(strength: np.ndarray, min_prominence: float) -> list:
    if strength.size == 0 or float(strength.max()) <= 0:
        return []

    s = strength / float(strength.max())
    peaks = []
    for y in range(1, len(s) - 1):
        if s[y] > s[y - 1] and s[y] > s[y + 1] and s[y] >= min_prominence:
            peaks.append(y)

    # merge too-close peaks, keep stronger
    merged = []
    for p in peaks:
        if not merged or abs(p - merged[-1]) > PEAK_MERGE_DIST:
            merged.append(p)
        else:
            if s[p] > s[merged[-1]]:
                merged[-1] = p
    return merged

def pick_first_line_strict_roi(peaks: list, strength: np.ndarray, first_y_prior: int) -> (int, dict):
    """
    ONLY pick the first line inside a strict ROI around the prior.
    If weak, relax prominence in ROI only.
    If still none, fallback to prior (NOT bottom-of-page).
    """
    h = len(strength)
    roi_lo = max(0, first_y_prior - FIRST_LINE_ROI_UP)
    roi_hi = min(h - 1, first_y_prior + FIRST_LINE_ROI_DOWN)

    debug = {
        "roi_lo": int(roi_lo),
        "roi_hi": int(roi_hi),
        "picked_from": None,
        "used_prominence": None,
        "fallback_to_prior": False,
    }

    for prom in np.linspace(PEAK_MIN_PROMINENCE_START, PEAK_MIN_PROMINENCE_MIN, num=5):
        prom = float(prom)
        pks = find_line_peaks(strength, min_prominence=prom)
        roi_peaks = [p for p in pks if roi_lo <= p <= roi_hi]
        if roi_peaks:
            first = int(max(roi_peaks, key=lambda p: strength[p]))
            debug["picked_from"] = "roi_strongest"
            debug["used_prominence"] = prom
            return first, debug

    debug["picked_from"] = "prior_fallback"
    debug["used_prominence"] = None
    debug["fallback_to_prior"] = True
    return int(first_y_prior), debug

def pick_next_lines(peaks: list,
                    strength: np.ndarray,
                    first_line_y: int,
                    num_rows: int,
                    expected_row_height: int,
                    table_height_px: int = TABLE_HEIGHT_PX,
                    table_height_margin: int = TABLE_HEIGHT_MARGIN) -> list:
    """
    Chain the next lines, but NEVER allow detection past the table bottom.
    Prevents leaking into general-info section below the people table.
    """
    if first_line_y < 0:
        return []

    # 🚧 hard bottom fence (table ends around here)
    table_bottom_limit = int(first_line_y + table_height_px + table_height_margin)

    boundaries = [int(first_line_y)]
    current = int(first_line_y)

    for _ in range(num_rows):
        target = current + expected_row_height

        lo = max(target - CHAIN_WINDOW, current + CHAIN_MIN_STEP)
        hi = min(target + CHAIN_WINDOW, current + CHAIN_MAX_STEP)

        # 🚧 apply bottom fence
        hi = min(hi, table_bottom_limit)
        if lo > table_bottom_limit:
            break

        candidates = [p for p in peaks if lo <= p <= hi]

        if not candidates:
            # broaden a bit, still fenced
            lo2 = current + CHAIN_MIN_STEP
            hi2 = min(len(strength) - 1, current + CHAIN_MAX_STEP + 50, table_bottom_limit)
            if lo2 <= hi2:
                candidates = [p for p in peaks if lo2 <= p <= hi2]

        if candidates:
            nxt = int(max(candidates, key=lambda p: strength[p]))
        else:
            nxt = int(min(current + expected_row_height, table_bottom_limit))

        if nxt <= current + 2:
            break

        boundaries.append(nxt)
        current = nxt

        if current >= table_bottom_limit:
            break

    # Fill missing boundaries conservatively, but don't cross the fence
    while len(boundaries) < num_rows + 1:
        nxt = int(min(boundaries[-1] + expected_row_height, table_bottom_limit))
        if nxt <= boundaries[-1] + 2:
            break
        boundaries.append(nxt)

    # strictly increasing
    fixed = [boundaries[0]]
    for b in boundaries[1:]:
        if b <= fixed[-1]:
            b = fixed[-1] + 1
        fixed.append(int(b))

    # final clamp to fence
    fixed = [min(int(y), table_bottom_limit) for y in fixed]

    return fixed

def detect_41_boundaries(gray: np.ndarray, first_row_y_prior: int, num_rows: int, expected_row_height: int):
    strength = horizontal_line_strength(gray)

    # peaks for chaining
    peaks = find_line_peaks(strength, min_prominence=PEAK_MIN_PROMINENCE_MIN)

    first, first_debug = pick_first_line_strict_roi(peaks, strength, first_row_y_prior)

    boundaries = pick_next_lines(
        peaks=peaks,
        strength=strength,
        first_line_y=first,
        num_rows=num_rows,
        expected_row_height=expected_row_height,
        table_height_px=TABLE_HEIGHT_PX,
        table_height_margin=TABLE_HEIGHT_MARGIN
    )

    debug = {
        "method": "peak_chain_strict_first_roi_bottom_fence",
        "peaks_count": int(len(peaks)),
        "first_line": int(first),
        "first_line_debug": first_debug,
        "expected_row_height": int(expected_row_height),
        "chain_window": int(CHAIN_WINDOW),
        "chain_min_step": int(CHAIN_MIN_STEP),
        "chain_max_step": int(CHAIN_MAX_STEP),
        "table_height_px": int(TABLE_HEIGHT_PX),
        "table_height_margin": int(TABLE_HEIGHT_MARGIN),
        "table_bottom_limit": int(first + TABLE_HEIGHT_PX + TABLE_HEIGHT_MARGIN),
    }
    return boundaries, debug, strength


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
    if ink_ratio < MIN_INK_RATIO:
        return False

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return False

    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest_area >= MIN_CC_AREA

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

    return is_head, tenure


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

    for col_name, (x1, x2) in columns.items():
        cv2.line(viz, (x1, 0), (x1, h), (255, 0, 0), 2)
        cv2.line(viz, (x2, 0), (x2, h), (255, 0, 0), 2)
        cv2.putText(viz, col_name, (x1, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

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

def save_row_strength_debug(strength: np.ndarray, out_path: str, roi_lo: int, roi_hi: int, first_line: int):
    """
    Saves a simple visualization of strength signal as an image.
    White = stronger. Marks ROI + first line.
    """
    s = strength.copy()
    if s.max() > 0:
        s = s / s.max()
    img = (s * 255).astype(np.uint8).reshape(-1, 1)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # mark ROI in blue
    for y in range(roi_lo, roi_hi):
        if 0 <= y < img.shape[0]:
            img[y, 0] = (255, 0, 0)

    # mark first line in green
    if 0 <= first_line < img.shape[0]:
        img[first_line, 0] = (0, 255, 0)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, img)


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

    # Deskew
    angle = estimate_skew_angle_degrees(gray)
    gray_ds = rotate_image(gray, -angle)

    # Detect row boundaries with strict-first-line ROI + bottom fence
    row_boundaries, debug_rows, strength = detect_41_boundaries(
        gray_ds,
        first_row_y_prior=FIRST_ROW_Y_PRIOR,
        num_rows=NUM_ROWS,
        expected_row_height=EXPECTED_ROW_HEIGHT
    )

    if not row_boundaries or len(row_boundaries) < 2:
        print("⚠️ Could not build row boundaries. Skipping.")
        return

    # Clamp within image
    row_boundaries = [max(0, min(gray_ds.shape[0] - 1, int(y))) for y in row_boundaries]
    rows_found = len(row_boundaries) - 1

    roi_lo = debug_rows["first_line_debug"]["roi_lo"]
    roi_hi = debug_rows["first_line_debug"]["roi_hi"]
    first_line = debug_rows["first_line"]
    bottom_limit = debug_rows["table_bottom_limit"]

    print(
        f"deskew_angle={angle:.3f}deg | rows_found={rows_found} | peaks={debug_rows['peaks_count']} | "
        f"first_line={first_line} | bottom_limit={bottom_limit} | first_pick={debug_rows['first_line_debug']['picked_from']}"
    )

    # Head detection
    rented_x1, rented_x2 = COLUMNS["rented"]
    owned_x1, owned_x2 = COLUMNS["owned"]

    head_rows = []
    head_row_tenure = {}
    for row_idx in range(rows_found):
        y1, y2 = row_boundaries[row_idx], row_boundaries[row_idx + 1]
        row_img = gray_ds[y1:y2, :]
        is_head, tenure = detect_head_row_from_tenure_cols(row_img, rented_x1, rented_x2, owned_x1, owned_x2)
        if is_head:
            head_rows.append(row_idx)
            head_row_tenure[row_idx] = tenure

    # Output
    img_out = os.path.join(OUTPUT_DIR, name)
    ensure_dir(img_out)

    if SAVE_VIZ:
        viz_path = os.path.join(img_out, "grid_overlay.png")
        title = f"{name} | deskew={angle:.2f}deg | rows={rows_found} | head={len(head_rows)}"
        draw_grid_overlay(gray_ds, COLUMNS, row_boundaries, head_rows, head_row_tenure, viz_path, title=title)
        print(f"✅ Grid visualization saved: {viz_path}")

    if SAVE_CELLS:
        extract_cells_for_image(gray_ds, COLUMNS, row_boundaries, head_rows, head_row_tenure, img_out)
        print(f"✅ Cells saved under: {img_out}/head_rows and {img_out}/non_head_rows")

    if SAVE_ROW_STRENGTH_DEBUG:
        dbg_path = os.path.join(img_out, "row_strength_debug.png")
        save_row_strength_debug(strength, dbg_path, roi_lo, roi_hi, first_line)
        print(f"✅ Row strength debug saved: {dbg_path}")

    report = {
        "image_name": name,
        "source_path": img_path,
        "image_shape_original": {"h": int(gray.shape[0]), "w": int(gray.shape[1])},
        "image_shape_deskewed": {"h": int(gray_ds.shape[0]), "w": int(gray_ds.shape[1])},
        "deskew_angle_deg_estimated": float(angle),
        "row_detection": debug_rows,
        "rows_found": int(rows_found),
        "row_boundaries": [int(y) for y in row_boundaries],
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

    print("=== SMART ADAPTIVE EXTRACTION v6 (STRICT ROI + BOTTOM FENCE) ===")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images found: {len(imgs)}")

    for i, p in enumerate(imgs, start=1):
        print(f"\n[{i}/{len(imgs)}]")
        process_one_image(p)

    print("\n🎯 DONE")


if __name__ == "__main__":
    main()
