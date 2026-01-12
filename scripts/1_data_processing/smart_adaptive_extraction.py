import os
import cv2
import json
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "data/from_jeremy/images_aligned_to_first"
OUTPUT_DIR = "data/processed/smart_adaptive_extraction_batch_v3"

# The census people-table has 40 rows (your statement)
NUM_ROWS = 40

# Priors (used ONLY to find the correct two anchor lines per image)
FIRST_ROW_Y_PRIOR = 1263
EXPECTED_ROW_HEIGHT = 78
BOTTOM_TABLE_Y_PRIOR = FIRST_ROW_Y_PRIOR + (EXPECTED_ROW_HEIGHT * NUM_ROWS)  # ~4383

# Column coordinates (after deskew, these become much more reliable)
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

# Ink detection knobs (tune if needed)
INK_PAD = 12
MIN_INK_RATIO = 0.010
MIN_CC_AREA = 60

# Horizontal-line detection knobs
ROW_LINE_PROJ_THRESH = 0.18   # lower -> more sensitive to lines
ROW_LINE_MERGE_DIST = 6

# How far we allow the chosen anchors to deviate from priors
TOP_ANCHOR_MAX_DELTA = 220     # pixels
BOTTOM_ANCHOR_MAX_DELTA = 260  # pixels


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

def robust_binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )

def group_nearby_positions(pos, merge_dist=6):
    if not pos:
        return []
    pos = sorted(pos)
    grouped = [pos[0]]
    for p in pos[1:]:
        if abs(p - grouped[-1]) <= merge_dist:
            grouped[-1] = int((grouped[-1] + p) / 2)
        else:
            grouped.append(p)
    return grouped


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
# Detect horizontal table lines, then enforce UNIFORM 40 rows
# ============================================================

def detect_horizontal_lines(gray: np.ndarray) -> list:
    """
    Returns list of y positions where strong horizontal lines exist.
    """
    bin_img = robust_binarize(gray)
    inv = 255 - bin_img

    h, w = inv.shape
    k_w = max(45, w // 22)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))

    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=2)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)), iterations=1)

    proj = np.sum(horiz > 0, axis=1).astype(np.float32)
    if proj.max() <= 0:
        return []

    proj_n = proj / proj.max()
    candidates = np.where(proj_n >= ROW_LINE_PROJ_THRESH)[0].tolist()
    if not candidates:
        return []

    # group consecutive rows into single y
    lines = []
    start = candidates[0]
    prev = candidates[0]
    for y in candidates[1:]:
        if y == prev + 1:
            prev = y
        else:
            lines.append(int((start + prev) / 2))
            start = y
            prev = y
    lines.append(int((start + prev) / 2))

    return group_nearby_positions(lines, merge_dist=ROW_LINE_MERGE_DIST)

def pick_anchor_line(lines: list, y_prior: int, max_delta: int) -> int:
    """
    Pick the detected line closest to y_prior, but require it's within max_delta.
    """
    if not lines:
        return -1
    best = min(lines, key=lambda y: abs(y - y_prior))
    if abs(best - y_prior) > max_delta:
        return -1
    return int(best)

def build_uniform_row_boundaries(lines: list, img_h: int) -> (list, dict):
    """
    Use 2 anchors (top, bottom) and enforce exactly 40 uniform rows between them.
    Returns (boundaries, debug).
    """
    debug = {
        "top_prior": int(FIRST_ROW_Y_PRIOR),
        "bottom_prior": int(BOTTOM_TABLE_Y_PRIOR),
        "lines_count": int(len(lines)),
        "top_anchor": None,
        "bottom_anchor": None,
        "method": None
    }

    top = pick_anchor_line(lines, FIRST_ROW_Y_PRIOR, TOP_ANCHOR_MAX_DELTA)
    bottom = pick_anchor_line(lines, BOTTOM_TABLE_Y_PRIOR, BOTTOM_ANCHOR_MAX_DELTA)

    # If bottom mistakenly picked above top (rare but possible), invalidate
    if top != -1 and bottom != -1 and bottom <= top + 200:
        bottom = -1

    if top != -1 and bottom != -1:
        debug["top_anchor"] = int(top)
        debug["bottom_anchor"] = int(bottom)
        debug["method"] = "uniform_between_anchors"

        table_height = bottom - top
        row_h = table_height / NUM_ROWS

        boundaries = [int(round(top + i * row_h)) for i in range(NUM_ROWS + 1)]
        boundaries[0] = max(0, min(img_h - 1, boundaries[0]))
        boundaries[-1] = max(0, min(img_h - 1, boundaries[-1]))

        # enforce strictly increasing
        fixed = [boundaries[0]]
        for b in boundaries[1:]:
            if b <= fixed[-1]:
                b = fixed[-1] + 1
            fixed.append(min(int(b), img_h - 1))

        return fixed, debug

    # Fallback: uniform from priors only (still consistent spacing, but less accurate anchors)
    debug["method"] = "uniform_from_priors_fallback"
    top_f = max(0, min(img_h - 1, FIRST_ROW_Y_PRIOR))
    bottom_f = max(0, min(img_h - 1, BOTTOM_TABLE_Y_PRIOR))
    if bottom_f <= top_f + 200:
        bottom_f = min(img_h - 1, top_f + int(EXPECTED_ROW_HEIGHT * NUM_ROWS))

    debug["top_anchor"] = int(top_f)
    debug["bottom_anchor"] = int(bottom_f)

    table_height = bottom_f - top_f
    row_h = table_height / NUM_ROWS
    boundaries = [int(round(top_f + i * row_h)) for i in range(NUM_ROWS + 1)]

    fixed = [boundaries[0]]
    for b in boundaries[1:]:
        if b <= fixed[-1]:
            b = fixed[-1] + 1
        fixed.append(min(int(b), img_h - 1))

    return fixed, debug


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

    # Columns
    for col_name, (x1, x2) in columns.items():
        cv2.line(viz, (x1, 0), (x1, h), (255, 0, 0), 2)
        cv2.line(viz, (x2, 0), (x2, h), (255, 0, 0), 2)
        cv2.putText(viz, col_name, (x1, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

    # Rows
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

    # Deskew
    angle = estimate_skew_angle_degrees(gray)
    gray_ds = rotate_image(gray, -angle)

    # Detect horizontal lines, then build uniform 40-row boundaries
    lines = detect_horizontal_lines(gray_ds)
    row_boundaries, debug_rows = build_uniform_row_boundaries(lines, gray_ds.shape[0])
    rows_found = len(row_boundaries) - 1

    print(f"deskew_angle={angle:.3f}deg | lines={len(lines)} | rows={rows_found} | row_method={debug_rows['method']}")
    print(f"anchors: top={debug_rows['top_anchor']} bottom={debug_rows['bottom_anchor']} (priors top={FIRST_ROW_Y_PRIOR} bottom={BOTTOM_TABLE_Y_PRIOR})")

    # Head detection
    rented_x1, rented_x2 = COLUMNS["rented"]
    owned_x1, owned_x2 = COLUMNS["owned"]

    head_rows = []
    head_row_tenure = {}

    for row_idx in range(rows_found):
        y1, y2 = row_boundaries[row_idx], row_boundaries[row_idx + 1]
        row_img = gray_ds[y1:y2, :]
        is_head, tenure, dbg = detect_head_row_from_tenure_cols(row_img, rented_x1, rented_x2, owned_x1, owned_x2)

        if is_head:
            head_rows.append(row_idx)
            head_row_tenure[row_idx] = tenure

    # Output
    img_out = os.path.join(OUTPUT_DIR, name)
    ensure_dir(img_out)

    if SAVE_VIZ:
        viz_path = os.path.join(img_out, "grid_overlay.png")
        title = f"{name} | deskew={angle:.2f}deg | rows={rows_found} | head={len(head_rows)} | {debug_rows['method']}"
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
        "detected_line_ys": [int(y) for y in lines],
        "rows_found": int(rows_found),
        "columns": {k: {"x1": int(v[0]), "x2": int(v[1])} for k, v in COLUMNS.items()},
        "row_boundaries": [int(y) for y in row_boundaries],
        "head_rows": [{"row_idx": int(i), "tenure": head_row_tenure.get(i, "NONE")} for i in head_rows],
        "head_rows_count": int(len(head_rows)),
        "ink_detection": {
            "pad": int(INK_PAD),
            "min_ink_ratio": float(MIN_INK_RATIO),
            "min_cc_area": int(MIN_CC_AREA),
        },
        "notes": [
            "Row spacing is enforced uniformly between detected top/bottom table anchors.",
            "Head row count is NOT a quality metric (some pages have 0 or 1 head rows).",
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

    print("=== SMART ADAPTIVE EXTRACTION v3 (DESKEW + UNIFORM 40 ROWS) ===")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images found: {len(imgs)}")

    for i, p in enumerate(imgs, start=1):
        print(f"\n[{i}/{len(imgs)}]")
        process_one_image(p)

    print("\n🎯 DONE")


if __name__ == "__main__":
    main()
