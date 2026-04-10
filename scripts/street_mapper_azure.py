import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time
import cv2

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential


DATASET_DIR = Path("data/training/head_rows_AzureTest")  # where Stage 1 wrote page folders


def get_client() -> DocumentIntelligenceClient:
    endpoint = os.environ.get("AZURE_DOCINTEL_ENDPOINT", "").strip()
    key = os.environ.get("AZURE_DOCINTEL_KEY", "").strip()
    if not endpoint or not key:
        raise RuntimeError(
            "Set AZURE_DOCINTEL_ENDPOINT and AZURE_DOCINTEL_KEY environment variables."
        )
    return DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def load_rows(rows_path: Path) -> List[Dict]:
    with open(rows_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    # expect list of {row_idx,y1,y2,y_center}
    return rows


def _polygon_to_xy(poly) -> List[Tuple[float, float]]:
    """
    Azure polygon sometimes comes as list of points with .x/.y,
    or list of dicts, or list of tuples.
    """
    pts = []
    if poly is None:
        return pts
    for p in poly:
        if hasattr(p, "x") and hasattr(p, "y"):
            pts.append((float(p.x), float(p.y)))
        elif isinstance(p, dict) and "x" in p and "y" in p:
            pts.append((float(p["x"]), float(p["y"])))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append((float(p[0]), float(p[1])))
    return pts


def _maybe_denormalize(pts: List[Tuple[float, float]], w: int, h: int) -> List[Tuple[float, float]]:
    if not pts:
        return pts
    max_x = max(x for x, _ in pts)
    max_y = max(y for _, y in pts)
    # Heuristic: if coords are ~0..1, treat as normalized
    if max_x <= 1.5 and max_y <= 1.5:
        return [(x * w, y * h) for x, y in pts]
    return pts


def azure_ocr_lines_with_ycenters(img_path: Path, client):
    """
    Returns: List[Tuple[text, y_center_px]]
    y_center is computed from the polygon.
    Handles polygon being either:
      - list of Point objects with .x/.y
      - flat list of numbers [x1,y1,x2,y2,...]
    """
    data = img_path.read_bytes()

    suf = img_path.suffix.lower()
    if suf == ".png":
        content_type = "image/png"
    elif suf in (".jpg", ".jpeg"):
        content_type = "image/jpeg"
    elif suf in (".tif", ".tiff"):
        content_type = "image/tiff"
    else:
        content_type = "application/octet-stream"

    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=data,
        content_type=content_type,
    )
    result = poller.result()

    # OPTIONAL: keep your full dump if you added it
    debug_full = img_path.with_name("street_azure_full.json")
    try:
        if hasattr(result, "as_dict"):
            with open(debug_full, "w", encoding="utf-8") as f:
                json.dump(result.as_dict(), f, indent=2)
        else:
            with open(debug_full, "w", encoding="utf-8") as f:
                f.write(repr(result))
    except Exception:
        pass

    def y_center_from_polygon(poly):
        if not poly:
            return None

        # Case 1: list of Point objects
        try:
            ys = []
            for p in poly:
                if hasattr(p, "y"):
                    ys.append(float(p.y))
            if ys:
                return float(sum(ys) / len(ys))
        except Exception:
            pass

        # Case 2: flat list [x1,y1,x2,y2,...]
        try:
            vals = list(poly)
            ys = [float(vals[i]) for i in range(1, len(vals), 2)]
            if ys:
                return float(sum(ys) / len(ys))
        except Exception:
            pass

        return None

    lines_out = []

    # Prefer lines if present
    for page in getattr(result, "pages", []) or []:
        for line in getattr(page, "lines", []) or []:
            text = (getattr(line, "content", "") or "").strip()
            if not text:
                continue

            poly = getattr(line, "polygon", None) or getattr(line, "bounding_polygon", None)
            yc = y_center_from_polygon(poly)

            # IMPORTANT: keep the line even if yc is None (rare), but we try to compute it
            if yc is None:
                continue

            lines_out.append((text, yc))

    # Fallback: if no lines, try words
    if not lines_out:
        for page in getattr(result, "pages", []) or []:
            for w in getattr(page, "words", []) or []:
                text = (getattr(w, "content", "") or "").strip()
                if not text:
                    continue
                poly = getattr(w, "polygon", None) or getattr(w, "bounding_polygon", None)
                yc = y_center_from_polygon(poly)
                if yc is None:
                    continue
                lines_out.append((text, yc))

    return lines_out

def get_extracted_row_indices(page_dir: Path) -> set:
    """
    Only keep rows that exist as folders: row_00, row_01, ...
    This matches your trigger logic (only important rows are saved).
    """
    rows = set()
    for p in page_dir.iterdir():
        if p.is_dir() and p.name.startswith("row_"):
            try:
                rows.add(int(p.name.split("_")[-1]))
            except Exception:
                pass
    return rows


def assign_lines_to_rows(lines: List[Tuple[str, float]], rows_meta: List[Dict], allowed_rows: set) -> Dict[str, str]:
    """
    Only return street values for rows that were actually extracted (allowed_rows).
    """
    rows_sorted = sorted(rows_meta, key=lambda r: float(r.get("y_center", 0.0)))

    buckets: Dict[int, List[str]] = {int(r["row_idx"]): [] for r in rows_sorted if int(r["row_idx"]) in allowed_rows}

    for text, yc in lines:
        yc = float(yc)

        hit = None
        for r in rows_sorted:
            ri = int(r["row_idx"])
            if ri not in allowed_rows:
                continue
            y1 = float(r["y1"])
            y2 = float(r["y2"])
            if y1 <= yc <= y2:
                hit = ri
                break

        if hit is None:
            # nearest allowed row center
            allowed_meta = [r for r in rows_sorted if int(r["row_idx"]) in allowed_rows]
            if not allowed_meta:
                break
            hit = int(min(allowed_meta, key=lambda r: abs(float(r["y_center"]) - yc))["row_idx"])

        buckets[hit].append(text)

    out: Dict[str, str] = {}
    for row_idx, parts in buckets.items():
        s = " ".join(parts).strip()
        s = " ".join(s.split())
        out[str(row_idx)] = s

    return out

def main():
    client = get_client()

    if not DATASET_DIR.exists():
        raise RuntimeError(f"DATASET_DIR not found: {DATASET_DIR}")

    page_dirs = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])

    print(f"Pages found: {len(page_dirs)}")
    for i, page in enumerate(page_dirs, 1):
        strip_path = page / "street_strip.png"
        rows_path = page / "rows.json"
        out_path = page / "street_map.json"

        if not strip_path.exists() or not rows_path.exists():
            continue  # skip pages that didn’t produce these

        

        print(f"[{i}/{len(page_dirs)}] {page.name}", flush=True)

        rows_meta = load_rows(rows_path)

        #  Heartbeat: tell us exactly where we're waiting
        print("   sending to Azure...", flush=True)
        lines = azure_ocr_lines_with_ycenters(strip_path, client)
        print("    Azure returned (writing debug + mapping rows)...", flush=True)

        debug_path = page / "street_azure_lines.json"
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump([{"text": t, "y_center": y} for t, y in lines], f, indent=2)
        print(f" returned {len(lines)} lines → {debug_path}", flush=True)

        allowed_rows = get_extracted_row_indices(page)
        street_map = assign_lines_to_rows(lines, rows_meta, allowed_rows)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(street_map, f, indent=2, ensure_ascii=False)

        print(f"  wrote {out_path.name} ({len(street_map)} rows)", flush=True)

        # Gentle throttle to reduce random stalls / rate-limits
        time.sleep(0.5)

    print(" DONE", flush=True)



if __name__ == "__main__":
    main()