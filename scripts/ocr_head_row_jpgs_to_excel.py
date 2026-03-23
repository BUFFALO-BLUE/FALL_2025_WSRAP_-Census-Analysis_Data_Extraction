import os
import glob
import json
import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

# ---- What we want in the final table
TARGET_FIELDS = ["gender", "race", "owned_rented", "house_number", "price"]

# ---- Map filenames (stems) → canonical field names
FIELD_MAP = {
    "gender": "gender",
    "race": "race",
    "price": "price",
    "house_number": "house_number",
    "housenumber": "house_number",
    "house": "house_number",
    "number": "house_number",
    "owned": "owned_rented",
    "rented": "owned_rented",
    "own": "owned_rented",
    "rent": "owned_rented",
    "owned_rented": "owned_rented",
    "street": "street",
    # debug/visual row image (we skip OCR on this)
    "row": "row_image",
    # common alternate naming
    "rented_or_owned": "owned_rented",
    "owned_or_rented": "owned_rented",
}


def get_azure_client() -> DocumentIntelligenceClient:
    endpoint = os.environ.get("AZURE_DOCINTEL_ENDPOINT", "").strip()
    key = os.environ.get("AZURE_DOCINTEL_KEY", "").strip()

    if not endpoint or not key:
        raise RuntimeError(
            "Set AZURE_DOCINTEL_ENDPOINT and AZURE_DOCINTEL_KEY environment variables."
        )

    return DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )


def preprocess(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def ocr_image_azure(img_path: Path, client) -> Tuple[str, float]:
    """
    OCR one cropped cell image with Azure Document Intelligence.
    Returns:
        text, avg_confidence
    """
    processed = preprocess(img_path)

    ok, encoded = cv2.imencode(".png", processed)
    if not ok:
        raise RuntimeError(f"Could not encode processed image for Azure: {img_path}")

    data = encoded.tobytes()

    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=data,
        content_type="image/png",
    )
    result = poller.result()

    words = []
    confidences: List[float] = []

    for page in getattr(result, "pages", []) or []:
        for w in getattr(page, "words", []) or []:
            txt = (getattr(w, "content", "") or "").strip()
            if txt:
                words.append(txt)

            conf = getattr(w, "confidence", None)
            if conf is not None:
                try:
                    confidences.append(float(conf))
                except Exception:
                    pass

    text = " ".join(words).strip()
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    return text, avg_conf


def parse_meta(root: Path, img_path: Path) -> Dict[str, str]:
    """
    Expected:
      root/<Census_Image>/row_XX/<field>.png
    Example:
      head_rows_version1/m-t0627-.../row_32/gender.png
    """
    rel = img_path.relative_to(root)
    parts = rel.parts

    census_image = parts[0] if len(parts) >= 1 else ""
    row_folder = parts[1] if len(parts) >= 2 else ""
    field_stem = img_path.stem.lower()

    field = FIELD_MAP.get(field_stem, field_stem)
    return {"Census_Image": census_image, "Row": row_folder, "Field": field}


# ---- Normalizers
def normalize_gender(text: str) -> str:
    t = re.sub(r"[^A-Za-z]", "", text).upper()
    for ch in t:
        if ch in ("M", "F"):
            return ch
    return ""


def normalize_race(text: str) -> str:
    t = re.sub(r"[^A-Za-z]", "", text).upper()
    for ch in t:
        if ch in ("W", "B"):
            return ch
    return ""


def normalize_owned_rented(text: str) -> str:
    t = text.strip().lower()
    if "own" in t:
        return "Owned"
    if "rent" in t:
        return "Rented"

    letters = re.sub(r"[^A-Za-z]", "", t).upper()
    if "O" in letters and "R" not in letters:
        return "Owned"
    if "R" in letters and "O" not in letters:
        return "Rented"
    return ""


def normalize_digits(text: str) -> str:
    t = text.upper().strip()
    t = t.replace("O", "0")
    t = t.replace("I", "1")
    t = t.replace("L", "1")
    t = t.replace("S", "5")
    return re.sub(r"[^0-9]", "", t)


def normalize_street(text: str) -> str:
    t = text.strip()
    return t if t else "0"


def normalize(field: str, text: str) -> str:
    if field == "gender":
        return normalize_gender(text)
    if field == "race":
        return normalize_race(text)
    if field == "owned_rented":
        return normalize_owned_rented(text)
    if field in ("house_number", "price"):
        return normalize_digits(text)
    return text.strip()


def fmt_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def write_excel(
    out_path: Path,
    structured: Dict[Tuple[str, str], Dict[str, object]],
    raw_rows: List[Dict[str, object]],
):
    structured_df = pd.DataFrame(list(structured.values()))
    if not structured_df.empty:
        structured_df = structured_df.sort_values(["Census_Image", "Row"])
    raw_df = pd.DataFrame(raw_rows)

    # make conf columns nicer (no -1)
    for f in TARGET_FIELDS:
        c = f + "_conf"
        if c in structured_df.columns:
            structured_df[c] = structured_df[c].apply(
                lambda x: "" if float(x) < 0 else x
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        structured_df.to_excel(writer, sheet_name="Structured", index=False)
        raw_df.to_excel(writer, sheet_name="Raw_OCR_Debug", index=False)


def load_processed_keys_from_many_excels(output_xlsx: Path) -> set:
    """
    Load already-processed (Census_Image, Row, Field) keys from ALL relevant Excel files
    in the output directory (checkpoints, partials, master/output).
    Dedupes automatically.
    """
    out_dir = output_xlsx.parent

    patterns = [
        str(out_dir / (output_xlsx.stem + ".checkpoint*.xlsx")),
        str(out_dir / (output_xlsx.stem + "_checkpoint*.xlsx")),
        str(out_dir / (output_xlsx.stem + "*PARTIAL*.xlsx")),
        str(out_dir / (output_xlsx.stem + ".xlsx")),
        str(out_dir / (output_xlsx.stem + "_MASTER.xlsx")),
    ]

    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    processed_keys = set()

    for f in sorted(set(files)):
        try:
            df = pd.read_excel(f, sheet_name="Raw_OCR_Debug", engine="openpyxl")
            if not {"Census_Image", "Row", "Field"}.issubset(df.columns):
                print(f"⚠️ {Path(f).name}: missing required columns, skipping.")
                continue

            before = len(processed_keys)
            for _, r in df.iterrows():
                processed_keys.add(
                    (str(r["Census_Image"]), str(r["Row"]), str(r["Field"]))
                )
            added = len(processed_keys) - before

            print(f"✅ Loaded {added:,} keys from {Path(f).name}")

        except ValueError as e:
            print(f"⚠️ {Path(f).name}: {e} (skipping)")
        except Exception as e:
            print(f"⚠️ Could not read {Path(f).name}: {e} (skipping)")

    print(f"✅ Total unique processed keys found: {len(processed_keys):,}")
    return processed_keys


def main():
    parser = argparse.ArgumentParser(
        description="OCR head-row crops into a structured Excel table using Azure."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-xlsx", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Debug: process first N images.")

    # Progress + ETA controls
    parser.add_argument(
        "--progress-every", type=int, default=200, help="Print progress every N processed images."
    )
    parser.add_argument("--show-current", action="store_true", help="Also print current file name at progress updates.")
    parser.add_argument("--warmup", type=int, default=50, help="Don’t print ETA until this many images are processed.")

    # Stop after N minutes and write partial output
    parser.add_argument(
        "--max-minutes",
        type=int,
        default=300,
        help="Stop after N minutes (default 300 = 5 hours). Writes PARTIAL Excel if reached.",
    )

    # Checkpoint writes
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="If >0, write a checkpoint Excel every N processed images.",
    )
    parser.add_argument(
        "--checkpoint-xlsx",
        type=Path,
        default=None,
        help="Where to write checkpoint file (default: output name + .checkpoint.xlsx).",
    )

    # Speed/filters
    parser.add_argument(
        "--only-target-fields",
        action="store_true",
        help="If set, OCR only the TARGET_FIELDS (skips everything else).",
    )

    # Resume support
    parser.add_argument(
        "--resume-from-xlsx",
        type=Path,
        default=None,
        help="Resume from a previous checkpoint Excel (skips already processed Census_Image+Row+Field).",
    )

    args = parser.parse_args()

    client = get_azure_client()
    root = args.input_dir

    street_maps = {}

    # Try loading street_map.json for each page folder
    for page_dir in root.iterdir():
        if page_dir.is_dir():
            map_path = page_dir / "street_map.json"
            if map_path.exists():
                with open(map_path, "r", encoding="utf-8") as f:
                    street_maps[page_dir.name] = json.load(f)

    exts = {".png", ".jpg", ".jpeg"}
    images = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])

    if args.limit:
        images = images[: args.limit]

    if not images:
        raise RuntimeError(f"No images found under {root}")

    print(f"Found {len(images)} images under: {root}")
    print("Using Azure Document Intelligence OCR")

    structured: Dict[Tuple[str, str], Dict[str, object]] = {}
    raw_rows: List[Dict[str, object]] = []

    start_time = time.time()

    ckpt_path = args.checkpoint_xlsx
    if args.checkpoint_every and not ckpt_path:
        ckpt_path = args.output_xlsx.with_name(args.output_xlsx.stem + ".checkpoint.xlsx")

    processed_keys = set()
    processed_keys |= load_processed_keys_from_many_excels(args.output_xlsx)

    if args.resume_from_xlsx and args.resume_from_xlsx.exists():
        print(f"🔁 Also adding explicit resume file: {args.resume_from_xlsx}")
        try:
            prev_raw = pd.read_excel(
                args.resume_from_xlsx, sheet_name="Raw_OCR_Debug", engine="openpyxl"
            )
            if {"Census_Image", "Row", "Field"}.issubset(prev_raw.columns):
                for _, r in prev_raw.iterrows():
                    processed_keys.add(
                        (str(r["Census_Image"]), str(r["Row"]), str(r["Field"]))
                    )
                print(f"✅ After explicit resume, total keys: {len(processed_keys):,}")
            else:
                print("⚠️ Explicit resume file missing required columns; ignoring it.")
        except Exception as e:
            print(f"⚠️ Could not load explicit resume file: {e}")

    processed = 0

    def progress_line(i_scanned: int, current_name: str):
        now = time.time()
        elapsed = now - start_time

        if processed < max(1, args.warmup):
            msg = (
                f"Scanned {i_scanned}/{len(images)} | processed {processed} | "
                f"elapsed {fmt_seconds(elapsed)} | warming up..."
            )
        else:
            rate = elapsed / processed
            remaining = rate * (len(images) - i_scanned)
            msg = (
                f"Scanned {i_scanned}/{len(images)} | processed {processed} | "
                f"elapsed {fmt_seconds(elapsed)} | ETA {fmt_seconds(remaining)}"
            )

        if args.show_current:
            msg += f" | current: {current_name}"
        print(msg)

    try:
        for i_scanned, img_path in enumerate(images, 1):
            if args.max_minutes and (time.time() - start_time) > args.max_minutes * 60:
                print(f"⏹ Time limit reached ({args.max_minutes} min). Writing PARTIAL Excel...")
                partial = args.output_xlsx.with_name(args.output_xlsx.stem + "_PARTIAL.xlsx")
                write_excel(partial, structured, raw_rows)
                print(f"✅ Partial Excel written to: {partial}")
                return

            meta = parse_meta(root, img_path)
            field = meta["Field"]

            # Skip street; it comes from street_map.json
            if field == "street":
                continue

            # Skip row image
            if img_path.stem.lower() == "row" or field == "row_image":
                continue

            # Optional filter
            if args.only_target_fields and field not in TARGET_FIELDS:
                continue

            resume_key = (meta["Census_Image"], meta["Row"], meta["Field"])
            if resume_key in processed_keys:
                continue

            census_image = meta["Census_Image"]
            row_folder = meta["Row"]

            try:
                text, conf = ocr_image_azure(img_path, client)
            except Exception as e:
                text, conf = "OCR_ERROR", 0.0
                meta["Error"] = str(e)

            norm = normalize(field, text) if field in TARGET_FIELDS else text.strip()

            raw_rows.append(
                {
                    **meta,
                    "Image_Name": img_path.name,
                    "OCR_Text": text,
                    "Normalized": norm,
                    "Confidence": conf,
                    "Image_Path": str(img_path),
                }
            )

            if field in TARGET_FIELDS:
                key = (census_image, row_folder)

                if key not in structured:
                    structured[key] = {"Census_Image": census_image, "Row": row_folder}
                    structured[key]["street"] = ""

                    for f in TARGET_FIELDS:
                        structured[key][f] = ""
                        structured[key][f + "_conf"] = -1.0

                row_idx = int(row_folder.split("_")[-1])

                street_value = ""
                if census_image in street_maps:
                    street_value = street_maps[census_image].get(str(row_idx), "")

                structured[key]["street"] = street_value

                prev_conf = float(structured[key][field + "_conf"])
                if float(conf) >= prev_conf:
                    structured[key][field] = norm
                    structured[key][field + "_conf"] = float(conf)

            processed += 1

            if processed % args.progress_every == 0:
                progress_line(i_scanned, img_path.name)

            if args.checkpoint_every and ckpt_path and (processed % args.checkpoint_every == 0):
                print(
                    f"💾 Writing checkpoint at processed={processed} "
                    f"(scan {i_scanned}/{len(images)}) → {ckpt_path}"
                )
                try:
                    write_excel(ckpt_path, structured, raw_rows)
                    print("✅ Checkpoint written.")
                except Exception as e:
                    print(f"⚠️ Checkpoint write failed: {e}")

        progress_line(len(images), "(done)")
        write_excel(args.output_xlsx, structured, raw_rows)

        total_time = time.time() - start_time
        print(f"✅ Excel written to: {args.output_xlsx}")
        if processed > 0:
            print(
                f"🏁 Total runtime: {fmt_seconds(total_time)} for {processed} processed images "
                f"(~{(total_time/processed):.3f}s/img)"
            )
        else:
            print(
                f"🏁 Total runtime: {fmt_seconds(total_time)} "
                f"(processed 0 images — check filters/resume keys)."
            )

    except KeyboardInterrupt:
        print("🛑 Interrupted by user (Ctrl+C). Writing PARTIAL Excel now...")
        partial = args.output_xlsx.with_name(args.output_xlsx.stem + "_PARTIAL.xlsx")
        try:
            write_excel(partial, structured, raw_rows)
            print(f"✅ Partial Excel written to: {partial}")
        except Exception as e:
            print(f"⚠️ Could not write partial Excel: {e}")


if __name__ == "__main__":
    main()