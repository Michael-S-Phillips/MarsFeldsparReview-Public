#!/usr/bin/env python3
"""
extract_new_spectra.py

Extract ratioed CRISM spectra from MTRDR IF files using CSI ROI coordinates,
then write them into new ENVI spectral library files in spectra/.

New observations:
  - 11 delivery-004 detections (from CSIfiles_concattable_newplag.csv)
  - HRL0000985E (from CSIfiles_concattable_redoplag.csv)

Output:
  data/spectra/plag_confident_new.hdr + binary   -- new plagioclase spectra
  data/spectra/other_endmem_alteration_new.hdr + binary -- new alteration endmembers

Usage (from the repository root, with MTRDR cubes under $CRISM_MTRDR_ROOT):
  python scripts/extraction/extract_new_spectra.py
"""

import re
import sys
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent   # MarsFeldsparReview/
SPECTRA_DIR = _ROOT / "data" / "spectra"

NEWPLAG_CSV = _ROOT / "data" / "rois" / "csi_newplag_rois.csv"
REDOPLAG_CSV = _ROOT / "data" / "rois" / "csi_redoplag_rois.csv"

import os
MTRDR_ROOTS = [Path(os.environ.get("CRISM_MTRDR_ROOT", "mtrdr"))]   # folder holding <id>_07_if*_mtr3/ cubes

# Reference wavelength grid (used for resampling all new spectra)
REF_HDR = SPECTRA_DIR / "plag_confident.hdr"

IGNORE_VAL = 65535.0   # CRISM data-ignore pixel value

# ── Target observations ────────────────────────────────────────────────────────
# Maps (CRISM_ID_upper, roi_num) -> (output_stem, label)
# output_stem: 'plag_confident_new' or 'other_endmem_alteration_new'
TARGETS = {
    # Delivery-004 plagioclase (from newplag CSV)
    ("FRT0000406B", 1): ("plag_confident_new", "Plagioclase"),
    ("FRT0000406B", 2): ("plag_confident_new", "Plagioclase"),
    ("FRT0000474A", 1): ("plag_confident_new", "Plagioclase"),
    ("FRT00004F2F", 1): ("plag_confident_new", "Plagioclase"),
    ("FRT00005068", 1): ("plag_confident_new", "Plagioclase"),
    ("FRT0000580C", 1): ("plag_confident_new", "Plagioclase"),
    ("FRT0000580C", 2): ("plag_confident_new", "Plagioclase"),
    ("FRT00005A3E", 3): ("plag_confident_new", "Plagioclase"),
    ("FRT00005F23", 1): ("plag_confident_new", "Plagioclase"),
    ("FRT00006246", 1): ("plag_confident_new", "Plagioclase"),
    ("HRL0000407E", 1): ("plag_confident_new", "Plagioclase"),
    ("HRL000042D9", 1): ("plag_confident_new", "Plagioclase"),
    ("HRL000042D9", 2): ("plag_confident_new", "Plagioclase"),
    ("HRS00004421", 1): ("plag_confident_new", "Plagioclase"),
    # Delivery-004 alteration endmembers (kaolinite)
    ("FRT00005A3E", 1): ("other_endmem_alteration_new", "Kaolinite"),
    ("FRT00005A3E", 2): ("other_endmem_alteration_new", "Kaolinite"),
    ("FRT00005F23", 2): ("other_endmem_alteration_new", "Kaolinite"),
    # Redo-plag plagioclase
    ("HRL0000985E", 1): ("plag_confident_new", "Plagioclase"),
}

# ── ENVI helpers ───────────────────────────────────────────────────────────────

def parse_envi_hdr(hdr_path):
    """Return dict with bands, samples, lines, wavelengths (nm), dtype."""
    text = hdr_path.read_text(encoding="latin-1")

    def get(key):
        m = re.search(rf"^{key}\s*=\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
        return m.group(1).strip() if m else None

    samples    = int(get("samples"))
    lines      = int(get("lines"))
    bands      = int(get("bands"))
    data_type  = int(get("data type"))
    byte_order = int(get("byte order"))

    m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text, re.DOTALL | re.IGNORECASE)
    wavelengths = (np.array([float(w) for w in m.group(1).split(",") if w.strip()])
                   if m else np.array([]))

    dtype_map = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4"}
    endian = "<" if byte_order == 0 else ">"
    dtype  = np.dtype(endian + dtype_map[data_type])

    return {"samples": samples, "lines": lines, "bands": bands,
            "wavelengths": wavelengths, "dtype": dtype}


def extract_roi_via_index(if_img_path, in_img_path, if_meta, in_meta,
                          center_row, center_col, sz_row, sz_col):
    """Extract spectrum using sensor-space ROI coordinates via the _in*_mtr3 index file.

    The CSI CSV row/col values are in CRISM detector (sensor) space, not map-projected
    pixel space. The _in*_mtr3 file maps each map pixel back to its sensor-frame
    coordinates:
      band 5 (0-indexed) = IR Sample  (detector column, 1-indexed)
      band 6 (0-indexed) = IR Line    (detector row,    1-indexed)
    NODATA sentinel = 65535.

    We find all map-projected pixels whose sensor coordinates fall within the
    sensor-space ROI box, then average their I/F spectra.
    """
    B_if = if_meta["bands"]
    L, S = if_meta["lines"], if_meta["samples"]
    B_in = in_meta["bands"]
    hw_r = sz_row // 2
    hw_c = sz_col // 2

    # Load IR Sample (band idx 5) and IR Line (band idx 6) from IN cube
    in_data = np.memmap(in_img_path, dtype=in_meta["dtype"], mode="r").reshape(B_in, L, S)
    ir_samp = in_data[5].astype(np.float32)
    ir_line = in_data[6].astype(np.float32)

    # Boolean mask: sensor-space coordinates within ROI, excluding NODATA
    mask = (
        (ir_samp < IGNORE_VAL * 0.99) & (ir_line < IGNORE_VAL * 0.99) &
        (ir_line  >= center_row - hw_r) & (ir_line  <= center_row + hw_r) &
        (ir_samp  >= center_col - hw_c) & (ir_samp  <= center_col + hw_c)
    )

    n_pix = mask.sum()
    if n_pix == 0:
        return np.full(B_if, np.nan), 0

    # Extract matching map pixels from IF cube
    if_data = np.memmap(if_img_path, dtype=if_meta["dtype"], mode="r").reshape(B_if, L, S)
    pixels  = if_data[:, mask].astype(np.float64)          # shape (B_if, n_pix)
    pixels[(pixels >= IGNORE_VAL * 0.99) | (pixels <= 0)] = np.nan

    return np.nanmean(pixels, axis=1), n_pix


def find_mtrdr_if(crism_id):
    """Search MTRDR_ROOTS for the IF MTRDR file matching crism_id (case-insensitive).
    Returns (if_hdr, if_img, in_hdr, in_img) or (None, None, None, None).
    """
    id_lower = crism_id.lower()
    for root in MTRDR_ROOTS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            if id_lower in d.name.lower() and "_if" in d.name.lower():
                for f in sorted(d.iterdir()):
                    if (f.suffix == ".hdr"
                            and "_if" in f.name.lower()
                            and not f.name.startswith(".")):
                        if_img = f.with_suffix(".img")
                        if not if_img.exists():
                            if_img = f.with_suffix("")
                        if not if_img.exists():
                            continue
                        # Find companion _in file
                        in_hdr = f.parent / f.name.replace("_if", "_in")
                        in_img = if_img.parent / if_img.name.replace("_if", "_in")
                        if in_hdr.exists() and in_img.exists():
                            return f, if_img, in_hdr, in_img
    return None, None, None, None


def get_ref_wavelengths(hdr_path):
    """Read the reference library wavelength grid.
    The plag_confident library has a non-monotonic jump at the VNIR→IR boundary.
    Returns (wl_array, jump_idx) where jump_idx is the index of the backward jump.
    """
    text = hdr_path.read_text(encoding="latin-1")
    m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text, re.DOTALL | re.IGNORECASE)
    wl = np.array([float(w) for w in m.group(1).split(",") if w.strip()])
    wl = wl[wl < 10000]
    diffs = np.diff(wl)
    bad   = np.where(diffs < 0)[0]
    jump  = int(bad[0]) if len(bad) else len(wl)
    return wl, jump


def resample_to_ref(spectrum, src_wl, ref_wl, jump_idx):
    """Resample a spectrum from src_wl (monotonic) to ref_wl (possibly non-monotonic).
    Wavelengths outside src_wl range get NaN.
    """
    n = len(ref_wl)
    result = np.full(n, np.nan)
    valid  = np.isfinite(spectrum) & (spectrum > 0)
    if valid.sum() < 2:
        return result
    sw  = src_wl[valid]
    sp  = spectrum[valid]

    # Segment 1: ref bands 0..jump_idx (monotonically increasing)
    s1  = ref_wl[: jump_idx + 1]
    in1 = (s1 >= sw[0]) & (s1 <= sw[-1])
    if in1.any():
        result[: jump_idx + 1][in1] = np.interp(s1[in1], sw, sp)

    # Segment 2: ref bands jump_idx+1..end (monotonically increasing IR)
    if jump_idx + 1 < n:
        s2  = ref_wl[jump_idx + 1:]
        in2 = (s2 >= sw[0]) & (s2 <= sw[-1])
        if in2.any():
            result[jump_idx + 1:][in2] = np.interp(s2[in2], sw, sp)

    return result


def write_envi_library(out_stem, spectra, names, ref_wl):
    """Write an ENVI spectral library: {out_stem}.hdr and {out_stem} (binary)."""
    hdr_path  = SPECTRA_DIR / f"{out_stem}.hdr"
    data_path = SPECTRA_DIR / out_stem
    n_spec, n_bands = spectra.shape
    assert n_bands == len(ref_wl)

    spectra.astype("<f8").tofile(data_path)

    wl_str    = ",\n  ".join(f"{w:.6f}" for w in ref_wl)
    names_str = ", \n ".join(names)
    hdr  = (
        "ENVI\n"
        "description = {\n"
        "  New spectral library file}\n"
        f"samples = {n_bands}\n"
        f"lines   = {n_spec}\n"
        "bands   = 1\n"
        "header offset = 0\n"
        "file type = ENVI Spectral Library\n"
        "data type = 5\n"
        "interleave = bsq\n"
        "sensor type = Unknown\n"
        "byte order = 0\n"
        "wavelength units = Nanometers\n"
        "reflectance scale factor = 1.000000\n"
        "z plot titles = {, Value}\n"
        "band names = {\n"
        " Spectral Library}\n"
        f"spectra names = {{\n {names_str}}}\n"
        f"wavelength = {{\n  {wl_str}}}\n"
    )
    hdr_path.write_text(hdr, encoding="latin-1")
    print(f"  → wrote {n_spec} spectra to spectra/{out_stem}.hdr")


# ── CSV parsing ────────────────────────────────────────────────────────────────

def parse_csi_csv(csv_path, has_header):
    """Parse a CSI CSV into a dict keyed by (CRISM_ID_upper, roi_num).
    Value: dict with roion_row, roion_col, roioff_row, roioff_col, roisz_row, roisz_col.
    """
    records = {}
    with open(csv_path) as f:
        lines = f.readlines()

    start = 1 if has_header else 0
    for line in lines[start:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 14:
            continue
        try:
            roi_num     = int(parts[0])
            raw_id      = parts[1].strip()         # e.g. FRT0000406B_07
            crism_id    = raw_id.upper().split("_")[0]  # e.g. FRT0000406B
            roion_row   = int(float(parts[8]))
            roion_col   = int(float(parts[9]))
            roioff_row  = int(float(parts[10]))
            roioff_col  = int(float(parts[11]))
            roisz_row   = int(float(parts[12]))
            roisz_col   = int(float(parts[13]))
        except (ValueError, IndexError):
            continue
        key = (crism_id, roi_num)
        records[key] = {
            "roion_row":  roion_row,
            "roion_col":  roion_col,
            "roioff_row": roioff_row,
            "roioff_col": roioff_col,
            "roisz_row":  roisz_row,
            "roisz_col":  roisz_col,
        }
    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading reference wavelength grid …")
    ref_wl, jump_idx = get_ref_wavelengths(REF_HDR)
    print(f"  {len(ref_wl)} bands, {ref_wl[0]:.1f}–{ref_wl[-1]:.1f} nm  "
          f"(VNIR–IR jump at index {jump_idx})")

    print("\nParsing CSI CSV files …")
    newplag  = parse_csi_csv(NEWPLAG_CSV,  has_header=True)
    redoplag = parse_csi_csv(REDOPLAG_CSV, has_header=False)
    all_csi  = {**newplag, **redoplag}
    print(f"  {len(newplag)} ROIs in newplag,  {len(redoplag)} in redoplag")

    # Collect spectra per output library
    out_libs = {}   # stem -> list of (spectrum, name)

    print("\nExtracting spectra …")
    for (crism_id, roi_num), (out_stem, label) in sorted(TARGETS.items()):
        key = (crism_id, roi_num)
        if key not in all_csi:
            print(f"  [SKIP] {crism_id} #{roi_num}: not in CSI CSV")
            continue

        roi = all_csi[key]
        if_hdr, if_img, in_hdr, in_img = find_mtrdr_if(crism_id)
        if if_hdr is None:
            print(f"  [SKIP] {crism_id} #{roi_num}: MTRDR IF/IN files not found")
            continue

        if_meta = parse_envi_hdr(if_hdr)
        in_meta = parse_envi_hdr(in_hdr)
        wl      = if_meta["wavelengths"]

        sp_on,  n_on  = extract_roi_via_index(
            if_img, in_img, if_meta, in_meta,
            roi["roion_row"],  roi["roion_col"],
            roi["roisz_row"],  roi["roisz_col"])
        sp_off, n_off = extract_roi_via_index(
            if_img, in_img, if_meta, in_meta,
            roi["roioff_row"], roi["roioff_col"],
            roi["roisz_row"],  roi["roisz_col"])

        if n_on == 0 or n_off == 0:
            print(f"  [WARN] {crism_id} #{roi_num}: ON={n_on} OFF={n_off} map pixels found")

        # Ratio: ON / OFF
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(
                np.isfinite(sp_on) & np.isfinite(sp_off) & (sp_off > 0),
                sp_on / sp_off,
                np.nan
            )

        valid_count = np.sum(np.isfinite(ratio))
        if valid_count < 10:
            print(f"  [WARN] {crism_id} #{roi_num}: only {valid_count} valid ratio bands")

        # Resample to reference grid
        ratio_resampled = resample_to_ref(ratio, wl, ref_wl, jump_idx)

        name = f"{crism_id}_07_#{roi_num}_{label}"
        out_libs.setdefault(out_stem, []).append((ratio_resampled, name))
        print(f"  {name:50s}  (ON={n_on:3d} OFF={n_off:3d} px, "
              f"{valid_count:3d} valid bands → resampled)")

    print("\nWriting ENVI spectral libraries …")
    for out_stem, items in out_libs.items():
        spectra = np.stack([sp for sp, _ in items], axis=0)
        names   = [n for _, n in items]
        write_envi_library(out_stem, spectra, names, ref_wl)

    if not out_libs:
        print("  No spectra were extracted.")
    else:
        print(f"\nDone. {sum(len(v) for v in out_libs.values())} spectra written.")


if __name__ == "__main__":
    main()
