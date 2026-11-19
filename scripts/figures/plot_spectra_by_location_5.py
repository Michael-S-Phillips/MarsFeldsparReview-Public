"""
Plot ENVI spectral library files as stacked / offset spectra in a single
multi-panel figure — one subplot per location.

Layout:
  - Locations are arranged alphabetically in a grid (NCOLS columns).
  - The last panel is reserved for a shared legend.
  - Within each panel, spectra are grouped and ordered by spectral category
    (nd at the bottom, ratioed high-confidence at the top).
  - Trace color tracks spectral category / confidence level.
  - Panel titles are colored by location (Spectral colormap, alphabetical,
    skipping the extreme red and blue ends).
  - Panel height is proportional to the number of spectra it contains so
    that visual density is comparable across locations.
  - Offsets between spectra are computed dynamically: the minimum vertical
    gap between any two consecutive traces is always MIN_GAP, preventing
    spectra from crossing.

Output is saved as:
    christina_spectra_by_location.pdf  /  .png

Configurable parameters:
  TIE_WAVELENGTH_UM  – wavelength at which every spectrum is normalised to 1.0
  MIN_GAP            – minimum reflectance-unit gap between consecutive traces
  NCOLS              – number of columns in the panel grid
  LINE_WIDTH         – trace line width
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import re
from pathlib import Path

_ROOT        = Path(__file__).parent.parent.parent     # MarsFeldsparReview/
SPECTRA_DIR  = _ROOT / "data" / "spectra"
SPREADSHEET  = _ROOT / "data/detections/figure_location_codes.csv"           # location codes
SPREADSHEET2 = _ROOT / "data/detections/figure_confidence_categories.csv"    # figure categories

# ── SETTINGS ──────────────────────────────────────────────────────────────────
TIE_WAVELENGTH_UM = 1.0    # µm – normalise every spectrum to 1.0 here
MIN_GAP           = 0.05   # minimum I/F gap between consecutive traces
NCOLS             = 4      # columns in the panel grid
LINE_WIDTH        = 2.2    # trace line width

OUT_DIR_BY_LOC = _ROOT / "output" / "spectra_by_location"
OUT_DIR_BY_LOC_MS = _ROOT / "output" / "spectra_by_location"
# ─────────────────────────────────────────────────────────────────────────────

# When a CRISM ID has both a ratioed and an unratioed spectrum, keep only
# the ratioed one.
RATIOED_STEM    = "plag_confident"
UNRATIOED_STEMS = {
    "unratioed_plag_highconfidence",
    "unratioed_plag_moderateconfidence_w_2micron",
}

CATEGORIES = {
    # plag_confident is split at load time into _high / _moderate via spreadsheet
    "plag_confident_high":                         "Plag Ratioed\nHigh Confidence",
    "plag_confident_moderate":                     "Plag Ratioed\nMod. Confidence",
    "plag_confident":                              "Plag Ratioed\n(confidence unknown)",
    "unratioed_plag_highconfidence":               "Plag Unratioed\nHigh Confidence",
    "unratioed_plag_moderateconfidence_w_2micron": "Plag Unratioed\nMod. Confidence",
    "dusty_plag":                                  "Plag Ratioed\nLow Confidence",
    "icy_plag":                                    "Icy Plagioclase",
    "nd":                                          "No Detection",
    "other_endmem_mafic":                          "Other Endmember\nMafic",
    "other_endmem_alteration":                     "Other Endmember\nAlteration",
}

# Stacking order within each panel: bottom → top
CAT_ORDER_STACK = [
    "nd",
    "other_endmem_alteration",
    "other_endmem_mafic",
    "dusty_plag",
    "icy_plag",
    "unratioed_plag_moderateconfidence_w_2micron",
    "plag_confident_moderate",
    "plag_confident",                              # fallback (confidence unknown)
    "unratioed_plag_highconfidence",
    "plag_confident_high",
]

# Color per category — matched to the project's confidence color scheme:
#   green  = high confidence  |  yellow = moderate confidence
#   orange = low / no-MTRDR   |  grey   = no detection
#   red    = other endmember
CAT_COLOR = {
    "plag_confident_high":                         "#4CAF50",  # green  – ratioed high confidence
    "plag_confident_moderate":                     "#FFD700",  # yellow – ratioed moderate confidence
    "plag_confident":                              "#90EE90",  # light green fallback
    "unratioed_plag_highconfidence":               "#4CAF50",  # green  – unratioed high confidence
    "unratioed_plag_moderateconfidence_w_2micron": "#FFD700",  # yellow – unratioed moderate confidence
    "dusty_plag":                                  "#FF5722",  # orange-red – low confidence
    "icy_plag":                                    "#FF9800",  # orange – no MTRDR
    "nd":                                          "#9E9E9E",  # grey   – no detection
    "other_endmem_mafic":                          "#F44336",  # red        – other endmember mafic
    "other_endmem_alteration":                     "#CC8899",  # puce       – other endmember alteration
}

# Location code → display name
LOCATIONS = {
    0:  "Not in table",
    1:  "Arcadia Dome Field",
    2:  "Holden Crater",
    3:  "N. Hellas / Tyrrhena Terra",
    4:  "Negril Crater",
    5:  "Nili Patera",
    6:  "Noachis Terra",
    7:  "Ritchey Crater",
    8:  "Terra Sabaea",
    9:  "Terra Sirenum",
    10: "Valles Marineris",
    11: "Xanthe Terra",
    12: "Nili Fossae",
    13: "Bond Crater",
    14: "South Isidis Basin Highlands",
}

# Detections removed from the paper (2026-10-21). HRL0000407E (Bond crater): denominator shows a
# pyroxene-like 2-um band and nearby pixels show a 2.3-um alteration band; fails the confidence criteria.
EXCLUDED_IDS = {"HRL0000407E"}

# Hardcoded location codes for delivery-004 new detections (supplement spreadsheet)
NEW_ID_LOCATIONS = {
    "FRT0000406B": 12, "FRT00005A3E": 12,
    "FRT0000474A": 2,  "FRT00004F2F": 2, "FRT00006246": 2, "FRT0000580C": 2,
    "FRT00005068": 3,  "HRL000042D9": 6,
    "HRS00004421": 10, "FRT00005F23": 10,
    "HRL0000985E": 11,
    "FRT00018FF2": 14,   # South Isidis Basin Highlands (uncertain plag, 87°E -1.8°N)
    "FRT00004185": 5,    # Nili Patera (plagioclase, Eggers 2021)
}

# Hardcoded confidence for delivery-004 new detections
NEW_ID_CONFIDENCE = {
    "FRT0000406B": "plag_confident_high",
    "FRT0000474A": "plag_confident_high",
    "FRT00004F2F": "plag_confident_high",
    "FRT00005068": "plag_confident_high",
    "FRT00005A3E": "plag_confident_high",
    "HRS00004421": "plag_confident_high",
    "FRT00006246": "plag_confident_moderate",
    "FRT0000580C": "plag_confident_moderate",
    "FRT00005F23": "plag_confident_moderate",
    "HRL000042D9": "plag_confident_moderate",
    "FRT00018FF2": "plag_confident_high",   # South Isidis Basin Highlands
    "FRT00004185": "plag_confident_moderate",   # Nili Patera (Eggers 2021)
    # Classes from C. Viviano's previous_and_new_plag.xlsx (2026-10-20), which override
    # "High" in previous_plag_table.xlsx
    "FRT00007F63": "plag_confident_moderate",
    "HRL0000A8F6": "plag_confident_moderate",
    "HRL0000638C": "plag_confident_moderate",
    # HRL0000985E: no confidence in original data, falls back to plain plag_confident
}

# Additional library files to merge into each stem
EXTRA_STEM_FILES = {
    "plag_confident":         ["plag_confident_new"],
    "other_endmem_alteration": ["other_endmem_alteration_new"],
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_confidence_lookup(path):
    """
    Returns a dict  CRISM_ID → stem_suffix  where suffix is one of:
      'plag_confident_high'     (High confidence)
      'plag_confident_moderate' (Moderate confidence)
    IDs not found or with other confidence values are not included
    (callers fall back to plain 'plag_confident').
    """
    mapping = dict(NEW_ID_CONFIDENCE)   # seed with hardcoded delivery-004 entries
    if not Path(path).exists():
        return mapping
    import csv
    for row in csv.DictReader(open(path)):   # data/detections/figure_confidence_categories.csv
        mapping.setdefault(row["crism_id"], row["figure_category"])
    return mapping


def load_spreadsheet(path):
    lookup = dict(NEW_ID_LOCATIONS)     # seed with hardcoded delivery-004 entries
    import csv
    for row in csv.DictReader(open(path)):   # data/detections/figure_location_codes.csv
        lookup.setdefault(row["crism_id"], int(row["location_code"]))
    return lookup


def parse_header(hdr_path):
    text = hdr_path.read_text(encoding="latin-1")

    def get_value(key):
        m = re.search(rf"^{key}\s*=\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
        return m.group(1).strip() if m else None

    samples    = int(get_value("samples"))
    lines      = int(get_value("lines"))
    data_type  = int(get_value("data type"))
    byte_order = int(get_value("byte order"))

    def parse_brace_list(key):
        m = re.search(rf"{key}\s*=\s*\{{([^}}]*)\}}", text, re.DOTALL | re.IGNORECASE)
        if not m:
            return []
        return [s.strip().strip('"') for s in m.group(1).split(",") if s.strip()]

    wavelengths   = np.array([float(w) for w in parse_brace_list("wavelength")])
    wavelengths   = wavelengths[wavelengths < 10000]
    spectra_names = parse_brace_list("spectra names")

    dtype_map = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4"}
    endian = "<" if byte_order == 0 else ">"
    dtype  = np.dtype(endian + dtype_map[data_type])

    return {"samples": samples, "lines": lines, "dtype": dtype,
            "wavelengths": wavelengths, "spectra_names": spectra_names}


def read_spectra(data_path, meta):
    raw  = np.fromfile(data_path, dtype=meta["dtype"])
    data = raw.reshape((meta["lines"], meta["samples"]))
    data = np.where(np.abs(data) > 1e30, np.nan, data)
    return data[:, :len(meta["wavelengths"])]


def correct_1um_offset(sp: np.ndarray, wav_um: np.ndarray,
                        overlap_lo: float = 1.015, overlap_hi: float = 1.050,
                        min_offset: float = 0.005) -> np.ndarray:
    """
    Correct the S/L detector seam at the VNIR/IR junction (~1 µm).

    The 544-band reference library has a non-monotonic wavelength grid:
      VNIR segment: indices 0..jump-1  (~364–1056 nm, S-detector)
      IR segment:   indices jump..end  (~1001–3937 nm, L-detector)

    Both segments cover the 1015–1050 nm overlap.  We interpolate the IR to
    the VNIR wavelengths in that range and compute offset = median(VNIR − IR).
    The VNIR segment is then shifted down by that offset, aligning the two
    detectors.  Corrections smaller than min_offset are skipped.
    """
    from scipy.interpolate import interp1d

    # Find the non-monotonic jump (where wavelength decreases)
    jump = len(wav_um)
    for i in range(len(wav_um) - 1):
        if wav_um[i + 1] < wav_um[i]:
            jump = i + 1
            break
    if jump == len(wav_um):
        return sp  # monotonic grid — no correction needed

    vnir_wl = wav_um[:jump]
    ir_wl   = wav_um[jump:]
    vnir_sp = sp[:jump]
    ir_sp   = sp[jump:]

    # Select bands in the overlap window from each segment
    v_mask = (vnir_wl >= overlap_lo) & (vnir_wl <= overlap_hi) & np.isfinite(vnir_sp)
    i_mask = (ir_wl   >= overlap_lo) & (ir_wl   <= overlap_hi) & np.isfinite(ir_sp)

    if v_mask.sum() < 2 or i_mask.sum() < 2:
        return sp

    # Interpolate IR values to the VNIR wavelengths
    f_ir = interp1d(ir_wl[i_mask], ir_sp[i_mask],
                    kind="linear", bounds_error=False, fill_value=np.nan)
    ir_at_vnir = f_ir(vnir_wl[v_mask])

    valid = np.isfinite(ir_at_vnir)
    if valid.sum() < 2:
        return sp

    offset = float(np.median(vnir_sp[v_mask][valid] - ir_at_vnir[valid]))
    if abs(offset) < min_offset:
        return sp

    corrected = sp.copy()
    corrected[:jump] -= offset
    return corrected


def normalize_spectrum(sp, wav_um, tie_wl):
    tie_idx = np.argmin(np.abs(wav_um - tie_wl))
    tie_val = sp[tie_idx]
    if np.isnan(tie_val) or tie_val == 0:
        finite  = sp[np.isfinite(sp)]
        tie_val = np.nanmedian(finite) if len(finite) else 1.0
    return sp / tie_val


# ── data loading ──────────────────────────────────────────────────────────────

def _load_library(stem, effective_stem, lookup, conf_lookup, ratioed_ids, by_location):
    """Load one ENVI library file into by_location. effective_stem is the CATEGORIES key."""
    hdr = SPECTRA_DIR / f"{stem}.hdr"
    dat = SPECTRA_DIR / stem
    if not hdr.exists() or not dat.exists():
        return

    meta    = parse_header(hdr)
    wav     = meta["wavelengths"] / 1000.0
    spectra = read_spectra(dat, meta)
    names   = meta["spectra_names"]

    for sp, name in zip(spectra, names):
        crism_id = name.split("_")[0]

        if effective_stem in UNRATIOED_STEMS and crism_id in ratioed_ids:
            continue

        cat = effective_stem
        if effective_stem == RATIOED_STEM and conf_lookup:
            cat = conf_lookup.get(crism_id, effective_stem)

        code = lookup.get(crism_id, 0)
        if code not in by_location:
            by_location[code] = []
        sp_corr = correct_1um_offset(sp, wav)
        by_location[code].append({
            "wav":  wav,
            "sp":   normalize_spectrum(sp_corr, wav, TIE_WAVELENGTH_UM),
            "stem": cat,
            "name": name,
        })


def collect_all_spectra(lookup, conf_lookup=None):
    # First pass: CRISM IDs that have a ratioed spectrum (main + supplement)
    ratioed_ids = set()
    for rstem in [RATIOED_STEM] + EXTRA_STEM_FILES.get(RATIOED_STEM, []):
        _hdr = SPECTRA_DIR / f"{rstem}.hdr"
        _dat = SPECTRA_DIR / rstem
        if _hdr.exists() and _dat.exists():
            _meta = parse_header(_hdr)
            for _name in _meta["spectra_names"]:
                ratioed_ids.add(_name.split("_")[0])

    by_location = {code: [] for code in LOCATIONS}

    for stem in CATEGORIES:
        _load_library(stem, stem, lookup, conf_lookup, ratioed_ids, by_location)
        for extra in EXTRA_STEM_FILES.get(stem, []):
            _load_library(extra, stem, lookup, conf_lookup, ratioed_ids, by_location)

    return by_location


# ── drawing ───────────────────────────────────────────────────────────────────

def compute_dynamic_offsets(records, mask, min_gap=MIN_GAP):
    """
    Tight-packing offset: for each new spectrum, find the vertical shift that
    brings it just into contact with the one below it (minimum gap = 0 at the
    closest wavelength), then back off by min_gap.

    At every valid wavelength j shared by spectrum i and i+1:
        gap(j) = (sp_{i+1}[j] + off_{i+1}) − (sp_i[j] + off_i)
    We want min_j(gap) = min_gap, which gives:
        off_{i+1} = off_i − min_j(sp_{i+1}[j] − sp_i[j]) + min_gap

    Falls back to the max/min approach when there is no valid wavelength overlap.
    """
    offsets = [0.0]
    for i in range(1, len(records)):
        sp_lo = records[i - 1]["sp"][mask]
        sp_hi = records[i]["sp"][mask]
        both  = np.isfinite(sp_lo) & np.isfinite(sp_hi)
        if both.any():
            diff   = sp_hi[both] - sp_lo[both]
            offset = offsets[i - 1] - float(np.min(diff)) + min_gap
        else:
            offset = offsets[i - 1] + float(np.nanmax(sp_lo)) - float(np.nanmin(sp_hi)) + min_gap
        offsets.append(offset)
    return offsets


def draw_location_panel(ax, code, records, show_crism_ids=False):
    """
    Draw stacked spectra for one location into the supplied Axes.
    Returns the list of category stems that appear in this panel.

    If show_crism_ids is True, place each CRISM ID immediately to the right of
    the end of its spectrum trace (outside the axes, clip_on=False).
    """
    loc_name = LOCATIONS.get(code, f"Code {code}")

    def sort_key(r):
        stem = r["stem"]
        return CAT_ORDER_STACK.index(stem) if stem in CAT_ORDER_STACK else len(CAT_ORDER_STACK)
    records = sorted(records, key=sort_key)

    wav  = records[0]["wav"]
    mask = (wav >= 0.40) & (wav <= 2.50)

    # Crop x-axis to actual valid data extent across all spectra
    wav_m = wav[mask]
    any_valid = np.zeros(mask.sum(), dtype=bool)
    for rec in records:
        any_valid |= np.isfinite(rec["sp"][mask])
    x_lo = float(wav_m[any_valid][0])  if any_valid.any() else 0.40
    x_hi = float(wav_m[any_valid][-1]) if any_valid.any() else 2.50

    offsets    = compute_dynamic_offsets(records, mask)
    used_stems = []
    y_all      = []

    for rec, offset in zip(records, offsets):
        sp     = rec["sp"]
        sp_off = sp + offset
        color  = CAT_COLOR.get(rec["stem"], "white")

        ax.plot(wav[mask], sp_off[mask],
                color=color, linestyle="-", alpha=0.75, lw=LINE_WIDTH, zorder=2)

        if show_crism_ids:
            crism_id = rec["name"].split("_")[0]
            valid = np.where(mask & np.isfinite(sp_off))[0]
            y_label = float(sp_off[valid[-1]]) if valid.size else offset + 1.0
            ax.text(x_hi + 0.02, y_label, crism_id,
                    fontsize=9, color=color, va="center", ha="left",
                    alpha=0.90, zorder=3, family="monospace", clip_on=False)

        finite_off = sp_off[mask][np.isfinite(sp_off[mask])]
        if finite_off.size:
            y_all.extend(finite_off.tolist())

        if rec["stem"] not in used_stems:
            used_stems.append(rec["stem"])

    ax.set_xlim(x_lo - 0.01, x_hi + 0.01)
    ax.tick_params(axis="x", labelsize=11)

    # Y-axis: tight to actual data; ticks set first, then ylim so ticks don't expand it
    y_lo = min(y_all) - MIN_GAP if y_all else 0.0
    y_hi = max(y_all) + MIN_GAP if y_all else 2.0
    major_ticks = np.arange(np.floor(y_lo / 0.5) * 0.5, y_hi + 0.5, 0.5)
    minor_ticks = np.arange(np.floor(y_lo / 0.1) * 0.1, y_hi + 0.1, 0.1)
    # Filter to range before setting so matplotlib doesn't expand ylim
    major_ticks = major_ticks[(major_ticks >= y_lo) & (major_ticks <= y_hi)]
    minor_ticks = minor_ticks[(minor_ticks >= y_lo) & (minor_ticks <= y_hi)]
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_yticklabels([f"{t:.1f}" for t in major_ticks], fontsize=10)
    ax.set_ylim(y_lo, y_hi)
    ax.tick_params(axis="y", which="major", length=5)
    ax.tick_params(axis="y", which="minor", length=2)
    ax.set_ylabel("Normalized CRISM I/F (offset)", fontsize=11)

    n_spec  = len(records)
    n_label = f"  (n={n_spec})" if n_spec > 3 else ""
    ax.set_title(f"{loc_name}{n_label}", fontsize=13, fontweight="bold",
                 pad=6, color="white")

    return used_stems, y_lo, y_hi


def draw_legend_panel(ax, all_used_stems):
    """Shared legend: category color + line-style swatches."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Legend", fontsize=8.5, fontweight="bold", pad=4)

    cat_present = [s for s in CAT_ORDER_STACK[::-1] if s in all_used_stems]
    row_h = 0.08   # fixed compact spacing regardless of panel height
    y     = 0.97

    ax.text(0.04, y, "Spectral category", fontsize=7.5, fontweight="bold",
            va="top", transform=ax.transAxes)
    y -= row_h

    for stem in cat_present:
        color = CAT_COLOR.get(stem, "white")
        ls    = "-"
        label = CATEGORIES[stem].replace("\n", " ")
        mid_y = y - row_h * 0.38
        line  = mlines.Line2D([0.04, 0.16], [mid_y, mid_y],
                               linestyle=ls, color=color, lw=1.5,
                               transform=ax.transAxes, clip_on=False)
        ax.add_line(line)
        ax.text(0.20, mid_y, label, fontsize=6.8, color="white",
                va="center", ha="left", transform=ax.transAxes)
        y -= row_h


# ── figure assembly ───────────────────────────────────────────────────────────

def make_figure():
    plt.style.use("dark_background")

    lookup      = load_spreadsheet(SPREADSHEET)
    conf_lookup = load_confidence_lookup(SPREADSHEET2)
    by_location = collect_all_spectra(lookup, conf_lookup)

    # Alphabetical order, skip empty locations
    named_sorted = sorted(
        [(code, name) for code, name in LOCATIONS.items()
         if code != 0 and by_location.get(code)],
        key=lambda x: x[1],
    )
    if by_location.get(0):
        named_sorted.append((0, LOCATIONS[0]))

    # Swap N. Hellas / Tyrrhena Terra with Xanthe Terra so the location with
    # many spectra lands in a row that can accommodate its height.
    def _swap(lst, name_a, name_b):
        ia = next((i for i, (_, n) in enumerate(lst) if n == name_a), None)
        ib = next((i for i, (_, n) in enumerate(lst) if n == name_b), None)
        if ia is not None and ib is not None:
            lst[ia], lst[ib] = lst[ib], lst[ia]
    _swap(named_sorted, "N. Hellas / Tyrrhena Terra", "Xanthe Terra")

    n_loc    = len(named_sorted)
    n_panels = n_loc + 1          # +1 legend
    ncols    = NCOLS
    nrows    = (n_panels + ncols - 1) // ncols

    # n_spectra per panel slot (legend gets the row-max so it isn't tiny)
    n_spec_list = [len(by_location[code]) for code, _ in named_sorted]

    # Height ratio per row = max n_spectra in that row
    row_heights = []
    for row in range(nrows):
        slot_counts = []
        for col in range(ncols):
            idx = row * ncols + col
            if idx < n_loc:
                slot_counts.append(n_spec_list[idx])
            elif idx == n_loc:          # legend slot
                slot_counts.append(max(n_spec_list))
        row_heights.append(max(slot_counts) if slot_counts else 1)

    # Scale figure height: ~0.10 inches per spectrum unit, min 1.5 in per row
    per_unit    = 0.10
    fig_h       = sum(max(1.5, h * per_unit) for h in row_heights) + 1.0
    fig_w       = 5.0 * ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": row_heights},
    )
    axes = axes.flatten()

    for ax in axes:
        ax.set_visible(False)

    all_used_stems = []
    for panel_idx, (code, _) in enumerate(named_sorted):
        ax      = axes[panel_idx]
        records = by_location[code]
        ax.set_visible(True)
        used, *_ = draw_location_panel(ax, code, records)
        for s in used:
            if s not in all_used_stems:
                all_used_stems.append(s)

    # x-axis label on bottom-row panels only
    bottom_start = (nrows - 1) * ncols
    for panel_idx in range(n_loc):
        if panel_idx >= bottom_start:
            axes[panel_idx].set_xlabel("Wavelength (μm)", fontsize=8)

    # Legend panel
    leg_ax = axes[n_loc]
    leg_ax.set_visible(True)
    draw_legend_panel(leg_ax, all_used_stems)

    fig.suptitle(
        "CRISM Plagioclase Spectra by Location\n"
        f"[normalized at {TIE_WAVELENGTH_UM:.2f} µm · min gap = {MIN_GAP}]",
        fontsize=12, fontweight="bold", y=0.99,
    )
    plt.tight_layout()

    for ext in ("pdf", "png"):
        out = Path(f"christina_spectra_by_location.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")

    plt.show()


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _legend_handles(used_stems):
    return [
        mlines.Line2D([], [], color=CAT_COLOR.get(stem, "white"),
                      lw=LINE_WIDTH, label=CATEGORIES[stem].replace("\n", " "))
        for stem in CAT_ORDER_STACK[::-1] if stem in used_stems
    ]


def make_per_location_figures():
    """One (or two-column) portrait figure per location, saved to OUT_DIR_BY_LOC."""
    plt.style.use("dark_background")
    OUT_DIR_BY_LOC.mkdir(exist_ok=True)
    OUT_DIR_BY_LOC_MS.mkdir(parents=True, exist_ok=True)

    lookup      = load_spreadsheet(SPREADSHEET)
    conf_lookup = load_confidence_lookup(SPREADSHEET2)
    by_location = collect_all_spectra(lookup, conf_lookup)

    named_sorted = sorted(
        [(code, name) for code, name in LOCATIONS.items()
         if code != 0 and by_location.get(code)],
        key=lambda x: x[1],
    )

    SCALE     = 8.0   # inches per reflectance unit of data span
    PAD       = 2.0   # fixed inches for title/xlabel/margins
    FIG_W     = 4.0   # single-column width
    FIG_COL_W = 3.5   # per-column width in two-column figures

    # Two-column splits: code → (left_predicate, left_subtitle, right_subtitle)
    TWO_COL = {
        9:  (lambda r: r["stem"] == "nd",
             "No Detection", "Other Detections"),
        10: (lambda r: r["stem"] == "plag_confident_moderate",
             "Plagioclase", "Other Endmember"),
    }

    for code, loc_name in named_sorted:
        records = by_location[code]
        slug    = _slugify(loc_name)

        if code in TWO_COL:
            pred_l, sub_l, sub_r = TWO_COL[code]
            recs_l = [r for r in records if pred_l(r)]
            recs_r = [r for r in records if not pred_l(r)]

            fig, (ax_l, ax_r) = plt.subplots(
                1, 2, figsize=(FIG_COL_W * 2, 10),
                gridspec_kw={"wspace": 0.45},
            )

            used_l, y_lo_l, y_hi_l = draw_location_panel(ax_l, code, recs_l, show_crism_ids=True)
            ax_l.set_title(f"{loc_name}\n{sub_l}", fontsize=12, fontweight="bold",
                           pad=6, color="white")
            ax_l.set_xlabel("Wavelength (μm)", fontsize=12)
            # Extend right xlim so CRISM ID labels land inside this panel's box,
            # not in the inter-panel gap where they'd overlap the right panel's y-ticks
            xl, xr = ax_l.get_xlim()
            ax_l.set_xlim(xl, xr + 0.4)

            used_r, y_lo_r, y_hi_r = draw_location_panel(ax_r, code, recs_r, show_crism_ids=True)
            ax_r.set_title(f"{loc_name}\n{sub_r}", fontsize=12, fontweight="bold",
                           pad=6, color="white")
            ax_r.set_xlabel("Wavelength (μm)", fontsize=12)
            ax_r.set_ylabel("")

            y_span = max(y_hi_l - y_lo_l, y_hi_r - y_lo_r)
            fig_h  = min(20.0, max(FIG_COL_W * 1.8, y_span * SCALE + PAD))
            fig.set_size_inches(FIG_COL_W * 2, fig_h)

            all_used = used_l + [s for s in used_r if s not in used_l]
            handles  = _legend_handles(all_used)
            if handles:
                fig.legend(handles=handles, fontsize=10,
                           bbox_to_anchor=(0.5, 0.06), loc="upper center",
                           framealpha=0.6, edgecolor="0.4", ncol=1)

            plt.tight_layout()

        else:
            fig, ax = plt.subplots(1, 1, figsize=(FIG_W, 10))
            used_stems, y_lo, y_hi = draw_location_panel(ax, code, records, show_crism_ids=True)

            y_span = y_hi - y_lo
            fig_h  = min(20.0, max(FIG_W * 1.8, y_span * SCALE + PAD))
            fig.set_size_inches(FIG_W, fig_h)

            ax.set_xlabel("Wavelength (μm)", fontsize=12)

            handles = _legend_handles(used_stems)
            if handles:
                ax.legend(handles=handles, fontsize=10,
                          bbox_to_anchor=(0.5, -0.08), loc="upper center",
                          framealpha=0.6, edgecolor="0.4", ncol=1)

            plt.tight_layout()

        for out_dir in (OUT_DIR_BY_LOC, OUT_DIR_BY_LOC_MS):
            for ext in ("png", "svg", "pdf"):
                out = out_dir / f"spectra_{slug}.{ext}"
                fig.savefig(out, bbox_inches="tight", dpi=150)
                print(f"Saved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    make_per_location_figures()
