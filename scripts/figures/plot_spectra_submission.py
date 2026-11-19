#!/usr/bin/env python3
"""
plot_spectra_submission.py

Four manuscript spectral figures.  Dark background; 3-column grid layout.
Each region occupies one row with ratio (col 0) and numerator (col 1) panels
side by side.  A third column holds overflow regions (e.g. Arcadia) or the
legend.  Spanning bold area titles sit above each row via fig.text().
"Ratio I/F" / "Numerator I/F" italic sub-labels appear inside each panel.
Legend is drawn into a dedicated axes cell.

Helper module for plot_spectra_r2.py (numerator collection and shared styling); the figures of the
originally submitted manuscript can still be drawn with it into output/.

Run (from the repository root):
    python scripts/figures/plot_spectra_submission.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

# ── shared infrastructure ─────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "psl5", _HERE / "plot_spectra_by_location_5.py")
psl5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(psl5)

OUT_DIR = _HERE.parent.parent / "output"
NEW_IDS = set(psl5.NEW_ID_LOCATIONS.keys())

# ── layout constants ──────────────────────────────────────────────────────────
COL_W   = 3.5    # inches per panel column
ROW_H   = 5.0    # inches per region row
WSPACE  = 0.45   # horizontal gap between columns  (fraction of avg col width)
HSPACE  = 0.65   # vertical gap between rows       (fraction of avg row height)
NUM_GAP = 0.08   # min gap between consecutive numerator traces

# ── palette / style ───────────────────────────────────────────────────────────
DARK_TEXT = "#eeeeee"   # primary text / fallback trace colour on dark bg
SPINE_CLR = "#888888"
GRID_CLR  = "#333333"
SUB_LBL   = "#aaaaaa"   # "Ratio I/F" / "Numerator I/F" label colour


def _setup_style():
    matplotlib.rcParams.update({
        "font.family":      "Helvetica Neue",
        "text.color":       DARK_TEXT,
        "axes.labelcolor":  DARK_TEXT,
        "axes.edgecolor":   SPINE_CLR,
        "axes.facecolor":   "#111111",
        "axes.grid":        True,
        "axes.grid.axis":   "y",
        "grid.color":       GRID_CLR,
        "grid.linewidth":   0.6,
        "xtick.color":      "#cccccc",
        "ytick.color":      "#cccccc",
        "xtick.labelsize":  9,
        "ytick.labelsize":  8,
        "figure.facecolor": "black",
        "legend.facecolor": "#222222",
        "legend.edgecolor": "#555555",
        "legend.fontsize":  8.5,
    })


def _color(stem):
    """Safe colour lookup."""
    return psl5.CAT_COLOR.get(stem, DARK_TEXT)


# ── helpers ───────────────────────────────────────────────────────────────────

def _poly_key(name):
    parts = name.split("_")
    poly  = next((p for p in parts if p.startswith("#")), "#0")
    return f"{parts[0]}_{poly}"


def _linestyle(crism_id):
    return "-" if crism_id in NEW_IDS else "--"


def _style_ax(ax, x_lo, x_hi, y_all):
    ax.set_xlim(x_lo - 0.01, x_hi + 0.01)
    y_lo = min(y_all) - psl5.MIN_GAP if y_all else 0.0
    y_hi = max(y_all) + psl5.MIN_GAP if y_all else 2.0
    major = np.arange(np.floor(y_lo / 0.5) * 0.5, y_hi + 0.5, 0.5)
    minor = np.arange(np.floor(y_lo / 0.1) * 0.1, y_hi + 0.1, 0.1)
    major = major[(major >= y_lo) & (major <= y_hi)]
    minor = minor[(minor >= y_lo) & (minor <= y_hi)]
    ax.set_yticks(major)
    ax.set_yticks(minor, minor=True)
    ax.set_yticklabels([f"{t:.1f}" for t in major], fontsize=8)
    ax.set_ylim(y_lo, y_hi)
    ax.tick_params(axis="y", which="major", length=4, color=SPINE_CLR)
    ax.tick_params(axis="y", which="minor", length=2, color=SPINE_CLR)
    ax.tick_params(axis="x", labelsize=9, color=SPINE_CLR)
    for spine in ax.spines.values():
        spine.set_color(SPINE_CLR)


# ── numerator loading ─────────────────────────────────────────────────────────

def collect_numerators(ratioed_ids):
    numerators = {}
    sd = psl5.SPECTRA_DIR
    for stem in psl5.UNRATIOED_STEMS:
        hdr = sd / f"{stem}.hdr"
        dat = sd / stem
        if not hdr.exists() or not dat.exists():
            continue
        meta = psl5.parse_header(hdr)
        wav  = meta["wavelengths"] / 1000.0
        for sp, name in zip(psl5.read_spectra(dat, meta), meta["spectra_names"]):
            crism_id = name.split("_")[0]
            if crism_id not in ratioed_ids:
                continue
            sp_c = psl5.correct_1um_offset(sp, wav)
            sp_n = psl5.normalize_spectrum(sp_c, wav, psl5.TIE_WAVELENGTH_UM)
            numerators[_poly_key(name)] = {"wav": wav, "sp": sp_n, "name": name}
    return numerators


def _ratioed_ids(by_location):
    return {r["name"].split("_")[0]
            for recs in by_location.values() for r in recs}


def _region_has_numerators(records, numerators):
    return any(numerators.get(_poly_key(r["name"])) is not None for r in records)


# ── core drawing: one region ──────────────────────────────────────────────────

def draw_area_pair(ax_ratio, ax_num, records, numerators,
                   show_ids=True, composite=True, slope_corr=None):
    """
    Draw ratio panel (ax_ratio) and, if ax_num is not None, numerator panel.
    Both panels use independent vertical stacking.
    Caller handles area title via _add_area_title().
    Returns list of category stems present.
    """
    def _sort(r):
        s = r["stem"]
        return (psl5.CAT_ORDER_STACK.index(s)
                if s in psl5.CAT_ORDER_STACK else len(psl5.CAT_ORDER_STACK))

    records = sorted(records, key=_sort)
    if slope_corr:
        records = [dict(r, sp=slope_corr(r["sp"], r["wav"])) for r in records]

    wav   = records[0]["wav"]
    mask  = (wav >= 0.40) & (wav <= 2.50)
    wav_m = wav[mask]

    any_valid = np.zeros(mask.sum(), dtype=bool)
    for rec in records:
        any_valid |= np.isfinite(rec["sp"][mask])
    x_lo = float(wav_m[any_valid][0])  if any_valid.any() else 0.40
    x_hi = float(wav_m[any_valid][-1]) if any_valid.any() else 2.50

    # ── ratio panel ──────────────────────────────────────────────────────────
    ratio_offsets = psl5.compute_dynamic_offsets(records, mask)
    used_stems    = []
    y_ratio       = []

    for rec, offset in zip(records, ratio_offsets):
        sp_off   = rec["sp"] + offset
        crism_id = rec["name"].split("_")[0]
        color    = _color(rec["stem"])
        ls       = _linestyle(crism_id) if composite else "-"

        ax_ratio.plot(wav[mask], sp_off[mask],
                      color=color, ls=ls, lw=psl5.LINE_WIDTH, alpha=0.90, zorder=2)
        if show_ids:
            valid = np.where(mask & np.isfinite(sp_off))[0]
            if valid.size:
                ax_ratio.text(x_hi + 0.02, float(sp_off[valid[-1]]), crism_id,
                              fontsize=7, color=color, va="center", ha="left",
                              zorder=3, family="monospace", clip_on=False)
        finite = sp_off[mask][np.isfinite(sp_off[mask])]
        if finite.size:
            y_ratio.extend(finite.tolist())
        if rec["stem"] not in used_stems:
            used_stems.append(rec["stem"])

    _style_ax(ax_ratio, x_lo, x_hi, y_ratio)
    ax_ratio.text(0.03, 0.97, "Ratio I/F",
                  transform=ax_ratio.transAxes,
                  fontsize=8, ha="left", va="top",
                  color=SUB_LBL, style="italic")

    if ax_num is None:
        return used_stems

    # ── numerator panel: independent stacking ────────────────────────────────
    num_recs = []
    for rec in records:
        num = numerators.get(_poly_key(rec["name"]))
        if num is not None:
            num_recs.append({"sp": num["sp"], "wav": num["wav"],
                             "name": rec["name"], "stem": rec["stem"]})

    y_num = []
    if num_recs:
        num_offsets = psl5.compute_dynamic_offsets(num_recs, mask, min_gap=NUM_GAP)
        for nr, offset in zip(num_recs, num_offsets):
            sp_off   = nr["sp"] + offset
            crism_id = nr["name"].split("_")[0]
            color    = _color(nr["stem"])
            ls       = _linestyle(crism_id) if composite else "-"
            ax_num.plot(wav[mask], sp_off[mask],
                        color=color, ls=ls, lw=psl5.LINE_WIDTH, alpha=0.90, zorder=2)
            if show_ids:
                valid_n = np.where(mask & np.isfinite(sp_off))[0]
                if valid_n.size:
                    ax_num.text(x_hi + 0.02, float(sp_off[valid_n[-1]]), crism_id,
                                fontsize=7, color=color, va="center", ha="left",
                                zorder=3, family="monospace", clip_on=False)
            finite_n = sp_off[mask][np.isfinite(sp_off[mask])]
            if finite_n.size:
                y_num.extend(finite_n.tolist())

    if y_num:
        _style_ax(ax_num, x_lo, x_hi, y_num)
    else:
        ax_num.set_xlim(x_lo - 0.01, x_hi + 0.01)
        ax_num.set_ylim(0, 1)
    ax_num.text(0.03, 0.97, "Numerator I/F",
                transform=ax_num.transAxes,
                fontsize=8, ha="left", va="top",
                color=SUB_LBL, style="italic")

    return used_stems


def _add_area_title(fig, ax_r, area_title, n_rec, ax_n=None):
    """Bold spanning area title in figure coordinates just above the row."""
    pos_r   = ax_r.get_position()
    x_left  = pos_r.x0
    x_right = ax_n.get_position().x1 if ax_n is not None else pos_r.x1
    fig.text((x_left + x_right) / 2, pos_r.y1 + 0.010,
             f"{area_title}  (n={n_rec})",
             ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=DARK_TEXT)


def _draw_legend_cell(ax, used_stems, include_ls=False):
    """Draw a self-contained legend into an axes cell and hide the frame."""
    ax.set_axis_off()
    handles = _color_handles(used_stems)
    if include_ls:
        handles += [mlines.Line2D([], [], color="none", label="")] + _ls_handles()
    ax.legend(handles=handles,
              loc="center left", bbox_to_anchor=(0.05, 0.5),
              fontsize=8.5, framealpha=0.6, edgecolor="#555555",
              handlelength=2.2, labelcolor=DARK_TEXT)


# ── legend handle builders ────────────────────────────────────────────────────

def _color_handles(stems):
    return [
        mlines.Line2D([], [], color=_color(s), lw=psl5.LINE_WIDTH, ls="-",
                      label=psl5.CATEGORIES[s].replace("\n", " "))
        for s in psl5.CAT_ORDER_STACK[::-1] if s in stems
    ]


def _ls_handles():
    return [
        mlines.Line2D([], [], color="#cccccc", lw=psl5.LINE_WIDTH, ls="-",
                      label="New detection (this study)"),
        mlines.Line2D([], [], color="#cccccc", lw=psl5.LINE_WIDTH, ls="--",
                      label="Previously reported"),
    ]


# ── axis labels ───────────────────────────────────────────────────────────────

def _xlabel(ax):
    ax.set_xlabel("Wavelength (µm)", fontsize=10, color=DARK_TEXT)


def _ylabel(ax):
    ax.set_ylabel("Normalized CRISM I/F (offset)", fontsize=10, color=DARK_TEXT)


# ── slope correction for Bond Crater ─────────────────────────────────────────

def slope_correct(sp, wav_um, wav_lo=0.52, wav_hi=2.45, frac=0.5):
    in_range = (wav_um >= wav_lo) & (wav_um <= wav_hi) & np.isfinite(sp)
    if in_range.sum() < 4:
        return sp
    order = np.argsort(wav_um[in_range])
    xr = wav_um[in_range][order]; yr = sp[in_range][order]
    n  = min(5, max(1, len(xr) // 8))
    x0, y0 = float(np.mean(xr[:n])),  float(np.mean(yr[:n]))
    x1, y1 = float(np.mean(xr[-n:])), float(np.mean(yr[-n:]))
    if x1 == x0:
        return sp
    line   = y0 + (y1 - y0) / (x1 - x0) * (wav_um - x0)
    result = sp.copy()
    v = np.isfinite(sp)
    with np.errstate(divide="ignore", invalid="ignore"):
        result[v] = (1 - frac) * sp[v] + frac * (sp[v] / line[v])
    return result


# ── figure builders ───────────────────────────────────────────────────────────

def make_n_hellas(by_location, numerators):
    """1 region; layout: ratio | numerator | legend  (1 × 3)."""
    records = by_location.get(3, [])
    has_num = _region_has_numerators(records, numerators)

    fig = plt.figure(figsize=(COL_W * 3, ROW_H + 0.8))
    gs  = GridSpec(1, 3, figure=fig, wspace=WSPACE,
                   left=0.09, right=0.97, top=0.88, bottom=0.13)

    ax_r = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[0, 1]) if has_num else None
    used = draw_area_pair(ax_r, ax_n, records, numerators,
                          show_ids=True, composite=False)
    _add_area_title(fig, ax_r, "N. Hellas / Tyrrhena Terra", len(records), ax_n)
    _ylabel(ax_r)
    _xlabel(ax_r)
    if ax_n is not None:
        _xlabel(ax_n)

    ax_leg = fig.add_subplot(gs[0, 2])
    _draw_legend_cell(ax_leg, used, include_ls=False)
    _save(fig, "spectra_n_hellas_tyrrhena_terra")


def make_terra_sirenum(by_location, numerators):
    """2 sub-regions; ratio-only rows + legend in bottom-right cell."""
    records  = by_location.get(9, [])
    recs_nd  = [r for r in records if r["stem"] == "nd"]
    recs_det = [r for r in records if r["stem"] != "nd"]
    sub_recs = [(recs_nd, "Terra Sirenum — No Detection"),
                (recs_det, "Terra Sirenum — Other Detections")]
    n_rows = len(sub_recs)

    # Check whether any sub-region has numerators to decide column count
    any_num = any(_region_has_numerators(recs, numerators) for recs, _ in sub_recs)
    n_cols  = 3 if any_num else 2   # col 0 ratio, col 1 num/empty, col 2 legend

    fig = plt.figure(figsize=(COL_W * n_cols, ROW_H * n_rows + 0.8))
    gs  = GridSpec(n_rows, n_cols, figure=fig, wspace=WSPACE, hspace=HSPACE,
                   left=0.09, right=0.97, top=0.93, bottom=0.07)

    all_used = []
    for row, (recs, title) in enumerate(sub_recs):
        if not recs:
            continue
        has_num = _region_has_numerators(recs, numerators)
        ax_r = fig.add_subplot(gs[row, 0])
        ax_n = fig.add_subplot(gs[row, 1]) if (has_num and any_num) else None
        used = draw_area_pair(ax_r, ax_n, recs, numerators,
                              show_ids=True, composite=False)
        _add_area_title(fig, ax_r, title, len(recs), ax_n)
        _ylabel(ax_r)
        _xlabel(ax_r)
        if ax_n is not None:
            _xlabel(ax_n)
        for s in used:
            if s not in all_used:
                all_used.append(s)

    ax_leg = fig.add_subplot(gs[n_rows - 1, n_cols - 1])
    _draw_legend_cell(ax_leg, all_used, include_ls=False)
    _save(fig, "spectra_terra_sirenum")


def make_ritchey_holden_bond(by_location, numerators):
    """4 regions × 3 cols; legend in [last_row, col 2]."""
    region_defs = [
        (7,  "Ritchey Crater",    None),
        (2,  "Holden Crater",     None),
        (13, "Bond Crater",       lambda sp, w: slope_correct(sp, w)),
        (10, "Valles Marineris",  None),
    ]
    n_rows = len(region_defs)

    fig = plt.figure(figsize=(COL_W * 3, ROW_H * n_rows + 0.8))
    gs  = GridSpec(n_rows, 3, figure=fig, wspace=WSPACE, hspace=HSPACE,
                   left=0.09, right=0.97, top=0.96, bottom=0.05)

    all_used = []
    for row, (code, name, corr) in enumerate(region_defs):
        records = by_location.get(code, [])
        has_num = _region_has_numerators(records, numerators)
        ax_r = fig.add_subplot(gs[row, 0])
        ax_n = fig.add_subplot(gs[row, 1]) if has_num else None
        used = draw_area_pair(ax_r, ax_n, records, numerators,
                              show_ids=True, composite=True, slope_corr=corr)
        _add_area_title(fig, ax_r, name, len(records), ax_n)
        _ylabel(ax_r)
        _xlabel(ax_r)
        if ax_n is not None:
            _xlabel(ax_n)
        for s in used:
            if s not in all_used:
                all_used.append(s)

    ax_leg = fig.add_subplot(gs[n_rows - 1, 2])
    _draw_legend_cell(ax_leg, all_used, include_ls=True)
    _save(fig, "spectra_ritchey_holden_bond")


def make_nili_noachis_arcadia(by_location, numerators):
    """
    3-row × 3-col layout:
      Row 0  Nili Patera:    [ratio | numerator | —      ]
      Row 1  Nili Fossae:    [ratio | numerator | Arcadia]
      Row 2  Noachis Terra:  [ratio | numerator | legend ]
    Arcadia (nd only, no numerators) fills [1, 2] so the grid stays full.
    """
    fig = plt.figure(figsize=(COL_W * 3, ROW_H * 3 + 0.8))
    gs  = GridSpec(3, 3, figure=fig, wspace=WSPACE, hspace=HSPACE,
                   left=0.09, right=0.97, top=0.96, bottom=0.05)

    all_used = []

    def _draw_row(row, code, title, corr=None):
        records = by_location.get(code, [])
        has_num = _region_has_numerators(records, numerators)
        ax_r = fig.add_subplot(gs[row, 0])
        ax_n = fig.add_subplot(gs[row, 1]) if has_num else None
        used = draw_area_pair(ax_r, ax_n, records, numerators,
                              show_ids=True, composite=True, slope_corr=corr)
        _add_area_title(fig, ax_r, title, len(records), ax_n)
        _ylabel(ax_r)
        _xlabel(ax_r)
        if ax_n is not None:
            _xlabel(ax_n)
        for s in used:
            if s not in all_used:
                all_used.append(s)
        return ax_r, ax_n

    _draw_row(0, 5,  "Nili Patera")
    _draw_row(1, 12, "Nili Fossae")
    _draw_row(2, 6,  "Noachis Terra")

    # Arcadia: ratio-only in [1, 2]
    recs_arc = by_location.get(1, [])
    ax_arc   = fig.add_subplot(gs[1, 2])
    used_arc = draw_area_pair(ax_arc, None, recs_arc, numerators,
                              show_ids=True, composite=True)
    _add_area_title(fig, ax_arc, "Arcadia Dome Field", len(recs_arc))
    _xlabel(ax_arc)
    for s in used_arc:
        if s not in all_used:
            all_used.append(s)

    # Legend in [2, 2]
    ax_leg = fig.add_subplot(gs[2, 2])
    _draw_legend_cell(ax_leg, all_used, include_ls=True)

    _save(fig, "spectra_nili_noachis_arcadia")


# ── save ──────────────────────────────────────────────────────────────────────

def _save(fig, slug):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{slug}.{ext}", bbox_inches="tight", dpi=150)
        print(f"Saved: {OUT_DIR}/{slug}.{ext}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    _setup_style()
    lookup      = psl5.load_spreadsheet(psl5.SPREADSHEET)
    conf_lookup = psl5.load_confidence_lookup(psl5.SPREADSHEET2)
    by_location = psl5.collect_all_spectra(lookup, conf_lookup)
    ratioed_ids = _ratioed_ids(by_location)
    numerators  = collect_numerators(ratioed_ids)
    print(f"Numerator spectra: {len(numerators)} entries, "
          f"{len({k.split('_')[0] for k in numerators})} unique CRISM IDs")
    make_n_hellas(by_location, numerators)
    make_terra_sirenum(by_location, numerators)
    make_ritchey_holden_bond(by_location, numerators)
    make_nili_noachis_arcadia(by_location, numerators)


if __name__ == "__main__":
    main()
