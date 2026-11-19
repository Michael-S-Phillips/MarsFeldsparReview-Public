#!/usr/bin/env python3
"""
plot_spectra_r2.py  (revision R2: layout / legibility redesign)

Regenerates the five regional spectral figures with a compact, print-oriented
layout.  Spectral data and processing are unchanged from plot_spectra_revision.py
(S/L seam correction near 1.0 um, normalization at 1.0 um, tight-packing vertical
offsets).

Design changes relative to *_r1:
  * charcoal background by default (FIG_THEME=light for white); figures drawn at final print size (6.8 in wide) so
    fonts are 7-9 pt on the page;
  * colorblind-safe Okabe-Ito/Tol palette; every category also has its own
    line style (plagioclase high/moderate = solid, low confidence = dashed,
    other endmember = dash-dot, no detection = thin solid gray);
  * new detections (this work) are marked by a bold ID label with an asterisk
    instead of a line style, which frees line style for the category;
  * ratio | numerator panels for each region in one row, rows stacked with a
    shared wavelength axis, panel heights scaled to the number of spectra;
  * single horizontal legend; ID labels de-overlapped with leader lines;
  * exact duplicate spectra (same name and identical values, loaded twice from
    plag_confident and plag_confident_new) are plotted once.

Outputs (project root, next to main.tex):
  spectra_n_hellas_tyrrhena_terra_r2.png, spectra_terra_sirenum_r2.png,
  spectra_ritchey_holden_bond_r2.png, spectra_valles_marineris_r2.png,
  spectra_nili_noachis_arcadia_r2.png   (+ .pdf)

Run (from the repository root; figures are written to output/):
    python scripts/figures/plot_spectra_r2.py
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import logging
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("pss", _HERE / "plot_spectra_submission.py")
pss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pss)
psl5 = pss.psl5

OUT_DIR = Path(os.environ.get("R2_OUT_DIR", _HERE.parent.parent / "output"))
DEDUP_EXACT = True

# ── style ────────────────────────────────────────────────────────────────────
FIG_W = 6.8           # inches (journal text width)
FS_TICK, FS_LAB, FS_TITLE, FS_ID, FS_LEG, FS_REF = 7.5, 8.5, 8.5, 6.6, 8, 6.8
THEME = os.environ.get("FIG_THEME", "dark")   # "dark" (charcoal, default) or "light"
if THEME == "dark":
    BG = "#1c1d21"
    TEXT, SPINE, REF, MUTED = "#e9eaed", "#9aa0aa", "#7d838d", "#b5bac2"
else:
    BG = "white"
    TEXT, SPINE, REF, MUTED = "#222222", "#555555", "#8a8a8a", "#6b6b6b"
XLIM = (0.40, 2.52)

# Low confidence (blue, dashed) and the other-endmember classes (red/violet,
# dash-dot) never share a hue family, and every class has its own line style.
if THEME == "dark":
    CAT_COLOR = {
        "plag_confident_high":     "#00e6a1",   # electric green
        "plag_confident_moderate": "#ffb000",   # amber
        "plag_confident":          "#5fd3ff",   # cyan (confidence not assigned)
        "dusty_plag":              "#3d8bff",   # electric blue
        "other_endmem_mafic":      "#ff4f6d",   # hot red
        "other_endmem_alteration": "#d58cff",   # violet
        "nd":                      "#9aa0a6",   # gray
    }
else:
    CAT_COLOR = {
        "plag_confident_high":     "#009E73",
        "plag_confident_moderate": "#E69F00",
        "plag_confident":          "#56B4E9",
        "dusty_plag":              "#0072B2",
        "other_endmem_mafic":      "#882255",
        "other_endmem_alteration": "#CC79A7",
        "nd":                      "#8C8C8C",
    }
CAT_LS = {
    "plag_confident_high":     "-",
    "plag_confident_moderate": "-",
    "plag_confident":          "-",
    "dusty_plag":              (0, (5, 1.6)),
    "other_endmem_mafic":      (0, (5, 1.4, 1.2, 1.4)),
    "other_endmem_alteration": (0, (5, 1.4, 1.2, 1.4)),
    "nd":                      "-",
}
CAT_LW = {k: 1.25 for k in CAT_COLOR}
CAT_LW["nd"] = 1.0
CAT_LABEL = {
    "plag_confident_high":     "Plagioclase, high confidence",
    "plag_confident_moderate": "Plagioclase, moderate confidence",
    "plag_confident":          "Plagioclase (confidence not assigned)",
    "dusty_plag":              "Plagioclase, low confidence",
    "other_endmem_mafic":      "Other endmember: mafic",
    "other_endmem_alteration": "Other endmember: alteration",
    "nd":                      "No detection",
}
LEG_ORDER = ["plag_confident_high", "plag_confident_moderate", "plag_confident",
             "dusty_plag", "other_endmem_mafic", "other_endmem_alteration", "nd"]

# FRT00004185 (Nili Patera) and HRL0000985E (Xanthe Terra) are in NEW_ID_LOCATIONS for their location codes,
# but both are reported observations reassessed in Supplementary Table 1, not new detections (2026-11-19).
NEW_IDS = set(psl5.NEW_ID_LOCATIONS.keys()) - {"FRT00004185", "HRL0000985E"}


def setup_style():
    matplotlib.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "Nimbus Sans",
                        "Liberation Sans", "DejaVu Sans"],
        "font.size": FS_LAB,
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "axes.edgecolor": SPINE, "axes.linewidth": 0.7,
        "axes.facecolor": BG, "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    })


def color(stem):
    return CAT_COLOR.get(stem, TEXT)


def short_id(name):
    return name.split("_")[0]


def is_new(name):
    return short_id(name) in NEW_IDS


# ── data ─────────────────────────────────────────────────────────────────────

def dedup(records):
    if not DEDUP_EXACT:
        return records
    out = []
    for r in records:
        dup = any(o["name"] == r["name"] and o["sp"].shape == r["sp"].shape and
                  np.allclose(o["sp"], r["sp"], equal_nan=True) for o in out)
        if dup:
            print(f"  [dedup] dropping exact duplicate {r['name']}")
            continue
        out.append(r)
    return out


def sort_records(records):
    def key(r):
        s = r["stem"]
        return (psl5.CAT_ORDER_STACK.index(s)
                if s in psl5.CAT_ORDER_STACK else len(psl5.CAT_ORDER_STACK))
    return sorted(records, key=key)


def load_all():
    lookup = psl5.load_spreadsheet(psl5.SPREADSHEET)
    conf = psl5.load_confidence_lookup(psl5.SPREADSHEET2)
    by_loc = psl5.collect_all_spectra(lookup, conf)
    by_loc = {k: dedup([r for r in v if r["name"].split("_")[0].upper() not in psl5.EXCLUDED_IDS])
              for k, v in by_loc.items()}
    nums = pss.collect_numerators(pss._ratioed_ids(by_loc))
    by_loc, nums = override_csi_r1(by_loc, nums)
    return by_loc, nums


# Revision R1 (2026-11-18/19): spectra from C. Viviano's CSI extraction (CSI_files_spectra_ratioed.sli and
# CSI_files_spectra_on.sli, copied to spectra/r1_csi_ratioed and spectra/r1_csi_on) so that each plotted region
# shows exactly the ROIs of Supplementary Table 1:
#   FRT0000B8E2: second denominator (sensor row 397, col 444) replaces the first (row 100); zeolite.
#   FRT0000BF69: MTRDR produced from TRDR data for this work; olivine.
#   FRT00008E76: the two Low ROIs of the table (#2, #4) replace library ROI #0, which is not a table ROI.
#   FRT0000C76C: ROI #5 duplicated ROI #1 (removed from the table); ROI #2 added.
#   FRT000092B4 #3, FRT00009AAE #3, FRT00009D31 #2, HRL0000A8F6 #1, FRT000082EE #2: table ROIs missing from
#     the libraries, added.
CSI_R1_STEM, CSI_R1_NUM_STEM = "r1_csi_ratioed", "r1_csi_on"
CSI_R1_REMOVE = {9: {"FRT00008E76_07_#0_Plagioclase"},
                 10: {"FRT0000B8E2_07_#1_Analcime", "FRT0000C76C_07_#5_Plagioclase"}}
CSI_R1 = {  # CSI name -> (location code, category stem, name used in the figure, add numerator)
    "FRT0000B8E2_07_#2_Analcime": (10, "other_endmem_alteration", "FRT0000B8E2_07_#2_Zeolite", False),
    "FRT0000BF69_07_#1_Fe_Olivine": (9, "other_endmem_mafic", "FRT0000BF69_07_#1_Fe_Olivine", False),
    "FRT00008E76_07_#2_Plagioclase": (9, "dusty_plag", "FRT00008E76_07_#2_Plagioclase", False),
    "FRT00008E76_07_#4_Plagioclase": (9, "dusty_plag", "FRT00008E76_07_#4_Plagioclase", False),
    "FRT00009AAE_07_#3_Fe_Olivine": (9, "other_endmem_mafic", "FRT00009AAE_07_#3_Fe_Olivine", False),
    "FRT00009D31_07_#2_Mg_Olivine HCP": (9, "other_endmem_mafic", "FRT00009D31_07_#2_Olivine_HCP", False),
    "FRT000092B4_07_#3_Plagioclase Kaolinite": (3, "plag_confident_high", "FRT000092B4_07_#3_Plagioclase", True),
    "FRT0000C76C_07_#2_Plagioclase": (10, "plag_confident_moderate", "FRT0000C76C_07_#2_Plagioclase", True),
    "HRL0000A8F6_07_#1_nd": (10, "plag_confident_moderate", "HRL0000A8F6_07_#1_Plagioclase", True),
    "FRT000082EE_07_#2_Al_smectite": (5, "other_endmem_alteration", "FRT000082EE_07_#2_Al_smectite", False),
}


def _csi_lib(stem):
    hdr, dat = psl5.SPECTRA_DIR / f"{stem}.hdr", psl5.SPECTRA_DIR / stem
    if not hdr.exists() or not dat.exists():
        return None, {}
    meta = psl5.parse_header(hdr)
    return meta["wavelengths"] / 1000.0, dict(zip(meta["spectra_names"], psl5.read_spectra(dat, meta)))


def _csi_prep(sp, wav):
    sp = np.where(sp >= 65535, np.nan, sp)  # ENVI bad-band value
    return psl5.normalize_spectrum(psl5.correct_1um_offset(sp, wav), wav, psl5.TIE_WAVELENGTH_UM)


def override_csi_r1(by_loc, nums):
    wav, ratio = _csi_lib(CSI_R1_STEM)
    if wav is None:
        return by_loc, nums
    wn, on = _csi_lib(CSI_R1_NUM_STEM)
    by_loc = {k: list(v) for k, v in by_loc.items()}
    for code, names in CSI_R1_REMOVE.items():
        by_loc[code] = [r for r in by_loc.get(code, []) if r["name"] not in names]
    for src, (code, stem, name, with_num) in CSI_R1.items():
        by_loc[code] = [r for r in by_loc.get(code, []) if r["name"] != name]
        by_loc[code].append(dict(wav=wav, sp=_csi_prep(ratio[src], wav), stem=stem, name=name))
        if with_num and src in on:
            nums[pss._poly_key(name)] = dict(wav=wn, sp=_csi_prep(on[src], wn), name=name)
    by_loc = {k: sort_records(v) for k, v in by_loc.items()}
    return by_loc, nums


# ── label placement ──────────────────────────────────────────────────────────

def _end_value(wav, y, mask):
    idx = np.where(mask & np.isfinite(y))[0]
    if not idx.size:
        return None, None
    tail = idx[-6:]
    return float(wav[idx[-1]]), float(np.median(y[tail]))


def place_labels(ax, items, x_text):
    """items: list of dict(y, x_end, text, color, bold). Draw de-overlapped
    labels to the right of the axes with short leader lines."""
    if not items:
        return
    fig = ax.figure
    dpi = fig.dpi
    sep = FS_ID * 1.18 * dpi / 72.0
    to_px = ax.transData.transform
    from_px = ax.transData.inverted().transform
    ys = np.array([to_px((0, it["y"]))[1] for it in items])
    order = np.argsort(ys)
    pos = ys[order].copy()
    bb = ax.get_window_extent()
    lo, hi = bb.y0 + sep * 0.3, bb.y1 + sep * 0.2
    for _ in range(200):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < sep:
                d = (sep - (pos[i] - pos[i - 1])) / 2 + 0.01
                pos[i] += d; pos[i - 1] -= d; moved = True
        if pos[0] < lo:
            pos += lo - pos[0]; moved = True
        if pos[-1] > hi and pos[-1] - pos[0] < hi - lo:
            pos -= pos[-1] - hi; moved = True
        if not moved:
            break
    x_px_text = to_px((x_text, 0))[0]
    for k, i in enumerate(order):
        it = items[i]
        y_lab = from_px((x_px_text, pos[k]))[1]
        txt = it["text"] + ("*" if it["bold"] else "")
        ax.text(x_text, y_lab, txt, fontsize=FS_ID, color=it["color"],
                fontweight="bold" if it["bold"] else "normal",
                va="center", ha="left", clip_on=False, zorder=4)
        ax.plot([it["x_end"] + 0.01, x_text - 0.012], [it["y"], y_lab],
                color=it["color"], lw=0.5, alpha=0.8, clip_on=False, zorder=3)


# ── axes styling ─────────────────────────────────────────────────────────────

def style_axes(ax, ylo, yhi, show_xlabels=True, xlabel=True):
    ax.set_xlim(*XLIM)
    span = yhi - ylo
    ax.set_ylim(ylo - 0.04 * span, yhi + 0.04 * span)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    step = np.diff(ax.get_yticks())[0] if len(ax.get_yticks()) > 1 else 0.5
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f" if step < 0.1 else "%.1f"))
    if not show_xlabels:
        ax.tick_params(axis="x", which="both", labelbottom=False)
    elif xlabel:
        ax.set_xlabel("Wavelength (µm)", fontsize=FS_LAB, labelpad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)


def ref_line(ax, x, label, show_label=False):
    ax.axvline(x, color=REF, ls=(0, (1, 1.5)), lw=0.8, zorder=1)
    if show_label:
        ax.text(x + 0.025, 0.025, label, transform=ax.get_xaxis_transform(),
                fontsize=FS_REF, color=MUTED, ha="left", va="bottom", zorder=1,
                bbox=dict(boxstyle="square,pad=0.08", fc=BG, ec="none", alpha=0.85))


def panel_title(ax, text):
    ax.set_title(text, fontsize=FS_TITLE, loc="left", pad=2.5, color=TEXT)


# ── stacked panel ────────────────────────────────────────────────────────────

def seam_correct(wav, sp, min_step=0.003):
    """S/L detector seam correction near 1 um for spectra with no overlap bands.

    The library's S segment ends at ~1.010 um and its L segment starts at
    ~1.047 um, so the overlap-based correct_1um_offset() in
    plot_spectra_by_location_5.py finds no common bands and returns the
    spectrum unchanged. Here each side is fit with a line (0.90-1.01 um and
    1.047-1.15 um), both fits are evaluated at 1.03 um, and the S segment
    (wavelengths < 1.03 um) is shifted by the difference. Steps smaller than
    min_step are left alone.
    """
    s = np.asarray(sp, dtype=float).copy()
    w = np.asarray(wav, dtype=float)
    v = (w >= 0.90) & (w <= 1.012) & np.isfinite(s)
    i = (w >= 1.045) & (w <= 1.15) & np.isfinite(s)
    if v.sum() < 4 or i.sum() < 4:
        return s
    step = np.polyval(np.polyfit(w[v], s[v], 1), 1.03) - np.polyval(np.polyfit(w[i], s[i], 1), 1.03)
    if abs(step) >= min_step:
        s[w < 1.03] -= step
    return s


def draw_stack(ax, entries, min_gap=psl5.MIN_GAP):
    """entries: list of dict(wav, sp, stem, name). Returns (ylo, yhi, label_items)."""
    entries = [dict(e, sp=seam_correct(e["wav"], e["sp"])) for e in entries]
    wav = entries[0]["wav"]
    mask = (wav >= 0.40) & (wav <= 2.50)
    offs = psl5.compute_dynamic_offsets(entries, mask, min_gap=min_gap)
    ys, items = [], []
    for e, off in zip(entries, offs):
        y = e["sp"] + off
        st = e["stem"]
        ax.plot(wav[mask], y[mask], color=color(st), ls=CAT_LS.get(st, "-"),
                lw=CAT_LW.get(st, 1.25), zorder=2, solid_capstyle="round",
                dash_capstyle="butt")
        f = y[mask][np.isfinite(y[mask])]
        ys.extend([f.min(), f.max()])
        xe, ye = _end_value(wav, y, mask)
        if xe is not None:
            items.append(dict(y=ye, x_end=xe, text=short_id(e["name"]) + e.get("label_suffix", ""),
                              color=color(st), bold=is_new(e["name"])))
    return min(ys), max(ys), items


def num_entries(records, nums):
    out = []
    for r in records:
        n = nums.get(pss._poly_key(r["name"]))
        if n is not None:
            out.append({"wav": n["wav"], "sp": n["sp"], "stem": r["stem"], "name": r["name"]})
    return out


# ── legend ───────────────────────────────────────────────────────────────────

def legend_handles(stems, include_new=True, ref_text=None):
    hs = [mlines.Line2D([], [], color=color(s), ls=CAT_LS[s], lw=1.6, label=CAT_LABEL[s])
          for s in LEG_ORDER if s in stems]
    if ref_text:
        hs.append(mlines.Line2D([], [], color=REF, ls=(0, (1, 1.5)), lw=1.0, label=ref_text))
    if include_new:
        hs.append(mlines.Line2D([], [], color="none",
                                label="ID*: new detection (this work)"))
    return hs


def top_legend(fig, stems, y, include_new=True, ncol=None, ref_text=None):
    hs = legend_handles(stems, include_new, ref_text)
    ncol = ncol or min(len(hs), 3)
    leg = fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
                     fontsize=FS_LEG, handlelength=2.6, columnspacing=1.2,
                     handletextpad=0.5, borderaxespad=0.0, labelspacing=0.3)
    if include_new:   # bold-face the "ID*" legend entry to match the labels
        leg.get_texts()[-1].set_fontweight("bold")
    fig.canvas.draw()
    bb = leg.get_window_extent()
    W = fig.get_window_extent().width
    assert bb.x0 >= 0 and bb.x1 <= W, f"legend clipped: {bb} vs width {W}"


# ── generic grid builder ─────────────────────────────────────────────────────

L_MARG, R_MARG = 0.50, 0.78       # inches (right margin holds ID labels)
COL_GAP = 1.12                    # inches between ratio and numerator panels
ROW_GAP_TITLE = 0.24              # gap between rows when x tick labels hidden
BOTTOM = 0.42
LEG_H = 0.42


def build(rows, slug, per_unit=0.25, min_units=4.0, legend="top", leg_ncol=None,
          leg_cell=None, panel_labels=False):
    if panel_labels:   # prefix (a), (b), ... in reading order
        k = 0
        for row in rows:
            for c in row:
                if isinstance(c, dict):
                    c["title"] = f"({chr(97 + k)}) " + c["title"]; k += 1
    cells = [c for row in rows for c in row if isinstance(c, dict)]
    any_new = any(is_new(e["name"]) for c in cells for e in c["entries"])
    any_num = any(c["kind"] == "num" for c in cells)
    ref_text = ("Dotted: 1.25 µm (ratio), 2 µm (numerator)" if any_num
                else "Dotted: 1.25 µm")
    stems0 = {e["stem"] for c in cells for e in c["entries"]}
    n_hand = len([s for s in LEG_ORDER if s in stems0]) + 1 + int(any_new)
    nc = leg_ncol or 3
    leg_h = 0.08 + 0.165 * int(np.ceil(n_hand / nc))
    """
    rows: list of rows; each row = list of 2 cells; a cell is None,
          'legend', or dict(kind='ratio'|'num', entries=[...], title=str,
          ref=(x,label)).
    """
    units = []
    for row in rows:
        n = max([len(c["entries"]) for c in row if isinstance(c, dict)] + [1])
        units.append(max(min_units, n + 1.5))
    row_h = [u * per_unit for u in units]
    top_pad = 0.22 + (leg_h if legend == "top" else 0.0)
    H = top_pad + sum(row_h) + ROW_GAP_TITLE * (len(rows) - 1) + BOTTOM
    fig = plt.figure(figsize=(FIG_W, H))
    pw = (FIG_W - L_MARG - COL_GAP - R_MARG) / 2.0
    x0s = [L_MARG, L_MARG + pw + COL_GAP]
    stems_used = []
    y_cursor = H - top_pad
    axes_rows = []
    for ri, (row, rh) in enumerate(zip(rows, row_h)):
        y0 = y_cursor - rh
        last = ri == len(rows) - 1
        ax_row = []
        for ci, cell in enumerate(row):
            rect = [x0s[ci] / FIG_W, y0 / H, pw / FIG_W, rh / H]
            if cell is None:
                ax_row.append(None)
                continue
            if cell == "legend":
                ax = fig.add_axes(rect); ax.set_axis_off()
                ax_row.append(("legend", ax))
                continue
            ax = fig.add_axes(rect)
            ylo, yhi, items = draw_stack(ax, cell["entries"], cell.get("min_gap", psl5.MIN_GAP))
            below_empty = (not last and (rows[ri + 1][ci] is None or rows[ri + 1][ci] == "legend"))
            style_axes(ax, ylo, yhi, show_xlabels=last or below_empty)
            if ci == 0:
                ax.set_ylabel("Normalized I/F (offset)" if len(rows) == 1 else "I/F (offset)",
                              fontsize=FS_LAB, labelpad=2)
            if cell.get("ref"):
                ref_line(ax, *cell["ref"])
            panel_title(ax, cell["title"])
            place_labels(ax, items, XLIM[1] + 0.05)
            for e in cell["entries"]:
                if e["stem"] not in stems_used:
                    stems_used.append(e["stem"])
            ax_row.append(ax)
        axes_rows.append(ax_row)
        y_cursor = y0 - ROW_GAP_TITLE
        if not last and any(isinstance(c, dict) for c in rows[ri + 1]) is False:
            pass
    if legend == "top":
        top_legend(fig, stems_used, (H - leg_h + 0.02) / H, ncol=nc,
                   include_new=any_new, ref_text=ref_text)
    elif legend == "cell":
        for row in axes_rows:
            for c in row:
                if isinstance(c, tuple):
                    hs = legend_handles(stems_used, any_new, ref_text)
                    lg = c[1].legend(handles=hs, loc="lower left", bbox_to_anchor=(-0.08, -0.02),
                                     fontsize=FS_LEG, handlelength=2.6, labelspacing=0.35,
                                     borderaxespad=0.0)
                    if any_new:
                        lg.get_texts()[-1].set_fontweight("bold")
    if len(rows) > 1:
        fig.text(0.012, (BOTTOM + (H - top_pad - BOTTOM) / 2) / H,
                 "Normalized I/F (offset)", rotation=90, va="center", ha="left",
                 fontsize=FS_LAB)
        for row in axes_rows:
            if row and hasattr(row[0], "set_ylabel"):
                row[0].set_ylabel("")
    save(fig, slug)
    return fig


def save(fig, slug):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{slug}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{slug}.pdf")
    print("saved", OUT_DIR / f"{slug}.png")
    plt.close(fig)


# ── figure definitions ───────────────────────────────────────────────────────

# Ratio spectra with weak 1.25-um bands, stretched about their value at 1.0 um for display
# Labels carry the factor; numerators are not stretched.
STRETCH = {"FRT00005A3E": 3, "FRT00008842": 3, "FRT0000474A": 3}
# individual spectra of any class, keyed by full library name
STRETCH_NAME = {"FRT00005A3E_07_#1_Kaolinite": 5, "FRT00005A3E_07_#2_Kaolinite": 5}


def stretch_recs(recs):
    out = []
    for r in recs:
        k = STRETCH_NAME.get(r["name"])
        if k is None and r["stem"].startswith(("plag", "dusty")):
            k = STRETCH.get(short_id(r["name"]))
        if k and "label_suffix" not in r:
            r = dict(r, sp=1 + k * (np.asarray(r["sp"], float) - 1), label_suffix=f" (×{k})")
        out.append(r)
    return out


def region_row(by_loc, nums, code, name, corr=None, n_override=None, ratio_gap=None):
    recs = sort_records(by_loc.get(code, []))
    if corr:
        recs = [dict(r, sp=corr(r["sp"], r["wav"])) for r in recs]
    ne = num_entries(recs, nums)
    n = len(recs)
    ratio = dict(kind="ratio", entries=stretch_recs(recs), title=f"{name}: ratio (n = {n})",
                 ref=(1.25, "1.25 µm"))
    if ratio_gap is not None:
        ratio["min_gap"] = ratio_gap
    num = (dict(kind="num", entries=ne, title=f"{name}: numerator (n = {len(ne)})",
                ref=(2.0, "2 µm"), min_gap=pss.NUM_GAP) if ne else None)
    return [ratio, num]


def make_n_hellas(by_loc, nums):
    # y stretched: taller rows, tighter ratio gaps
    build([region_row(by_loc, nums, 3, "N. Hellas / Tyrrhena Terra", ratio_gap=0.02)],
          "spectra_n_hellas_tyrrhena_terra_r2", per_unit=0.50)


def make_terra_sirenum(by_loc, nums):
    # Split so that small features are not compressed by the large-amplitude
    # olivine spectra: FRT00008E76 (low
    # confidence + mafic ROIs) and the alteration spectra get their own axes.
    recs = sort_records(by_loc.get(9, []))
    nd = [r for r in recs if r["stem"] == "nd"]
    e76 = [r for r in recs if r["stem"] != "nd" and "8E76" in r["name"].upper()]
    alt = [r for r in recs if r["stem"] == "other_endmem_alteration"]
    maf = [r for r in recs if r["stem"] not in ("nd", "other_endmem_alteration")
           and "8E76" not in r["name"].upper()]
    g = 0.012
    rows = [[dict(kind="ratio", entries=e76,
                  title=f"FRT00008E76: low confidence and mafic (n = {len(e76)})",
                  ref=(1.25, "1.25 µm"), min_gap=0.006),
             dict(kind="ratio", entries=alt,
                  title=f"Other endmember, alteration (n = {len(alt)})",
                  ref=(1.25, "1.25 µm"), min_gap=g)],
            [dict(kind="ratio", entries=maf,
                  title=f"Other endmember, mafic (n = {len(maf)})",
                  ref=(1.25, "1.25 µm"), min_gap=g),
             dict(kind="ratio", entries=nd,
                  title=f"No detection (n = {len(nd)})",
                  ref=(1.25, "1.25 µm"), min_gap=g)]]
    build(rows, "spectra_terra_sirenum_r2", per_unit=0.30, min_units=5.0, leg_ncol=2)


# HRL0000407E (Bond crater): spectra re-extracted from the MTRDR cube for C. Viviano's ROI in Supplementary
# Table 2 (numerator sensor row 246, col 200; denominator row 336, col 200; 3x3; 1-based centers), matching
# her CSI extraction (2026-10-21). Replaces the library entry, which came from a different (Varda) ROI.
ROI_407E = _HERE.parent.parent / "data" / "crism" / "HRL0000407E_sensorROI_spectra.npz"


def override_407e(by_loc, nums):
    if not ROI_407E.exists() or not by_loc.get(13):
        return by_loc, nums
    d = np.load(ROI_407E); w = d["wav"]; ref = by_loc[13][0]["wav"]
    def onto(sp):
        out = np.interp(ref, w, sp, left=np.nan, right=np.nan)
        out[(ref > 1.0105) & (ref < 1.0465)] = np.nan
        return out
    def norm(sp):
        return sp / sp[np.nanargmin(np.abs(ref - psl5.TIE_WAVELENGTH_UM))]
    recs = [r for r in by_loc[13] if "407E" not in r["name"].upper()]
    for k in [k for k in nums if "407E" in k.upper()]:
        nums.pop(k)
    name = "HRL0000407E_07_#1_Plagioclase"
    recs.append(dict(wav=ref, sp=norm(onto(d["num"] / d["den"])), stem="plag_confident_high", name=name))
    nums[pss._poly_key(name)] = dict(wav=ref, sp=norm(onto(d["num"])), name=name)
    by_loc = dict(by_loc); by_loc[13] = sort_records(recs)
    return by_loc, nums


def make_ritchey_holden_bond(by_loc, nums):
    # Bond crater (HRL0000407E) removed 2026-10-21; see psl5.EXCLUDED_IDS.
    rows = [region_row(by_loc, nums, 7, "Ritchey"),
            region_row(by_loc, nums, 2, "Holden")]
    build(rows, "spectra_ritchey_holden_bond_r2", per_unit=0.26)


# HRL0000638C: spectra re-extracted from the MTRDR cube (IF182J) for the ROI in
# Supplementary Table 1 (C. Viviano): numerator sensor row 283, col 60; denominator
# row 276, col 60; 5x5 px, 1-based box centers in sensor space, mapped to the
# map-projected cube through the IR line/sample backplanes of IN182J. Moderate
# confidence, as in Supp. Table 1. Replaces the two library entries (MSP, 2026-10-20).
ROI_638C = Path(os.environ.get("ROI_638C", _HERE.parent.parent / "data" / "crism" / "HRL0000638C_sensorROI_spectra.npz"))


STRETCH_638C = 5


def override_638c(by_loc, nums):
    if not ROI_638C.exists():
        return by_loc, nums
    d = np.load(ROI_638C)
    recs = [r for r in by_loc.get(10, []) if "638C" not in r["name"].upper()]
    for k in [k for k in nums if "638C" in k.upper()]:
        nums.pop(k)
    ref = by_loc[10][0]["wav"]
    w = d["wav"]
    def onto(sp):
        out = np.interp(ref, w, sp, left=np.nan, right=np.nan)
        out[(ref > 1.0105) & (ref < 1.0465)] = np.nan
        return out
    def norm(sp):
        return sp / sp[np.nanargmin(np.abs(ref - psl5.TIE_WAVELENGTH_UM))]
    num, den = d["center1_num"], d["center1_den"]
    name = "HRL0000638C_07_#1_Plagioclase"
    # The 1.25-um band is weak (depth ~0.005), so the ratio is stretched about its
    # value at 1.0 um for display.
    r = norm(onto(num / den))
    recs.append(dict(wav=ref, sp=1 + STRETCH_638C * (r - 1), stem="plag_confident_moderate",
                     name=name, label_suffix=f" (×{STRETCH_638C:g})"))
    nums[pss._poly_key(name)] = dict(wav=ref, sp=norm(onto(num)), name=name)
    recs = sort_records(recs)
    by_loc = dict(by_loc); by_loc[10] = recs
    return by_loc, nums


# Viviano-Beck et al. (2017) "Plagioclase LCP" observations reassessed in MTRDR data (2026-10-20);
# spectra extracted by revision_R1/viviano2017_mtrdr/extract_viviano2017_mtrdr.py at the 2017 ROIs.
VB6_DIR = _HERE.parent.parent / "data" / "crism" / "viviano2017_mtrdr"
VB6_CLASS = {  # (observation, ROI): (stem, label) -- classes as in Supplementary Table 1
    ("FRT00005F84", 1): "other_endmem_mafic", ("FRT00005F84", 2): "other_endmem_mafic",
    ("FRT00005F84", 3): "other_endmem_mafic", ("FRT00005F84", 4): "other_endmem_mafic",
    ("FRT00007AF6", 1): "other_endmem_mafic", ("FRT0000A16B", 1): "other_endmem_mafic",
    ("FRT0000B98D", 1): "other_endmem_mafic", ("HRL00019DC2", 1): "other_endmem_mafic",
    ("HRS00004787", 1): "other_endmem_mafic", ("HRS00004787", 2): "other_endmem_mafic",
}


def vb6_records(ref, nums):
    recs = []
    for (cid, k), stem in VB6_CLASS.items():
        f = VB6_DIR / f"{cid}_mtrdr_spectra.npz"
        if not f.exists():
            continue
        d = np.load(f); w = d["wav"]
        def onto(sp):
            out = np.interp(ref, w, sp, left=np.nan, right=np.nan)
            out[(ref > 1.0105) & (ref < 1.0465)] = np.nan
            return out
        def norm(sp):
            return sp / sp[np.nanargmin(np.abs(ref - psl5.TIE_WAVELENGTH_UM))]
        num, den = d[f"roi{k}_num"], d[f"roi{k}_den"]
        name = f"{cid}_07_#{k}_VB2017"
        recs.append(dict(wav=ref, sp=norm(onto(num / den)), stem=stem, name=name))
        nums[pss._poly_key(name)] = dict(wav=ref, sp=norm(onto(num)), name=name)
    return sort_records(recs)


def make_valles_marineris(by_loc, nums):
    by_loc, nums = override_638c(by_loc, nums)
    # y stretched
    row1 = region_row(by_loc, nums, 10, "Valles Marineris", ratio_gap=0.02)
    vb = vb6_records(by_loc[10][0]["wav"], nums)
    rows = [row1]
    if vb:
        ne = num_entries(vb, nums)
        rows.append([dict(kind="ratio", entries=vb, title=f"Plagioclase + LCP sites (2017): ratio (n = {len(vb)})",
                          ref=(1.25, "1.25 µm"), min_gap=0.02),
                     dict(kind="num", entries=ne, title=f"Plagioclase + LCP sites (2017): numerator (n = {len(ne)})",
                          ref=(2.0, "2 µm"), min_gap=pss.NUM_GAP)])
    build(rows, "spectra_valles_marineris_r2", per_unit=0.30, panel_labels=True)


def make_nili_noachis_arcadia(by_loc, nums):
    rows = [region_row(by_loc, nums, 5, "Nili Patera"),
            region_row(by_loc, nums, 12, "Nili Fossae"),
            region_row(by_loc, nums, 6, "Noachis Terra"),
            region_row(by_loc, nums, 1, "Arcadia Planitia")]
    rows[-1][1] = "legend"
    build(rows, "spectra_nili_noachis_arcadia_r2", per_unit=0.24, legend="cell")


def main():
    setup_style()
    by_loc, nums = load_all()
    make_n_hellas(by_loc, nums)
    make_terra_sirenum(by_loc, nums)
    make_ritchey_holden_bond(by_loc, nums)
    make_valles_marineris(by_loc, nums)
    make_nili_noachis_arcadia(by_loc, nums)


if __name__ == "__main__":
    main()
