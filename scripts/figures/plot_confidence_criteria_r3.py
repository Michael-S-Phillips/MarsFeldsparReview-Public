#!/usr/bin/env python3
"""
plot_confidence_criteria_r3.py  (revision R3: ratio, numerator and denominator
stacked in one panel for every example)

Example CRISM MTRDR spectra that illustrate the confidence classes
(Section 2.3.1).  One panel per example; each panel shows the ratio spectrum
(top, left axis, normalized at 1.0 um) and the unratioed numerator and
denominator ROI spectra (bottom, right axis, I/F, not normalized, not offset
from each other).  The two groups are separated vertically by giving each axis
its own range, so every trace is stretched independently.

Examples (same ROIs as confidence_criteria_r2):
  high  FRT000092B4 #2   plag_confident / unratioed_plag_highconfidence
  mod   HRS00012C57 #1   plag_confident / unratioed_plag_moderateconfidence_w_2micron
  low   FRT00008E76 #2   r1_csi_ratioed (CSI)       (Supplementary Table 1 Low ROI, 5 x 7 px)
  ol    FRT00009AAE #2   other_endmem_mafic
  nd    FRT0000A8B7 #1   nd

Data sources
  * Ratio: always the spectral-library ratio (the spectrum shown in every
    other figure of the paper).
  * High and moderate: numerator = library unratioed spectrum; denominator =
    numerator / library ratio (exact by construction).
  * Low, olivine, no detection: the library holds no unratioed spectra, so the
    numerator and denominator are extracted from the MTRDR I/F cubes with the
    same routine used for the delivery-004 spectra
    (scripts/extraction/extract_new_spectra.py: extract_roi_via_index, i.e.
    the mean of all map-projected pixels whose IR Sample / IR Line in the
    _in*_mtr3 index cube fall inside the sensor-space ROI box; 65535 and <= 0
    masked to NaN; resample_to_ref onto the 544-band library grid).  The
    MTRDR ratio (num/den) of all five examples is compared against the library
    ratio and the residuals are printed.

The extracted spectra are cached in data/crism/confidence_criteria_r3_spectra.npz
so the figure can be rebuilt without the CRISM drive mounted.  Delete the cache
(or set R3_REEXTRACT=1) to re-extract.

All spectra get the S/L seam correction near 1.0 um (seam_correct in plot_spectra_r2.py: line fits on each detector, extrapolated to 1.03 um).
No continuum removal.

Output: output/confidence_criteria_r3.png / .pdf

Run (from the repository root):
    python scripts/figures/plot_confidence_criteria_r3.py
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import to_rgb
from matplotlib.ticker import MultipleLocator, MaxNLocator, FormatStrFormatter, AutoMinorLocator

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
_spec = importlib.util.spec_from_file_location("ps2", _HERE / "plot_spectra_r2.py")
ps2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps2)
psl5 = ps2.psl5
SD = psl5.SPECTRA_DIR
OUT_DIR = Path(os.environ.get("R2_OUT_DIR", ROOT / "output"))

_spec = importlib.util.spec_from_file_location(
    "xns", _HERE.parent / "extraction" / "extract_new_spectra.py")
xns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xns)

CACHE = ROOT / "data" / "crism" / "confidence_criteria_r3_spectra.npz"

# CRISM MTRDR root (only needed if the cached spectra in data/crism are deleted): $CRISM_MTRDR_ROOT.
_ROOTS = [Path(p) for p in (os.environ.get("CRISM_MTRDR_ROOT", ""),) if p]
# (IF base, IN base) relative to the FeldsReview root, without extension
CUBES = {
    "FRT000092B4": ("CartPoulet2013/frt000092b4_07_if164j_mtr3",
                    "frt000092b4_07_if164j_mtr3/frt000092b4_07_in164j_mtr3"),
    "FRT000064B3": ("quantin2012/frt000064b3_07_if164j_mtr3/frt000064b3_07_if164j_mtr3",
                    "quantin2012/frt000064b3_07_if164j_mtr3/frt000064b3_07_in164j_mtr3"),
    "HRS00012C57": ("hrs00012c57_07_if173j_mtr3/hrs00012c57_07_if173j_mtr3",
                    "hrs00012c57_07_if173j_mtr3/hrs00012c57_07_in173j_mtr3"),
    "FRT00008E76": ("frt00008e76_07_if164j_mtr3/frt00008e76_07_if164j_mtr3",
                    "frt00008e76_07_if164j_mtr3/frt00008e76_07_in164j_mtr3"),
    "FRT00009AAE": ("frt00009aae_07_if164j_mtr3/frt00009aae_07_if164j_mtr3",
                    "frt00009aae_07_if164j_mtr3/frt00009aae_07_in164j_mtr3"),
    "FRT0000A8B7": ("frt0000a8b7_07_if164j_mtr3/frt0000a8b7_07_if164j_mtr3",
                    "frt0000a8b7_07_if164j_mtr3/frt0000a8b7_07_in164j_mtr3"),
}

# ROI coordinates (sensor space) from CSIfiles_concattable_redoplag.csv,
# (on_row, on_col, off_row, off_col, sz_row, sz_col).  The ROI number matches
# the #N of the library spectrum name except for FRT00008E76 (#0 in the
# library = CSI ROI 1; identified by the best ratio match among ROIs 1-4).
EX = {
    "high": dict(cid="FRT000092B4", stem="plag_confident", ustem="unratioed_plag_highconfidence",
                 name="FRT000092B4_07_#2_Plagioclase", roi=(337, 384, 198, 384, 5, 5),
                 cat="plag_confident_high", title="High confidence", loc="N. Hellas rim"),
    "mod":  dict(cid="HRS00012C57", stem="plag_confident",
                 ustem="unratioed_plag_moderateconfidence_w_2micron",
                 name="HRS00012C57_07_#1_Plagioclase", roi=(174, 308, 84, 308, 5, 5),
                 cat="plag_confident_moderate", title="Moderate confidence", loc="Valles Marineris"),
    # FRT00008E76 ROI 2, a Low ROI of Supplementary Table 1 (2026-11-19); library ROI #0 used before is not a
    # table ROI (its numerator is the ROI 1 box, which the table classes as olivine).
    "low":  dict(cid="FRT00008E76", stem="r1_csi_ratioed", ustem=None,
                 name="FRT00008E76_07_#2_Plagioclase", roi=(315, 294, 413, 294, 5, 7),
                 cat="dusty_plag", title="Low confidence (dusty)", loc="Terra Sirenum"),
    "ol":   dict(cid="FRT00009AAE", stem="other_endmem_mafic", ustem=None,
                 name="FRT00009AAE_07_#2_Fe_Olivine", roi=(361, 406, 263, 406, 5, 5),
                 cat="other_endmem_mafic", title="Reinterpreted as olivine", loc="Terra Sirenum"),
    "nd":   dict(cid="FRT0000A8B7", stem="nd", ustem=None,
                 name="FRT0000A8B7_07_#1_nd", roi=(157, 470, 327, 470, 5, 5),
                 cat="nd", title="No detection (dusty)", loc="Terra Sirenum"),
}
ORDER = ["high", "mod", "low", "ol", "nd"]


# ── data ─────────────────────────────────────────────────────────────────────

def lib(stem, name):
    meta = psl5.parse_header(SD / f"{stem}.hdr")
    for sp, nm in zip(psl5.read_spectra(SD / stem, meta), meta["spectra_names"]):
        if nm == name:
            sp = sp.astype(float)
            return meta["wavelengths"], np.where(sp >= 65535, np.nan, sp)  # ENVI bad-band value (CSI library)
    raise KeyError(name)


def find_cube(cid):
    for r in _ROOTS:
        ifb, inb = (r / p for p in CUBES[cid])
        if all(Path(str(b) + s).exists() for b in (ifb, inb) for s in (".hdr", ".img")):
            return ifb, inb
    return None


def extract(cid, roi, ref_wl, jump):
    """Numerator and denominator mean I/F from the MTRDR, resampled to the library grid."""
    found = find_cube(cid)
    if found is None:
        return None
    ifb, inb = found
    im = xns.parse_envi_hdr(Path(str(ifb) + ".hdr"))
    nm = xns.parse_envi_hdr(Path(str(inb) + ".hdr"))
    r_on, c_on, r_off, c_off, szr, szc = roi
    args = (Path(str(ifb) + ".img"), Path(str(inb) + ".img"), im, nm)
    with np.errstate(all="ignore"):
        on, n_on = xns.extract_roi_via_index(*args, r_on, c_on, szr, szc)
        off, n_off = xns.extract_roi_via_index(*args, r_off, c_off, szr, szc)
    cw = im["wavelengths"]
    return dict(num=xns.resample_to_ref(on, cw, ref_wl, jump),
                den=xns.resample_to_ref(off, cw, ref_wl, jump),
                n_on=n_on, n_off=n_off, src=str(ifb))


def load_all():
    ref_wl, jump = xns.get_ref_wavelengths(SD / "plag_confident.hdr")
    cache = {}
    if CACHE.exists() and not os.environ.get("R3_REEXTRACT"):
        z = np.load(CACHE, allow_pickle=True)
        cache = z["data"].item()
    changed = False
    out = {}
    for k in ORDER:
        e = EX[k]
        wl, ratio = lib(e["stem"], e["name"])
        assert np.allclose(wl, ref_wl)
        if k not in cache:
            x = extract(e["cid"], e["roi"], ref_wl, jump)
            if x is not None:
                cache[k] = x
                changed = True
        x = cache.get(k)
        rec = dict(wl=wl / 1000.0, ratio=ratio)
        if e["ustem"]:
            _, n = lib(e["ustem"], e["name"])
            with np.errstate(divide="ignore", invalid="ignore"):
                rec.update(num=n, den=n / ratio, src="library (numerator; denominator = numerator / ratio)")
        elif x is not None:
            # keep the band set of the library ratio (drops the S/L overlap bands
            # that resample_to_ref fills in both segments, and the 0.65-0.7 um gap)
            keep = np.isfinite(ratio)
            rec.update(num=np.where(keep, x["num"], np.nan), den=np.where(keep, x["den"], np.nan),
                       src="MTRDR " + Path(x["src"]).name)
        else:
            print(f"  [WARN] {e['cid']}: no MTRDR cube and no cache; numerator/denominator omitted")
        # verification: MTRDR ratio vs library ratio
        if x is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                rm = x["num"] / x["den"]
            w = wl / 1000.0
            ok = np.isfinite(rm) & np.isfinite(ratio) & (w >= 0.42) & (w <= 2.6)
            d = np.abs(rm - ratio)[ok]
            msg = (f"  {e['name']:32s} ON={x['n_on']:3d} OFF={x['n_off']:3d} map px | "
                   f"MTRDR ratio vs library: max|d|={d.max():.4f} median|d|={np.median(d):.4f}")
            if e["ustem"]:
                dn = np.abs(x["num"] - rec["num"])[ok]
                msg += f" | MTRDR num vs library num: max|d|={np.nanmax(dn):.4f} I/F"
            print(msg)
        out[k] = rec
    if changed:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, data=np.array(cache, dtype=object))
        print("  cached MTRDR extractions ->", CACHE)
    return out


# ── drawing ──────────────────────────────────────────────────────────────────

def tint(c, f=0.45):
    r, g, b = to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


NUM_LS = "-"
DEN_LS = (0, (3.2, 1.6))
DEN_CLR = "#c9ccd1" if ps2.THEME == "dark" else "#4a4a4a"
RANGE = (0.40, 2.50)
LAB_X = (1.40, 1.95)
FS_TR = 6.6          # direct trace labels


def seg(w, sp):
    m = (w >= RANGE[0]) & (w <= RANGE[1])
    return w[m], sp[m]


def nice_ticks(lo, hi, n=3):
    loc = MaxNLocator(nbins=n, steps=[1, 2, 5, 10])
    t = loc.tick_values(lo, hi)
    return t[(t >= lo - 1e-9) & (t <= hi + 1e-9)]


def fmt(t):
    step = t[1] - t[0] if len(t) > 1 else 0.1
    return "%.2f" if step < 0.1 - 1e-9 else "%.1f"


def pw_in(ax):
    return ax.get_position().width * ax.figure.get_figwidth()


def panel(ax, rec, e, letter, refs, show_xlab, left_lab, right_lab,
          r_frac=(0.55, 0.88), n_frac=(0.11, 0.44), if_min_wl=0.70):
    """Ratio on ax (left axis) in the band r_frac of the panel height; numerator
    and denominator on a twin axis (right) in the band n_frac."""
    clr = ps2.color(e["cat"])
    w = rec["wl"]
    s = psl5.normalize_spectrum(ps2.seam_correct(w, rec["ratio"]), w, 1.0)
    ws, ss = seg(w, s)
    ax.plot(ws, ss, color=clr, ls=ps2.CAT_LS[e["cat"]], lw=1.25, zorder=3)
    f = ss[np.isfinite(ss)]
    lo, hi = f.min(), f.max()
    span = hi - lo
    # map [lo, hi] onto the r_frac band of the axes
    a0, a1 = r_frac
    ylo = lo - a0 * span / (a1 - a0)
    yhi = ylo + span / (a1 - a0)
    ax.set_ylim(ylo, yhi)
    t = nice_ticks(lo, hi)
    ax.set_yticks(t)
    ax.yaxis.set_major_formatter(FormatStrFormatter(fmt(t)))
    ax.tick_params(axis="y", colors=clr, labelcolor=clr, which="both")
    ax.spines["left"].set_color(ps2.SPINE)
    # label: right end of ratio trace
    mL = (ws >= 0.42) & (ws <= 0.85)
    ax.text(0.44, np.nanmax(ss[mL]) + 0.03 * (yhi - ylo), "Ratio", color=clr, fontsize=FS_TR,
            ha="left", va="bottom", zorder=5)

    ax2 = ax.twinx()
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    h = {}
    if "num" in rec:
        n = ps2.seam_correct(w, rec["num"])
        d = ps2.seam_correct(w, rec["den"])
        wn, nn = seg(w, n)
        _, dd = seg(w, d)
        ax2.plot(wn, nn, color=tint(clr), ls=NUM_LS, lw=1.0, zorder=2)
        ax2.plot(wn, dd, color=DEN_CLR, ls=DEN_LS, lw=1.0, zorder=2)
        m = wn >= if_min_wl
        vals = np.concatenate([nn[m], dd[m]])
        vals = vals[np.isfinite(vals)]
        lo2, hi2 = vals.min(), vals.max()
        sp2 = hi2 - lo2
        b0, b1 = n_frac
        y2lo = lo2 - b0 * sp2 / (b1 - b0)
        y2hi = y2lo + sp2 / (b1 - b0)
        ax2.set_ylim(y2lo, y2hi)
        t2 = nice_ticks(lo2, hi2)
        ax2.set_yticks(t2)
        ax2.yaxis.set_major_formatter(FormatStrFormatter(fmt(t2)))
        h = dict(nn=nn, dd=dd, wn=wn)
        # direct labels: above the upper trace / below the lower trace over 1.35-2.45 um
        mw = (wn >= LAB_X[0]) & (wn <= LAB_X[1])
        gap = 0.03 * (y2hi - y2lo)
        upper_is_num = np.nanmedian(nn[mw]) >= np.nanmedian(dd[mw])
        up, dn = (nn, dd) if upper_is_num else (dd, nn)
        lab_up, lab_dn = ("Numerator", "Denominator") if upper_is_num else ("Denominator", "Numerator")
        c_up, c_dn = (tint(clr), DEN_CLR) if upper_is_num else (DEN_CLR, tint(clr))
        xm = 0.5 * (LAB_X[0] + LAB_X[1])
        ax2.text(xm, np.nanmax(up[mw]) + gap, lab_up, color=c_up, fontsize=FS_TR, ha="center",
                 va="bottom", zorder=5)
        ax2.text(xm, np.nanmin(dn[mw]) - gap, lab_dn, color=c_dn, fontsize=FS_TR, ha="center",
                 va="top", zorder=5)
    ax2.tick_params(axis="y", colors=ps2.MUTED, labelcolor=ps2.TEXT, which="both", direction="in",
                    length=3)
    ax2.spines["right"].set_color(ps2.SPINE)
    for sp_ in ("top", "bottom", "left"):
        ax2.spines[sp_].set_visible(False)

    ax.set_xlim(*ps2.XLIM)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    for x, lab in refs:
        ax.axvline(x, color=ps2.REF, ls=(0, (1, 1.5)), lw=0.8, zorder=1)
        ax.text(x + 0.02, 0.975, lab, transform=ax.get_xaxis_transform(), fontsize=6.4,
                color=ps2.MUTED, ha="left", va="top", zorder=4)
    ax.set_title(f"({letter}) {e['title']}\n{e['cid']} ({e['loc']})", fontsize=8, loc="left",
                 pad=2.5, linespacing=1.15)
    if show_xlab:
        ax.set_xlabel("Wavelength (µm)", fontsize=ps2.FS_LAB, labelpad=1.5)
    if left_lab:
        ax.set_ylabel("Ratio\n(norm. at 1.0 µm)", fontsize=7.6, linespacing=1.1)
        ax.yaxis.set_label_coords(-0.335 / pw_in(ax), 0.5 * sum(r_frac))
    if right_lab:
        ax2.set_ylabel("I/F", fontsize=7.6, rotation=270, va="bottom")
        ax2.yaxis.set_label_coords(1.0 + 0.33 / pw_in(ax), 0.5 * sum(n_frac))
    return ax2


def main():
    print("Loading spectra and verifying MTRDR extractions ...")
    data = load_all()
    ps2.setup_style()
    W, H = 6.8, 5.2
    fig = plt.figure(figsize=(W, H))
    L, R, B, T = 0.56, 0.46, 0.40, 0.40
    gx, gy = 0.62, 0.62
    pw = (W - L - R - 2 * gx) / 3
    ph = (H - B - T - gy) / 2

    def ax_at(r, c):
        return fig.add_axes([(L + c * (pw + gx)) / W, (B + (1 - r) * (ph + gy)) / H, pw / W, ph / H])

    plag_refs = [(0.9, "0.9"), (1.25, "1.25"), (2.0, "2.0")]
    spec = [("high", 0, 0, "a", plag_refs),
            ("mod", 0, 1, "b", plag_refs),
            ("low", 0, 2, "c", plag_refs),
            ("ol", 1, 0, "d", [(0.7, "0.7"), (1.25, "1.25"), (2.0, "2.0")]),
            ("nd", 1, 1, "e", plag_refs)]
    for k, r, c, letter, refs in spec:
        ax = ax_at(r, c)
        panel(ax, data[k], EX[k], letter, refs, show_xlab=(r == 1 or c == 2),
              left_lab=(c == 0), right_lab=True)

    # key in the sixth slot
    kx = (L + 2 * (pw + gx)) / W
    ky = B / H
    kax = fig.add_axes([kx, ky, pw / W, ph / H])
    kax.axis("off")
    stems = ["plag_confident_high", "plag_confident_moderate", "dusty_plag", "other_endmem_mafic", "nd"]
    hs = [mlines.Line2D([], [], color=ps2.color(s), ls=ps2.CAT_LS[s], lw=1.5, label=ps2.CAT_LABEL[s])
          for s in stems]
    hs.append(mlines.Line2D([], [], color=tint("#bbbbbb", 0.3), ls=NUM_LS, lw=1.0,
                            label="Numerator (tint of class color)"))
    hs.append(mlines.Line2D([], [], color=DEN_CLR, ls=DEN_LS, lw=1.0, label="Denominator"))
    hs.append(mlines.Line2D([], [], color=ps2.REF, ls=(0, (1, 1.5)), lw=1.0,
                            label="Wavelength (µm) referred to in text"))
    kax.legend(handles=hs, loc="upper left", bbox_to_anchor=(-0.06, 1.02), fontsize=6.8,
               handlelength=2.4, labelspacing=0.45, borderaxespad=0.0, frameon=False)
    kax.text(-0.04, 0.0, "Upper trace: ratio, left axis.\nLower traces: unratioed ROI\nspectra, right axis (I/F).",
             transform=kax.transAxes, fontsize=6.6, color=ps2.MUTED, ha="left", va="bottom",
             linespacing=1.25)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "confidence_criteria_r3.png", dpi=300)
    fig.savefig(OUT_DIR / "confidence_criteria_r3.pdf")
    print("saved", OUT_DIR / "confidence_criteria_r3.png")


if __name__ == "__main__":
    main()
