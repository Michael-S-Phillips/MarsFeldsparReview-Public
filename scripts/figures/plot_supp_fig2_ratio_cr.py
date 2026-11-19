#!/usr/bin/env python3
"""Supplementary Figure 2: continuum removal over joined S+L detectors vs the L detector only,
for CRISM MTRDR ratio spectra from this work: olivine (FRT0000CAF3 ROI 1, the deepest olivine ratio band), a no-detection
(HRL00012E0D ROI 1, Fig. 6), and high-confidence plagioclase (FRT000110B7 ROI 1).
Usage (from the repository root):
  python3 scripts/figures/plot_supp_fig2_ratio_cr.py output/supp_figure_2_ratio_cr.png"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_spectra_r2 as P

by_loc, _ = P.load_all()
REC = {r["name"]: r for v in by_loc.values() for r in v}
from scipy.signal import medfilt, savgol_filter
def get(name):   # 5-band median filter removes single-band spikes (e.g. 1.11 and 1.66 um) before the hull is fit
    r = REC[name]; w = np.asarray(r["wav"], float); y = np.asarray(r["sp"], float).copy()
    jump = int(np.argmax(np.diff(w) < 0)) + 1 if np.any(np.diff(w) < 0) else len(w)
    for a, b in ((0, jump), (jump, len(w))):
        seg = y[a:b]; ok = np.isfinite(seg)
        if ok.sum() > 7: seg[ok] = medfilt(seg[ok], 5); y[a:b] = seg
    return w, y
SPEC = {"Olivine": get("FRT0000CAF3_07_#1_Fe_Olivine"), "No detection": get("HRL00012E0D_07_#1_nd")}
SRC = {"Olivine": "FRT0000CAF3 ROI 1", "No detection": "HRL00012E0D ROI 1"}
SHORT = {"Olivine": "olivine", "No detection": "no det."}
wp, sp_pl = get("FRT000110B7_07_#1_Plagioclase")

BG, FG, MUTED, REF = "#1c1d21", "#e8e8ea", "#9aa0a6", "#6b6f76"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                     "axes.edgecolor": MUTED, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "font.size": 8})
COL = {"Olivine": "#ff4f6d", "No detection": "#d9c27a"}; PLC = "#5fd3ff"
K_PL, K_OLS = 3, {"Olivine": 3, "No detection": 3}   # display stretch of CR spectra about 1 (plagioclase, olivine); depths in titles are unscaled

def hull(x, y):
    pts = []
    for p in zip(x, y):
        while len(pts) >= 2 and (pts[-1][0]-pts[-2][0])*(p[1]-pts[-2][1]) - (pts[-1][1]-pts[-2][1])*(p[0]-pts[-2][0]) >= 0:
            pts.pop()
        pts.append(p)
    hx, hy = zip(*pts); return np.interp(x, hx, hy)

def prep(w, s, lo):
    s = P.seam_correct(w, s) if lo < 1.0 else np.asarray(s, float)
    m = (w >= lo) & (w <= 2.60) & np.isfinite(s) & ~((w > 0.63) & (w < 0.71)) & ~((w > 1.0105) & (w < 1.0465))
    x, y = w[m], s[m]; h = hull(x, y); return x, y, h, y / h

fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.6), sharex=True)
for r, name in enumerate(["Olivine", "No detection"]):
    for c, (lab, lo) in enumerate([("S + L (0.44–2.60 µm)", 0.44), ("L only (1.05–2.60 µm)", 1.045)]):
        a = axs[r, c]
        xo, yo, ho, cro = prep(*SPEC[name], lo)
        xp, yp, hp, crp = prep(wp, sp_pl, lo)
        ko, kp = np.nanmax(ho), np.nanmax(hp)
        # three stacks, top to bottom: CR (overlaid), olivine ratio + hull, plagioclase ratio + hull
        lo_ol = np.nanmin(yo / ko); lo_pl = np.nanmin(yp / kp)
        off_pl = -(1.0 - lo_ol) - 0.10                                  # plagioclase I/F stack below olivine
        crp_d = 1 + K_PL * (crp - 1); cro_d = 1 + K_OLS[name] * (cro - 1)
        off_cr = 1.0 - min(np.nanmin(cro_d), np.nanmin(crp_d)) + 0.10       # CR stack above olivine
        a.plot(xo, yo / ko, color=COL[name], lw=0.8, alpha=0.6); a.plot(xo, ho / ko, color=FG, lw=0.6, ls="--")
        a.plot(xp, yp / kp + off_pl, color=PLC, lw=0.8, alpha=0.6); a.plot(xp, hp / kp + off_pl, color=FG, lw=0.6, ls="--")
        a.plot(xp, crp_d + off_cr, color=PLC, lw=1.3)
        a.plot(xo, cro_d + off_cr, color=COL[name], lw=1.3)
        bo, bp = (xo >= 0.75) & (xo <= 2.0), (xp >= 0.75) & (xp <= 2.0)   # skip the hull artifact on the visible rise
        sm = lambda v: savgol_filter(v, 15, 2) if len(v) > 15 else v      # band minimum from a smoothed CR spectrum
        io = np.where(bo)[0][np.argmin(sm(cro[bo]))]; ip = np.where(bp)[0][np.argmin(sm(crp[bp]))]
        a.axvline(xo[io], color=COL[name], ls=":", lw=0.8); a.axvline(xp[ip], color=PLC, ls=":", lw=0.8)
        tx = 2.63
        a.text(tx, np.nanmedian(cro_d[-15:]) + off_cr - 0.02, "CR", color=FG, fontsize=7, va="top")
        a.text(tx, np.nanmedian(yo[-15:] / ko), f"{SHORT[name]}\nratio", color=COL[name], fontsize=6.5, va="center")
        a.text(tx, np.nanmedian(yp[-15:] / kp) + off_pl, "plag.\nratio", color=PLC, fontsize=6.5, va="center")
        a.set_title(f"{name} vs plagioclase, {lab}\nCR minima: {SHORT[name]} {xo[io]:.2f} µm ({1-cro[io]:.2f}), "
                    f"plagioclase {xp[ip]:.2f} µm ({1-crp[ip]:.2f})", fontsize=7.5)
        a.axvline(1.03, color=MUTED, ls="-.", lw=0.8, zorder=0)            # S/L detector boundary
        a.text(1.03, 0.015, " S | L ", transform=a.get_xaxis_transform(), ha="center", va="bottom",
               fontsize=6.5, color=MUTED, bbox=dict(boxstyle="square,pad=0.1", fc=BG, ec="none"))
        a.set_xlim(0.4, 2.85); a.set_yticks([])
        if c == 0: a.set_ylabel("Ratio / CR (offset)")
        if r == 1: a.set_xlabel("Wavelength (µm)")
from matplotlib.lines import Line2D
hs = [Line2D([], [], color=COL["Olivine"], lw=1.3, label=f"Olivine, FRT0000CAF3 ROI 1; CR ×{K_OLS['Olivine']}"),
      Line2D([], [], color=COL["No detection"], lw=1.3, label=f"No detection, HRL00012E0D ROI 1; CR ×{K_OLS['No detection']}"),
      Line2D([], [], color=PLC, lw=1.3, label=f"Plagioclase, FRT000110B7 ROI 1; CR ×{K_PL}"),
      Line2D([], [], color=FG, lw=0.6, ls="--", label="Continuum (upper convex hull)"),
      Line2D([], [], color=MUTED, lw=0.8, ls="-.", label="S/L detector boundary (~1.03 µm)")]
fig.legend(handles=hs, loc="lower center", ncol=2, fontsize=7.2, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.075, 1, 1)); fig.savefig(sys.argv[1], dpi=300); print("saved")
