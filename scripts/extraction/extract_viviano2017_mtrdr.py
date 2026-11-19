#!/usr/bin/env python3
"""Extract MTRDR ratio spectra at the Viviano-Beck et al. (2017) "Plagioclase LCP" ROIs for the six
Valles Marineris observations plotted as High in Fig. 1 but absent from Supplementary Table 1.

ROIs (sensor space) from data/crism/Viviano2017_IDs.xlsx, treated as 1-based box centers (the convention
that reproduces the HRL0000638C library spectrum, r = 0.987). Sensor row/col are mapped to the
map-projected MTRDR cube through the IR Line / IR Sample backplanes of the IN product.

Usage: python3 extract_viviano2017_mtrdr.py <MTRDR root> <out dir>
"""
import glob, os, re, sys
import numpy as np

ROOT, OUT = sys.argv[1], sys.argv[2]
IDS = ["FRT00005F84", "FRT00007AF6", "FRT0000A16B", "FRT0000B98D", "HRL00019DC2", "HRS00004787"]
X = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "rois", "viviano2017_rois.csv")

def hdr(p):
    h = open(p).read()
    g = lambda k: int(re.search(rf"^{k}\s*=\s*(\d+)", h, re.M).group(1))
    wl = re.search(r"wavelength = \{([^}]*)\}", h)
    names = re.search(r"band names = \{([^}]*)\}", h)
    return dict(S=g("samples"), L=g("lines"), B=g("bands"),
                wl=np.array([float(x) for x in wl.group(1).split(",")]) / 1000 if wl else None,
                names=[n.strip() for n in names.group(1).split(",")] if names else None)

rois = {}
seen = set()
import csv
for r in csv.DictReader(open(X)):   # data/rois/viviano2017_rois.csv, in ROI order
    rois.setdefault(r["crism_id"], []).append(tuple(int(r[k]) for k in
        ("num_row", "num_col", "den_row", "den_col", "roi_rows", "roi_cols")))

os.makedirs(OUT, exist_ok=True)
for cid in IDS:
    d = glob.glob(os.path.join(ROOT, cid.lower() + "_07_if*_mtr3"))
    if not d:
        print(cid, "not downloaded"); continue
    d = d[0]
    ifp = glob.glob(os.path.join(d, "*_IF*_MTR3.IMG")); inp = glob.glob(os.path.join(d, "*_IN*_MTR3.IMG"))
    if not ifp or not inp:
        print(cid, "incomplete:", "IF" if not ifp else "", "IN" if not inp else ""); continue
    H = hdr(ifp[0][:-4] + ".HDR"); HI = hdr(inp[0][:-4] + ".HDR")
    if os.path.getsize(ifp[0]) != H["S"] * H["L"] * H["B"] * 4:
        print(cid, "IF still downloading"); continue
    cube = np.memmap(ifp[0], dtype="<f4", mode="r", shape=(H["B"], H["L"], H["S"]))
    ib = np.fromfile(inp[0], dtype="<f4").reshape(HI["B"], HI["L"], HI["S"])
    isamp, iline = np.rint(ib[HI["names"].index("IR Sample")]), np.rint(ib[HI["names"].index("IR Line")])
    def box(r, c, nr, nc):
        rs = range(r - nr // 2 - 1, r + nr // 2)      # 1-based center -> 0-based rows
        cs = range(c - nc // 2 - 1, c + nc // 2)
        m = np.isin(iline, list(rs)) & np.isin(isamp, list(cs))
        ll, ss = np.where(m)
        sp = np.array([cube[:, l, s] for l, s in zip(ll, ss)], dtype=float)
        sp[sp > 1e3] = np.nan
        return np.nanmean(sp, 0), len(ll)
    out = dict(wav=H["wl"])
    for k, (nr_, nc_, dr, dc, sr, sc) in enumerate(rois[cid], 1):
        n, nn = box(nr_, nc_, sr, sc); dn, dd = box(dr, dc, sr, sc)
        out[f"roi{k}_num"], out[f"roi{k}_den"] = n, dn
        out[f"roi{k}_meta"] = np.array([nr_, nc_, dr, dc, sr, sc, nn, dd])
        print(cid, f"ROI {k}: num {nr_}/{nc_} den {dr}/{dc} {sr}x{sc}  map px {nn}/{dd}")
    np.savez(os.path.join(OUT, f"{cid}_mtrdr_spectra.npz"), **out)
