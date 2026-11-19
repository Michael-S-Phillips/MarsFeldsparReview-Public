#!/usr/bin/env python3
"""Extract CRISM MTRDR spectra at the ROIs of Supplementary Tables 1 and 2.

Pixel convention. ROW and COL in the tables are 1-based, non-map-projected (sensor) coordinates: the IR Line and
IR Sample values of the MTRDR IN backplane plus one. A rectangular ROI of ROWS x COLS pixels is centered on
(ROW, COL). Because the MTRDR is map projected, the ROI is found by selecting every map-projected pixel whose IN
backplane values fall inside the sensor-space box; the spectrum is the mean I/F of those pixels. Values >= 6e4
(the CRISM data-ignore value 65535) and values <= 0 are treated as missing.

Polygon ROIs (FRT00018FF2, FRT00004185; data/rois/*.gpkg) were drawn on the map-projected MTRDR. Their pixels are
the map-projected pixels whose centers fall inside the polygon. Table ROW and COL give the mean sensor coordinates
of those pixels, and the table ROI value gives their number.

Files expected for each observation (from the PDS Geosciences Node, mro-m-crism-5-rdr-mptargeted-v1):
    <id>_07_if*_mtr3.img/.hdr   I/F cube (BSQ, float32)
    <id>_07_in*_mtr3.img/.hdr   IN backplane (band 6 = IR Sample, band 7 = IR Line, 0-based values)
    <id>_07_de*_mtr1.img/.hdr   DE backplane (band 4 = latitude, band 5 = longitude, deg E)

Examples
    # rectangular ROI ratio (Supplementary Table 2, FRT00004B38) -> .npz with wav, num, den
    python roi_tools.py box path/to/frt00004b38_07_if164j_mtr3 45 81 84 81 5 7 -o 4B38.npz
    # polygon ROI summary (pixel count, mean sensor row/col, mean lon/lat)
    python roi_tools.py polygon ../../data/rois/FRT00004185_polygon_rois.gpkg path/to/frt00004185_07_if165j_mtr3 4
"""
import argparse, glob, os, re, sqlite3, struct
import numpy as np


def hdr(path):
    t = open(path).read()
    g = lambda k: int(re.search(rf"^\s*{k}\s*=\s*(\d+)", t, re.M | re.I).group(1))
    out = dict(S=g("samples"), L=g("lines"), B=g("bands"))
    m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", t)
    if m:
        w = np.array([float(x) for x in m.group(1).split(",")])
        out["wav"] = w / 1000 if w.max() > 100 else w
    m = re.search(r"map info\s*=\s*\{([^}]*)\}", t)
    if m:
        mi = [x.strip() for x in m.group(1).split(",")]
        out["map"] = dict(ref_x=float(mi[1]), ref_y=float(mi[2]), e=float(mi[3]), n=float(mi[4]),
                          dx=float(mi[5]), dy=float(mi[6]))
    return out


def open_cube(folder):
    f = lambda pat: sorted(p for p in glob.glob(os.path.join(folder, pat)) if not os.path.basename(p).startswith("._"))[0]
    ifp, inp, dep = f("*_if*_mtr3.img"), f("*_in*_mtr3.img"), f("*_de*_mtr1.img")
    h = hdr(ifp[:-4] + ".hdr")
    S, L, B = h["S"], h["L"], h["B"]
    return dict(h=h, S=S, L=L, B=B,
                cube=np.memmap(ifp, "<f4", "r", shape=(B, L, S)),
                inb=np.memmap(inp, "<f4", "r", shape=(9, L, S)),
                de=np.memmap(dep, "<f4", "r", shape=(14, L, S)))


def _clean(a):
    a = np.asarray(a, float)
    a[(a > 6e4) | (a <= 0)] = np.nan
    return a


def box_mask(C, row, col, nrow, ncol):
    """Map-projected pixels whose sensor (IN) coordinates fall in the 1-based box centered on (row, col)."""
    irl, irs = np.asarray(C["inb"][6]), np.asarray(C["inb"][5])
    r0, c0 = row - 1 - nrow // 2, col - 1 - ncol // 2
    return (irl >= r0) & (irl <= r0 + nrow - 1) & (irs >= c0) & (irs <= c0 + ncol - 1)


def mean_spectrum(C, mask):
    ys, xs = np.where(mask)
    if not len(ys):
        return None, 0
    return np.nanmean(_clean(np.array([C["cube"][:, y, x] for y, x in zip(ys, xs)])), 0), len(ys)


def polygon_mask(gpkg, C, number):
    """Pixels of polygon `number` ("Polygon Number" field) of a Varda ROI GeoPackage."""
    from matplotlib.path import Path
    con = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
    table = con.execute("select table_name from gpkg_contents").fetchone()[0]
    geom = [g for n, g in con.execute(f'select "Polygon Number", geom from "{table}"') if str(n) == str(number)][0]
    flags = geom[3]; hl = 8 + [0, 32, 48, 48, 64][(flags >> 1) & 7]; w = geom[hl:]
    bo = "<" if w[0] == 1 else ">"
    n = struct.unpack(bo + "I", w[9:13])[0]
    ring = np.frombuffer(w[13:13 + 16 * n], dtype=bo + "f8").reshape(n, 2)
    m = C["h"]["map"]
    yy, xx = np.mgrid[0:C["L"], 0:C["S"]]
    cx = m["e"] + (xx + 1.5 - m["ref_x"]) * m["dx"]   # pixel centers; ENVI reference pixel is 1-based
    cy = m["n"] - (yy + 1.5 - m["ref_y"]) * m["dy"]
    return Path(ring).contains_points(np.c_[cx.ravel(), cy.ravel()]).reshape(C["L"], C["S"])


def describe(C, mask):
    lat, lon = np.asarray(C["de"][3])[mask], np.asarray(C["de"][4])[mask]
    lon = np.where(lon > 180, lon - 360, lon)
    return dict(pixels=int(mask.sum()),
                sensor_row=float(np.asarray(C["inb"][6])[mask].mean() + 1),
                sensor_col=float(np.asarray(C["inb"][5])[mask].mean() + 1),
                lon=float(lon.mean()), lat=float(lat.mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("box"); b.add_argument("cube_dir")
    for k in ("num_row", "num_col", "den_row", "den_col", "roi_rows", "roi_cols"):
        b.add_argument(k, type=int)
    b.add_argument("-o", "--out", required=True)
    p = sub.add_parser("polygon"); p.add_argument("gpkg"); p.add_argument("cube_dir"); p.add_argument("number")
    a = ap.parse_args()
    C = open_cube(a.cube_dir)
    if a.cmd == "box":
        mn, md = box_mask(C, a.num_row, a.num_col, a.roi_rows, a.roi_cols), box_mask(C, a.den_row, a.den_col, a.roi_rows, a.roi_cols)
        num, n1 = mean_spectrum(C, mn); den, n2 = mean_spectrum(C, md)
        np.savez(a.out, wav=C["h"]["wav"], num=num, den=den, ratio=num / den, n_num=n1, n_den=n2)
        print(f"numerator {n1} px, denominator {n2} px ->", a.out)
    else:
        print(describe(C, polygon_mask(a.gpkg, C, a.number)))
