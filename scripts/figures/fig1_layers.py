"""Build the map layers for Fig. 1 (global map of spectrally feldspathic detections).

Outputs, per central meridian:
  basemap_lon<L>.png   shaded-relief basemap in the Robinson projection (RGBA, transparent outside the globe)
  features_lon<L>.json projected pixel coordinates of detections, landing sites, outlines, graticule, labels

Inputs:
  HRSC-MOLA blended DEM v2 (Fergason et al., 2018), read at a reduced overview level
  data/detections/literature_detections.csv (literature detections)
  data/tables/supp_table_1_reassessed_literature.csv (reassessed class per observation; best class across ROIs)
  data/tables/supp_table_2_new_detections.csv (new detections from this work)

Usage (from the repository root; omit --dem to rebuild the features_*.json files only):
  python scripts/figures/fig1_layers.py --dem Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif --out output/fig1 --lon0 0 70
"""
import argparse
import csv
import json
import os
import re

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling

R_MARS = 3396190.0
CLASS_RANK = {"High": 1, "Moderate": 2, "Low": 3, "Other": 4, "nd": 5, "No": 6}

# The six Viviano-Beck et al. (2017) observations formerly hard-coded here as "High" (FRT00005F84,
# FRT00007AF6, FRT0000A16B, FRT0000B98D, HRL00019DC2, HRS00004787) were reassessed in MTRDR data on
# 2026-10-20 and are now classed in Supplementary Table 1.
CLASS_OVERRIDES = {}

LANDING_SITES = [  # name, lat (N), lon (E)
    ("Mars Pathfinder", 19.13, -33.22),
    ("Spirit", -14.568, 175.473),
    ("Opportunity", -1.946, -5.527),
    ("Curiosity", -4.590, 137.442),
    ("Perseverance", 18.445, 77.451),
    ("InSight", 4.502, 135.623),
    ("Zhurong", 25.066, 109.925),
]

BASINS = [("Hellas", -42.4, 70.5), ("Argyre", -49.7, -43.0), ("Isidis", 12.9, 87.0),
          ("Utopia", 46.7, 117.2), ("Tharsis", 2.0, -112.0)]

# Terra Sirenum/Cimmeria study region of Payre et al. (2022), drawn as a lat/lon box
TSC_BOX = dict(lon_min=140.0, lon_max=212.0, lat_min=-40.0, lat_max=-12.0)


def norm_lon(lon):
    lon = float(str(lon).replace("−", "-"))
    return ((lon + 180.0) % 360.0) - 180.0


def norm_lat(lat):
    return float(str(lat).replace("−", "-"))


def hillshade(dem, lat, az=315.0, alt=35.0, z=6.0):
    h, w = dem.shape
    dy = np.pi * R_MARS / h
    dx = 2 * np.pi * R_MARS / w * np.cos(np.radians(lat))[:, None]
    gy, gx = np.gradient(dem.astype(np.float32) * z)
    gx = gx / np.maximum(dx, 1.0)
    gy = -gy / dy
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azr, altr = np.radians(360.0 - az + 90.0), np.radians(alt)
    shade = np.sin(altr) * np.cos(slope) + np.cos(altr) * np.sin(slope) * np.cos(azr - aspect)
    return np.clip(shade, 0, 1)


def tint(elev):
    """Low-chroma hypsometric tint: lowlands cool gray, highlands warm light gray."""
    lo = np.array([150, 160, 170], np.float32)
    mid = np.array([196, 192, 184], np.float32)
    hi = np.array([232, 226, 214], np.float32)
    t = np.clip((elev + 6000.0) / 12000.0, 0, 1)[..., None]
    a = np.clip(t * 2, 0, 1)
    b = np.clip(t * 2 - 1, 0, 1)
    return lo * (1 - a) + mid * a * (1 - b) + hi * b


def read_supp(path, newtab=False):
    rows, cur = [], None
    for line in open(path, encoding="utf-8"):
        if "&" not in line or line.lstrip().startswith("%") or "\\thead" in line:
            continue
        c = [x.strip().rstrip("\\").strip() for x in line.split("&")]
        if len(c) < 12:
            continue
        oid = re.sub(r"\$\^\{[a-z]\}\$", "", c[1]).replace("\\", "").strip().upper()  # drop footnote marks such as $^{c}$
        if re.match(r"^(FRT|HRL|HRS|FRS)[0-9A-F]+$", oid):
            cur = oid
        if cur is None:
            continue
        rows.append((cur, c))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", default=None, help="omit to rebuild features_*.json only (basemaps kept)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lon0", type=float, nargs="+", default=[0.0])
    ap.add_argument("--fac", type=int, default=16, help="DEM decimation factor")
    ap.add_argument("--width", type=int, default=2400, help="output basemap width (px)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- detections -----------------------------------------------------------
    lit = [dict(id=r["crism_id"], lat=norm_lat(r["lat"]), lon=norm_lon(r["lon"]), ref=r["reference"],
                region=r["location"] or None)
           for r in csv.DictReader(open("data/detections/literature_detections.csv"))]
    st1 = list(csv.DictReader(open("data/tables/supp_table_1_reassessed_literature.csv")))
    st2 = list(csv.DictReader(open("data/tables/supp_table_2_new_detections.csv")))
    best = {}
    for r in st1:
        oid, cls = r["crism_id"], r["confidence"].split()[0] if r["confidence"] else ""
        if cls in CLASS_RANK and (oid not in best or CLASS_RANK[cls] < CLASS_RANK[best[oid]]):
            best[oid] = cls
    for d in lit:
        d["cls"] = best.get(d["id"], CLASS_OVERRIDES.get(d["id"]))  # None = no class recorded
    new = {}
    for r in st2:
        oid = r["crism_id"]
        if "Plagioclase" in r["signature"] and oid not in new and r["lon"]:
            new[oid] = dict(id=oid, lat=float(r["lat"]), lon=norm_lon(r["lon"]), region=r["region"],
                            cls=r["confidence"].split()[0] if r["confidence"] else "")
    new = list(new.values())

    # ---- DEM -> shaded relief (equirectangular) -----------------------------
    if a.dem:
        with rasterio.open(a.dem) as ds:
            h, w = ds.height // a.fac, ds.width // a.fac
            dem = ds.read(1, out_shape=(h, w), resampling=Resampling.average).astype(np.float32)
        lat_c = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
        shade = hillshade(dem, lat_c)
        rgb_eq = tint(dem) * (0.30 + 0.70 * shade[..., None])
        rgb_eq = np.clip(rgb_eq, 0, 255).astype(np.uint8)
        del shade

    for lon0 in a.lon0:
        crs = f"+proj=robin +lon_0={lon0} +R={R_MARS} +units=m +no_defs"
        fwd = Transformer.from_crs(f"+proj=longlat +R={R_MARS} +no_defs", crs, always_xy=True)
        inv = Transformer.from_crs(crs, f"+proj=longlat +R={R_MARS} +no_defs", always_xy=True)
        xmax = fwd.transform(lon0 + 179.9999, 0)[0]
        ymax = fwd.transform(lon0, 90)[1]
        W = a.width
        H = int(round(W * ymax / xmax))
        xs = (np.arange(W) + 0.5) / W * 2 * xmax - xmax
        ys = ymax - (np.arange(H) + 0.5) / H * 2 * ymax
        out = np.zeros((H, W, 4), np.uint8)
        for j0 in (range(0, H, 256) if a.dem else []):  # rows in blocks to limit memory
            yy, xx = np.meshgrid(ys[j0:j0 + 256], xs, indexing="ij")
            lon, lat = inv.transform(xx, yy)
            ok = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= 90)
            # Robinson inverse returns values outside the outline; reject by round trip
            fx, _ = fwd.transform(np.where(ok, lon, 0), np.where(ok, lat, 0))
            ok &= np.abs(fx - xx) < 2 * (2 * xmax / W)
            col = (((np.where(ok, lon, 0) + 180.0) % 360.0) / 360.0 * w).astype(int) % w
            row = np.clip(((90.0 - np.where(ok, lat, 0)) / 180.0 * h).astype(int), 0, h - 1)
            blk = out[j0:j0 + 256]
            blk[..., :3] = rgb_eq[row, col]
            blk[..., 3] = np.where(ok, 255, 0)
        tag = f"lon{int(lon0)}"
        if a.dem:
            Image.fromarray(out, "RGBA").save(os.path.join(a.out, f"basemap_{tag}.png"), optimize=True)

        def px(lat, lon):
            x, y = fwd.transform(lon, lat)
            return [round((x + xmax) / (2 * xmax) * W, 1), round((ymax - y) / (2 * ymax) * H, 1)]

        def line(lats, lons):
            pts = [px(la, lo) for la, lo in zip(lats, lons)]
            # break the line where it crosses the map edge
            segs, seg = [], [pts[0]]
            for p, q in zip(pts[:-1], pts[1:]):
                if abs(q[0] - p[0]) > W / 2:
                    segs.append(seg)
                    seg = []
                seg.append(q)
            segs.append(seg)
            return [s for s in segs if len(s) > 1]

        grat = []
        for lo in range(-180, 181, 30):
            lats = np.linspace(-90, 90, 91)
            grat += line(lats, np.full_like(lats, lo + 1e-6 if lo == -180 else lo))
        for la in range(-60, 61, 30):
            lons = np.linspace(lon0 - 179.999, lon0 + 179.999, 361)
            grat += line(np.full_like(lons, la), lons)
        b = TSC_BOX
        tl = np.r_[np.linspace(b["lon_min"], b["lon_max"], 60), np.full(30, b["lon_max"]),
                   np.linspace(b["lon_max"], b["lon_min"], 60), np.full(30, b["lon_min"])]
        tt = np.r_[np.full(60, b["lat_max"]), np.linspace(b["lat_max"], b["lat_min"], 30),
                   np.full(60, b["lat_min"]), np.linspace(b["lat_min"], b["lat_max"], 30)]
        tsc = line(tt, [norm_lon(v) for v in tl])
        outline = line(np.r_[np.linspace(-90, 90, 181), np.linspace(90, -90, 181)],
                       np.r_[np.full(181, lon0 - 179.999), np.full(181, lon0 + 179.999)])
        feat = dict(
            lon0=lon0, width=W, height=H,
            literature=[dict(d, xy=px(d["lat"], d["lon"])) for d in lit],
            new=[dict(d, xy=px(d["lat"], d["lon"])) for d in new],
            landing=[dict(name=n, lat=la, lon=lo, xy=px(la, lo)) for n, la, lo in LANDING_SITES],
            basins=[dict(name=n, lat=la, lon=lo, xy=px(la, lo)) for n, la, lo in BASINS],
            graticule=grat, tsc=tsc, outline=outline,
            lon_labels=[dict(lon=lo, xy=px(0, lo)) for lo in range(-150, 181, 30)],
            lat_labels=[dict(lat=la, xy=px(la, lon0 - 179.9)) for la in (-60, -30, 0, 30, 60)],
        )
        with open(os.path.join(a.out, f"features_{tag}.json"), "w") as f:
            json.dump(feat, f)
        print(tag, W, H, "literature", len(lit), "new", len(new))
    missing = sorted({d["id"] for d in lit if d["cls"] is None})
    print("literature observations not in Supplementary Table 1:", missing)


if __name__ == "__main__":
    main()
