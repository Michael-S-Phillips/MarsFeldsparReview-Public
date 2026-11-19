#!/usr/bin/env python3
"""Recompute the DCI column of Supplementary Tables 1 and 2.

DCI is the TES dust cover index (Ruff and Christensen, 2002) at the ROI center, taken from the nearest pixel of
the 16 pixel-per-degree global map (lon -180..180 E, lat 90..-90). The map is not redistributed here; pass the
path to a GeoTIFF of it (any raster with that global 16 ppd grid works).

Usage (from the repository root):
    python scripts/tables/sample_dci.py path/to/tes_dci_16ppd.tif
Prints each ROI whose recomputed value differs from the table by more than 0.0005.
"""
import csv, sys
import numpy as np
import rasterio

with rasterio.open(sys.argv[1]) as ds:
    D = ds.read(1).astype(float)
NR, NC = D.shape


def dci(lat, lon):
    lon = ((lon + 180.0) % 360.0) - 180.0
    r = min(max(int(np.floor((90.0 - lat) * 16)), 0), NR - 1)
    c = min(max(int(np.floor((lon + 180.0) * 16)), 0), NC - 1)
    return D[r, c]


n = bad = 0
for fn in ("data/tables/supp_table_1_reassessed_literature.csv", "data/tables/supp_table_2_new_detections.csv"):
    for r in csv.DictReader(open(fn)):
        if not r["lon"] or not r["dci"]:
            continue
        v = dci(float(r["lat"]), float(r["lon"]))
        n += 1
        if abs(v - float(r["dci"])) > 0.0005:
            bad += 1
            print(f"{fn.split('/')[-1]} {r['crism_id']} {r['num_row']}/{r['num_col']}: table {r['dci']}, map {v:.4f}")
print(f"{n} ROIs checked, {bad} differ")
