"""
LiDAR Tree & Building Detector
------------------------------
Open-source LiDAR analysis that runs anywhere (no ArcGIS Pro needed). From a
LAS/LAZ point cloud it:

  1. Rasterizes a bare-earth DEM, a surface DSM, and a Canopy Height Model
     (CHM = DSM - DEM), the height of everything above the ground.
  2. Detects BUILDINGS  -> large, flat, elevated regions in the CHM.
  3. Detects TREES      -> local height maxima (treetops) in the CHM that are
     not part of a building.
  4. Writes GeoTIFFs, a detections summary, and a labeled visualization image.

Libraries: laspy, numpy, scipy, rasterio, matplotlib  (all pip-installable, Mac-friendly)

Run:
    python lidar_tree_building_detector.py data/sample_lidar.las --cell 1.0
"""

import sys, argparse
import numpy as np
import laspy
from scipy import ndimage
from scipy.ndimage import maximum_filter, label, generic_filter
import rasterio
from rasterio.transform import from_origin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ----------------------------------------------------------------------
# 1. Read + rasterize
# ----------------------------------------------------------------------
def read_las(path):
    las = laspy.read(path)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    cls = np.asarray(las.classification) if hasattr(las, "classification") else np.zeros(len(x), int)
    return x, y, z, cls


def rasterize(x, y, z, cell, reducer):
    """Bin points into a grid; reduce z per cell (min/max/mean). Returns grid, transform."""
    xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()
    ncols = int(np.ceil((xmax - xmin) / cell))
    nrows = int(np.ceil((ymax - ymin) / cell))
    col = np.clip(((x - xmin) / cell).astype(int), 0, ncols - 1)
    row = np.clip(((ymax - y) / cell).astype(int), 0, nrows - 1)  # north-up

    grid = np.full((nrows, ncols), np.nan)
    flat_idx = row * ncols + col
    order = np.argsort(flat_idx)
    fi, zz = flat_idx[order], z[order]
    # split into groups per cell
    bounds = np.searchsorted(fi, np.arange(fi.max() + 2))
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        if e > s:
            r, c = divmod(k, ncols)
            grid[r, c] = reducer(zz[s:e])
    transform = from_origin(xmin, ymax, cell, cell)
    return grid, transform


def fill_nans(a):
    """Fill empty cells with nearest-neighbor so rasters are continuous."""
    mask = np.isnan(a)
    if mask.any():
        idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
        a = a[tuple(idx)]
    return a


# ----------------------------------------------------------------------
# 2-3. Detect buildings + trees from the CHM
# ----------------------------------------------------------------------
def detect(chm, cell, min_height=2.0, tree_min=3.0):
    """Return building mask, list of building dicts, and tree (row,col,height) list."""
    from scipy.ndimage import gaussian_filter, uniform_filter
    chm_s = gaussian_filter(chm, sigma=1.0)        # smooth to suppress noise peaks

    # local roughness: std of height in a 3x3 window (buildings are smooth, trees rough)
    mean = uniform_filter(chm_s, size=3)
    mean_sq = uniform_filter(chm_s ** 2, size=3)
    roughness = np.sqrt(np.clip(mean_sq - mean ** 2, 0, None))

    # ---- BUILDINGS: elevated AND smooth (flat roof) AND sizeable ----
    flat_elevated = (chm_s > 2.5) & (roughness < 0.5)
    lbl, n = label(flat_elevated)
    building_mask = np.zeros_like(flat_elevated, bool)
    buildings = []
    cell_area = cell * cell
    for i in range(1, n + 1):
        blob = lbl == i
        area = blob.sum() * cell_area
        if area < 150:                             # buildings are sizeable
            continue
        building_mask |= blob
        ys, xs = np.where(blob)
        buildings.append({
            "area_m2": round(area, 1),
            "mean_height": round(float(np.mean(chm_s[blob])), 1),
            "row0": ys.min(), "row1": ys.max(), "col0": xs.min(), "col1": xs.max(),
        })

    # ---- TREES: well-separated local maxima (treetops), not on a building ----
    win = 5                                         # ~min spacing between treetops
    localmax = maximum_filter(chm_s, size=win)
    peaks = (chm_s > tree_min) & (~building_mask) & (chm_s == localmax)
    ys, xs = np.where(peaks)
    trees = [(int(r), int(c), round(float(chm[r, c]), 1)) for r, c in zip(ys, xs)]
    return building_mask, buildings, trees


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------
def write_tif(path, grid, transform):
    with rasterio.open(path, "w", driver="GTiff", height=grid.shape[0], width=grid.shape[1],
                       count=1, dtype="float32", transform=transform) as dst:
        dst.write(grid.astype("float32"), 1)


def make_viz(chm, buildings, trees, out_png):
    fig, ax = plt.subplots(figsize=(9, 8), dpi=130)
    im = ax.imshow(chm, cmap="YlGn", vmin=0, vmax=max(5, np.nanpercentile(chm, 99)))
    plt.colorbar(im, ax=ax, shrink=0.7, label="Height above ground (m)")
    # trees
    if trees:
        tr = np.array(trees)
        ax.scatter(tr[:, 1], tr[:, 0], s=12, c="#c0392b", marker="^",
                   label=f"Trees ({len(trees)})")
    # buildings
    for b in buildings:
        ax.add_patch(Rectangle((b["col0"], b["row0"]), b["col1"] - b["col0"], b["row1"] - b["row0"],
                     fill=False, edgecolor="#2c3e50", lw=2))
    if buildings:
        ax.add_patch(Rectangle((0, 0), 0, 0, fill=False, edgecolor="#2c3e50", lw=2,
                     label=f"Buildings ({len(buildings)})"))
    ax.set_title("LiDAR Canopy Height Model with detected trees and buildings")
    ax.legend(loc="upper right"); ax.axis("off")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Reusable pipeline (used by main() and by the ArcGIS notebook)
# ----------------------------------------------------------------------
def process(las_path, cell):
    """Run the full pipeline; return dem, dsm, chm, transform, buildings, trees."""
    x, y, z, cls = read_las(las_path)
    ground = cls == 2
    if ground.sum() > 100:
        dem, tr = rasterize(x[ground], y[ground], z[ground], cell, np.min)
    else:
        dem, tr = rasterize(x, y, z, cell, np.min)
    dem = fill_nans(dem)
    dsm, _ = rasterize(x, y, z, cell, np.max)
    dsm = fill_nans(dsm)
    chm = np.clip(dsm - dem, 0, None)
    building_mask, buildings, trees = detect(chm, cell)
    return dem, dsm, chm, tr, buildings, trees


def world_features(trees, buildings, transform, wkid=26918):
    """Convert pixel detections to real-world coordinates.
    Returns (tree_points, building_polygons, wkid). Default CRS UTM 18N (EPSG:26918)."""
    tree_points = []
    for (r, c, h) in trees:
        wx, wy = transform * (c + 0.5, r + 0.5)   # cell center
        tree_points.append({"x": wx, "y": wy, "height_m": h})
    building_polys = []
    for b in buildings:
        x0, y0 = transform * (b["col0"], b["row0"])
        x1, y1 = transform * (b["col1"] + 1, b["row1"] + 1)
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        building_polys.append({"ring": ring, "area_m2": b["area_m2"], "height_m": b["mean_height"]})
    return tree_points, building_polys, wkid


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("las", nargs="?", default="data/sample_lidar.las")
    ap.add_argument("--cell", type=float, default=1.0, help="raster cell size (map units)")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)
    print(f"Reading {args.las} ...")
    x, y, z, cls = read_las(args.las)
    print(f"  {len(x):,} points")

    # DEM from ground points (class 2 if available, else lowest returns)
    ground = cls == 2
    if ground.sum() > 100:
        dem, tr = rasterize(x[ground], y[ground], z[ground], args.cell, np.min)
    else:
        dem, tr = rasterize(x, y, z, args.cell, np.min)
    dem = fill_nans(dem)

    # DSM from all points (max = tops of features)
    dsm, _ = rasterize(x, y, z, args.cell, np.max)
    dsm = fill_nans(dsm)

    chm = np.clip(dsm - dem, 0, None)   # canopy height model

    write_tif(f"{args.out}/dem.tif", dem, tr)
    write_tif(f"{args.out}/dsm.tif", dsm, tr)
    write_tif(f"{args.out}/chm.tif", chm, tr)

    building_mask, buildings, trees = detect(chm, args.cell)
    make_viz(chm, buildings, trees, f"{args.out}/detections.png")

    # ---- summary ----
    print("\n" + "=" * 56)
    print("LiDAR DETECTION SUMMARY")
    print("=" * 56)
    print(f"Grid cell size        : {args.cell} m")
    print(f"CHM max height        : {chm.max():.1f} m")
    print(f"Trees detected        : {len(trees)}")
    if trees:
        hs = [t[2] for t in trees]
        print(f"  tallest tree        : {max(hs):.1f} m")
        print(f"  mean tree height    : {np.mean(hs):.1f} m")
    print(f"Buildings detected    : {len(buildings)}")
    for i, b in enumerate(buildings, 1):
        print(f"  building {i}: {b['area_m2']} m^2, ~{b['mean_height']} m tall")
    print("=" * 56)
    print(f"\nOutputs in ./{args.out}/ : dem.tif, dsm.tif, chm.tif, detections.png")


if __name__ == "__main__":
    main()
