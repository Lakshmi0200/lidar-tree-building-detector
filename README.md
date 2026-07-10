# LiDAR Tree & Building Detector (ArcGIS-integrated)

LiDAR analysis that detects **trees and buildings** from a point cloud, then
displays the results on an interactive **ArcGIS** map using the **ArcGIS API for
Python**.

The heavy LiDAR processing runs with open-source Python (so it works on macOS
without ArcGIS Pro); the ArcGIS API for Python handles the spatial display over a
real Esri basemap.

> The sample tile is placed near Albany, NY (UTM Zone 18N) so it maps in the
> right place. Sample data is synthetic; the tool works on real LAS/LAZ too.

## What it does
1. Builds a **DEM**, **DSM**, and **Canopy Height Model** (CHM = DSM - DEM)
2. Detects **trees** - treetops as well-separated local maxima in the CHM
3. Detects **buildings** - large, smooth, elevated regions (rough canopy vs
   smooth roofs, using local roughness)
4. Displays detected trees (points) and building footprints (polygons) on an
   **ArcGIS map** via the ArcGIS API for Python

On the sample it finds **1 building (~8 m)** and **~260 treetops**.

## Files
```
.
├── lidar_tree_building_detector.py   # LiDAR processing + detection (open-source)
├── lidar_arcgis_map.ipynb            # ArcGIS API for Python: display on an ArcGIS map
├── make_sample_las.py                # regenerates the synthetic sample (optional)
├── data/
│   └── sample_lidar.las              # 141k-pt sample near Albany (UTM 18N)
├── LICENSE
└── README.md
```

## Setup (macOS, using the conda `arcgis` environment)

```bash
conda activate arcgis        # the env where you installed arcgis
pip install laspy rasterio scipy matplotlib
```

## Run

**Option A - command line (rasters + detection summary + PNG):**
```bash
python lidar_tree_building_detector.py data/sample_lidar.las --cell 1.0
```
Creates `output/` with dem.tif, dsm.tif, chm.tif, and detections.png.

**Option B - ArcGIS map (the portfolio piece):**
```bash
jupyter notebook lidar_arcgis_map.ipynb
```
Run the cells top to bottom. The last cell shows the detected trees (green points)
and building (red polygon) on an interactive ArcGIS basemap near Albany.
Screenshot it for LinkedIn.

## Using real LiDAR
Download free LAS/LAZ tiles (they carry their own coordinate system) from
OpenTopography, USGS 3DEP, or the NY State GIS Clearinghouse. Point the tool at a
tile and set --cell to the point spacing.

## License
MIT. Sample data is synthetic and for demonstration only.
