"""
Generate a small SYNTHETIC LiDAR point cloud (.las) for testing the toolkit
without having to download real data. It simulates:
  - a gently rolling bare-earth surface (classified as ground, class 2)
  - clusters of "trees" as raised vegetation returns (class 5, high vegetation)
  - one flat-roofed "building" block (class 6)

Coordinates are in a local projected meter grid (UTM-like). All values invented.

Output: data/sample_lidar.las
"""
import numpy as np
import laspy

rng = np.random.default_rng(7)

# --- ground surface over a 500 x 500 m tile ---
N_GROUND = 120_000
gx = rng.uniform(0, 500, N_GROUND)
gy = rng.uniform(0, 500, N_GROUND)
# rolling terrain: a couple of low hills
gz = (100
      + 6*np.sin(gx/90) + 4*np.cos(gy/70)
      + 2*np.sin((gx+gy)/50)
      + rng.normal(0, 0.15, N_GROUND))
gcls = np.full(N_GROUND, 2, dtype=np.uint8)     # 2 = ground

def terrain_z(x, y):
    return 100 + 6*np.sin(x/90) + 4*np.cos(y/70) + 2*np.sin((x+y)/50)

# --- trees: clusters of raised points 4-22 m above ground ---
tx, ty, tz = [], [], []
for _ in range(140):                            # 140 tree clumps
    cx, cy = rng.uniform(20, 480), rng.uniform(20, 480)
    n = rng.integers(60, 200)
    px = cx + rng.normal(0, 2.5, n)
    py = cy + rng.normal(0, 2.5, n)
    height = rng.uniform(4, 22)
    pz = terrain_z(px, py) + rng.uniform(0.3, 1.0, n) * height
    tx.append(px); ty.append(py); tz.append(pz)
tx = np.concatenate(tx); ty = np.concatenate(ty); tz = np.concatenate(tz)
tcls = np.full(len(tx), 5, dtype=np.uint8)      # 5 = high vegetation

# --- one building: flat roof ~8 m tall ---
bx = rng.uniform(210, 260, 4000)
by = rng.uniform(210, 250, 4000)
bz = terrain_z(bx, by) + 8 + rng.normal(0, 0.1, 4000)
bcls = np.full(len(bx), 6, dtype=np.uint8)      # 6 = building

# --- combine ---
X = np.concatenate([gx, tx, bx])
Y = np.concatenate([gy, ty, by])
Z = np.concatenate([gz, tz, bz])
C = np.concatenate([gcls, tcls, bcls])

# place the 500 x 500 m tile at a REAL location: UTM Zone 18N (EPSG:26918),
# near Albany, NY, so it maps correctly on an ArcGIS basemap.
EAST0, NORTH0 = 600000.0, 4724000.0
X = X + EAST0
Y = Y + NORTH0

# --- write LAS 1.2, point format 3 ---
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = [X.min(), Y.min(), Z.min()]
header.scales = [0.01, 0.01, 0.01]
las = laspy.LasData(header)
las.x = X; las.y = Y; las.z = Z
las.classification = C
las.return_number = np.ones(len(X), dtype=np.uint8)
las.number_of_returns = np.ones(len(X), dtype=np.uint8)
las.write("data/sample_lidar.las")

print(f"Wrote data/sample_lidar.las with {len(X):,} points")
print(f"  ground: {len(gx):,}  vegetation: {len(tx):,}  building: {len(bx):,}")
print(f"  extent: X 0-500  Y 0-500  Z {Z.min():.1f}-{Z.max():.1f} m")
