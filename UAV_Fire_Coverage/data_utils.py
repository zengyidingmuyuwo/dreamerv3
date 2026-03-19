"""
data_utils.py — Data loading & coordinate conversion for UAV fire-coverage tasks.

Supports:
  - Circle-centre CSV  (latitude, longitude, radius_m)
  - Fire-point file    (.shp via pyshp  OR  .csv with latitude/longitude columns)
  - Elevation GeoTIFF  (.tif via rasterio)  — optional, for Circle 8

All geographic coordinates are projected to a flat 2-D local metre grid
centred at the circle centre.
"""

import csv
import os

import numpy as np


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def latlon_to_xy(lat, lon, lat0, lon0):
    """
    Convert (lat, lon) to local (x, y) metres relative to origin (lat0, lon0).
    Uses an equi-rectangular approximation suitable for areas < ~100 km.
    """
    R = 6_371_000.0                          # Earth radius [m]
    lat_rad = np.radians((lat + lat0) / 2)   # mid-latitude for x scaling
    dy = R * np.radians(lat - lat0)
    dx = R * np.cos(lat_rad) * np.radians(lon - lon0)
    return float(dx), float(dy)


# ---------------------------------------------------------------------------
# Circle centre
# ---------------------------------------------------------------------------

def load_circle_center(csv_path):
    """
    Load the first row of a circle-centre CSV.

    Expected header (case-insensitive):  latitude, longitude, radius_m

    Returns
    -------
    lat, lon : float
    radius_m : float
    """
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        row = next(reader)
    keys = {k.strip().lower(): v for k, v in row.items()}
    lat = float(keys['latitude'])
    lon = float(keys['longitude'])
    radius = float(keys.get('radius_m', keys.get('radius', 5000.0)))
    return lat, lon, radius


# ---------------------------------------------------------------------------
# Fire points
# ---------------------------------------------------------------------------

def load_fire_points(points_file, lat0, lon0):
    """
    Load fire points and return them as a (N, 2) float32 array of local
    (x, y) metres relative to the circle centre.

    Accepts:
      - .shp files  (requires pyshp / shapefile)
      - .csv files  (must have 'latitude' and 'longitude' columns)
    """
    ext = os.path.splitext(points_file)[1].lower()

    if ext == '.shp':
        return _load_shp(points_file, lat0, lon0)
    else:
        return _load_csv_points(points_file, lat0, lon0)


def _load_shp(path, lat0, lon0):
    try:
        import shapefile
    except ImportError:
        raise ImportError(
            "pyshp is required to read .shp files.  "
            "Install with:  pip install pyshp"
        )
    sf = shapefile.Reader(path)
    pts = []
    for shape in sf.shapes():
        if shape.shapeType in (1, 11, 21):   # Point / PointZ / PointM
            lon, lat = shape.points[0]
        elif shape.shapeType in (3, 5, 13, 15, 23, 25):
            # Take centroid of first part
            xs = [p[0] for p in shape.points]
            ys = [p[1] for p in shape.points]
            lon, lat = np.mean(xs), np.mean(ys)
        else:
            continue
        x, y = latlon_to_xy(lat, lon, lat0, lon0)
        pts.append([x, y])
    if not pts:
        raise ValueError(f"No point features found in {path}")
    return np.array(pts, dtype=np.float32)


def _load_csv_points(path, lat0, lon0):
    pts = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        keys_lower = None
        for row in reader:
            if keys_lower is None:
                keys_lower = {k.strip().lower(): k for k in row.keys()}
            lat_k = keys_lower.get('latitude', keys_lower.get('lat'))
            lon_k = keys_lower.get('longitude', keys_lower.get('lon'))
            if lat_k is None or lon_k is None:
                raise ValueError(
                    "CSV must contain 'latitude' and 'longitude' columns"
                )
            lat = float(row[lat_k])
            lon = float(row[lon_k])
            x, y = latlon_to_xy(lat, lon, lat0, lon0)
            pts.append([x, y])
    if not pts:
        raise ValueError(f"No rows found in {path}")
    return np.array(pts, dtype=np.float32)


# ---------------------------------------------------------------------------
# Elevation / obstacle map
# ---------------------------------------------------------------------------

def load_obstacle_map(tif_path, lat0, lon0, radius_m, grid_size=200,
                      threshold_m=2000.0):
    """
    Load a GeoTIFF elevation file and build a binary obstacle grid.

    Parameters
    ----------
    tif_path    : str    Path to elevation GeoTIFF.
    lat0, lon0  : float  Circle centre coordinates.
    radius_m    : float  Circle radius in metres.
    grid_size   : int    Number of grid cells per side (default 200).
    threshold_m : float  Elevation threshold; cells above this are obstacles.

    Returns
    -------
    obstacle_grid : np.ndarray  shape (grid_size, grid_size), dtype bool
                    True where the terrain blocks the UAV.
    cell_size     : float  metres per grid cell.
    """
    try:
        import rasterio
        from rasterio.transform import rowcol
    except ImportError:
        raise ImportError(
            "rasterio is required to read GeoTIFF elevation maps.  "
            "Install with:  pip install rasterio"
        )

    cell_size = 2 * radius_m / grid_size
    obstacle_grid = np.zeros((grid_size, grid_size), dtype=bool)

    with rasterio.open(tif_path) as src:
        crs_epsg = src.crs.to_epsg() if src.crs else None

        # Iterate over grid cells
        for iy in range(grid_size):
            for ix in range(grid_size):
                # Local metres → geographic
                x = -radius_m + (ix + 0.5) * cell_size
                y = -radius_m + (iy + 0.5) * cell_size
                # Approximate back-projection
                R = 6_371_000.0
                lat_c = np.radians(lat0)
                dlat = np.degrees(y / R)
                dlon = np.degrees(x / (R * np.cos(lat_c)))
                geo_lat = lat0 + dlat
                geo_lon = lon0 + dlon

                try:
                    row, col = rowcol(src.transform, geo_lon, geo_lat)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        elev = src.read(1)[row, col]
                        if elev > threshold_m:
                            obstacle_grid[iy, ix] = True
                except Exception:
                    pass   # out-of-bounds → not an obstacle

    return obstacle_grid, cell_size


def make_synthetic_obstacle_map(radius_m, grid_size=200):
    """
    Generate a synthetic obstacle map for use with sample data.
    Places a rectangular high-elevation region in one corner.
    """
    obstacle_grid = np.zeros((grid_size, grid_size), dtype=bool)
    # Add a barrier strip in the upper-right quadrant
    obstacle_grid[grid_size * 3 // 4:, grid_size * 3 // 4:] = True
    # Add a diagonal strip
    for i in range(grid_size // 4, grid_size // 2):
        j = i + grid_size // 8
        if j < grid_size:
            obstacle_grid[i, j:j + 8] = True
    cell_size = 2 * radius_m / grid_size
    return obstacle_grid, cell_size
