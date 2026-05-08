#!/usr/bin/env python3
"""Precompute Circle8 DEM obstacle mask (>2000m) for clean rendering."""

import argparse
import math
import os
import sys

import numpy as np


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
UAV_DIR = os.path.join(REPO_DIR, 'UAV_Fire_Coverage')
PREPARE_DIR = os.path.join(UAV_DIR, 'prepare')
if UAV_DIR not in sys.path:
    sys.path.insert(0, UAV_DIR)

from data_utils import load_circle_center_csv, resolve_elevation_raster_path  # noqa: E402


def _pick_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _find_dem_candidate(prepare_dir):
    if not os.path.isdir(prepare_dir):
        return None
    names = sorted(os.listdir(prepare_dir))
    for name in names:
        low = name.lower()
        if low.endswith(('.zip', '.tif', '.tiff')):
            return os.path.join(prepare_dir, name)
    elev_dir = os.path.join(prepare_dir, 'elevation')
    if os.path.isdir(elev_dir):
        for name in sorted(os.listdir(elev_dir)):
            low = name.lower()
            if low.endswith(('.zip', '.tif', '.tiff')):
                return os.path.join(elev_dir, name)
    return None


def _crop_dem_to_circle8_mask(
    dem_path,
    lat_center,
    lon_center,
    radius_m=15000.0,
    threshold_m=2000.0,
    resolution_m=50.0,
):
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling
    from pyproj import Transformer

    raster_path = resolve_elevation_raster_path(dem_path)
    size_pix = max(2, int(round((2.0 * radius_m) / max(resolution_m, 1e-6))))
    dlat = radius_m / 111_000.0
    dlon = radius_m / (111_000.0 * math.cos(math.radians(lat_center)))
    min_lon, max_lon = lon_center - dlon, lon_center + dlon
    min_lat, max_lat = lat_center - dlat, lat_center + dlat

    with rasterio.open(raster_path) as src:
        if src.crs and str(src.crs).upper() != 'EPSG:4326':
            to_src = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
            min_x, min_y = to_src.transform(min_lon, min_lat)
            max_x, max_y = to_src.transform(max_lon, max_lat)
            left, right = sorted([min_x, max_x])
            bottom, top = sorted([min_y, max_y])
            window = from_bounds(left, bottom, right, top, src.transform)
        else:
            window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
        elev = src.read(
            1,
            window=window,
            out_shape=(size_pix, size_pix),
            resampling=Resampling.bilinear,
        )
        nodata = src.nodata

    if nodata is not None:
        elev = np.where(np.isclose(elev, nodata), np.nan, elev)
    mask = np.isfinite(elev) & (elev > float(threshold_m))
    xs = np.linspace(-radius_m, radius_m, size_pix, dtype=np.float32)
    ys = np.linspace(-radius_m, radius_m, size_pix, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    circle = (xx ** 2 + yy ** 2) <= (radius_m ** 2)
    return (mask & circle).astype(bool)


def main():
    parser = argparse.ArgumentParser(description='Preprocess Circle8 DEM to binary obstacle mask.')
    parser.add_argument('--circle_id', type=int, default=8)
    parser.add_argument('--radius_m', type=float, default=15000.0)
    parser.add_argument('--threshold_m', type=float, default=2000.0)
    parser.add_argument('--resolution_m', type=float, default=50.0)
    parser.add_argument('--center_csv', type=str, default='')
    parser.add_argument('--dem', type=str, default='')
    parser.add_argument('--output', type=str, default=os.path.join(PREPARE_DIR, 'circle8_obstacle_mask.npy'))
    args = parser.parse_args()

    center_csv = args.center_csv or _pick_existing(
        os.path.join(PREPARE_DIR, 'circle_8_center.csv'),
    )
    if center_csv is None:
        raise FileNotFoundError('circle_8_center.csv not found in UAV_Fire_Coverage/prepare.')
    dem_src = args.dem or _find_dem_candidate(PREPARE_DIR)
    if dem_src is None:
        raise FileNotFoundError('No DEM .zip/.tif found under UAV_Fire_Coverage/prepare.')

    lat_c, lon_c, csv_radius = load_circle_center_csv(center_csv, circle_id=args.circle_id)
    radius_m = float(args.radius_m or csv_radius)
    mask = _crop_dem_to_circle8_mask(
        dem_path=dem_src,
        lat_center=lat_c,
        lon_center=lon_c,
        radius_m=radius_m,
        threshold_m=args.threshold_m,
        resolution_m=args.resolution_m,
    )
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, mask.astype(bool))

    print(f'[preprocess_dem] center_csv={center_csv}')
    print(f'[preprocess_dem] dem={dem_src}')
    print(f'[preprocess_dem] output={out_path}')
    print(f'[preprocess_dem] shape={mask.shape}, obstacle_pixels={int(mask.sum())}')


if __name__ == '__main__':
    main()
