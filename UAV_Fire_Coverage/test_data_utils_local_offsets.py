import numpy as np

from data_utils import local_offsets_from_projected_xy

LOCAL_COORD_ABS_P95_MAX_M = 300_000.0


def test_local_offsets_from_projected_xy_matches_center_relative_translation():
  lat_c = 25.0
  lon_c = 101.0
  pts_abs = np.array([
      [500000.0, 2765000.0],
      [500150.0, 2764800.0],
      [499900.0, 2765200.0],
  ], dtype=np.float64)

  local = local_offsets_from_projected_xy(pts_abs, lat_c, lon_c)

  assert local.shape == (3, 2)
  # Ensure values are local-scale (hundreds of km max), not raw projected
  # northings/eastings in the multi-million meter range.
  assert np.percentile(np.abs(local), 95) < LOCAL_COORD_ABS_P95_MAX_M
  # Offsets among points should be preserved by translation.
  delta_abs = pts_abs[1] - pts_abs[0]
  delta_local = local[1] - local[0]
  np.testing.assert_allclose(delta_local, delta_abs, rtol=1e-6, atol=1e-4)
