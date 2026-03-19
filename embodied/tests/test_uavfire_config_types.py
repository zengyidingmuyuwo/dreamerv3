from pathlib import Path

import ruamel.yaml as yaml


def _load_configs():
  root = Path(__file__).resolve().parents[2]
  text = (root / 'dreamerv3' / 'configs.yaml').read_text()
  return yaml.YAML(typ='safe').load(text)


def test_uavfire_optional_overrides_are_not_null():
  cfg = _load_configs()
  uavfire = cfg['defaults']['env']['uavfire']
  assert uavfire['circle1_center_csv'] == ''
  assert uavfire['circle1_points_file'] == ''
  assert uavfire['circle1_circle_id'] == -1
  assert uavfire['circle8_center_csv'] == ''
  assert uavfire['circle8_points_file'] == ''
  assert uavfire['circle8_circle_id'] == -1
  assert uavfire['elevation_tif'] == ''


def test_uavfire_circle_id_sentinel_normalization():
  from UAV_Fire_Coverage.dreamer_uav_env import _normalize_circle_id
  assert _normalize_circle_id(-1) is None
  assert _normalize_circle_id('-1') is None
  assert _normalize_circle_id('') is None
  assert _normalize_circle_id(None) is None
  assert _normalize_circle_id(8) == 8
