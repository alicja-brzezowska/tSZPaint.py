import yaml
import abacusnbody.hod.abacus_hod as _hod_module
from abacusnbody.hod.abacus_hod import AbacusHOD

path2config = '/home/ab2927/rds/tSZPaint.py/abacusHOD/config.yaml'

config = yaml.safe_load(open(path2config))
sim_params = config['sim_params']
HOD_params = config['HOD_params']

# abacusutils >= 2.1.2 expects 'LightConeOrigins' but older data files use 'LCOrigins'.
# Monkey-patch asdf.open so the header transparently aliases the old key.
import asdf as _asdf
_original_open = _asdf.open

def _patched_open(filename, *args, **kwargs):
    af = _original_open(filename, *args, **kwargs)
    if 'header' in af:
        h = af['header']
        if 'LightConeOrigins' not in h and 'LCOrigins' in h:
            h['LightConeOrigins'] = h['LCOrigins']
    return af

_asdf.open = _patched_open
_hod_module.asdf = _asdf

ball = AbacusHOD(sim_params, HOD_params)

mock_dict = ball.run_hod(ball.tracers, want_rsd=True, write_to_disk=True, Nthread=8)

lrgs = mock_dict['LRG']
n_total = len(lrgs['x'])
print(f"Generated {n_total} LRGs")
print(f"Keys: {list(lrgs.keys())}")
print(f"Output written to: {ball.mock_dir}")
