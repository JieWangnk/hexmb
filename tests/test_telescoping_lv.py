"""Tests for the parameterized telescoping-LV entry point.

Self-contained: builds on a synthetic prolate-spheroid ``surface`` so the zone
construction (the numpy-only part) is exercised without OpenFOAM. The merge/couple
path needs OpenFOAM and is exercised by the ``cap_to_mesh`` demo on CAP data.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexmb.services.quality import scaled_jac  # noqa: E402
from hexmb.templates.telescoping_lv import (  # noqa: E402
    TelescopingLVConfig, build_telescoping_lv, valve_despike,
)
from _synthetic import synthetic_lv  # noqa: E402


def test_config_defaults_are_the_validated_1M():
    cfg = TelescopingLVConfig()
    assert cfg.body_NT == 340 and cfg.body_NR == 24        # the ~1M configuration
    assert cfg.base_cut == 0.98 and cfg.valve_despike       # the valve fixes are on by default
    assert cfg.approx_cells > 900_000                       # body alone is ~1M


def test_build_three_valid_zones_with_patches():
    G, axis, e1, e2, mu = synthetic_lv()
    cfg = TelescopingLVConfig(body_NT=88, body_NR=6, body_levels=30)   # small, fast
    zones = build_telescoping_lv(G, axis, e1, e2, cfg, mu=mu)
    assert set(zones) == {"apex", "body", "base"}
    for name, m in zones.items():
        assert scaled_jac(m.points, m.hexes).min() > 0, f"{name} zone inverted"
    # patch tags the AMI coupling + BCs rely on
    assert "apexTop" in zones["apex"].patch_key_map()[0].values()
    bodyp = set(zones["body"].patch_key_map()[0].values())
    assert {"bodyBot", "bodyTop"} <= bodyp
    basep = set(zones["base"].patch_key_map()[0].values())
    assert {"baseBot", "inlet", "outlet1"} <= basep


def test_valve_despike_preserves_size_and_reduces_spikes():
    """The de-spike must cut high-frequency spikes without shrinking the annulus
    (the whole reason it is Fourier, not Laplacian)."""
    G, axis, e1, e2, mu = synthetic_lv()
    # add a spike to the top rings, then de-spike
    Gtop = G[-6:].copy()
    Gtop[:, ::7, 0] += 4.0                       # radial spikes every 7th point
    ring = Gtop[-1]
    r_before = np.linalg.norm(ring[:, :2] - ring[:, :2].mean(0), axis=1).mean()
    fixed = valve_despike(np.vstack([G[:-6], Gtop]))[-1]
    r_after = np.linalg.norm(fixed[:, :2] - fixed[:, :2].mean(0), axis=1).mean()
    # high-frequency content dropped, mean radius (size) essentially unchanged
    def hf(r):
        F = np.fft.rfft(r, axis=0); F[7:] = 0
        return np.linalg.norm(r - np.fft.irfft(F, n=len(r), axis=0), axis=1).max()
    assert hf(fixed) < hf(ring)
    assert abs(r_after - r_before) / r_before < 0.05
