"""Regression coverage for motion NPZ units, early input rejection and wrapped sectors."""
from types import SimpleNamespace

import numpy as np
import pytest

from _synthetic import synthetic_lv
from hexmb import cli
from hexmb.services.motion import build_lv_motion
from hexmb.templates.lv import LVChamberTemplate


@pytest.mark.parametrize('unit,factor', [('mm', 1), ('cm', .1), ('m', .001)])
def test_motion_npz_units_and_frame(tmp_path, unit, factor):
    G, axis, e1, e2, mu = synthetic_lv()
    path = tmp_path / 'motion.npz'
    np.savez(path, grids=np.array([G, G * .95]) * factor,
             axis=axis, e1=e1, e2=e2, mu=mu * factor)
    a = cli.build_parser().parse_args(['motion', '--surfaces', str(path), '--units', unit, '-o', str(tmp_path/'out')])
    grids, frame = cli._load_phase_grids(a, a.surfaces)
    np.testing.assert_allclose(grids[0], G)
    np.testing.assert_allclose(frame[3], mu)
    np.testing.assert_allclose(frame[1], e1)


def test_motion_npz_wrong_units_and_empty_are_rejected(tmp_path):
    G = synthetic_lv()[0]
    path = tmp_path / 'motion.npz'
    np.savez(path, grids=np.array([G, G]) * .001)
    a = cli.build_parser().parse_args(['motion', '--surfaces', str(path), '-o', str(tmp_path/'out')])
    with pytest.raises(ValueError, match='--units m'):
        cli._load_phase_grids(a, a.surfaces)
    np.savez(path, grids=np.empty((0, 40, 56, 3)))
    with pytest.raises(ValueError, match='NPZ grids'):
        cli._load_phase_grids(a, a.surfaces)


@pytest.mark.parametrize('kw', [dict(times=[0, np.nan]), dict(times=[0, np.inf]),
    dict(times=[-.1, .2]), dict(period=np.nan), dict(period=-1),
    dict(reference=-1), dict(reference=2), dict(scale=np.nan), dict(interpolation='cubic')])
def test_motion_rejects_bad_metadata_before_building(tmp_path, kw):
    G = synthetic_lv()[0]
    args = dict(times=[0, .4], builder='single', out=tmp_path/'case', verbose=False)
    args.update(kw)
    with pytest.raises(ValueError):
        build_lv_motion([G, G], **args)
    assert not (tmp_path/'case').exists()


def test_motion_rejects_nonfinite_coordinates():
    G = synthetic_lv()[0]
    G[1, 1, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        build_lv_motion([G, G], [0, .4], verbose=False)


@pytest.mark.parametrize('text', ['nan,0,0', 'inf,1,2'])
def test_nonfinite_landmark(text):
    with pytest.raises(ValueError):
        cli._xyz(text)


@pytest.mark.parametrize('text', ['nan,20', '10,inf'])
def test_nonfinite_sector(text):
    with pytest.raises(ValueError):
        cli._sector(text, 'mitral_deg')


def test_single_valve_sector_wraps_through_zero():
    G, axis, e1, e2, mu = synthetic_lv()
    template = LVChamberTemplate(G, axis, e1, e2, mu=mu,
                                 aortic_deg=(350, 20), mitral_deg=(100, 250))
    point = lambda deg: mu + np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0])
    assert template._base_patch(point(355)) == 'outlet1'
    assert template._base_patch(point(10)) == 'outlet1'
    assert template._base_patch(point(170)) == 'inlet'
    assert template._base_patch(point(300)) == 'wall'


@pytest.mark.parametrize('n', ['0', '1', '21'])
def test_demo_rejects_invalid_phase_count(tmp_path, n):
    out = tmp_path / 'demo'
    assert cli.main(['demo', '--phases', n, '-o', str(out)]) == 1
    assert not out.exists()


def test_static_phase_builder_rejects_nonfinite_grid():
    from hexmb.services.motion import build_phases
    G, axis, e1, e2, mu = synthetic_lv()
    G[3, 4, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_phases([G], (axis, e1, e2, mu), verbose=False)
