"""--mitral-deg / --aortic-deg reach the conformal template config (valve-position sensitivity runs)."""
from types import SimpleNamespace

import pytest

from hexmb import cli


def test_sector_options_set_config():
    cfg = cli._conformal_cfg(SimpleNamespace(mitral_deg="200,340", aortic_deg="350,125"))
    assert cfg.mitral_deg == (200.0, 340.0)
    assert cfg.aortic_deg == (350.0, 125.0)          # a sector may wrap through 0


def test_sector_defaults_unchanged():
    cfg = cli._conformal_cfg(SimpleNamespace())
    assert cfg.mitral_deg == (150.0, 340.0) and cfg.aortic_deg == (0.0, 125.0)


def test_sector_bad_input():
    with pytest.raises(ValueError, match="FROM,TO"):
        cli._sector("1,2,3", "mitral_deg")


def test_motion_parser_accepts_sectors():
    p = cli.build_parser() if hasattr(cli, "build_parser") else cli._parser()
    a = p.parse_args(["motion", "--surfaces", "x.exnode", "-o", "out", "--mitral-deg", "150,290"])
    assert a.mitral_deg == "150,290"
