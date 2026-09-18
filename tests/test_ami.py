"""Guard tests for the non-conformal AMI assembler.

The point of these is defensive: the module's source was once lost in a cleanup
and survived only as compiled bytecode. These tests fail loudly if that happens
again -- if the source file disappears (leaving only a .pyc), or if the public
API drifts from what the telescoping-mesh callers rely on.

They do NOT invoke OpenFOAM (the numpy-only test env has none); the functional
merge/couple/check path is exercised by the ``cap_to_mesh`` demo on CAP data.
"""
import inspect

from hexmb.assembly.ami import HybridAssembler
import hexmb.assembly.ami as hybrid


def test_source_present_not_only_bytecode():
    """The module must exist as readable .py source, not only a compiled .pyc."""
    assert hybrid.__file__.endswith(".py"), f"hybrid imported from {hybrid.__file__}, not source"


def test_public_api_matches_callers():
    """The callers (build_telescoping_ami, telescope_flow_run) use exactly these."""
    for name in ("merge", "couple", "check", "verify_flow"):
        assert callable(getattr(HybridAssembler, name)), f"missing method {name}"


def test_couple_signature():
    """couple(patch_pairs, check_overlap=True) and returns a list of triples --
    the caller does `for pa, pb, n in res`."""
    sig = inspect.signature(HybridAssembler.couple)
    params = list(sig.parameters)
    assert params[:2] == ["self", "patch_pairs"]
    assert sig.parameters["check_overlap"].default is True


def test_instantiable(tmp_path):
    a = HybridAssembler(tmp_path / "master")
    assert a.case == tmp_path / "master"
