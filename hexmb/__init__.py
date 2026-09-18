"""hexmb — general multi-block structured hexahedral (O-grid) mesh engine."""
from .core.block import Block, BoundaryFace
from .core.multiblock import MultiBlockMesh
from .core.topology import FaceKey

__all__ = ["Block", "BoundaryFace", "MultiBlockMesh", "FaceKey"]
__version__ = "0.1.0"
