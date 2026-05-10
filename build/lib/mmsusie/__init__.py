"""Top-level package exports for mmsusie."""

from .gmatrix import agmat
from .mmsusie_sp import MMSuSiESp
from .mmsusie_dense import MMSuSiEDense
from .varcom import WeightEMAI, prepare_varcom_inputs

# Backward-compatible alias (old name before the Sp/Dense split)
MMSuSiE = MMSuSiESp

__version__ = "1.0"

__all__ = [
    "MMSuSiESp",
    "MMSuSiEDense",
    "MMSuSiE",
    "WeightEMAI",
    "prepare_varcom_inputs",
    "agmat",
    "__version__",
]
