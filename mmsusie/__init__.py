"""Top-level package exports for mmsusie."""

from .gmatrix import agmat
from .mmsusie import MMSuSiE
from .varcom import WeightEMAI, prepare_varcom_inputs
from .mmsusie_main import MMSuSiE

__version__ = "0.1.0"

__all__ = [
    "MMSuSiE",
    "WeightEMAI",
    "prepare_varcom_inputs",
    "agmat",
    "__version__",
]
