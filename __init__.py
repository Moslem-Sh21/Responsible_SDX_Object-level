from ._version import __version__
from .experiment import *
from .heatmap import *
from .hook import *
from .utils import *
from .trace import *

# SDXL compatibility note
__sdxl_compatible__ = True
__supported_pipelines__ = [
    "StableDiffusionPipeline",
    "StableDiffusionXLPipeline"
]