"""Mobile AI Apps - iOS & Android.

A modern Python package for mobile AI development with SOTA models.
"""

__version__ = "0.1.0"
__author__ = "Mobile AI Team"
__email__ = "team@mobile-ai.dev"

from mobile_ai.core import MobileAI
from mobile_ai.models import YOLODetector, SAMSegmenter

__all__ = [
    "MobileAI",
    "YOLODetector",
    "SAMSegmenter",
    "__version__",
]
