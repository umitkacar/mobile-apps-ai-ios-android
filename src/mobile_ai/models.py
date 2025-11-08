"""AI models for mobile deployment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from pathlib import Path


class Detection(BaseModel):
    """Object detection result."""

    class_id: int = Field(..., description="Class ID of detected object")
    class_name: str = Field(..., description="Class name of detected object")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: list[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")


class YOLODetector:
    """YOLO object detector for mobile devices.

    Supports YOLOv8, YOLOv9, YOLOv10, and YOLOv11 models.

    Attributes:
        model_path: Path to the YOLO model file.
        confidence_threshold: Minimum confidence for detections.

    Example:
        >>> detector = YOLODetector(Path("yolov11n.onnx"))
        >>> detections = detector.detect(image)
    """

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model file.
            confidence_threshold: Minimum confidence threshold.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model: Any | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model."""
        # Implementation would load actual YOLO model

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect objects in image.

        Args:
            image: Input image as numpy array.

        Returns:
            List of Detection objects.

        Raises:
            ValueError: If model is not loaded.
            TypeError: If image is not a numpy array.
        """
        if self.model is None:
            msg = "Model not loaded"
            raise ValueError(msg)

        if not isinstance(image, np.ndarray):
            msg = "Image must be a numpy array"
            raise TypeError(msg)

        # Implementation would perform actual detection
        return []

    def export_to_coreml(self, output_path: Path) -> Path:
        """Export model to Core ML format for iOS.

        Args:
            output_path: Path to save Core ML model.

        Returns:
            Path to exported Core ML model.
        """
        # Implementation would convert to Core ML
        return output_path

    def export_to_tflite(self, output_path: Path) -> Path:
        """Export model to TensorFlow Lite for Android.

        Args:
            output_path: Path to save TFLite model.

        Returns:
            Path to exported TFLite model.
        """
        # Implementation would convert to TFLite
        return output_path


class SegmentationMask(BaseModel):
    """Segmentation mask result."""

    mask: list[list[int]] = Field(..., description="Binary segmentation mask")
    class_id: int = Field(..., description="Class ID of segmented object")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Segmentation confidence")


class SAMSegmenter:
    """Segment Anything Model (SAM) for mobile devices.

    Supports SAM, SAM 2, Mobile-SAM, and FastSAM models.

    Attributes:
        model_path: Path to the SAM model file.
        model_type: Type of SAM model (sam, sam2, mobile-sam, fast-sam).

    Example:
        >>> segmenter = SAMSegmenter(Path("mobile_sam.onnx"), model_type="mobile-sam")
        >>> masks = segmenter.segment(image, points=[[100, 100]])
    """

    def __init__(
        self,
        model_path: Path,
        model_type: str = "mobile-sam",
    ) -> None:
        """Initialize SAM segmenter.

        Args:
            model_path: Path to SAM model file.
            model_type: Type of SAM model.

        Raises:
            ValueError: If model_type is not supported.
        """
        supported_models = {"sam", "sam2", "mobile-sam", "fast-sam"}
        if model_type not in supported_models:
            msg = f"Unsupported model type: {model_type}"
            raise ValueError(msg)

        self.model_path = model_path
        self.model_type = model_type
        self.model: Any | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load SAM model."""
        # Implementation would load actual SAM model

    def segment(
        self,
        image: np.ndarray,
        points: list[list[int]] | None = None,
        boxes: list[list[int]] | None = None,
    ) -> list[SegmentationMask]:
        """Segment objects in image using prompts.

        Args:
            image: Input image as numpy array.
            points: List of point prompts [[x, y], ...].
            boxes: List of box prompts [[x1, y1, x2, y2], ...].

        Returns:
            List of SegmentationMask objects.

        Raises:
            ValueError: If model is not loaded or no prompts provided.
        """
        if self.model is None:
            msg = "Model not loaded"
            raise ValueError(msg)

        if points is None and boxes is None:
            msg = "Must provide either points or boxes for segmentation"
            raise ValueError(msg)

        # Implementation would perform actual segmentation
        return []

    def segment_everything(self, image: np.ndarray) -> list[SegmentationMask]:
        """Segment all objects in image automatically.

        Args:
            image: Input image as numpy array.

        Returns:
            List of SegmentationMask objects.

        Raises:
            ValueError: If model is not loaded.
        """
        if self.model is None:
            msg = "Model not loaded"
            raise ValueError(msg)

        # Implementation would perform automatic segmentation
        return []
