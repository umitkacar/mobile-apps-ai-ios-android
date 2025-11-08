"""Tests for models module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mobile_ai.models import Detection, SAMSegmenter, SegmentationMask, YOLODetector

if TYPE_CHECKING:
    pass


class TestDetection:
    """Tests for Detection model."""

    def test_detection_creation(self) -> None:
        """Test creating a detection object."""
        detection = Detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=[100.0, 100.0, 200.0, 200.0],
        )

        assert detection.class_id == 0
        assert detection.class_name == "person"
        assert detection.confidence == 0.95
        assert detection.bbox == [100.0, 100.0, 200.0, 200.0]

    def test_detection_validation_confidence(self) -> None:
        """Test confidence validation."""
        with pytest.raises(ValueError):
            Detection(
                class_id=0,
                class_name="person",
                confidence=1.5,  # Invalid
                bbox=[100.0, 100.0, 200.0, 200.0],
            )


class TestYOLODetector:
    """Tests for YOLODetector."""

    def test_initialization(self, model_path: Path) -> None:
        """Test YOLO detector initialization."""
        detector = YOLODetector(model_path, confidence_threshold=0.6)

        assert detector.model_path == model_path
        assert detector.confidence_threshold == 0.6

    def test_detect_without_model(
        self,
        model_path: Path,
        sample_image: np.ndarray,
    ) -> None:
        """Test detection without loaded model."""
        detector = YOLODetector(model_path)

        with pytest.raises(ValueError, match="Model not loaded"):
            detector.detect(sample_image)

    def test_detect_invalid_image(self, model_path: Path) -> None:
        """Test detection with invalid image."""
        detector = YOLODetector(model_path)
        detector.model = "dummy"  # Mock loaded model

        with pytest.raises(ValueError, match="Image must be a numpy array"):
            detector.detect("not_an_array")  # type: ignore[arg-type]

    def test_export_to_coreml(self, model_path: Path, tmp_path: Path) -> None:
        """Test exporting to Core ML format."""
        detector = YOLODetector(model_path)
        output_path = tmp_path / "model.mlmodel"

        result = detector.export_to_coreml(output_path)

        assert result == output_path

    def test_export_to_tflite(self, model_path: Path, tmp_path: Path) -> None:
        """Test exporting to TensorFlow Lite format."""
        detector = YOLODetector(model_path)
        output_path = tmp_path / "model.tflite"

        result = detector.export_to_tflite(output_path)

        assert result == output_path

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_detect_with_gpu(
        self,
        model_path: Path,
        sample_image: np.ndarray,
    ) -> None:
        """Test detection on GPU (requires GPU)."""
        pytest.skip("Requires GPU")


class TestSegmentationMask:
    """Tests for SegmentationMask model."""

    def test_mask_creation(self) -> None:
        """Test creating a segmentation mask."""
        mask = SegmentationMask(
            mask=[[0, 1], [1, 0]],
            class_id=1,
            confidence=0.89,
        )

        assert mask.mask == [[0, 1], [1, 0]]
        assert mask.class_id == 1
        assert mask.confidence == 0.89


class TestSAMSegmenter:
    """Tests for SAMSegmenter."""

    def test_initialization_mobile_sam(self, model_path: Path) -> None:
        """Test Mobile-SAM initialization."""
        segmenter = SAMSegmenter(model_path, model_type="mobile-sam")

        assert segmenter.model_path == model_path
        assert segmenter.model_type == "mobile-sam"

    def test_initialization_sam2(self, model_path: Path) -> None:
        """Test SAM 2 initialization."""
        segmenter = SAMSegmenter(model_path, model_type="sam2")

        assert segmenter.model_type == "sam2"

    def test_unsupported_model_type(self, model_path: Path) -> None:
        """Test initialization with unsupported model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            SAMSegmenter(model_path, model_type="invalid-model")

    def test_segment_without_model(
        self,
        model_path: Path,
        sample_image: np.ndarray,
        detection_points: list[list[int]],
    ) -> None:
        """Test segmentation without loaded model."""
        segmenter = SAMSegmenter(model_path)

        with pytest.raises(ValueError, match="Model not loaded"):
            segmenter.segment(sample_image, points=detection_points)

    def test_segment_without_prompts(
        self,
        model_path: Path,
        sample_image: np.ndarray,
    ) -> None:
        """Test segmentation without prompts."""
        segmenter = SAMSegmenter(model_path)
        segmenter.model = "dummy"  # Mock loaded model

        with pytest.raises(ValueError, match="Must provide either points or boxes"):
            segmenter.segment(sample_image)

    def test_segment_with_points(
        self,
        model_path: Path,
        sample_image: np.ndarray,
        detection_points: list[list[int]],
    ) -> None:
        """Test segmentation with point prompts."""
        segmenter = SAMSegmenter(model_path)
        segmenter.model = "dummy"  # Mock loaded model

        # Should not raise an error (actual implementation would return masks)
        result = segmenter.segment(sample_image, points=detection_points)
        assert isinstance(result, list)

    def test_segment_with_boxes(
        self,
        model_path: Path,
        sample_image: np.ndarray,
        detection_boxes: list[list[int]],
    ) -> None:
        """Test segmentation with box prompts."""
        segmenter = SAMSegmenter(model_path)
        segmenter.model = "dummy"  # Mock loaded model

        result = segmenter.segment(sample_image, boxes=detection_boxes)
        assert isinstance(result, list)

    def test_segment_everything_without_model(
        self,
        model_path: Path,
        sample_image: np.ndarray,
    ) -> None:
        """Test automatic segmentation without loaded model."""
        segmenter = SAMSegmenter(model_path)

        with pytest.raises(ValueError, match="Model not loaded"):
            segmenter.segment_everything(sample_image)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_segmentation_pipeline(
        self,
        model_path: Path,
        sample_image: np.ndarray,
    ) -> None:
        """Test complete segmentation pipeline."""
        pytest.skip("Requires actual SAM model file")


@pytest.mark.ios
class TestiOSIntegration:
    """Integration tests for iOS deployment."""

    def test_ios_model_export(self, model_path: Path, tmp_path: Path) -> None:
        """Test exporting models for iOS."""
        detector = YOLODetector(model_path)
        output = tmp_path / "yolo.mlmodel"

        result = detector.export_to_coreml(output)
        assert result.suffix == ".mlmodel"


@pytest.mark.android
class TestAndroidIntegration:
    """Integration tests for Android deployment."""

    def test_android_model_export(self, model_path: Path, tmp_path: Path) -> None:
        """Test exporting models for Android."""
        detector = YOLODetector(model_path)
        output = tmp_path / "yolo.tflite"

        result = detector.export_to_tflite(output)
        assert result.suffix == ".tflite"
