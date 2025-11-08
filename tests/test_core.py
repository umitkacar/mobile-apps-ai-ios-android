"""Tests for core module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mobile_ai.core import MobileAI, MobileAIConfig


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class TestMobileAIConfig:
    """Tests for MobileAIConfig."""

    def test_config_creation(self, model_path: Path) -> None:
        """Test creating a valid configuration."""
        config = MobileAIConfig(
            model_path=model_path,
            device="cpu",
            batch_size=1,
            confidence_threshold=0.5,
        )

        assert config.model_path == model_path
        assert config.device == "cpu"
        assert config.batch_size == 1
        assert config.confidence_threshold == 0.5

    def test_config_validation_batch_size(self, model_path: Path) -> None:
        """Test batch size validation."""
        with pytest.raises(ValueError):
            MobileAIConfig(
                model_path=model_path,
                batch_size=0,  # Invalid: must be >= 1
            )

    def test_config_validation_confidence(self, model_path: Path) -> None:
        """Test confidence threshold validation."""
        with pytest.raises(ValueError):
            MobileAIConfig(
                model_path=model_path,
                confidence_threshold=1.5,  # Invalid: must be <= 1.0
            )

    def test_config_defaults(self, model_path: Path) -> None:
        """Test default configuration values."""
        config = MobileAIConfig(model_path=model_path)

        assert config.device == "cpu"
        assert config.batch_size == 1
        assert config.confidence_threshold == 0.5


class TestMobileAI:
    """Tests for MobileAI."""

    def test_initialization(self, mobile_ai_config: MobileAIConfig) -> None:
        """Test MobileAI initialization."""
        ai = MobileAI(mobile_ai_config)

        assert ai.config == mobile_ai_config
        assert ai.model is None  # Model not actually loaded in tests

    def test_predict_without_model(
        self,
        mobile_ai_config: MobileAIConfig,
        sample_image: np.ndarray,
    ) -> None:
        """Test prediction without loaded model."""
        ai = MobileAI(mobile_ai_config)

        with pytest.raises(ValueError, match="Model not loaded"):
            ai.predict(sample_image)

    def test_export_for_ios(
        self,
        mobile_ai_config: MobileAIConfig,
        tmp_path: Path,
    ) -> None:
        """Test exporting model for iOS."""
        ai = MobileAI(mobile_ai_config)
        output_path = tmp_path / "model.mlmodel"

        result = ai.export_for_mobile(output_path, platform="ios")

        assert result == output_path

    def test_export_for_android(
        self,
        mobile_ai_config: MobileAIConfig,
        tmp_path: Path,
    ) -> None:
        """Test exporting model for Android."""
        ai = MobileAI(mobile_ai_config)
        output_path = tmp_path / "model.tflite"

        result = ai.export_for_mobile(output_path, platform="android")

        assert result == output_path

    def test_export_unsupported_platform(
        self,
        mobile_ai_config: MobileAIConfig,
        tmp_path: Path,
    ) -> None:
        """Test exporting to unsupported platform."""
        ai = MobileAI(mobile_ai_config)
        output_path = tmp_path / "model.bin"

        with pytest.raises(ValueError, match="Unsupported platform"):
            ai.export_for_mobile(output_path, platform="webos")


@pytest.mark.slow
class TestMobileAIIntegration:
    """Integration tests for MobileAI."""

    @pytest.mark.integration
    def test_full_pipeline(
        self,
        mobile_ai_config: MobileAIConfig,
        sample_image: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """Test complete AI pipeline from load to export."""
        # This would be a full integration test
        # Skipped in unit tests as it requires actual model files
        pytest.skip("Requires actual model file")
