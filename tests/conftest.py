"""Pytest configuration and fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest


if TYPE_CHECKING:
    from pathlib import Path

    from mobile_ai.core import MobileAIConfig


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample test image.

    Returns:
        Random RGB image as numpy array.
    """
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Create a temporary model file path.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to temporary model file.
    """
    model_file = tmp_path / "model.onnx"
    model_file.touch()
    return model_file


@pytest.fixture
def mobile_ai_config(model_path: Path) -> MobileAIConfig:
    """Create a sample MobileAI configuration.

    Args:
        model_path: Path to model file.

    Returns:
        MobileAIConfig instance.
    """
    from mobile_ai.core import MobileAIConfig

    return MobileAIConfig(
        model_path=model_path,
        device="cpu",
        batch_size=1,
        confidence_threshold=0.5,
    )


@pytest.fixture
def detection_points() -> list[list[int]]:
    """Sample detection points for segmentation.

    Returns:
        List of [x, y] coordinates.
    """
    return [[100, 100], [200, 200], [300, 300]]


@pytest.fixture
def detection_boxes() -> list[list[int]]:
    """Sample bounding boxes for segmentation.

    Returns:
        List of [x1, y1, x2, y2] coordinates.
    """
    return [[50, 50, 150, 150], [200, 200, 300, 300]]
