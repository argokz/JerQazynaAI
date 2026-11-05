"""Схема валидации: пространственные разбиения и кривые обнаружения."""

from jerqazyna.validation.metrics import (
    CaptureEstimate,
    bootstrap_capture,
    capture_at,
    capture_at_matched_area,
    detection_curve,
    estimate_capture,
)
from jerqazyna.validation.splits import blind_region_split, spatial_block_folds, spatial_blocks

__all__ = [
    "CaptureEstimate",
    "blind_region_split",
    "bootstrap_capture",
    "capture_at",
    "capture_at_matched_area",
    "detection_curve",
    "estimate_capture",
    "spatial_block_folds",
    "spatial_blocks",
]
