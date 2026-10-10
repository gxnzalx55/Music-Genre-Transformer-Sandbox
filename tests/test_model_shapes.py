"""Tests for model forward-pass output shapes."""
import pytest


def test_ast_scratch_output_shape(tiny_cfg):
    """AST scratch: (2, 1, 128, 1000) input → (2, num_classes) output."""
    pytest.skip("Implemented in Phase 3")


def test_ast_pretrained_output_shape(tiny_cfg):
    """HuggingFace AST wrapper: (2, 1, 128, 1000) → (2, num_classes)."""
    pytest.skip("Implemented in Phase 3")


def test_ast_scratch_param_count(tiny_cfg):
    """AST scratch with default config should have 5-8 M trainable params."""
    pytest.skip("Implemented in Phase 3")


def test_spec_augment_shape(tiny_cfg):
    """SpecAugment must preserve input tensor shape."""
    pytest.skip("Implemented in Phase 3")
