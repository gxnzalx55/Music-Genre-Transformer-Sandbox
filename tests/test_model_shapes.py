"""Tests for model forward-pass output shapes."""
import pytest


def test_ast_scratch_output_shape(tiny_cfg):
    """AST scratch: (2, 1, 128, 1000) input → (2, num_classes) output."""
    import torch
    from models.factory import build_model

    model = build_model(tiny_cfg.model, tiny_cfg.dataset).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 128, 1001))
    assert out.shape == (2, tiny_cfg.dataset.num_classes)


def test_ast_pretrained_output_shape(tiny_cfg):
    """HuggingFace AST wrapper: (2, 1, 128, 1000) → (2, num_classes)."""
    pytest.skip("Implemented in Phase 3")


def test_ast_scratch_param_count(tiny_cfg):
    """AST scratch with default config should have 5-8 M trainable params."""
    pytest.skip("Implemented in Phase 3")


def test_spec_augment_shape(tiny_cfg):
    """SpecAugment must preserve input tensor shape."""
    import torch
    from data.transforms import SpecAugment

    aug = SpecAugment().train()
    for shape in [(3, 1, 128, 1001), (1, 128, 1001)]:
        x = torch.ones(*shape)
        out = aug(x)
        assert out.shape == x.shape
        assert (out == 0).any()
