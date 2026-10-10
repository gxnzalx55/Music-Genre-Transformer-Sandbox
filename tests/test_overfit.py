"""Sanity check: model must overfit a single synthetic batch."""
import pytest


def test_overfit_single_batch(tiny_cfg):
    """Train on one batch of 8 synthetic samples for 200 steps.

    Expects train accuracy ≥ 0.95 — validates the full forward/backward chain.
    """
    pytest.skip("Implemented in Phase 4")
