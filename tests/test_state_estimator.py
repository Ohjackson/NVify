"""Smoke tests for state estimator scaffolding."""

import pytest

from engine import state_estimator


def test_estimate_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        state_estimator.estimate({})
