"""Smoke tests for utilities."""

import pytest

from engine import utils


def test_cosine_similarity_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        utils.cosine_similarity([1.0], [1.0])
