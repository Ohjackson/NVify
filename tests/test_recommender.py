"""Smoke tests for recommender scaffolding."""

import pytest

from engine import recommender


def test_recommend_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        recommender.recommend(state={"v": 0.0, "a": 0.0})


def test_rerank_two_users_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        recommender.rerank_two_users((0.0, 0.0), (0.0, 0.0))
