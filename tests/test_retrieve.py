from __future__ import annotations

from agent.retrieve import _vector_to_list


class PgVectorLike:
    def __init__(self, values):
        self.values = values

    def to_list(self):
        return self.values


def test_vector_to_list_accepts_iterables() -> None:
    assert _vector_to_list((1, 2.5, "3")) == [1.0, 2.5, 3.0]


def test_vector_to_list_accepts_pgvector_like_objects() -> None:
    assert _vector_to_list(PgVectorLike([1, 2.5, "3"])) == [1.0, 2.5, 3.0]


def test_vector_to_list_accepts_none() -> None:
    assert _vector_to_list(None) is None
