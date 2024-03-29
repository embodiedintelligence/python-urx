from collections import defaultdict
from typing import Dict, List

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st
from urx.transform import Transform, get_pose_mat

# some light weight tests

@st.composite
def float_strategy(draw, min_value: float = -100, max_value: float = 100) -> st.SearchStrategy[float]:
    """Generates a float.

    Raises
    ------
    ValueError: If `max_value` < `min_value`.
    """
    if max_value < min_value:
        raise ValueError("`max_value` should not be less than `min_value`!")
    length = draw(st.floats(min_value, max_value, width=32), label="length")
    return length


@st.composite
def pose_vector_strategy(
    draw, min_value: float = -10, max_value: float = 10
) -> st.SearchStrategy[np.array]:
    """Generates a pose as represented by a vector of length 6.

    The first 3 values represent the position. The last 3 values are a rotation vector.
    """

    pos_vector = draw(st.lists(float_strategy(min_value, max_value), min_size=3, max_size=3), label="dimension_3d")
    rot_vector = draw(
        st.lists(
            float_strategy(min_value, max_value), min_size=3, max_size=3
        ).filter(lambda x: np.linalg.norm(x) > 1 / 1024),
        label="dimension_3d"
    )
    return np.array(pos_vector + rot_vector)

def test_transform_empty():
    """Tests that instantiating a empty Transform is an identity transform."""
    t = Transform()
    assert np.allclose(t.pose, np.eye(4))
    assert np.allclose(t.pose_vector, np.zeros(6,))


@given(pose_vector=pose_vector_strategy())
def test_transform_vector(pose_vector: np.array):
    """Tests that instantiating a Transform as a pose vector works."""

    t = Transform(pose_vector)
    assert np.allclose(get_pose_mat(t.pose_vector), get_pose_mat(pose_vector))
    t.pos == pose_vector[:3]
    t.ori
    assert np.allclose((t.inverse * t).pose, np.eye(4))
    assert t.dist(t) == 0


@given(pose_vector=pose_vector_strategy())
def test_transform_pose(pose_vector: np.array):
    """Tests that instantiating a Transform as a matrix works."""
    pose_mat = get_pose_mat(pose_vector)
    t = Transform(pose_mat)
    assert np.allclose(t.pose, pose_mat)
    assert np.allclose(get_pose_mat(t.pose_vector), get_pose_mat(pose_vector))
    t.pos == pose_vector[:3]
    t.ori
    assert np.allclose((t.inverse * t).pose, np.eye(4))
    assert t.dist(t) == 0
