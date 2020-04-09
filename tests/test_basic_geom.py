from collections import defaultdict
from typing import Dict, List

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st
from urx.robot import Transform3D, get_pose_mat

# some light weight tests

@st.composite
def float_strategy(draw, min_value: float = -100, max_value: float = 100) -> st.SearchStrategy[float]:
    """Generates a float.

    Raises
    ------
    ValueError: If `min_value` is not positive or `max_value` < `min_value`.
    """
    if max_value < min_value:
        raise ValueError("`max_value` should not be less than `min_value`!")
    length = draw(st.floats(min_value, max_value, width=32), label="length")
    return length


@st.composite
def pose_vector_strategy(
    draw, min_value: float = -10, max_value: float = 10
) -> st.SearchStrategy[np.array]:
    """Generates a `Dimension3D` object with coordinates in [min_value, max_value]."""

    pos_vector = draw(st.lists(float_strategy(min_value, max_value), min_size=3, max_size=3), label="dimension_3d")
    rot_vector = draw(
        st.lists(
            float_strategy(min_value, max_value), min_size=3, max_size=3
        ).filter(lambda x: np.linalg.norm(x) > 1 / 1024),
        label="dimension_3d"
    )
    return np.array(pos_vector + rot_vector)

@given(pose_vector=pose_vector_strategy())
def test_Transform3D_vector(pose_vector: np.array):
    t = Transform3D(pose_vector)
    assert np.allclose(get_pose_mat(t.pose_vector), get_pose_mat(pose_vector))
    t.pos == pose_vector[:3]
    t.ori
    assert np.allclose((t.inverse * t).pose, np.eye(4))
    assert t.dist(t) == 0

@given(pose_vector=pose_vector_strategy())
def test_Transform3D_pose(pose_vector: np.array):
    pose_mat = get_pose_mat(pose_vector)
    t = Transform3D(pose_mat)
    assert np.allclose(t.pose, pose_mat)
    assert np.allclose(get_pose_mat(t.pose_vector), get_pose_mat(pose_vector))
    t.pos == pose_vector[:3]
    t.ori
    assert np.allclose((t.inverse * t).pose, np.eye(4))
    assert t.dist(t) == 0
