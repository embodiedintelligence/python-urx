from collections import defaultdict
from typing import Dict, List

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st
from urx import Transform3D, mul_tf

# some light weight tests

@st.composite
def pose_vector_strategy(
    draw, min_value: float = 1.0 / 1024, max_value: float = 10
) -> st.SearchStrategy[np.array]:
    """Generates a `Dimension3D` object with coordinates in [min_value, max_value]."""

    dimension = draw(st.lists(length_strategy(min_value, max_value), min_size=6, max_size=6), label="dimension_3d")
    assume(all(dimension))
    return np.array(dimension)

@given(pose_vector=pose_vector_strategy())
def test_dimension_3d_scale(pose_vector: np.array):
    t = Transform3D(pose_vector)
    assert t.pose_vector == pose_vector
    t.pos == pose_vector[:3]
    t.ori
    assert mul_tf(t.inverse, t).pose == np.eye(4)
    assert t.dist(t) == 0
