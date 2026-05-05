import numpy as np

from rxnfit.lsoda_p_vector import build_lsoda_parameter_vector_p


def test_p_order_example():
    p = build_lsoda_parameter_vector_p(
        {"k2": 2.0, "k1": 1.0, "k3": 3.0, "k4": 4.0},
        ("k2", "k1"),
    )
    assert list(p) == [2.0, 1.0, 3.0, 4.0]
