import numpy as np
from rl.utils.np_utils import cross_diag


def test_cross_diag():
    v = np.random.random(size=(5, ))
    y = cross_diag(v)
    t = np.array([[0.0, 0.0, 0.0, 0.0, v[0]],
                  [0.0, 0.0, 0.0, v[1], 0.0],
                  [0.0, 0.0, v[2], 0.0, 0.0],
                  [0.0, v[3], 0.0, 0.0, 0.0],
                  [v[4], 0.0, 0.0, 0.0, 0.0]])

    assert np.array_equal(t, y)
