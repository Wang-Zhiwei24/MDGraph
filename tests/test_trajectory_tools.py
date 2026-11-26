import numpy as np
import pytest

from mdgraph.trajectory_tools import pairs_within


@pytest.mark.parametrize("tilt", [0.0, 1.8])
def test_periodic_pair_tools_agree_across_boundary(tilt):
    pytest.importorskip("MDAnalysis")
    cell = np.array([[10.0, 0.0, 0.0], [tilt, 9.0, 0.0], [0.5, 0.2, 8.0]])
    positions = (
        np.array([[0.02, 0.03, 0.2], [0.96, 0.04, 0.2], [0.03, 0.91, 0.2], [0.4, 0.45, 0.7]]) @ cell
    )
    a = dict(
        zip(
            zip(*pairs_within("ase", positions, cell, 2.0)[:2]),
            pairs_within("ase", positions, cell, 2.0)[2],
        )
    )
    i, j, d = pairs_within("mdanalysis", positions, cell, 2.0)
    b = dict(zip(zip(i, j), d))
    assert len(a) == 6
    assert b == pytest.approx(a, abs=1e-5)


def test_mdanalysis_can_return_no_pairs():
    pytest.importorskip("MDAnalysis")
    i, j, d = pairs_within(
        "mdanalysis", np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]]), np.eye(3) * 10, 1.0
    )
    assert not len(i) and not len(j) and not len(d)
