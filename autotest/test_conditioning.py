import dany
from flopy.discretization import StructuredGrid
import numpy as np
from pathlib import Path


script_ws = Path(__file__).parent
data_ws = script_ws / "../data/dem_conditioning"

nrow = 5
ncol = 5
nlay = 1
sgrid = StructuredGrid(
    delc=np.full((nrow,), 10),
    delr=np.full((ncol,), 10),
    top=np.ones((nrow, ncol)),
    botm=np.zeros((nlay, nrow, ncol)),
    idomain=np.ones((nlay, nrow, ncol), dtype=int),
    nlay=nlay
)


def test_imp_eps_priority_flood():
    data_ws = Path("../data/dem_conditioning")
    test_array = data_ws / "sink_array.txt"
    cond_array = data_ws / "eps_conditioned_array.txt"

    ta = np.genfromtxt(test_array, delimiter=",")
    ca = np.genfromtxt(cond_array, delimiter=",")


    ca0 = dany.fill_sinks(sgrid, ta, method="priority")
    ca0 = ca0.reshape((nrow, ncol))

    if not np.allclose(ca, ca0):
        raise AssertionError(
            "Improved-EPS Priority flood method not returning proper values"
        )


def test_complete_priority_flood():
    test_array = data_ws / "sink_array.txt"
    cond_array = data_ws / "complete_conditioned_array.txt"

    ta = np.genfromtxt(test_array, delimiter=",")
    ca = np.genfromtxt(cond_array, delimiter=",")

    ca0 = dany.fill_sinks(sgrid, ta, method="complete")
    ca0 = ca0.reshape((nrow, ncol))

    if not np.allclose(ca, ca0):
        raise AssertionError(
            "Complete fill Priority flood method not returning proper values"
        )


def test_flood_and_drain_fill():
    test_array = data_ws / "sink_array.txt"
    cond_array = data_ws / "fd_fill_conditioned_array.txt"

    ta = np.genfromtxt(test_array, delimiter=",")
    ca = np.genfromtxt(cond_array, delimiter=",")

    ca0 = dany.fill_sinks(sgrid, ta, method="drain")
    ca0 = ca0.reshape((nrow, ncol))

    if not np.allclose(ca, ca0):
        raise AssertionError(
            "Flood and drain fill method not returning proper values"
        )


def test_nan_fill():
    array = np.array(
        [[1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, np.nan, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9]]
    )
    filled = dany.fill_nan_values(sgrid, array)
    if not np.isclose(filled[12], 5):
        raise AssertionError("Mean filling not working as intended")