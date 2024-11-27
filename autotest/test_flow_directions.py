import dany
from flopy.discretization import StructuredGrid
import numpy as np
from pathlib import Path


data_ws = Path("../data/freyberg_synthetic")
dem = data_ws / "freyberg_multi-streams_synthetic.txt"
idomain = data_ws / "idomain.txt"

dem = np.genfromtxt(dem)
idomain = np.genfromtxt(idomain).astype(int)

nlay = 1
nrow = 40
ncol = 20
dx = dy = 250
delr = np.full((ncol,), dx)
delc = np.full((nrow,), dy)
top = dem
botm = np.zeros((nlay, nrow, ncol)) + dem

grid = StructuredGrid(
    delc=delc,
    delr=delr,
    top=top,
    botm=botm,
    idomain=idomain,
)


def test_flow_directions_filled():
    fdir_file = data_ws / "filled_flow_dirs.txt"
    valid_fdirs = np.genfromtxt(fdir_file, dtype=int)

    filled_dem = dany.fill_sinks(grid, dem)
    fdobj = dany.FlowDirections(grid, filled_dem)
    fdir = fdobj.flow_directions()

    if not np.allclose(fdir, valid_fdirs):
        raise AssertionError(
            "Flow directions routine not calculating the correct flow directions"
        )


def test_flow_directions_unfilled():
    fdir_file = data_ws / "unfilled_flow_dirs.txt"
    valid_fdirs = np.genfromtxt(fdir_file, dtype=int)

    fdobj = dany.FlowDirections(grid, dem)
    fdir = fdobj.flow_directions()
    np.savetxt(fdir_file, fdir, fmt="%d")

    if not np.allclose(fdir, valid_fdirs):
        raise AssertionError(
            "Flow directions routine not calculating the correct flow directions"
        )


def test_get_watershed():
    pass


def test_get_subbassins():
    pass


def test_flow_accumulation():
    pass


def test_flow_direction_attributes():
    pass


if __name__ == "__main__":
    test_flow_directions_filled()
    test_flow_directions_unfilled()

