import dany
from flopy.discretization import StructuredGrid
import numpy as np
from pathlib import Path


script_ws = Path(__file__)
data_ws = script_ws / "../data/freyberg_synthetic"
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

    if not np.allclose(fdir, valid_fdirs):
        raise AssertionError(
            "Flow directions routine not calculating the correct flow directions"
        )


def test_get_watershed():
    watershed_file = data_ws / "watershed_boundary.txt"
    valid_wshed = np.genfromtxt(watershed_file, dtype=int)

    fdobj = dany.FlowDirections(grid, dem)
    fdir = fdobj.flow_directions()
    point = (grid.xcellcenters[39, 15], grid.ycellcenters[39, 15])

    watershed = fdobj.get_watershed_boundary(point)

    if not np.allclose(watershed, valid_wshed):
        raise AssertionError(
            "get_watershed not calculating the correct divide locations"
        )


def test_get_subbassins():
    subbasin_file = data_ws / "subbasins.txt"
    valid_subs = np.genfromtxt(subbasin_file, dtype=int)

    locs = [[39, 15], [30, 12], [29, 13]]
    points = [[grid.xcellcenters[r, c], grid.ycellcenters[r, c]] for r, c in locs]

    fdobj = dany.FlowDirections(grid, dem)
    fdir = fdobj.flow_directions()
    subbasins = fdobj.get_subbasins(points)

    if not np.allclose(subbasins, valid_subs):
        raise AssertionError(
            "get_subbasins not calculating the correct contrib. areas"
        )


def test_flow_accumulation_as_area():
    fa_file = data_ws / "flow_accumulation_area.txt"
    valid_fa = np.genfromtxt(fa_file, dtype=int)

    fdobj = dany.FlowDirections(grid, dem)
    fdir = fdobj.flow_directions()
    facc = fdobj.flow_accumulation()


    if not np.allclose(facc, valid_fa):
        raise AssertionError(
            "flow accumulation as area not returning correct values"
        )


def test_flow_accumulation_as_cells():
    fa_file = data_ws / "flow_accumulation_as_cells.txt"
    valid_fa = np.genfromtxt(fa_file, dtype=int)

    fdobj = dany.FlowDirections(grid, dem)
    fdir = fdobj.flow_directions()
    facc = fdobj.flow_accumulation(as_cells=True)

    if not np.allclose(facc, valid_fa):
        raise AssertionError(
            "flow accumulation as cells not returning correct values"
        )


def test_flow_direction_attributes():
    slopes_file = data_ws / "slope.txt"
    aspect_file = data_ws / "aspect.txt"
    hru_len_file = data_ws / "hru_len.txt"
    vu_file = data_ws / "vector_u_file.txt"
    vv_file = data_ws / "vector_v_file.txt"

    valid_slope = np.genfromtxt(slopes_file)
    valid_aspect = np.genfromtxt(aspect_file)
    valid_hru_len = np.genfromtxt(hru_len_file)
    valid_vu = np.genfromtxt(vu_file)
    valid_vv = np.genfromtxt(vv_file)

    fdobj = dany.FlowDirections(grid, dem)
    fdir = fdobj.flow_directions()

    if not np.allclose(fdobj.slope, valid_slope):
        raise AssertionError(
            "Slope calculation is not correct"
        )

    if not np.allclose(fdobj.aspect, valid_aspect):
        raise AssertionError(
            "Aspect calculation is not correct"
        )

    if not np.allclose(fdobj.hru_len, valid_hru_len):
        raise AssertionError(
            "hru_len calculation is not correct"
        )

    u, v = fdobj.vectors

    if not np.allclose(u, valid_vu) or not np.allclose(v, valid_vv):
        raise AssertionError(
            "unit circle vector calculation is not correct"
        )
