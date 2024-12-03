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

contrib_area = 2.5e6
valid_stream_count = 66


def test_contributing_area():
    ca_as_cells = int(contrib_area / (dx * dy))

    filled_dem = dany.fill_sinks(grid, dem)
    fdobj = dany.FlowDirections(grid, filled_dem)
    fdobj.flow_directions()
    fdobj.flow_accumulation()

    strms0 = dany.PrmsStreams(grid, fdobj)
    strm_array = strms0.delineate_streams(contrib_area)
    count = np.count_nonzero(strm_array)

    if count != valid_stream_count:
        raise AssertionError(
            "Contributing area thresholding returning invalid number of stream cells"
        )

    filled_dem = dany.fill_sinks(grid, dem)
    fdobj = dany.FlowDirections(grid, filled_dem)
    fdobj.flow_directions()
    fdobj.flow_accumulation(as_cells=True)

    strms0 = dany.PrmsStreams(grid, fdobj)
    strm_array = strms0.delineate_streams(ca_as_cells)
    count = np.count_nonzero(strm_array)

    if count != valid_stream_count:
        raise AssertionError(
            "Contributing cells thresholding returning invalid number of stream cells"
        )


def test_stream_connectivity():
    valid_graph = {6: 7, 5: 7, 3: 4, 2: 4, 1: 8, 7: 8, 8: 9, 4: 9, 9: 10, 10: 10}

    filled_dem = dany.fill_sinks(grid, dem)
    fdobj = dany.FlowDirections(grid, filled_dem)
    fdobj.flow_directions()
    fdobj.flow_accumulation()

    strms0 = dany.PrmsStreams(grid, fdobj)
    strm_array = strms0.delineate_streams(contrib_area)
    grouped_graph = strms0.get_stream_connectivity(strm_array, group_segments=True)
    ungrouped_graph = strms0.get_stream_connectivity(strm_array, group_segments=False)

    if len(grouped_graph) != 10:
        raise AssertionError(
            "Number of grouped stream segments is incorrect"
        )

    if len(ungrouped_graph) != valid_stream_count:
        raise AssertionError(
            "Number of ungrouped stream segments not equal to number of stream cells"
        )

    for k, v in valid_graph.items():
        if grouped_graph[k] != v:
            raise AssertionError(
                "Topological sort is incorrect for stream connectivity"
            )


def test_prms_streams_cascades():
    hru_up_file = data_ws / "hru_up_id.txt"
    hru_dn_file = data_ws / "hru_down_id.txt"
    hru_pct_file = data_ws / "hru_pct_id.txt"
    hru_strm_file = data_ws / "hru_strmseg_dn_id.txt"

    valid_hru_up = np.genfromtxt(hru_up_file, dtype=int)
    valid_hru_dn = np.genfromtxt(hru_dn_file, dtype=int)
    valid_hru_pct = np.genfromtxt(hru_pct_file)
    valid_hru_strm = np.genfromtxt(hru_strm_file, dtype=int)

    filled_dem = dany.fill_sinks(grid, dem)
    fdobj = dany.FlowDirections(grid, filled_dem)
    fdobj.flow_directions()
    fdobj.flow_accumulation()

    strms0 = dany.PrmsStreams(grid, fdobj)
    strms0.delineate_streams(contrib_area)
    hru_up, hru_dn, hru_pct, hru_strm_dn = strms0.get_cascades(group_segments=True)

    if not np.allclose(hru_up, valid_hru_up):
        raise AssertionError(
            "Cascades hru_up_id not correct"
        )

    if not np.allclose(hru_dn, valid_hru_dn):
        raise AssertionError(
            "Cascades hru_dn_id not correct"
        )

    if not np.allclose(hru_pct, valid_hru_pct):
        raise AssertionError(
            "hru percent flow is not correct"
        )

    if not np.allclose(hru_strm_dn, valid_hru_strm):
        raise AssertionError(
            "hrus not draining to the correct stream segments"
        )



