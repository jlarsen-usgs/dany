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


def test_cascades_builder_object():
    hru_up_file = data_ws / "hru_up_id.txt"
    hru_dn_file = data_ws / "hru_down_id.txt"
    hru_pct_file = data_ws / "hru_pct_id.txt"
    hru_strm_file = data_ws / "hru_strmseg_dn_id.txt"
    slopes_file = data_ws / "slope.txt"
    aspect_file = data_ws / "aspect.txt"

    valid_hru_up = np.genfromtxt(hru_up_file, dtype=int)
    valid_hru_dn = np.genfromtxt(hru_dn_file, dtype=int)
    valid_hru_pct = np.genfromtxt(hru_pct_file)
    valid_hru_strm = np.genfromtxt(hru_strm_file, dtype=int)
    valid_slope = np.genfromtxt(slopes_file)
    valid_aspect = np.genfromtxt(aspect_file)

    fdobj = dany.FlowDirections(grid, dem)
    fdobj.flow_directions()
    fdobj.flow_accumulation()

    strms0 = dany.PrmsStreams(grid, fdobj)
    stream_array = strms0.delineate_streams(contrib_area)
    cascades_obj = strms0.get_pygsflow_builder_object(
        stream_array,
        group_segments=True
    )

    if cascades_obj.dany_flag != 1:
        raise AssertionError("Dany flag not set for pyGSFLOW")

    if not np.allclose(cascades_obj.hru_up_id, valid_hru_up):
        raise AssertionError(
            "Cascades hru_up_id not correct"
        )

    if not np.allclose(cascades_obj.hru_down_id, valid_hru_dn):
        raise AssertionError(
            "Cascades hru_dn_id not correct"
        )

    if not np.allclose(cascades_obj.hru_pct_up, valid_hru_pct):
        raise AssertionError(
            "hru percent flow is not correct"
        )

    if not np.allclose(cascades_obj.hru_strmseg_down_id, valid_hru_strm):
        raise AssertionError(
            "hrus not draining to the correct stream segments"
        )

    if not np.allclose(cascades_obj.hru_slope, valid_slope.ravel()):
        raise AssertionError(
            "Slope calculation is not correct"
        )

    if not np.allclose(cascades_obj.hru_aspect, valid_aspect.ravel()):
        raise AssertionError(
            "Aspect calculation is not correct"
        )


def test_sfr2005_reach_data():
    reach_data_file = data_ws / "reach_data.npy"
    fdobj = dany.FlowDirections(grid, dem)
    fdobj.flow_directions()
    fdobj.flow_accumulation()

    sfrstrms = dany.Sfr2005(grid, fdobj)
    strm_array = sfrstrms.delineate_streams(contrib_area)
    reach_data = sfrstrms.reach_data(group_segments=True)

    if len(reach_data) != valid_stream_count:
        raise AssertionError(
            "reach data block length is not consistent with number of valid reaches"
        )

    valid_reach_data = np.fromfile(reach_data_file, dtype=reach_data.dtype)

    if not np.allclose(reach_data.tolist(), valid_reach_data.tolist()):
        raise AssertionError(
            "SFR 2005 reach data is not consistent with valid reach data array"
        )


def test_sfr2005_segment_data():
    segment_data_file = data_ws / "segment_data.npy"
    fdobj = dany.FlowDirections(grid, dem)
    fdobj.flow_directions()
    fdobj.flow_accumulation()

    sfrstrms = dany.Sfr2005(grid, fdobj)
    strm_array = sfrstrms.delineate_streams(contrib_area)
    reach_data = sfrstrms.reach_data(group_segments=True)
    segment_data = sfrstrms.segment_data(group_segments=True)

    if segment_data.size != 10:
        raise AssertionError(
            "Segment data block has an incorrect number of segments for problem"
        )

    valid_segment_data = np.fromfile(segment_data_file, dtype=segment_data.dtype)

    if not np.allclose(segment_data.tolist(), valid_segment_data.tolist()):
        raise AssertionError(
            "SFR 2005 segment data is not consistent with valid segment data array"
        )


def test_mf6_connection_data():
    connection_data_file = data_ws / "connection_data.txt"
    valid_connection_data = []
    with open(connection_data_file) as cdf:
        for line in cdf:
            rec = tuple([int(i) for i in line.strip().split()])
            valid_connection_data.append(rec)

    fdobj = dany.FlowDirections(grid, dem)
    fdobj.flow_directions()
    fdobj.flow_accumulation()

    sfrstrms = dany.Sfr6(grid, fdobj)
    strm_array = sfrstrms.delineate_streams(contrib_area)

    connection_data = sfrstrms.connectiondata()

    for ix, rec in enumerate(connection_data):
        if not np.allclose(rec, valid_connection_data[ix]):
            raise AssertionError(
                "SFR6 connection data is not consistent with stored connection data"
            )


def test_mf6_package_data():
    package_data_file = data_ws / "package_data.npy"
    fdobj = dany.FlowDirections(grid, dem)
    fdobj.flow_directions()
    fdobj.flow_accumulation()

    sfrstrms = dany.Sfr6(grid, fdobj)
    strm_array = sfrstrms.delineate_streams(contrib_area)

    connection_data = sfrstrms.connectiondata()
    package_data = sfrstrms.packagedata(detuple=True)
    valid_package_data = np.fromfile(package_data_file, dtype=package_data.dtype)

    if not np.allclose(package_data.tolist(), valid_package_data.tolist()):
        raise AssertionError(
            "MF6 package data is not consistent with valid package data block"
        )
