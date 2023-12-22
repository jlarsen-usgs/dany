import os
import flopy
import matplotlib.pyplot as plt
from flopy.plot import PlotMapView
import numpy as np
import shapefile
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dem_conditioning import fill_sinks
from flow_directions import FlowDirections
from stream_util import PrmsStreams


def build_grid_instance(shp):
    """

    Parameters
    ----------
    shp : str
    """
    verts = {}
    icverts = []
    xcyc = []
    ivcnt = 0
    with shapefile.Reader(shp) as r:
        for ix, shape in enumerate(r.shapes()):
            icv = []
            points = shape.points[::-1]
            poly = np.array(list(points))
            center = list(np.mean(poly, axis=0))
            xcyc.append(center)
            for vert in points:
                if vert in verts:
                    iv = verts[vert]
                    if iv not in icv:
                        icv.append(iv)
                else:
                    icv.append(ivcnt)
                    verts[vert] = ivcnt
                    ivcnt += 1

            icverts.append(icv)

        ncpl = len(icverts)
        cell2d = []
        for ix, (xc, yc) in enumerate(xcyc):
            iverts = icverts[ix]
            cell2d_rec = [ix, xc, yc, len(iverts)] + list(iverts)
            cell2d.append(cell2d_rec)

        verts = [[v, k[0], k[1]] for k, v in verts.items()]
        idomain = np.ones((ncpl,))
        top = np.ones((ncpl,))
        botm = np.ones((1, ncpl,))

        grid = flopy.discretization.VertexGrid(
            vertices=verts,
            cell2d=cell2d,
            top=top,
            botm=botm,
            idomain=idomain,
            nlay=1,
            ncpl=ncpl
        )
        return grid


def make_stream_mask(shp, modelgrid):
    gix = flopy.utils.GridIntersect(modelgrid)
    mask = np.zeros((modelgrid.ncpl,), dtype=int)
    with shapefile.Reader(shp) as r:
        for shape in r.shapes():
            result = gix.intersect(shape)
            cellids = result.cellids.astype(int)
            mask[cellids] = 1

    return mask


if __name__ == "__main__":
    resample = False
    approach = 2
    ws = os.path.abspath(os.path.dirname(__file__))
    data_ws = os.path.join(ws, "data", "Lolo_Voronoi_Grid")
    grid_shp = os.path.join(data_ws, "LoloCr_voronoiGrid.shp")
    dem_file = os.path.join(data_ws, "USGS_13_lolo_utm.tif")
    dem_ascii = os.path.join(data_ws, "USGS_13_lolo_utm_dem_ascii.csv")
    fa_dem_grid_supplied = os.path.join(data_ws, "streams_grid_supplied_no_strm_correction.csv")

    vgrid = build_grid_instance(grid_shp)

    if resample:
        raster = flopy.utils.Raster.load(dem_file)
        dem = raster.resample_to_grid(vgrid, band=raster.bands[0], method="min")
        nanval = raster.nodatavals[0]
        idx = np.where(dem == nanval)[0]
        for ix in idx:
            neighs = vgrid.neighbors(ix)
            elevs = dem[neighs]
            fixed_elev = np.min(elevs)
            dem[ix] = fixed_elev
        np.savetxt(dem_ascii, dem)
    else:
        dem = np.genfromtxt(dem_ascii).ravel()

    # pmv = PlotMapView(modelgrid=vgrid)
    # pc = pmv.plot_array(conditioned_dem)
    # pmv.plot_grid()
    # plt.colorbar(pc)
    # plt.show()
    if approach == 1:
        conditioned_dem = fill_sinks(vgrid, dem)
        fobj = FlowDirections(vgrid, conditioned_dem)
        fdir = fobj.flow_direction_array
        facc = fobj.flow_accumulation()
        fx = np.log10(facc)
        cell_area = fobj._area

        contrib_area = np.full_like(facc, 10000000)
        contrib_area[cell_area < 10000] = 5000000

        strms = PrmsStreams(vgrid, fobj)
        strm_array = strms.delineate_streams(contrib_area=contrib_area)

        pmv = PlotMapView(modelgrid=vgrid)
        pc = pmv.plot_array(strm_array)
        # pmv.plot_grid()
        plt.colorbar(pc)
        plt.show()

        np.savetxt(fa_dem_grid_supplied, strm_array, fmt="%d")

    if approach == 2:
        stream_vector_file = os.path.join(data_ws, "Lolo_Streams_UTM.shp")
        stream_mask = make_stream_mask(stream_vector_file, vgrid)
        stream_v_cells = np.where(stream_mask > 0)[0]

        conditioned_dem = fill_sinks(vgrid, dem, stream_mask=stream_mask)
        conditioned_dem = fill_sinks(vgrid, conditioned_dem)

        fobj = FlowDirections(vgrid, conditioned_dem)
        # fdir = fobj.flow_direction_array
        facc = fobj.flow_accumulation()
        cell_area = fobj._area

        stream_cell_area = np.max(cell_area[stream_v_cells])
        print(stream_cell_area)
        contrib_area = np.full_like(facc, 50000000)
        contrib_area[cell_area < stream_cell_area * 1.25] = 1000000

        strms = PrmsStreams(vgrid, fobj)
        strm_array = strms.delineate_streams(contrib_area=contrib_area)

        pmv = PlotMapView(modelgrid=vgrid)
        pc = pmv.plot_array(strm_array)
        # pmv.plot_grid()
        plt.colorbar(pc)
        plt.show()

        print('break')

    if approach == 3:
        pass
        # this is a cellular approach, re-implement cellular NIDP for flow_accumulation & contrib_area
    # todo: consider developing a workflow starting at structured FA
    #   define stream vectors then create a voronoi and re-FA.