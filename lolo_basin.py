import os
import flopy
import matplotlib.pyplot as plt
from flopy.plot import PlotMapView
from flopy.utils.triangle import Triangle
from flopy.utils.voronoi import VoronoiGrid
from flopy.discretization import VertexGrid
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import shapefile
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dem_conditioning import fill_sinks
from flow_directions import FlowDirections
from stream_util import PrmsStreams

from gsflow.builder import GenerateFishnet



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


def get_grid_bounds(shp):
    with shapefile.Reader(shp) as r:
        bounds = r.bbox

    polygon = [
        (bounds[0], bounds[1]),
        (bounds[0], bounds[3]),
        (bounds[2], bounds[3]),
        (bounds[2], bounds[1]),
        (bounds[0], bounds[1])
    ]
    return [bounds[0], bounds[2], bounds[1], bounds[3]], polygon


if __name__ == "__main__":
    resample = False
    approach = 4
    ws = os.path.abspath(os.path.dirname(__file__))
    data_ws = os.path.join(ws, "data", "Lolo_Voronoi_Grid")
    tri_ws = os.path.join(ws, "data", "triangle_grid")
    grid_shp = os.path.join(data_ws, "LoloCr_voronoiGrid.shp")
    dem_file = os.path.join(data_ws, "USGS_13_lolo_utm.tif")
    dem_ascii = os.path.join(data_ws, "USGS_13_lolo_utm_dem_ascii.csv")
    fa_dem_grid_supplied = os.path.join(data_ws, "streams_grid_supplied_no_strm_correction.csv")

    vgrid = build_grid_instance(grid_shp)
    bbox, bounds_polygon = get_grid_bounds(grid_shp)

    if resample:
        raster = flopy.utils.Raster.load(dem_file)
        raster.crop(bounds_polygon)
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

    if approach == 1:
        # accumulate flow to existing grid
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

    elif approach == 2:
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

    elif approach == 3:
        # cellular NIDP (pyGSFLOW method)
        conditioned_dem = fill_sinks(vgrid, dem)
        fobj = FlowDirections(vgrid, conditioned_dem)
        fdir = fobj.flow_direction_array
        facc = fobj.flow_accumulation(as_cells=True)

        strms = PrmsStreams(vgrid, fobj)
        strm_array = strms.delineate_streams(contrib_area=75)

        pmv = PlotMapView(modelgrid=vgrid)
        pc = pmv.plot_array(conditioned_dem)
        plt.colorbar(pc)
        plt.show()

        pmv = PlotMapView(modelgrid=vgrid)
        pc = pmv.plot_array(strm_array)
        # pmv.plot_grid()
        plt.colorbar(pc)
        plt.show()

    elif approach == 4:
        # todo: need to trim DEM
        pploc = (723400, 5181350)
        sgrid = GenerateFishnet(bbox, xcellsize=250, ycellsize=250)

        raster = flopy.utils.Raster.load(dem_file)
        raster.crop(bounds_polygon)
        dem = raster.resample_to_grid(sgrid, band=raster.bands[0], method="min")
        dem = np.where(dem == raster.nodatavals[0], np.nan, dem)

        # now delineate initial stream locations and watershed
        conditioned_dem = fill_sinks(sgrid, dem)
        fdir = FlowDirections(sgrid, conditioned_dem)
        facc = fdir.flow_accumulation()
        watershed = fdir.get_watershed_boundary(pploc)
        facc[watershed == 0] = np.nan

        pmv = PlotMapView(modelgrid=sgrid)
        lfacc = np.log10(facc)
        pc = pmv.plot_array(lfacc)
        # pmv.plot_grid()
        plt.colorbar(pc)
        plt.show()

        strms = PrmsStreams(sgrid, fdir)
        strm_array = strms.delineate_streams(contrib_area=4e6).astype(float)
        strm_array[watershed == 0] = np.nan
        vectors = strms.create_stream_vectors(strm_array)

        geom = [LineString(v) for v in vectors.values()]
        segs = [k for k in vectors.keys()]
        gdf = gpd.GeoDataFrame({"geometry": geom, "segments": segs})
        gdf = gdf.dissolve()
        gdf["geometry"] = gdf.geometry.buffer(400, cap_style=2, join_style=3) # cs2 for flat ends
        fig, ax = plt.subplots(figsize=(8, 6))
        pmv = PlotMapView(modelgrid=sgrid, ax=ax)
        strm_array[watershed == 0] = np.nan
        pc = pmv.plot_array(strm_array)
        gdf.plot(ax=ax, alpha=0.5)
        plt.show()

        sgdf = sgrid.geo_dataframe
        iloc = np.where(watershed.ravel() > 0)[0]
        sgdf = sgdf.iloc[iloc]
        sgdf = sgdf.dissolve()

        gdf = gpd.overlay(gdf, sgdf, how="intersection")

        fig, ax = plt.subplots(figsize=(8, 6))
        pmv = PlotMapView(modelgrid=sgrid, ax=ax)
        strm_array[watershed == 0] = np.nan
        pc = pmv.plot_array(strm_array)
        # sgdf.plot(ax=ax)
        gdf.plot(ax=ax, alpha=0.5)
        plt.show()

        wsloc = (710000, 5172000)
        srloc = (712000, 5181600)
        tri_ws = os.path.join(ws, "data", "triangle_grid")
        tri = flopy.utils.triangle.Triangle(angle=30, model_ws=tri_ws)

        tri.add_polygon(sgdf.geometry.values[0])
        tri.add_polygon(gdf.geometry.values[0], ignore_holes=True)
        tri.add_region(wsloc, 0, maximum_area=250 * 250 * 100)
        tri.add_region(pploc, 1, maximum_area=250 * 250)
        tri.build()

        tri.plot()
        plt.show()

        vor = flopy.utils.voronoi.VoronoiGrid(tri)
        gridprops = vor.get_gridprops_vertexgrid()
        vgrid = VertexGrid(**gridprops)

        print('break')
    # todo: consider developing a workflow starting at structured FA
    #   define stream vectors then create a voronoi and re-FA.