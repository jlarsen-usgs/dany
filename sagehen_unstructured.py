import os
import flopy
from pathlib import Path
from dem_conditioning import fill_nan_values, fill_sinks
from flow_directions import FlowDirections
from stream_util import PrmsStreams
from flopy.utils.voronoi import VoronoiGrid
from flopy.utils.triangle import Triangle

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gsflow
from gsflow.builder import GenerateFishnet
from gsflow.builder import PrmsBuilder
import gsflow.builder.builder_utils as bu


def build_lut(f, dtype=int):
    d = {}
    with open(f) as foo:
        for line in foo:
            temp = line.strip().split("#")[0]
            if not temp:
                continue
            else:
                l = temp.split(":")
                d[dtype(l[0])] = float(l[1])
    return d


if __name__ == "__main__":
    resample_rasters = True
    output_ws = Path("data/sagehen_voronoi")
    dem_file = Path("data/dem.img")
    pour_point = Path("data/model_points.shp")
    geospatial = Path("data/geospatial")
    awc = geospatial / "awc.img"
    ksat = geospatial / "ksat.img"
    clay = geospatial / "clay.img"
    sand = geospatial / "sand.img"

    nlcd2011 = geospatial / "nlcd2011_imp_utm.img"
    lf_cover = geospatial / "us_140evc_utm.img"
    lf_vegtype = geospatial / "us_140evt_utm.img"

    ppt_rstrs = [geospatial / f"climate/ppt_utm/PRISM_ppt_30yr_normal_800mM2_{i :02d}_bil.img" for i in range(1, 13)]
    tmax_rstrs = [geospatial / f"climate/tmax_utm/PRISM_tmax_30yr_normal_800mM2_{i :02d}_bil.img" for i in range(1, 13)]
    tmin_rstrs = [geospatial / f"climate/tmin_utm/PRISM_tmin_30yr_normal_800mM2_{i :02d}_bil.img" for i in range(1, 13)]

    # generate a structured fishnet grid
    cellsize = 50
    sgrid = GenerateFishnet(str(dem_file), xcellsize=cellsize, ycellsize=cellsize)

    rstr = flopy.utils.Raster.load(dem_file)
    dem = rstr.resample_to_grid(sgrid, rstr.bands[0], method="min")
    dem[dem == rstr.nodatavals[0]] = np.nan

    conditioned_dem = fill_sinks(sgrid, dem)
    fdir = FlowDirections(sgrid, conditioned_dem)
    fdir.flow_directions()
    facc = fdir.flow_accumulation()

    gdf = gpd.read_file(pour_point)
    pploc = gdf.geometry.values[0]

    watershed = fdir.get_watershed_boundary(pploc)
    print(facc.shape, watershed.shape)
    facc[watershed == 0] = np.nan

    strms = PrmsStreams(sgrid, fdir)
    strm_array = strms.delineate_streams(contrib_area=810000).astype(float)
    strm_array[watershed == 0] = np.nan
    vectors = strms.create_stream_vectors(strm_array)

    geom = [LineString(v) for v in vectors.values()]
    segs = [k for k in vectors.keys()]
    gdf = gpd.GeoDataFrame({"geometry": geom, "segments": segs})
    gdf = gdf.dissolve()
    gdf["geometry"] = gdf.geometry.buffer(50, cap_style=2, join_style=3)

    sgdf = sgrid.geo_dataframe
    iloc = np.where(watershed.ravel() > 0)[0]
    sgdf = sgdf.iloc[iloc]
    sgdf = sgdf.dissolve()

    igdf = gpd.overlay(gdf, sgdf, how="intersection")

    # Generate a triangular mesh using the `Triangle` module
    # Note: Must define a watershed point `wsloc` and a stream refinement point `srloc`

    wsloc = (220000, 4368000)
    srloc = (219250, 4370000)

    tri_ws = Path("data/sagehen_tri_grid")
    tri = Triangle(angle=30, model_ws=tri_ws)
    tri.add_polygon(sgdf.geometry.values[0])
    tri.add_polygon(gdf.geometry.values[0], ignore_holes=True)
    tri.add_region(wsloc, 0, maximum_area=100 * 100 * 3)
    tri.add_region(srloc, 1, maximum_area=40 * 40)
    tri.build()

    vor = VoronoiGrid(tri)
    gridprops = vor.get_gridprops_vertexgrid()
    vgrid = flopy.discretization.VertexGrid(nlay=1, **gridprops)
    verts = np.array(gridprops["vertices"])[:, 1:]

    raster = flopy.utils.Raster.load(dem_file)
    dem = raster.resample_to_grid(vgrid, band=raster.bands[0], method="min")
    dem = np.where(dem == raster.nodatavals[0], np.nan, dem)

    filled_dem = fill_nan_values(vgrid, dem, method="mean")
    conditioned_dem = fill_sinks(vgrid, filled_dem, eps=0)

    fdir = FlowDirections(vgrid, conditioned_dem)
    fdir.flow_directions()
    facc = fdir.flow_accumulation()
    flen = fdir.hru_len

    strms = PrmsStreams(vgrid, fdir)
    stream_array = strms.delineate_streams(contrib_area=1.2e2)
    cascades = strms.get_pygsflow_builder_object(stream_array, group_segments=False)

    prms_build = PrmsBuilder(None, cascades, vgrid, conditioned_dem)
    parameters = prms_build.build("voronoi_sagehen")

    nhru = parameters.nhru.values[0]
    # todo: pygsflow input data (climate, soils, etc...)
    #   builder utilities for other PRMS parameters (soil zone, etc...)

    veg_cov_file = output_ws / "lf_veg_cover.txt"
    veg_type_file = output_ws / "lf_veg_type.txt"
    imperv_file = output_ws / "impervious.txt"
    awc_file = output_ws / "soil_awc.txt"
    ksat_file = output_ws / "soil_ksat.txt"
    sand_file = output_ws / "soil_sand.txt"
    clay_file = output_ws / "soil_clay.txt"
    ppt_file = output_ws / "prism_ppt.txt"
    tmax_file = output_ws / "prism_tmax.txt"
    tmin_file = output_ws / "prism_tmin.txt"
    if resample_rasters:
        raster = flopy.utils.Raster.load(lf_cover)
        veg_cov = raster.resample_to_grid(
            vgrid,
            band=raster.bands[0],
            method="nearest",
        )
        veg_cov = veg_cov.astype(int)
        np.savetxt(veg_cov_file, veg_cov, fmt="%d")

        raster = flopy.utils.Raster.load(lf_vegtype)
        veg_type = raster.resample_to_grid(
            vgrid,
            band=raster.bands[0],
            method="nearest",
        )
        veg_type = veg_type.astype(int)
        np.savetxt(veg_type_file, veg_type, fmt="%d")

        raster = flopy.utils.Raster.load(awc)
        awc = raster.resample_to_grid(
            vgrid,
            band=raster.bands[0],
            method="median",
        )
        awc[awc == raster.nodatavals[0]] = np.nanmedian(awc)
        np.savetxt(awc_file, awc)

        raster = flopy.utils.Raster.load(ksat)
        ksat = raster.resample_to_grid(
            vgrid,
            band=raster.bands[0],
            method="median",
        )
        ksat[ksat == raster.nodatavals[0]] = np.nanmedian(ksat)
        np.savetxt(ksat_file, ksat)

        raster = flopy.utils.Raster.load(sand)
        sand = raster.resample_to_grid(
            vgrid,
            band=raster.bands[0],
            method="median",
        )
        sand[sand == raster.nodatavals[0]] = np.nanmedian(sand)
        sand /= 100
        np.savetxt(sand_file, sand)

        raster = flopy.utils.Raster.load(clay)
        clay = raster.resample_to_grid(
            vgrid,
            band=raster.bands[0],
            method="median",
        )
        clay[clay == raster.nodatavals[0]] = np.nanmedian(clay)
        clay /= 100
        np.savetxt(clay_file, clay)

        raster = flopy.utils.Raster.load(nlcd2011)
        impervious = raster.resample_to_grid(
            vgrid,
            band=raster.bands[0],
            method="median",
        )
        impervious /= 100
        np.savetxt(imperv_file, impervious)

        ppt = []
        for rstr in ppt_rstrs:
            raster = flopy.utils.Raster.load(rstr)
            tppt = raster.resample_to_grid(
                vgrid,
                band=raster.bands[0],
                method="linear",
            )
            ppt.append(tppt.ravel())
        ppt = np.array(ppt)
        np.savetxt(ppt_file, ppt)

        tmin = []
        for rstr in tmin_rstrs:
            raster = flopy.utils.Raster.load(rstr)
            ttmin = raster.resample_to_grid(
                vgrid,
                band=raster.bands[0],
                method="linear",
            )
            tmin.append(ttmin.ravel())
        tmin = np.array(tmin)
        np.savetxt(tmin_file, tmin)

        tmax = []
        for rstr in tmax_rstrs:
            raster = flopy.utils.Raster.load(rstr)
            ttmax = raster.resample_to_grid(
                vgrid,
                band=raster.bands[0],
                method="linear",
            )
            tmax.append(ttmax.ravel())
        tmax = np.array(tmax)
        np.savetxt(tmax_file, tmax)

    else:
        veg_type = np.genfromtxt(veg_type_file, dtype=int)
        veg_cov = np.genfromtxt(veg_cov_file, dtype=int)
        awc = np.genfromtxt(awc_file)
        ksat = np.genfromtxt(ksat_file)
        sand = np.genfromtxt(sand_file)
        clay = np.genfromtxt(clay_file)
        impervious = np.genfromtxt(imperv_file)
        ppt = np.genfromtxt(ppt_file)
        tmax = np.genfromtxt(tmax_file)
        tmin = np.genfromtxt(tmin_file)
        ppt.shape = (12, nhru)
        tmax.shape = (12, nhru)
        tmin.shape = (12, nhru)

    # load the remap files
    gsf_ws = os.path.split(gsflow.__file__)[0]
    covtype_remap = os.path.join(gsf_ws, "..", "examples", "data", 'remaps', "landfire", "covtype.rmp")
    covden_sum_remap = os.path.join(gsf_ws, "..", "examples", "data", 'remaps', "landfire", "covdensum.rmp")
    covden_win_remap = os.path.join(gsf_ws, "..", "examples", "data", 'remaps', "landfire", "covdenwin.rmp")
    root_depth_remap = os.path.join(gsf_ws, "..", "examples", "data", 'remaps', "landfire", 'rtdepth.rmp')
    snow_intcp_remap = os.path.join(gsf_ws, "..", "examples", "data", "remaps", "landfire", "snow_intcp.rmp")
    srain_intcp_remap = os.path.join(gsf_ws, "..", "examples", "data", "remaps", "landfire", "srain_intcp.rmp")

    covtype_lut = build_lut(covtype_remap)
    covden_sum_lut = build_lut(covden_sum_remap)
    covden_win_lut = build_lut(covden_win_remap)
    root_depth_lut = build_lut(root_depth_remap)
    snow_intcp_lut = build_lut(snow_intcp_remap)
    srain_intcp_lut = build_lut(srain_intcp_remap)

    # build vegatative cover parameters
    covtype = bu.covtype(veg_type, covtype_lut)
    covden_sum = bu.covden_sum(veg_cov, covden_sum_lut)
    covden_win = bu.covden_win(covtype.values, covden_win_lut)
    rad_trncf = bu.rad_trncf(covden_win.values)
    snow_intcp = bu.snow_intcp(veg_type, snow_intcp_lut)
    srain_intcp = bu.srain_intcp(veg_type, srain_intcp_lut)
    wrain_intcp = bu.wrain_intcp(veg_type, snow_intcp_lut)

    # add veg to param_obj
    parameters.add_record_object(covtype, True)
    parameters.add_record_object(covden_sum, True)
    parameters.add_record_object(covden_win, True)
    parameters.add_record_object(rad_trncf, True)
    parameters.add_record_object(snow_intcp, True)
    parameters.add_record_object(srain_intcp, True)
    parameters.add_record_object(wrain_intcp, True)

    # build soil parameters
    hru_slope = fdir.slope
    hru_aspect = fdir.aspect
    hru_len = fdir.hru_len

    root_depth = bu.root_depth(veg_type, root_depth_lut)
    soil_type = bu.soil_type(clay, sand)
    soil_moist_max = bu.soil_moist_max(awc, root_depth)
    soil_moist_init = bu.soil_moist_init(soil_moist_max.values)
    soil_rech_max = bu.soil_rech_max(awc, root_depth)
    ssr2gw_rate = bu.ssr2gw_rate(ksat, sand, soil_moist_max.values)
    ssr2gw_sq = bu.ssr2gw_exp(nhru)
    soil_rech_init = bu.soil_rech_init(soil_rech_max.values)
    slowcoef_lin = bu.unstructured_slowcoef_lin(ksat, hru_aspect, hru_len)

    slowcoef_sq = bu.unstructured_slowcoef_sq(
        ksat,
        hru_aspect,
        sand,
        soil_moist_max.values,
        hru_len
    )

    parameters.add_record_object(soil_type, replace=True)
    parameters.add_record_object(soil_moist_max, replace=True)
    parameters.add_record_object(soil_moist_init, replace=True)
    parameters.add_record_object(soil_rech_max, replace=True)
    parameters.add_record_object(soil_rech_init, replace=True)
    parameters.add_record_object(ssr2gw_rate, replace=True)
    parameters.add_record_object(ssr2gw_sq, replace=True)
    parameters.add_record_object(slowcoef_lin, replace=True)
    parameters.add_record_object(slowcoef_sq, replace=True)

    # imperviousness parameters
    hru_percent_imperv = bu.hru_percent_imperv(impervious)
    carea_max = bu.carea_max(impervious)

    parameters.add_record_object(hru_percent_imperv, replace=True)
    parameters.add_record_object(carea_max, replace=True)

    print('break')