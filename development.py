import flopy
import matplotlib.pyplot as plt
import os

import numpy as np
from flopy.utils import Raster
from dem_conditioning import fill_sinks
from flow_directions import FlowDirections
from stream_util import Sfr6, Sfr2005, PrmsStreams
from gsflow.builder import GenerateFishnet
import shapefile


resample = False
dem_file = os.path.join("data", "dem.img")
resampled_dem = os.path.join("data", "dem50.txt")
pp_file = os.path.join("data", "model_points.shp")
modelgrid = GenerateFishnet(dem_file, 50, 50)

if resample:
    raster = Raster.load(dem_file)
    dem_data = raster.resample_to_grid(
        modelgrid,
        raster.bands[0],
        method="min"
    )

    np.savetxt(resampled_dem, dem_data, fmt="%.4f")
else:
    dem_data = np.genfromtxt(resampled_dem)

with shapefile.Reader(pp_file) as r:
    pour_point = r.shape(0)

# for subbasin development
pp2_loc = (73, 111)
pp2_point = (
    modelgrid.xcellcenters[pp2_loc[0], pp2_loc[1]],
    modelgrid.ycellcenters[pp2_loc[0], pp2_loc[1]]
)

pour_points = np.array([pour_point.points[0], list(pp2_point)])
contrib_area = 810000

wf = fill_sinks(modelgrid, dem_data)

fa = FlowDirections(modelgrid, wf)
fa.flow_directions()
fdir = fa.flow_direction_array
nidp = fa.get_nidp()
facc = fa.flow_accumulation()
wshed = fa.get_watershed_boundary(pour_point)
sbsin = fa.get_subbasins(pour_points)

prms_strms = PrmsStreams(modelgrid, fa)
strm_array = prms_strms.delineate_streams(contrib_area, wshed)
cascades = prms_strms.get_cascades(strm_array, basin_boundary=wshed, many2many=True)


strms = Sfr6(modelgrid, fa)
strm_array = strms.delineate_streams(contrib_area, wshed)
strm_connectivity = strms.get_stream_connectivity(strm_array)
connectiondata = strms.connectiondata()
packagedata = strms.packagedata()
print('break')

strms = Sfr2005(modelgrid, fa)
strm_array = strms.delineate_streams(contrib_area, wshed)
stream_connectivity = strms.get_stream_connectivity(strm_array)
reach_data = strms.reach_data()
print('break')
strm_array = strms._stream_array.reshape(modelgrid.shape)[0]
strm_array = strm_array.astype(float)
strm_array[strm_array == 0] = np.nan
plt.imshow(wshed, interpolation=None)
plt.imshow(strm_array, interpolation=None, cmap="cool")
plt.show()

plt.imshow(fdir, interpolation=None)
plt.colorbar()
plt.show()

plt.imshow(facc, interpolation=None)
plt.colorbar()
plt.show()

x, y = pour_point.points[0]
pmv = flopy.plot.PlotMapView(modelgrid=modelgrid)
pc = pmv.plot_array(wshed)
plt.plot([x,], [y,], "bo", ms=4)
plt.colorbar(pc)
plt.show()

facc[wshed == 0] = np.nan
pmv = flopy.plot.PlotMapView(modelgrid=modelgrid)
pc = pmv.plot_array(facc)
plt.scatter(pour_points.T[0], pour_points.T[1], c="k")
plt.colorbar(pc)
plt.show()

pmv = flopy.plot.PlotMapView(modelgrid=modelgrid)
pc = pmv.plot_array(sbsin)
plt.scatter(pour_points.T[0], pour_points.T[1], c="k")
plt.show()

# todo: need to create a reachdata array and connectiondata array
# todo: add methods to fill in data that does not exist...
