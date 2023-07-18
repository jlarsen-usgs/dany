import flopy
import matplotlib.pyplot as plt
import os

import numpy as np
from flopy.utils import Raster
from flow_directions import FlowDirections
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

fa = FlowDirections(modelgrid, dem_data)

fdir = fa.flow_directions
nidp = fa.get_nidp()
facc = fa.flow_acculumation()
wshed = fa.get_watershed_boundary(pour_point)

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
