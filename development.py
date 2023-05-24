import flopy as fp
import matplotlib.pyplot as plt
import os

import numpy as np
from flopy.utils import Raster
from flow_directions import FlowDirections
from gsflow.builder import GenerateFishnet


resample = False
dem_file = os.path.join("data", "dem.img")
resampled_dem = os.path.join("data", "dem50.txt")
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

fa = FlowDirections(modelgrid, dem_data)

fdir = fa.flow_directions

plt.imshow(fdir, interpolation=None)
plt.colorbar()
plt.show()

