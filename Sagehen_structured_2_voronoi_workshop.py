#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


ws = os.path.join(".")
dem_file = os.path.join(ws, "data", "dem.img")
pour_point = os.path.join(ws, "data", "model_points.shp")


# Generate a `StructuredGrid` instance via pyGSFLOW's `GenerateFishnet` class 

# In[28]:


cellsize = 50
sgrid = GenerateFishnet(dem_file, xcellsize=cellsize, ycellsize=cellsize)
sgrid


# Use FloPy's `Raster` class to resample the DEM to the model grid

# In[29]:


rstr = flopy.utils.Raster.load(dem_file)
dem = rstr.resample_to_grid(sgrid, rstr.bands[0], method="min")
dem[dem == rstr.nodatavals[0]] = np.nan


# Condition the DEM and run flow accumulation

# In[31]:


conditioned_dem = fill_sinks(sgrid, dem)
fdir = FlowDirections(sgrid, conditioned_dem)
facc = fdir.flow_accumulation()


# In[32]:


# fig, ax = plt.subplots(figsize=(8, 6))
# pmv = PlotMapView(modelgrid=sgrid, ax=ax)
# lfacc = np.log10(facc)
# pc = pmv.plot_array(facc)
# pmv.plot_grid()
# plt.colorbar(pc);


# Read in the gage location at the end of the basin and delineate the watershed

# In[33]:


gdf = gpd.read_file(pour_point)
pploc = gdf.geometry.values[0]
type(pploc)
pploc.__geo_interface__


# In[34]:


watershed = fdir.get_watershed_boundary(pploc)
print(facc.shape, watershed.shape)
facc[watershed == 0] = np.nan


# In[37]:


# fig, ax = plt.subplots(figsize=(8, 6))
# pmv = PlotMapView(modelgrid=sgrid, ax=ax)
# lfacc = np.log10(facc)
# pc = pmv.plot_array(facc)
# pmv.plot_grid()
# plt.colorbar(pc);


# Delineate the structured stream locations and create vectors

# In[38]:


strms = PrmsStreams(sgrid, fdir)
strm_array = strms.delineate_streams(contrib_area=810000).astype(float)
strm_array[watershed == 0] = np.nan
vectors = strms.create_stream_vectors(strm_array)


# Plot these vectors with the stream array

# In[39]:


geom = [LineString(v) for v in vectors.values()]
segs = [k for k in vectors.keys()]
gdf = gpd.GeoDataFrame({"geometry": geom, "segments": segs})
gdf = gdf.dissolve()
gdf["geometry"] = gdf.geometry.buffer(50, cap_style=2, join_style=3)


# In[40]:


# fig, ax = plt.subplots(figsize=(8, 6))
# pmv = PlotMapView(modelgrid=sgrid, ax=ax)
# strm_array[watershed == 0] = np.nan
# pc = pmv.plot_array(strm_array)
# gdf.plot(ax=ax, alpha=0.75);


# Create a geodataframe of the active watershed and dissolve it

# In[41]:


sgdf = sgrid.geo_dataframe
iloc = np.where(watershed.ravel() > 0)[0]
sgdf = sgdf.iloc[iloc]
sgdf = sgdf.dissolve()


# In[42]:


# sgdf.plot();


# Trim the buffered stream vectors by overlaying with the active watershed

# In[43]:


igdf = gpd.overlay(gdf, sgdf, how="intersection")


# In[44]:


# fig, ax = plt.subplots(figsize=(8, 6))
# pmv = PlotMapView(modelgrid=sgrid, ax=ax)
# strm_array[watershed == 0] = np.nan
# pc = pmv.plot_array(strm_array)
# igdf.plot(ax=ax);


# Generate a triangular mesh using the `Triangle` module
# 
# Note: Must define a watershed point `wsloc` and a stream refinement point `srloc`

# In[45]:


wsloc = (220000, 4368000)
srloc = (219250, 4370000)


# In[102]:


tri_ws = os.path.join(ws, "data", "sagehen_tri_grid")
tri = flopy.utils.triangle.Triangle(angle=30, model_ws=tri_ws)
tri.add_polygon(sgdf.geometry.values[0])
tri.add_polygon(gdf.geometry.values[0], ignore_holes=True)
tri.add_region(wsloc, 0, maximum_area=100*100*5)
tri.add_region(srloc, 1, maximum_area=30*30)
tri.build()


# In[103]:


# fig, ax = plt.subplots(figsize=(8, 6))
# tri.plot(ax=ax)
# igdf.geometry.plot(ax=ax, alpha=0.25, zorder=5);


# Now generate a voronoi mesh from the triangular mesh

# In[104]:


vor = VoronoiGrid(tri)
# todo: check Voronoi for empty iverts (772) and then pop and renumber? the cell number if needed
gridprops = vor.get_gridprops_vertexgrid()
vgrid = VertexGrid(nlay=1, **gridprops)

verts = np.array(gridprops["vertices"])[:, 1:]

check = np.unique(verts, axis=0)

if verts.shape != check.shape:
    # duplicate vertices!!!!! We need to fix this somehow!
    # potentially create a search and replace operation. It seems likely that some voronoi cells have duplicate verts.
    print(verts.shape, check.shape)

# vpd = vgrid.geo_dataframe
# In[105]:


fig, ax = plt.subplots(figsize=(8, 6))
pmv = PlotMapView(modelgrid=vgrid, ax=ax)
pmv.plot_grid(color="k")
igdf.geometry.plot(ax=ax, alpha=0.50, zorder=5)
plt.show()

# Now we can begin the process of performing raster resampling and flow accumulation on the voronoi grid

# In[106]:


raster = flopy.utils.Raster.load(dem_file)
raster.resample_raster_resolution(scaling_factor=30)
dem = raster.resample_to_grid(vgrid, band=raster.bands[0], method="min")
dem = np.where(dem == raster.nodatavals[0], np.nan, dem)


# In[ ]:


conditioned_dem = fill_sinks(vgrid, dem)

# perform area based flow accumulation
fdir = FlowDirections(vgrid, conditioned_dem)
facc = fdir.flow_accumulation()
# perform cellular flow accumulation
fdir2 = FlowDirections(vgrid, conditioned_dem)
facc2 = fdir2.flow_accumulation(as_cells=True)


# In[ ]:





# In[ ]:




