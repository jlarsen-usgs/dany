import random
import cProfile, pstats
import numpy as np
import flopy


def fill_sinks(modelgrid, dem, eps=1e-04, seed=0):
    """
    Iterative sink fill method based on Planchon and Darboux, 2001:  "A fast
    simple and versitile algorithm to fill the depressions of digital
    elevation models"

    Parameters
    ----------
    modelgrid : flopy.discretization.Grid
    dem : np.ndarray
        digital elevation array
    eps : float
        small fill value to raise cells for digital drainage
    seed : int
        pour point location for seeding the fill operation
    Returns
    -------
        np.ndarray: filled dem
    """
    dem = dem.ravel()
    neighbors = modelgrid.neighbors(method="queen", as_nodes=True)

    wf = np.ones(dem.size) * 1e+10
    wf[seed] = dem[seed]

    modified = True
    niter = 0
    while modified:
        niter += 1
        modified, wf = _sink_fill(dem, wf, eps, neighbors)
        print(niter)

    return wf


# potentially implement network analysis method to speed this process up...
def _sink_fill(dem, wf, eps, neighbors):
    """
    Method based on Planchon and Darboux, 2001:  "A fast simple and versitile
    algorithm to fill the depressions of digital elevation models"

    Need to do additonal evaluations to find the method's gotchas.
    First limitation is that a pour point must be defined for it to properly
    run. Watershed delineation might be in order prior to running this, or we
    could set it up as a method that you run for -2 cells within the model...

    Parameters
    ----------
    dem : np.ndarray
        numpy array of digital elevations
    wf : np.ndarray
        numpy array of filled elevations and flood elevations
    eps : float
        small fill value to raise cells for digital drainage
    neighbors : dict
        dictionary of node : neighbors

    Returns
    -------
    tuple (bool, np.ndarray) returns a tuple that includes a flag to indicate
    if modifications were performed and the filled/flooded elevations.
    """
    modified = False
    for c in range(dem.size):
        nc = neighbors[c]
        if wf[c] > dem[c]:
            nidx = np.where(dem[c] >= wf[nc] + eps)[0]
            if len(nidx) > 0:
                wf[c] = dem[c]
                modified = True
            else:
                nidx = np.where(wf[nc] + eps < wf[c])[0]
                if len(nidx > 0):
                    nc = [nc[i] for i in nidx]
                    wn = np.max(wf[nc])
                    wf[c] = wn + eps
                    modified = True
    return modified, wf


def priority_flood(modelgrid, dem, seed=0, eps=1e-06):
    """
    Priority flood method for sink fill operations based on Barnes and others,
    2014, "Priority-flood: An optimal depression-filling and watershed-labeling
    algorithm for digital elevation models"

    This implementation uses the fast eps-improved priority flood method

    Parameters
    ----------
    modelgrid : flopy.discretization.Grid object
    dem : np.ndarray
        numpy array of DEM elevations
    seed : cell number for beginning the flood fill (pour point)
    eps : float
        epsilon difference for filled cells. This raises the cell by a small
        value compared to neighbors, this can be set to zero for a flat fill.

    Returns
    -------
    np.array : filled numpy array of Digital elevations

    """
    import heapq
    dem = dem.ravel()
    newdem = dem.copy()
    newdem[seed] = dem[seed]

    pit = []
    open = []
    heapq.heappush(open, (dem[seed], seed))
    closed = np.zeros(dem.size, dtype=bool)
    closed[seed] = True
    neighbors = modelgrid.neighbors(method="queen", as_nodes=True)

    while open or pit:
        if pit:
            c = pit.pop(0)
        else:
            elev, c = heapq.heappop(open)

        neighs = neighbors[c]
        for n in neighs:
            if closed[n]:
                continue
            closed[n] = True
            if newdem[n] <= newdem[c]:
                newdem[n] = newdem[c] + eps
                pit.append(n)
            else:
                heapq.heappush(open, (newdem[n], n))

    return newdem


# todo: test on actual watersheds with divides. Seed method will might
#   cause failures as edge nodes were the preferred seeds in the literature.
#   We need to overcome that because we are not able to easily identify edges
#   in unstructured and vertex grids
#   we could use the number of vertexes vs. the number of neighbors to determine
#   if a cell is an edge cell...


def identify_edge_nodes(modelgrid):
    pass



nrow = 5
ncol = 4
dem = np.array([[100, 90, 95, 100],
                [91, 45, 46, 89],
                [90, 41, 40, 90],
                [85, 70, 88, 89],
                [69.1, 72, 84, 85]])
seed = 16

nrow = 20
ncol = 20
dem = np.abs(np.random.random(nrow*ncol) * 100)
seed = (nrow * (ncol - 1)) + np.argmin(dem[nrow*(ncol - 1):])

idomain = np.ones((1, nrow, ncol), dtype=int)
botm = np.zeros((1, nrow, ncol))
delc = np.ones((nrow,)) * 500
delr = np.ones((ncol,)) * 500


grid = flopy.discretization.StructuredGrid(
    top=dem,
    botm=botm,
    delc=delc,
    delr=delr,
    idomain=idomain,
    nlay=1,
    nrow=nrow,
    ncol=ncol
)

pro = cProfile.Profile()
pro.enable()

wf = priority_flood(grid, dem, seed=seed)

# wf = fill_sinks(grid, dem, eps=0.1, seed=seed)
pro.disable()
stats = pstats.Stats(pro)
stats.print_stats()
wf = wf.reshape((nrow, ncol))

import matplotlib.pyplot as plt
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 8))
vmin = dem.min()
vmax = dem.max()

ax0.imshow(dem.reshape(nrow, ncol), interpolation="None", vmin=vmin, vmax=vmax)
obj = ax1.imshow(wf, interpolation="None", vmin=vmin, vmax=vmax)
plt.colorbar(obj)
plt.show()
print('break')