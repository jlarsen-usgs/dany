import numpy as np
from collections import defaultdict


def sink_fill(modelgrid, dem, eps=1e-06, method="priority"):
    """
    Method to fill digital sinks within a DEM. Default is the eps-improved
    priority-flood method. This method is compatible with d-n (any number of
    neighbor) grids, such as vertex and unstructured grids. Grid cell neighbors
    are calculated using a queen-neighbor method (shared vertex).

    Parameters
    ----------
    modelgrid : flopy.discetization.Grid object
    dem : np.ndarray
        numpy array of digital elevations
    eps : float
        epsilon fill value to remove digital flat areas.
    method : str
        sink fill method. "priority" runs the improved priority-flood algorithm
        described in Barnes and others (2014) which is an efficient single
        pass filling method epsilon filling method in this implementation.
        "complete" also runs the improved priority-flood algorithm, however
        does not use epsilon filling to correct digitally flat areas of the
        DEM. "drain" uses the direct implementation of Planchon and Darboux,
        2001 to perform epsilon filling. This method is iterative and is
        slower than the priority flood method. It is included however for it's
        robustness.

    Returns
    -------
        np.array : filled DEM elevations
    """
    edge_nodes = identify_edge_nodes(modelgrid)
    if method.lower() in ("priority", "complete"):
        if method.lower() == "complete":
            eps = 0.

        filled_dem = priority_flood(modelgrid, dem, eps, seed=edge_nodes)

    else:
        filled_dem = flood_and_drain_fill(modelgrid, dem, eps, seed=edge_nodes)

    return filled_dem


def flood_and_drain_fill(modelgrid, dem, eps=1e-06, seed=0):
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
    seed : list, int
        pour point or grid edge cell locations (more robust) for seeding the
        fill operation

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
        modified, wf = _flood_and_drain_fill(dem, wf, eps, neighbors)
        print(niter)

    return wf


# potentially implement network analysis method to speed this process up...
def _flood_and_drain_fill(dem, wf, eps, neighbors):
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


def priority_flood(modelgrid, dem, eps=1e-06, seed=0):
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
    seed : list, int
        cell number(s) for beginning the flood fill (pour point) or grid
        edges (more robust)
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

    if isinstance(seed, int):
        seed = [seed]

    newdem[seed] = dem[seed]

    pit = []
    open = []
    for n in seed:
        heapq.heappush(open, (dem[n], n))

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


def identify_edge_nodes(modelgrid):
    """
    Method to identify model grid "edge" nodes. This method adapts the
    rook neighbor calculation to identify cell edges with no neighbor
    connection and then creates a set of those nodes.

    Parameters
    ----------
    modelgrid : flopy.discretization.Grid object

    Returns
    -------
        list : list of node numbers
    """
    geoms = []
    node_nums = []
    for node_num, poly in enumerate(modelgrid.iverts):
        if poly[0] == poly[-1]:
            poly = poly[:-1]
        for v in range(len(poly)):
            geoms.append(tuple(sorted([poly[v - 1], poly[v]])))
        node_nums += [node_num] * len(poly)

    edges_and_nodes = defaultdict(set)
    for i, item in enumerate(geoms):
        edges_and_nodes[item].add(node_nums[i])

    edge_nodes = set()
    for nodes in edges_and_nodes.values():
        if len(nodes) == 1:
            edge_nodes = edge_nodes | nodes

    return list(sorted(edge_nodes))


if __name__ == "__main__":
    import cProfile, pstats
    import flopy
    import matplotlib.pyplot as plt

    nrow = 5
    ncol = 4
    dem = np.array([[100, 90, 95, 100],
                    [91, 45, 46, 89],
                    [90, 41, 40, 90],
                    [85, 70, 88, 89],
                    [69.1, 72, 84, 85]])


    oseed = 16

    nrow = 1000
    ncol = 1000
    dem = np.abs(np.random.random(nrow*ncol) * 100)
    # seed = (nrow * (ncol - 1)) + np.argmin(dem[nrow*(ncol - 1):])

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

    e_nodes = identify_edge_nodes(grid)

    pro = cProfile.Profile()
    pro.enable()

    wf = sink_fill(grid, dem, eps=1e-04, method="priority")

    pro.disable()
    stats = pstats.Stats(pro)
    stats.print_stats()
    wf = wf.reshape((nrow, ncol))


    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 8))
    vmin = dem.min()
    vmax = dem.max()

    ax0.imshow(dem.reshape(nrow, ncol), interpolation="None", vmin=vmin, vmax=vmax)
    obj = ax1.imshow(wf, interpolation="None", vmin=vmin, vmax=vmax)
    ax0.set_title("A. Original DEM", loc="left")
    ax1.set_title("B. Filled DEM", loc="left")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(obj, cax=cbar_ax)

    plt.show()
    print('break')