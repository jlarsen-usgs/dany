import numpy as np
from collections import defaultdict


def fill_sinks(modelgrid, dem, eps=2e-06, stream_mask=None, method="priority"):
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
    stream_mask : np.ndarray or None
        numpy array of boolean values to deliniate existing stream line-work.
        stream_mask only works with the priority flood fill method ('priority'
        or 'complete')
    method : str
        sink fill method. "priority" runs the improved priority-flood algorithm
        described in Barnes and others (2014) which is an efficient single
        pass epsilon filling method in this implementation.
        "complete" also runs the improved priority-flood algorithm, however
        does not use epsilon filling to correct digitally flat areas of the
        DEM. "drain" uses the direct implementation of Planchon and Darboux,
        2001 to perform epsilon filling. This method is iterative and is
        slower than the priority flood method. It is included however for its
        robustness.

    Returns
    -------
        np.array : filled DEM elevations
    """
    edge_nodes = _identify_edge_nodes(modelgrid)
    dem = np.copy(dem)
    if stream_mask is not None:
        if method.lower() not in ("priority", "complete"):
            method = "priority"

    if method.lower() in ("priority", "complete"):
        if method.lower() == "complete":
            eps = 0.

        if stream_mask is not None:
            seed = np.where(stream_mask.ravel() > 0)[0]
            dem = _priority_flood(modelgrid, dem, eps, seed=seed, streams=True)
            edge_nodes = np.unique(edge_nodes + list(seed))

        filled_dem = _priority_flood(modelgrid, dem, eps, seed=edge_nodes)

    else:
        filled_dem = _flood_and_drain_fill(modelgrid, dem, eps, seed=edge_nodes)

    return filled_dem


def _flood_and_drain_fill(modelgrid, dem, eps=2e-06, seed=0):
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
    neighbors = modelgrid.neighbors(method="queen", as_nodes=True, reset=True)

    wf = np.ones(dem.size) * 1e+10
    wf[seed] = dem[seed]

    modified = True
    niter = 0
    while modified:
        niter += 1
        modified, wf = _inner_flood_and_drain_fill(dem, wf, eps, neighbors)
        print(f"Flood and Fill conditioning: iteration {niter}")

    return wf


# potentially implement network analysis method to speed this process up...
def _inner_flood_and_drain_fill(dem, wf, eps, neighbors):
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


def _priority_flood(modelgrid, dem, eps=2e-06, seed=0, streams=False):
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
    streams : bool
        boolean flag to indicate that priority flood is being performed on
        existing stream lines. Method adapted from Condon and Maxwell 2019
        to support queen neighbor connections.

    Returns
    -------
    np.array : filled numpy array of Digital elevations

    """
    import heapq
    newdem = dem.ravel().copy()

    if isinstance(seed, int):
        seed = [seed]

    pit = []
    open = []
    for n in seed:
        heapq.heappush(open, (newdem[n], n))

    closed = np.zeros(dem.size, dtype=bool)
    closed[seed] = True
    neighbors = modelgrid.neighbors(method="queen", as_nodes=True)

    while open or pit:
        if pit:
            c = pit.pop(0)
        else:
            elev, c = heapq.heappop(open)

        neighs = neighbors[c]
        if streams:
            if c in seed:
                # pit fill stream cell if needed
                elevs = [newdem[n] for n in neighs if n in seed]
                if len(elevs) > 1 and np.min(elevs) > newdem[c]:
                    newdem[c] = np.min(elevs) + eps

        for n in neighs:
            if n >= dem.size:
                continue
            if closed[n]:
                continue
            closed[n] = True
            if newdem[n] <= newdem[c]:
                newdem[n] = newdem[c] + eps
                pit.append(n)
            # continue testing, but I think this elif should be removed
            #   it seems to really mess up the stream_mask conditioning...
            # elif not streams:
            #     heapq.heappush(open, (newdem[n], n))
            else:
                heapq.heappush(open, (newdem[n], n))
                # pass

    return newdem


def _identify_edge_nodes(modelgrid):
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


def fill_nan_values(modelgrid, dem, method="mean"):
    """
    Method to fill nan's in resampled raster. Sets the cell elevation to the
    based on the elevation of neighboring cells

    Parameters
    ----------
    modelgrid : flopy.discretization.Grid object

    dem : np.array
        array of DEM values

    method : str
        valid methods are "mean" (default), "median", "min", and "max"

    Returns
    -------
    filled_dem : np.array
        nan filled dem
    """
    filled_dem = np.ravel(dem).copy()
    stack = list(np.where(np.isnan(filled_dem))[0])
    neighbors = modelgrid.neighbors(as_nodes=True, method="queen")
    method = method.lower()

    while stack:
        node = stack.pop()
        nn = neighbors[node]
        elevs = filled_dem[nn]
        if method == "mean":
            elev = np.nanmean(elevs)
        elif method == "median":
            elev = np.nanmedian(elevs)
        elif method == "min":
            elev = np.nanmin(elevs)
        elif method == "max":
            elev = np.nanmax(elevs)
        else:
            raise NotImplementedError(f"{method} has not been implemented")

        if np.isnan(elev):
            stack.append(node)
        else:
            filled_dem[node] = elev

    return filled_dem
