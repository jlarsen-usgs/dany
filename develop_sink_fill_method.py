import random
import cProfile, pstats
import numpy as np
import flopy


def fill_sinks(modelgrid, dem, eps=1e-04, seed=0):
    """

    :param modelgrid:
    :param dem:
    :return:
    """
    z = dem.ravel()
    neighbors = modelgrid.neighbors(method="queen", as_nodes=True)

    w = np.ones(z.size) * 1e+10
    w[seed] = z[seed]

    modified = True
    niter = 0
    while modified:
        niter += 1
        modified, w = _sink_fill(z, w, eps, neighbors)
        print(niter)

    return w

# potentially do network analysis to speed this process up...
def _sink_fill(z, w, eps, neighbors):
    """
    Method based on Planchon and Darboux, 2001:  "A fast simple and versitile
    algorithm to fill the depressions of digital elevation models"

    Need to do additonal evaluations to find the method's gotchas.
    First limitation is that a pour point must be defined for it to properly
    run. Watershed delineation might be in order prior to running this, or we
    could set it up as a method that you run for -2 cells within the model...


    :param z:
    :param w:
    :param eps:
    :param neighbors:
    :return:
    """
    modified = False
    for c in range(z.size):
        nc = neighbors[c]
        if w[c] > z[c]:
            nidx = np.where(z[c] >= w[nc] + eps)[0]
            if len(nidx) > 0:
                w[c] = z[c]
                modified = True
            else:
                nidx = np.where(w[nc] + eps < w[c])[0]
                if len(nidx > 0):
                    nc = [nc[i] for i in nidx]
                    wn = np.max(w[nc])
                    w[c] = wn + eps
                    modified = True
    return modified, w


def priority_flood(modelgrid, dem, seed=0):
    """
    Priority flood method

    :return:
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
            c = pit.pop()
        else:
            elev, c = heapq.heappop(open)

        neighs = neighbors[c]
        for n in neighs:
            if closed[n]:
                continue
            closed[n] = True
            if newdem[n] <= dem[c]:
                newdem[n] = dem[c]
                pit.append(n)
            else:
                heapq.heappush(open, (dem[n], n))

    return newdem



nrow = 5
ncol = 4
dem = np.array([[100, 90, 95, 100],
                [91, 45, 46, 89],
                [90, 41, 40, 90],
                [85, 70, 88, 89],
                [69.1, 72, 84, 85]])
seed = 16

# nrow = 20
# ncol = 20
# dem = np.abs(np.random.random(nrow*ncol) * 100)
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