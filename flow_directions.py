import numpy as np





class FlowDirections:
    """

    """
    def __init__(self, modelgrid, dem):
        self._modelgrid = modelgrid
        self._grid_type = modelgrid.grid_type

        if self._grid_type in ("structured", "vertex"):
            self._shape = self._modelgrid.shape[1:]
        else:
            self._shape = self._modelgrid.shape

        self._neighbors = modelgrid.neighbors(method="queen")
        self._fneighbors = None
        # self._dem = dem.ravel()
        self._dem = np.array(list(dem.ravel()) + [1e+10])
        self._xcenters = np.array(list(modelgrid.xcellcenters.ravel()) + [np.mean(modelgrid.xcellcenters) + 0.1])
        self._ycenters = np.array(list(modelgrid.ycellcenters.ravel()) + [np.mean(modelgrid.ycellcenters) + 0.1])

        self._fdir = np.full(self._dem.size - 1, -1)
        self._fillval = self._dem[-1]
        self._fillidx = self._modelgrid.ncpl
        self._fill_irregular_neighbor_array()
        slopes = self._calculate_slopes()
        self._calculate_flowcell(slopes)

    def _fill_irregular_neighbor_array(self):
        """
        Method to create a regular np.array of neighbors for broadcasting
        operations

        """
        axis0 = len(self._neighbors)
        axis1 = 0
        for _, n in self._neighbors.items():
            if len(n) > axis1:
                axis1 = len(n)

        self._fneighbors = np.zeros((axis0, axis1), dtype=int)
        for node, n in self._neighbors.items():
            if len(n) < axis1:
                n += [self._fillidx] * (axis1 - len(n))

            self._fneighbors[node] = n

        self._fmask = np.where(self._fneighbors == self._fillidx)

    def _calculate_slopes(self, threshold=1e-04):
        """

        :param threshold:
        :return:
        """
        cell_elevation = np.expand_dims(self._dem[:-1], axis=1)
        neighbor_elevation = self._dem[self._fneighbors]
        x0 = np.expand_dims(self._xcenters[:-1], axis=1)
        y0 = np.expand_dims(self._ycenters[:-1], axis=1)
        x1 = self._xcenters[self._fneighbors]
        y1 = self._ycenters[self._fneighbors]

        drop = neighbor_elevation - cell_elevation
        drop = np.where(drop < threshold, threshold, drop)
        asq = (x1 - x0) ** 2
        bsq = (y1 - y0) ** 2
        dist = np.sqrt(asq + bsq)

        slopes = np.where(
            self._fneighbors == self._fillidx,
            1e+10,
            drop / dist
        )

        return slopes

    def _calculate_flowcell(self, slopes):
        """
        Method to calculate the flow direction of an array of slopes

        :param slopes:
        :return:
        """
        fcells = [
            list(np.where(slope == np.min(slope))[0]) for slope in slopes
        ]
        for ix, node in enumerate(self._fdir):
            if node != -1:
                continue
            flow_to = fcells[ix]
            if len(flow_to) == 1:
                self._fdir[ix] = self._fneighbors[ix, flow_to[0]]

            else:
                # todo: now implement dijkstra's algorithm
                #  using path distance as a weight.
                dest = None
                stack = {}
                conns = self._fneighbors[flow_to]
                stack[ix] = conns
                # try to find a destination
                for conn in conns:
                    if conn in stack:
                        continue
                    cell = fcells[conn]


        print('break')