import numpy as np
from collections import defaultdict


class FlowDirections:
    """
    Ahhhh!!!!
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

    @property
    def flow_directions(self):
        return self._fdir.reshape(self._shape)

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
                sink = True
                self._stack = defaultdict(set)
                conns = self._fneighbors[ix, flow_to]
                n = 0
                for conn in conns:
                    self._stack[conn].add(ix)
                # todo: need to create a stack that we iterate through
                tmp_stack = list(conns)
                visited = []
                while True:
                    n = 0
                    for cell in tmp_stack:
                        if cell not in visited:
                            flow_to = fcells[cell]
                            dest, conns, sink = self._resolve_flats(cell, flow_to, dest)
                            if dest is None:
                                tmp_stack += list(conns)

                            visited.append(cell)
                        else:
                            if dest is not None:
                                continue
                            else:
                                n += 1
                                if n >= len(self._stack):
                                    sink = True

                    if dest is not None:
                        sink = False
                        break
                    if sink:
                        break
                if sink:
                    # now that we have sinks,
                    # we can apply hydrologic conditioning...
                    self._fdir[ix] = -2
                    for cell in self._stack.keys():
                        self._fdir[cell] = -2
                else:
                    # create a weighted distance array
                    flow_trace = {list(self._stack[dest])[0]: [dest, 0]}
                    visited = []
                    ldest = [dest, ]
                    while self._stack:
                        node_pop = []
                        # todo: now iterate over ldest to make sure we properly weight the algorithm
                        # for node_to, nodes_from in self._stack.items():
                        for node_to in ldest:
                            nodes_from = self._stack[node_to]
                            visited.append(node_to)
                            for node in nodes_from:
                                if node_to == dest:
                                    flow_trace[node] = [node_to, 0]
                                    node_pop.append(node_to)
                                else:
                                    if node_to in flow_trace:
                                        dist0 = 1e6
                                        if node in flow_trace:
                                            dist0 = flow_trace[node][1]

                                        dist = flow_trace[node_to][1] + 1
                                        if dist < dist0:
                                            flow_trace[node] = [node_to, dist]
                                            node_pop.append(node_to)
                                    else:
                                        flow_trace[node] = [node_to, 999]
                                        node_pop.append(node_to)

                        self._stack.pop(node_to)

                        ldest = list(nodes_from)
                        if not ldest:
                            # todo: need to advance the algorithm for weird splits
                            #   etc....
                            ldest = [list(self._stack.keys())[-1],]

                        # todo: may need to improve this algorithm to provide
                        #  better mapping. flow trace does not solve for all
                        #  possible cells in the map, it only solves for
                        #  a routing distance
                        # break

                    if ix not in flow_trace:
                        print('break')
                    for node, (node_to, dist) in flow_trace.items():
                        self._fdir[node] = node_to

    def _resolve_flats(self, cell, flow_to, dest):
        """

        :param cell:
        :param flow_neighbors:
        :return:
        """
        sink = True
        conns = self._fneighbors[cell, flow_to]
        if len(conns) == 1:
            dest = conns[0]
        for conn in conns:
            if conn in self._stack:
                continue
            self._stack[conn].add(cell)
            sink = False
            if self._fdir[conn] not in (-1, -2):
                dest = conn

        return dest, conns, sink

    def get_nidp(self, cell):
        """
        Method to calculate the number of input drainage paths of a cell

        :param cell:
        :return:
        """
        pass