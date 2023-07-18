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
        self._area = self._shoelace_area()
        self._fneighbors = None
        # self._dem = dem.ravel()
        self._dem = np.array(list(dem.ravel()) + [1e+10])
        self._xcenters = np.array(list(modelgrid.xcellcenters.ravel()) + [np.mean(modelgrid.xcellcenters) + 0.1])
        self._ycenters = np.array(list(modelgrid.ycellcenters.ravel()) + [np.mean(modelgrid.ycellcenters) + 0.1])

        self._fdir = np.full(self._dem.size - 1, -1)
        self._fdir_r = None
        self._facc = None
        self._fillval = self._dem[-1]
        self._fillidx = self._modelgrid.ncpl
        self._fill_irregular_neighbor_array()
        slopes = self._calculate_slopes()
        self._calculate_flowcell(slopes)

    @property
    def flow_directions(self):
        return self._fdir.reshape(self._shape)

    @property
    def flow_accumulation(self):
        return self._facc.reshape(self._shape)

    def _shoelace_area(self):
        """
        Use shoelace algorithm for non-self-intersecting polygons to
        calculate area.

        Returns
        -------

        """
        # irregular_shape_patch
        from flopy.plot.plotutil import UnstructuredPlotUtilities
        # when looping through to create determinants, need to start at -1
        xverts, yverts = self._modelgrid.cross_section_vertices
        xverts, yverts = UnstructuredPlotUtilities.irregular_shape_patch(
            xverts, yverts
        )
        area_x2 = np.zeros((1, len(xverts)))
        for i in range(xverts.shape[-1]):
            # calculate the determinant of each line in polygon
            area_x2 += xverts[:, i - 1] * yverts[:, i] - yverts[:, i - 1] * xverts[:, i]

        area = area_x2 / 2.
        return np.ravel(area)

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

    def _calculate_slopes(self, threshold=1e-06):
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
        drop = np.where((drop < threshold) & (drop > 0), 0, drop)
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
            elif len(flow_to) == 2:
                self._fdir[ix] = self._fneighbors[ix, flow_to[0]]
            elif len(flow_to) == 3:
                self._fdir[ix] = self._fneighbors[ix, flow_to[-1]]
            else:
                dest = None
                sink = True
                self._stack = defaultdict(set)
                conns = self._fneighbors[ix, flow_to]
                for conn in conns:
                    self._stack[conn].add(ix)

                tmp_stack = list(conns)
                visited = []
                print('running dijkstra')
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
        :param flow_to:
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

    def _reverse_flow_directions(self):
        """
        Method to calculate an array of reversed flow directions for
        watershed delineation

        Returns
        -------
        None
        """
        if self._fdir_r is not None:
            return self._fdir_r

        else:
            max_nidp = np.max(self.get_nidp())
            self._fdir_r = np.full((max_nidp, self._fdir.size), -1, dtype=int)
            for ix, fdir in enumerate(self._fdir):
                for jx in range(max_nidp):
                    if self._fdir_r[jx, fdir] == -1:
                        self._fdir_r[jx, fdir] = ix
                        break
            return self._fdir_r

    def get_nidp(self):
        """
        Method to calculate the number of input drainage paths of a cell

        :return:
        """
        nidp_array = np.zeros(self._fdir.size, dtype=int)

        for cell in self._fdir:
            nidp_array[cell] += 1

        return nidp_array

    def flow_acculumation(self):
        """

        :return:
        """
        nidp_array = self.get_nidp()
        flow_acc = np.ones(self._shape).ravel()
        flow_acc_area = np.copy(self._area) # np.zeros(self._shape).ravel()
        for ix, nidp_val in enumerate(nidp_array):
            if nidp_val != 0:
                continue

            n = ix
            naccu = 0
            area = 0
            while True:
                flow_acc[n] += naccu
                flow_acc_area[n] += area
                naccu = flow_acc[n]
                area = flow_acc_area[n]
                if nidp_array[n] >= 2:
                    nidp_array[n] -= 1
                    break
                n = self._fdir[n]

        self._facc = flow_acc
        return flow_acc.reshape(self._shape)

    def get_watershed_boundary(self, point):
        """

        point :


        :return:
        """
        from flopy.utils.geospatial_utils import GeoSpatialUtil

        if isinstance(point, (tuple, list, np.ndarray)):
            if len(point) != 2:
                raise AssertionError(
                    "point must be an interable with only x, y values"
                )
        else:
            point = GeoSpatialUtil(point, shapetype='point').points

        cellid = self._modelgrid.intersect(*point)
        if isinstance(cellid, tuple):
            cellid = (0,) + cellid
            cellid = self._modelgrid.get_node([cellid,])[0]

        fdir_r = self._reverse_flow_directions()
        nidp = self.get_nidp()
        subbasins = np.zeros(self._fdir.shape, dtype=int)
        subbasins[cellid] = 1
        nidp_cell = nidp[cellid]
        stack = fdir_r[:nidp_cell, cellid].tolist()
        while stack:
            cellid = stack[0]
            if subbasins[cellid] != 0:
                pass
            else:
                subbasins[cellid] = 1
                nidp_cell = nidp[cellid]
                if nidp_cell == 0:
                    pass
                else:
                    stack += fdir_r[:nidp_cell, cellid].tolist()

            stack.pop(0)

        return subbasins.reshape(self._shape)
