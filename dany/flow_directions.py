import numpy as np
from collections import defaultdict
from .stream_util import Topology


class FlowDirections:
    """
    Flow direction and flow accumulation class that works with FloPy's
    StructuredGrid, VertexGrid, and/or UnstructuredGrid. This class performs
    d-n flow accumulation by using a queen neighbor algorithm to map
    potential paths for flow.

    Parameters
    ----------
    modelgrid : flopy.discretization.Grid instance
    dem : np.ndarray
        numpy array of resampled DEM elevations
    check_elevations : bool
        boolean flag that asserts a single unique minimum elevation in the DEM

    """
    def __init__(
        self,
        modelgrid,
        dem,
        check_elevations=True,
    ):
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
        self._dem = np.array(list(dem.ravel().copy()) + [1e+10])
        self._xcenters = np.array(
            list(modelgrid.xcellcenters.ravel()) +
            [np.mean(modelgrid.xcellcenters) + 0.1]
        )
        self._ycenters = np.array(
            list(modelgrid.ycellcenters.ravel()) +
            [np.mean(modelgrid.ycellcenters) + 0.1]
        )

        min_cell = np.where(self._dem == np.nanmin(self._dem))[0]
        if check_elevations:
            if min_cell.size > 1:
                raise AssertionError(
                   "multiple cells have the same minimum elevation, please "
                   "reduce the elevation at the basin outlet"
                )
        self._min_cell = min_cell
        self._fdir = None
        self._fdir_r = None
        self._facc = None
        self._fillval = self._dem[-1]
        self._fillidx = self._modelgrid.ncpl
        self._threshold = 1e-6

    @property
    def area(self):
        """
        Method to return the area of each cell

        """
        return self._area

    @property
    def flow_direction_array(self):
        """
        Method to return a previously calculated flow direction array

        """
        return self._fdir.reshape(self._shape)

    @property
    def flow_accumulation_array(self):
        """
        Method that returns a previously calculated flow accumulation array

        """
        return self._facc.reshape(self._shape)

    def _shoelace_area(self):
        """
        Use shoelace algorithm for non-self-intersecting polygons to
        calculate area.

        Returns
        -------
            area : np.ndarray
                numpy array of cell areas in L^2
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

        area = np.abs(area_x2 / 2.)
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

    def _calculate_slopes(self):
        """
        Internal method that calculates slopes on a "filled"/uniform numpy
        array of neighboring elevations

        Returns
        -------
        slopes : np.ndarray
            uniform array of neighboring slopes, "fake cells" are given a
            very large upward slope as fill.
        """
        cell_elevation = np.expand_dims(self._dem[:-1], axis=1)
        neighbor_elevation = self._dem[self._fneighbors]
        x0 = np.expand_dims(self._xcenters[:-1], axis=1)
        y0 = np.expand_dims(self._ycenters[:-1], axis=1)
        x1 = self._xcenters[self._fneighbors]
        y1 = self._ycenters[self._fneighbors]

        drop = neighbor_elevation - cell_elevation
        drop = np.where((drop < self._threshold) & (drop > 0), 0, drop)
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
        Method to calculate the d-n flow direction from an array
        of slopes. Method does not use direction encoding and instead
        maps the downslope cell flow goes to.

        Parameters
        ----------
        slopes : np.ndarray
            uniform numpy array of neighboring slopes

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
        fdir_r : np.ndarray
            array of reversed flow directions for calculating watershed
            boundaries
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

        Returns
        -------
        nidp_array : np.ndarray
            array of the number of input drainage paths for each cell
        """
        nidp_array = np.zeros(self._fdir.size, dtype=int)

        for cell in self._fdir:
            nidp_array[cell] += 1

        return nidp_array

    def flow_directions(self, burn_threshold=1e-6):
        """
        Method to calculate d-n flow directions from DEM data

        Parameters
        ----------
        burn_threshold : np.ndarray
            digital threshold for small numerical artificats. Upslope elevation
            differences between neighbors within the digital threshold will
            be treated as digitally flat.

        Returns
        -------
        flow_directions : np.ndarray
            numpy array of cell numbers that each individual cell flows to

        """
        self._threshold = burn_threshold
        self._fdir = np.full(self._dem.size - 1, -1)
        self._fdir[self._min_cell] = self._min_cell
        self._fill_irregular_neighbor_array()
        slopes = self._calculate_slopes()
        self._calculate_flowcell(slopes)
        return self.flow_direction_array

    def flow_accumulation(self, as_cells=False):
        """
        Method to perform an accumulation of upslope areas or cells for
        each cell in the analysis

        Parameters
        ----------
        as_cells : bool
            boolean flag to accumulate based on the number of incoming drainage
            cells instead of drainage area. This is how pyGSFLOW
            (Larsen et al., 2022) performed accumulation. Default is False.

        Returns
        -------
        flow_accumulation : np.ndarray
            numpy array of upslope contributing area or upslope contribing
            cells

        """
        nidp_array = self.get_nidp()
        flow_acc = np.ones(self._shape).ravel()
        flow_acc_area = np.copy(self._area)
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

        if as_cells:
            self._facc = flow_acc
            return flow_acc.reshape(self._shape)

        else:
            self._facc = flow_acc_area
            return flow_acc_area.reshape(self._shape)

    def get_watershed_boundary(self, point, **kwargs):
        """
        Method to digitally determine a watershed boundary based on
        outflow location ("pour point")

        Parameters
        ----------
        point : geospatial object
            point can accept any of the following objects:
                shapefile.Shape object
                flopy.utils.geometry objects
                list of vertices
                geojson geometry objects
                shapely.geometry objects

        **kwargs :
            "subbasins" : np.ndarray
                keyword argument for subbasin delineation, this is an internal
                kwarg that is called by the `get_subbasins()` method and
                should not be provided by the user.

        Returns
        -------
        subbasins : np.ndarray
            array of active and inactive areas for a given pour point
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
            if len(cellid) > 2:
                cellid = cellid[1:]
            cellid = (0,) + cellid
            cellid = self._modelgrid.get_node([cellid,])[0]

        fdir_r = self._reverse_flow_directions()
        nidp_array = self.get_nidp()
        subbasins = kwargs.pop("subbasins", np.zeros(self._fdir.shape, dtype=int))
        subbasin_id = np.max(subbasins) + 1
        subbasins[cellid] = subbasin_id
        nidp = nidp_array[cellid]
        stack = fdir_r[:nidp, cellid].tolist()
        while stack:
            cellid = stack[0]
            if subbasins[cellid] != 0:
                pass
            else:
                subbasins[cellid] = subbasin_id
                nidp = nidp_array[cellid]
                if nidp == 0:
                    pass
                else:
                    stack += fdir_r[:nidp, cellid].tolist()

            stack.pop(0)

        return subbasins.reshape(self._shape)

    def get_subbasins(self, points):
        """
        Method to delineate inactive and active model area as well as
        subbasins within the model based on subbasin outlet points.

        Parameters
        ----------
        points : collection object
            points can accept the following types

            str : shapefile path
            PathLike : shapefile path
            shapefile.Reader object
            list of [shapefile.Shape, shapefile.Shape,]
            shapefile.Shapes object
            flopy.utils.geometry.Collection object
            list of [flopy.utils.geometry, ...] objects
            geojson.GeometryCollection object
            geojson.FeatureCollection object
            shapely.GeometryCollection object
            list of [[vertices], ...].

        :return:
        """
        from flopy.utils.geospatial_utils import GeoSpatialCollection

        if isinstance(points, (tuple, list, np.ndarray)):
            for point in points:
                if len(point) != 2:
                    raise AssertionError(
                        "each point must be an interable with only x, y values"
                    )
        else:
            points = GeoSpatialCollection(points, shapetype='point').points

        cellids = []
        for point in points:
            cellid = self._modelgrid.intersect(*point)
            if isinstance(cellid, tuple):
                cellid = (0,) + cellid
                cellid = self._modelgrid.get_node([cellid, ])[0]

            cellids.append(cellid)

        graph = {}
        visited = []
        for cellid in cellids:
            current = cellid
            while True:
                if cellid in visited:
                    graph[current] = -1
                    break
                elif cellid not in cellids or cellid == current:
                    if cellid != current:
                        visited.append(cellid)
                    cellid = self._fdir[cellid]
                else:
                    graph[current] = cellid
                    break

        topo = Topology()
        for cellid, cellid_to in graph.items():
            topo.add_connection(cellid, cellid_to)

        topo.add_connection(-1, -1)
        solution_order = topo.sort()[:-1]
        solution_idxs = [cellids.index(cellid) for cellid in solution_order]

        subbasins = np.zeros(self._fdir.shape, dtype=int)
        for ix, point_idx in enumerate(solution_idxs):
            point = points[point_idx]
            subbasins = self.get_watershed_boundary(
                point, subbasins=subbasins.ravel()
            )
        return subbasins

    def _get_shared_vertex(self, iverts0, iverts1, verts):
        """

        :return:
        """
        shared = []
        for iv in iverts0:
            if iv in iverts1:
                shared.append(verts[iv])

        if len(shared) > 1:
            shared = [np.mean(shared, axis=0)]

        return shared[0]

    def _calculate_hru_len(self):
        """
        Calculates hru len from either face centers or vetex depending on
        the connectivity of uphill and downhill neighbor connections

        Returns
        -------
            hru_len : np.ndarray
        """
        fdir = self.flow_direction_array.ravel()
        verts = self._modelgrid.verts
        iverts = self._modelgrid.iverts
        hru_len = []
        for node, dn_node in enumerate(fdir):
            shared_vrts = []
            up_nodes = np.where(fdir == node)[0]
            ivs = iverts[node]
            dnivs = iverts[dn_node]
            shared_vrts.append(self._get_shared_vertex(ivs, dnivs, verts))

            for up_node in up_nodes:
                upivs = iverts[up_node]
                shared_vrts.append(self._get_shared_vertex(ivs, upivs, verts))

            shared_vrts = np.array(shared_vrts).T
            xc, yc = self._xcenters[node], self._ycenters[node]
            asq = (shared_vrts[0] - xc) ** 2
            bsq = (shared_vrts[1] - yc) ** 2
            dist = np.sqrt(asq + bsq)
            hlen = np.sum(dist) / (len(dist) / 2)
            hru_len.append(hlen)

        return np.array(hru_len)

    @property
    def hru_len(self):
        """
        Returns the routed length through the hru based on uphill and downhill
        flow direction

        Returns
        -------
        hru_len : np.ndarray
        """
        return self._calculate_hru_len()

    @property
    def slope(self):
        """
        Returns the watershed slopes for each hru

        Returns
        -------
        slopes : np.ndarray
            slope of the cell's drainage path
        """
        fdir = self.flow_direction_array.ravel()
        dem = self._dem[:-1]

        x0, y0 = self._xcenters[:-1], self._ycenters[:-1]
        x1, y1 = x0[fdir], y0[fdir]
        drop = dem - dem[fdir]
        drop = np.where((drop < 1e-06) & (drop > 0), 0, drop)
        asq = (x0 - x1) ** 2
        bsq = (y0 - y1) ** 2
        dist = np.sqrt(asq + bsq)
        slopes = drop / dist
        # fix for outlet slope
        slopes = np.where(np.isnan(slopes), 0, slopes)
        return slopes.reshape(self._shape)

    @property
    def aspect(self):
        """
        Returns the hru's aspect based on the calculated flow direction

        Returns
        -------
        aspect : np.ndarray
            aspect of the cell's drainage path

        """
        fdir = self.flow_direction_array.ravel()
        xc, yc = self._xcenters[:-1], self._ycenters[:-1]
        xp = xc[fdir] - xc
        yp = yc[fdir] - yc
        aspect = np.arctan2(yp, xp) * (180 / np.pi)
        aspect = np.where(aspect < 0, aspect + 360, aspect)
        return aspect.reshape(self._shape)

    @property
    def vectors(self):
        """
        Return the quiver vectors, U, V for matplotlib quiver

        Returns
        -------

        tuple of vector offsets (U, V) on a unit circle with a radius of 1

        """
        aspect = self.aspect.ravel()
        aspect_radians = aspect * (np.pi / 180)
        u = np.cos(aspect_radians)
        v = np.sin(aspect_radians)
        return u, v
