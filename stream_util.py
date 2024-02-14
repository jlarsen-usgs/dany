import numpy as np


class StreamBase:
    """
    Base class for delineating stream locations, generating stream
    connectivity graphs, and generating model inputs.
    Not to be instantiated by the user.

    Parameters
    ----------
    modelgrid : flopy.discretization.Grid
        flopy grid object (StructuredGrid, VertexGrid, or UnstructuredGrid)
    fdir : np.ndarray
        array of flow directions
    facc : np.ndarray
        flow accumulation array
    shape : np.ndarray
        user returned shape of the fdir and facc arrays

    """
    def __init__(self, modelgrid, fdir, facc, shape):
        self._modelgrid = modelgrid
        self._fdir = fdir
        self._facc = facc
        self._shape = shape
        self._stream_array = None
        self._graph = None

        self._shape = self._modelgrid.shape[1:]
        self._nnodes = self._modelgrid.ncpl
        if self._modelgrid.grid_type == "unstructured":
            self._nnodes = self._modelgrid.nnodes
            self._shape = (self._nnodes,)

    @property
    def stream_array(self):
        """
        Method to return a copy of the delineated stream array

        Returns
        -------
        stream_array : np.ndarray
            returns a copy of the delineated stream array

        """
        if self._stream_array is None:
            raise AssertionError(
                "delineate_streams() or set_stream_array() must be run prior "
                "to getting a stream array"
            )

        return self._stream_array.copy().reshape(self._shape)

    def set_stream_array(self, stream_array):
        """
        Method to set a custom stream array

        Parameters
        ----------
        stream_array : np.ndarray
            optional array of stream locations, if None is provided method
            looks for an internally stored stream array

        """
        # need to create a trap for unstrucutred grids
        if stream_array.size != self._nnodes:
            raise ValueError(f"Array size is incompatible with modelgrid size {self._nnodes}")

        self._stream_array = stream_array.reshape(self._shape)

    def delineate_streams(self, contrib_area, basin_boundary=None):
        """
        Method to binarize flow accumulation by a contributing area
        and delineate streams.

        Parameters
        ----------
        contrib_area : int, float, np.ndarray
            contributing area threshold to binarize flow accumulation
            into streams and landscape.

        basin_boundary : np.ndarray

        Returns
        -------
            np.ndarray : binaray numpy array of stream cell locations

        """
        if isinstance(contrib_area, np.ndarray):
            contrib_area = contrib_area.ravel()
            if contrib_area.size != self._facc.size:
                raise AssertionError(
                    f"contrib_area array size {contrib_area.size} is not "
                    f"compatable with flow accumulation size {self._facc.size}"
                )
        stream_array = np.where(self._facc >= contrib_area, 1, 0).astype(int)
        if basin_boundary is not None:
            stream_array[basin_boundary.ravel() == 0] = 0

        self._stream_array = stream_array.reshape(self._shape)
        return self._stream_array

    def _mf6_stream_connectivity(self, stream_array=None):
        """
        Method to determine MF6 stream connectivity by reach

        Parameters
        ----------
        stream_array : None or np.ndarray
            optional array of stream locations, if None is provided method
            looks for an internally stored stream array

        Returns
        -------
        stream_graph : dict
            graph representation of MF6 reach connectivity

        """
        if stream_array is None:
            if self._stream_array is None:
                raise AssertionError(
                    "delineate_streams() must be run prior to mapping the "
                    "stream connectivity or a binary array of stream cells "
                    "must be provided"
                )
            stream_array = self._stream_array

        stream_array = stream_array.ravel()
        strm_nodes = np.where(stream_array)[0]
        # assign an initial reach number to stream cells
        for i in range(1, len(strm_nodes) + 1):
            ix = strm_nodes[i - 1]
            stream_array[ix] = i

        # create a connectivity graph.... via flow directions
        topo = Topology()
        for node in strm_nodes:
            rchid = stream_array[node]
            flow_to = self._fdir[node]
            rchto = stream_array[flow_to]
            topo.add_connection(rchid, rchto)

        rchid_mapper = {rchid: ix + 1 for ix, rchid in enumerate(topo.sort())}

        # now remap the topology tree...
        old_map = topo.topology
        new_map = {}
        for old_rch, rch_from in rchid_mapper.items():
            old_rchto = old_map[old_rch]
            if old_rchto == 0:
                rchto = 0
            else:
                rchto = rchid_mapper[old_rchto]
            new_map[rch_from] = rchto

        # renumber the stream array
        for node in strm_nodes:
            old = stream_array[node]
            new = rchid_mapper[old]
            stream_array[node] = new

        self._stream_array = stream_array
        self._graph = new_map
        return new_map

    def _mf2005_stream_connectivity(self, stream_array=None):
        """
        Method to return MF2005 SFR connectivity by segment

        Parameters
        ----------
        stream_array :
            optional array of stream locations, if None is provided method
            looks for an internally stored stream array

        Returns
        -------
        segment_graph : dict
            graph representation of stream connectivity by MF2005 stream
            segment

        """
        if stream_array is None:
            if self._stream_array is None:
                raise AssertionError(
                    "delineate_streams() must be run prior to mapping the "
                    "stream connectivity or a binary array of stream cells "
                    "must be provided"
                )
            stream_array = self._stream_array

        # 1a) create graph of connectivity via node numbers...
        stream_array = stream_array.ravel()
        strm_nodes = np.where(stream_array)[0]

        graph = {}
        for node in strm_nodes:
            node_dn = self._fdir[node]
            if node_dn not in strm_nodes:
                node_dn = None
            graph[node] = node_dn

        nodesup = list(graph.keys())
        nodesdn = [i for i in graph.values() if i is not None]

        # figure out where segments start then use those to map connectivity
        segstrts = {}
        iseg = 1
        for node in nodesup:
            if node not in nodesdn:
                segstrts[node] = iseg
                iseg += 1

        for node in nodesdn:
            x = np.where(nodesdn == node)[0]
            if len(x) > 1:
                if node not in segstrts:
                    segstrts[node] = iseg
                    iseg += 1

        # node, seg, rch --->??
        # segmap --->?
        nd_seg_rch = []
        seg_graph = {}
        for node, seg in segstrts.items():
            rch = 1
            while True:
                nd_seg_rch.append([node, seg, rch])
                stream_array[node] = seg
                rch += 1

                dnnode = graph[node]
                if dnnode in segstrts or dnnode is None:
                    if dnnode is not None:
                        seg_graph[seg] = segstrts[dnnode]
                    else:
                        seg_graph[seg] = 0
                    break

                node = dnnode

        topo = Topology()
        for iseg, ioutseg in seg_graph.items():
            topo.add_connection(iseg, ioutseg)

        segid_mapper = {segid: ix + 1 for ix, segid in enumerate(topo.sort())}
        segid_mapper[0] = 0

        # remap_node_seg_reach (move to reach data)
        for i in range(len(nd_seg_rch)):
            oldseg = nd_seg_rch[i][1]
            nd_seg_rch[i][1] = segid_mapper[oldseg]

        # remap seg_graph
        seg_graph = {
            segid_mapper[iseg]: segid_mapper[ioutseg]
            for iseg, ioutseg in seg_graph.items()
        }

        seg_array = np.zeros((stream_array.shape), dtype=int)
        for iseg, new_seg in segid_mapper.items():
            idx = np.where(
                stream_array == iseg,
            )[0]
            seg_array[idx] = new_seg

        self._stream_array = seg_array.reshape(
            tuple(self._modelgrid.shape[1:])
        )
        self._graph = graph
        self._seg_graph = seg_graph
        self._node_seg_rch = np.array(nd_seg_rch)
        return seg_graph

    def create_stream_vectors(self, stream_array=None):
        """
        Method to create stream vectors (lines) from a binary
        stream array

        Parameter
        ---------
        strm_array : np.ndarray
            optional array of stream locations, if None is provided method
            looks for an internally stored stream array

        Returns
        -------
        dict : {seg : [[x0, y0],...[xn, yn]]}
            dictionary of segment number and line vertices that can be
            used to produce shapefiles, geopandas GeoDataFrames or
            other geospatial objects.

        """
        if stream_array is not None:
            stream_array = stream_array.copy()
            stream_array[np.isnan(stream_array)] = 0
        seg_graph = self._mf2005_stream_connectivity(stream_array)
        # get headwater segs
        headwaters = []
        seg_graph_r = {v: k for k, v in seg_graph.items()}
        for k, v in seg_graph.items():
            if k not in seg_graph_r:
                headwaters.append(k)

        vectors = {}
        processed = []
        stack = []
        nsr = self._node_seg_rch.copy()
        while headwaters or stack:
            if stack:
                seg = stack.pop(0)
            else:
                seg = headwaters.pop(0)
            bool_slice = nsr[:, 1] == seg
            current_seg = nsr[bool_slice, :]
            nodes = list(current_seg[:, 0])

            # add downstream connection
            dn_seg = seg_graph[seg]
            if dn_seg != 0:
                if dn_seg not in processed:
                    stack.append(dn_seg)
                bool_slice = nsr[:, 1] == dn_seg
                down_seg = nsr[bool_slice, :]
                nodes.append(down_seg[0, 0])

            xverts = self._modelgrid.xcellcenters.ravel()[nodes]
            yverts = self._modelgrid.ycellcenters.ravel()[nodes]
            vector = list(zip(xverts, yverts))
            if len(vector) > 1:
                vectors[seg] = vector
                processed.append(seg)

        return vectors


class Sfr6(StreamBase):
    """
    Class for generating MODFLOW-6 SFR pacakge stream networks.

    Parameters
    ----------
    modelgrid : flopy.discretization.Grid

    faobj : FlowDirection object
        A flow direction object that has at `flow_directions` and
        `flow_accumulation` calculated.

    """
    def __init__(self, modelgrid, faobj, **kwargs):
        if faobj is not None:
            super().__init__(modelgrid, faobj._fdir, faobj._facc, faobj._shape)
            self.connection_data = None
            self.package_data = None

        else:
            pass

    def make_connection_data(self, graph=None):
        """
        Method to create the modflow 6 connection data block from a graph
        of reach connectivity

        Parameters
        ----------
        graph : dict, None
            graph of {reach from: reach to}

        """
        if graph is None:
            if self._graph is None:
                if self._stream_array is None:
                    raise AssertionError(
                        "get_stream_connectivity() must be run or a graph of "
                        "stream connectivity must be provided prior to creating"
                        "the connection data array"
                    )
                else:
                    self._mf6_stream_connectivity()

            graph = self._graph

        conn_dict = {i: [] for i in sorted(graph.keys())}
        for reach, reach_to in graph.items():
            if reach_to != 0:
                conn_dict[reach].append(-1 * reach_to)
                conn_dict[reach_to].insert(0, reach)

        connection_data = []
        for k, v in conn_dict.items():
            connection_data.append((k,) + tuple(v))

        return connection_data

    def make_package_data(self, stream_array=None, **kwargs):
        # cellid and add complexity
        pass


class Sfr2005(StreamBase):
    """
    Class for generating MODFLOW-2005 SFR pacakge stream networks.

    Parameters
    ----------
    modelgrid : flopy.discretization.Grid

    faobj : FlowDirection object
        A flow direction object that has at `flow_directions` and
        `flow_accumulation` calculated.

    """
    def __init__(self, modelgrid, faobj, **kwargs):
        if faobj is not None:
            super().__init__(modelgrid, faobj._fdir, faobj._facc, faobj._shape)

        self._node_seg_rch = None
        self._seg_graph = None

    def get_stream_connectivity(self, stream_array=None):
        """
        Method to get modflow 2005 SFR connectivity based on Segments

        Parameters
        ----------
        stream_array : None, np.ndarray
            optional array of stream locations, if None is provided method
            looks for an internally stored stream array

        Returns
        -------
        segment_graph : dict
            graph representation of stream connectivity by MF2005 stream
            segment

        """
        return self._mf2005_stream_connectivity(stream_array=stream_array)

    def reach_data(self, stream_array=None):
        """
        Method to generate a basic reach data recarray for Modflow-2005 SFR
        package. Method fills, KRCH, IRCH, JRCH, ISEG, IREACH, and provides
        a simple assumption of 1.5 x the connectivity length for RCHLEN.
        User will need to fill in additional reach parameters and update
        RCHLEN if 1.5 x connectivity does not represent the system being
        simulated.

        Parameters
        ----------
        stream_array : np.ndarray
            optional array of stream locations, if None is provided method
            looks for an internally stored stream array

        Returns
        -------
        basic_rec : np.recarray
            numpy recarray, from flopy, of reach_length input.

        """
        from flopy.modflow import ModflowSfr2
        if stream_array is None:
            if self._stream_array is None:
                raise AssertionError(
                    "delineate_streams() must be run prior to mapping the "
                    "stream connectivity or a binary array of stream cells "
                    "must be provided"
                )
            stream_array = self._stream_array

        if self._seg_graph is None:
            self._mf2005_stream_connectivity(stream_array)

        basic_rec = []
        xcenters = self._modelgrid.xcellcenters
        ycenters = self._modelgrid.ycellcenters
        reachid = 1
        seg_rid_lut = {}
        for iseg, outseg in sorted(self._seg_graph.items()):
            if iseg != 0:
                idx = np.where(self._node_seg_rch[:, 1] == iseg)[0]
                nd_seg_rch = self._node_seg_rch[idx]
                numrch = len(nd_seg_rch) - 1
                for ix, rec in enumerate(nd_seg_rch):
                    if ix != numrch:
                        nd_dn = nd_seg_rch[ix + 1][0]
                    else:
                        if outseg != 0:
                            idx = np.where(
                                self._node_seg_rch[:, 1] == outseg
                            )[0][0]
                            nd_dn = self._node_seg_rch[idx][0]
                        else:
                            nd_dn = None

                    _, irch, jrch = self._modelgrid.get_lrc(int(rec[0]))[0]
                    if nd_dn is not None:
                        _, irch_dn, jrch_dn = self._modelgrid.get_lrc(int(nd_dn))[0]
                        a2 = (
                            xcenters[irch, jrch] - xcenters[irch_dn, jrch_dn]
                        ) ** 2
                        b2 = (
                            ycenters[irch, jrch] - ycenters[irch_dn, jrch_dn]
                        ) ** 2
                        dist = np.sqrt(a2 + b2)
                        rchlen = dist * 1.5
                    else:
                        rchlen = 1.5 * np.mean(
                            [self._modelgrid.delr[jrch], self._modelgrid.delc[irch]]
                        )

                    node = self._modelgrid.get_node([(0, irch, jrch),])[0]
                    record = [node, 0, irch, jrch, iseg, rec[-1], rchlen, reachid, reachid + 1]
                    rec_len = len(record)
                    if ix == 0:
                        seg_rid_lut[iseg] = reachid
                    elif ix == numrch:
                        record.pop(-1)
                    else:
                        pass

                    basic_rec.append(record)
                    reachid += 1

        for ix in range(len(basic_rec)):
            rec = basic_rec[ix]
            if len(rec) < rec_len:
                iseg = rec[3]
                if iseg not in self._seg_graph:
                    downreach = 0
                else:
                    outseg = self._seg_graph[iseg]
                    downreach = seg_rid_lut[outseg]
                basic_rec[ix].append(downreach)

        nreach = len(basic_rec)
        basic_rec = np.array(basic_rec).T

        reach_data = ModflowSfr2.get_empty_reach_data(nreaches=nreach)
        reach_data["node"] = basic_rec[0]
        reach_data["k"] = basic_rec[1]
        reach_data["i"] = basic_rec[2]
        reach_data["j"] = basic_rec[3]
        reach_data["iseg"] = basic_rec[4]
        reach_data["ireach"] = basic_rec[5]
        reach_data["rchlen"] = basic_rec[6]
        reach_data["reachID"] = basic_rec[-2]
        reach_data["outreach"] = basic_rec[-1]

        return basic_rec

    def segment_data(self, stream_array=None):
        """

        :return:
        """
        from flopy.modflow import ModflowSfr2
        if stream_array is None:
            if self._stream_array is None:
                raise AssertionError(
                    "delineate_streams() must be run prior to mapping the "
                    "stream connectivity or a binary array of stream cells "
                    "must be provided"
                )
            stream_array = self._stream_array

        if self._seg_graph is None:
            self._mf2005_stream_connectivity(stream_array)

        nseg = len(self._seg_graph)
        segids = []
        outsegs = []
        for seg, outseg in sorted(self._seg_graph):
            segids.append(seg)
            outsegs.append(outseg)

        recarray = ModflowSfr2.get_empty_segment_data(nsegments=nseg)
        recarray["nseg"] = outsegs
        recarray["outseg"] = outsegs
        return recarray


class PrmsStreams(StreamBase):
    """
    Class for generating PRMS/pywatershed stream and cascade connectivity.

    Parameters
    ----------
    modelgrid : flopy.discretization.Grid
        flopy model grid object (StructuredGrid, VertexGrid, or
        UnstructuredGrid)
    faobj : FlowDirection object
        A flow direction object that has at `flow_directions` and
        `flow_accumulation` calculated.

    """
    def __init__(self, modelgrid, faobj, **kwargs):
        if faobj is not None:
            super().__init__(modelgrid, faobj._fdir, faobj._facc, faobj._shape)
            self._faobj = faobj

    def get_stream_connectivity(self, stream_array=None, group_segments=False):
        """
        Method to calculate stream connectivity for PRMS streams

        Parameters
        ----------
        stream_array : None or np.ndarray
            optional array of stream locations, if None is provided method
            looks for an internally stored stream array

        group_segments : bool
            boolean flag that groups stream cells into segments based on
            stream confluences. This produces segments in the MF2005 based
            framework. Default is False and each cell is treated as an
            individual segment/reach as in the MF6 framework.

        Returns
        -------
        segment_graph : dict
            a graph representation of the stream connectivity throughout
            the basin.

        """
        if not group_segments:
            return self._mf6_stream_connectivity(stream_array=stream_array)
        else:
            return self._mf2005_stream_connectivity(stream_array=stream_array)

    def get_cascades(
        self,
        stream_array=None,
        basin_boundary=None,
        many2many=False,
    ):
        """
        Method to get PRMS/pyWatershed cascades

        Parameters
        ----------
        stream_array : np.ndarray
            optional array of stream locations, if None is provided method
            looks for an internally stored stream array
        basin_boundary : np.ndarray
            optional array of active and inactive cells within the basin of
            interest. It is highly recommended that the user pass this
            array when the discretization does not conform to the basin
            boundary (ex. StructuredGrid/DIS models) or a stream array that
            has been masked to the basin boundary has not been provided.
        many2many : bool
            flag to calculate many to many cascades, default is many to one

        Returns
        -------
        tuple : (
            hru_up_id : np.ndarray
                index of hru numbers that contain a cascading area
            hru_down_id : np.ndarray
                index number of the downslope HRU to which the corresponding
                upslope hru is connected to
            hru_pct_up : np.ndarray
                fraction of area used to compute flow to a downslope
                hru or stream segment.
            hru_strmseg_down_id : np.ndarray
                Index number of the stream segment that cascade area
                contributes flow
        )


        """
        if stream_array is None:
            stream_array = self.stream_array

        stream_array = stream_array.ravel()

        if not many2many:
            hru_up_id, hru_down_id, hru_pct_up = \
                self._build_many_to_one_cascades(basin_boundary=basin_boundary)
        else:
            hru_up_id, hru_down_id, hru_pct_up = \
                self._build_many_to_many_cascades(
                    stream_array=stream_array,
                    basin_boundary=basin_boundary,
                )

        hru_up_id = np.array(hru_up_id, dtype=int)
        hru_down_id = np.array(hru_down_id, dtype=int)
        hru_pct_up = np.array(hru_pct_up)

        hru_strmseg_down_id = []
        for ix, hru_id in enumerate(hru_down_id):
            if stream_array[hru_id] == 0:
                hru_strmseg_down_id.append(0)
            else:
                hru_strmseg_down_id.append(stream_array[hru_id])

        hru_down_id += 1
        hru_up_id += 1
        hru_strmseg_down_id = np.array(hru_strmseg_down_id)
        return hru_up_id, hru_down_id, hru_pct_up, hru_strmseg_down_id

    def _build_many_to_one_cascades(self, basin_boundary=None):
        """
        Method to generate many to one cascades

        Parameters
        ----------
        basin_boundary : None or np.ndarray
            optional array of active and inactive cells within the basin of
            interest. It is highly recommended that the user pass this
            array when the discretization does not conform to the basin
            boundary (ex. StructuredGrid/DIS models)

        Returns
        -------
        tuple : (
            hru_up_id : np.ndarray
                index of hru numbers that contain a cascading area
            hru_down_id : np.ndarray
                index number of the downslope HRU to which the corresponding
                upslope hru is connected to
            hru_pct_up : np.ndarray
                fraction of area used to compute flow to a downslope
                hru or stream segment.
        )
        """
        fdir = self._fdir.copy().ravel()
        if basin_boundary is not None:
            fdir[basin_boundary.ravel() == 0] = 0

        hru_up_id = []
        hru_down_id = []
        hru_pct_up = []
        for hru_down in fdir:
            if hru_down == 0:
                continue

            idxs = np.where(fdir == hru_down)[0]
            if len(idxs) == 0:
                continue

            if fdir[hru_down] == 0:
                # trap for outlets, need to set hru_up and hru_dn to same val
                idxs = [hru_down]

            hru_up = list(idxs)
            hru_down = [hru_down,] * len(hru_up)
            hru_pct = [1.0,] * len(hru_up)

            hru_up_id.extend(hru_up)
            hru_down_id.extend(hru_down)
            hru_pct_up.extend(hru_pct)

        return hru_up_id, hru_down_id, hru_pct_up

    def _build_many_to_many_cascades(
        self,
        stream_array,
        basin_boundary=None,
    ):
        """
        Method to generate many to many cascades

        Parameters
        ----------
        stream_array :
        basin_boundary :
            optional array of active and inactive cells within the basin of
            interest. It is highly recommended that the user pass this
            array when the discretization does not conform to the basin
            boundary (ex. StructuredGrid/DIS models)

        Returns
        -------
        tuple : (
            hru_up_id : np.ndarray
                index of hru numbers that contain a cascading area
            hru_down_id : np.ndarray
                index number of the downslope HRU to which the corresponding
                upslope hru is connected to
            hru_pct_up : np.ndarray
                fraction of area used to compute flow to a downslope
                hru or stream segment.
        )

        """
        fdir = self._fdir.copy().ravel()
        if basin_boundary is not None:
            fdir[basin_boundary.ravel() == 0] = 0

        hru_up_id = []
        hru_down_id = []
        hru_pct_up = []
        stream_nodes = np.where(stream_array > 0)[0]
        slopes = self._faobj._calculate_slopes()
        fcells = [
            list(np.where(slope <= 0)[0]) for slope in slopes
        ]

        for node, hru_down in enumerate(fdir):
            if hru_down == 0:
                continue

            if node in stream_nodes:
                # force many to one connection for stream cells
                if fdir[hru_down] == 0:
                    # trap for outlets; set hru_up and hru_down to same val
                    hru_downs = node
                else:
                    hru_downs = hru_down

                hru_up_id.append(node)
                hru_down_id.append(hru_downs)
                hru_pct_up.append(1)
            else:
                # many to many connection for all landscape cells
                flow_to = fcells[node]
                conns = self._faobj._fneighbors[node, flow_to]

                # fix the watershed divide case where there could be cascade
                #  links that point out of the active model extent based
                #  on slope
                flow_to = np.array(flow_to)
                keep_idx = [ix for ix, conn in enumerate(conns) if fdir[conn] != 0]
                flow_to = list(flow_to[keep_idx])
                conns = conns[keep_idx]

                # get the slopes and calculate % flows
                dn_slopes = slopes[node, flow_to]
                hru_pcts = np.abs(dn_slopes) / np.sum(np.abs(dn_slopes))

                # remove links with low fraction of flow
                keep_idx = np.where(hru_pcts > 0.01)[0]
                dn_slopes = dn_slopes[keep_idx]
                hru_pcts = np.abs(dn_slopes) / np.sum(np.abs(dn_slopes))
                hru_downs = conns[keep_idx]

                hru_up_id.extend([node,] * len(hru_pcts))
                hru_down_id.extend(list(hru_downs))
                hru_pct_up.extend(hru_pcts)

        return hru_up_id, hru_down_id, hru_pct_up


class Topology(object):
    """
    A topological sort method that uses a modified Khan algorithm to sort
    stream networks by connectivity

    Parameters
    ----------
    n_segments : int
        number of sfr segments in network

    """

    def __init__(self, nss=None):
        self.topology = dict()
        self.nss = nss

    def add_connection(self, iseg, ioutseg):
        """
        Method to add a topological connection

        Parameters
        ----------
        iseg : int
            current segment number
        ioutseg : int
            output segment number
        """
        self.topology[iseg] = ioutseg

    def _sort_util(self, seg, visited, stack):
        """
        Recursive function used by topological
        sort to perform sorting

        Parameters
        ----------
        seg : int
            segment number
        visited : list
            list of bools to indicate if location visited
        stack : list
            stack of sorted segment numbers

        """
        visited[seg] = True
        if seg == 0:
            ioutseg = 0
        else:
            ioutseg = self.topology[seg]

        if not visited[ioutseg]:
            self._sort_util(ioutseg, visited, stack)

        if seg == 0:
            pass
        elif ioutseg == 0:
            stack.append(seg)
        else:
            stack.insert(0, seg)

    def sort(self):
        """
        Method to perform a topological sort
        on the streamflow network

        Returns
        -------
            stack: list of ordered nodes

        """
        visited = {0: False}
        for key in self.topology:
            visited[key] = False

        stack = []
        for i in sorted(visited):
            if i == 0:
                pass
            else:
                if not visited[i]:
                    self._sort_util(i, visited, stack)

        return stack
