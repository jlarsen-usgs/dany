import numpy as np
import flopy


class SfrBase:
    def __init__(self, modelgrid, fdir, facc, shape,):
        self._modelgrid = modelgrid
        self._fdir = fdir
        self._facc = facc
        self._shape = shape
        self._stream_array = None
        self._graph = None

    def delineate_streams(self, contrib_area, basin_boundary=None):
        """

        contrib_area : int, float, np.ndarray
            contributing area threshold to binarize flow accumulation
            into streams and landscape.

        basin_boundary : np.ndarray

        Returns
        -------
            np.ndarray : binaray numpy array of stream cell locations
        """
        if isinstance(contrib_area, np.ndarray):
            contrib_area = contrib_area.ravel
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

    def get_stream_conectivity(self, stream_array=None):
        """

        stream_array :

        :return:
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


class Sfr6(SfrBase):
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

        graph : dict, None
            graph of {reach from: reach to}

        """
        if graph is None:
            if self._graph is None:
                raise AssertionError(
                    "get_stream_connectivity() must be run or a graph of "
                    "stream connectivity must be provided prior to creating"
                    "the connection data array"
                )

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

    def make_package_data(self, ):
        # cellid and add complexity
        pass


class Sfr2005(SfrBase):
    def __init__(self, modelgrid, faobj, **kwargs):
        if faobj is not None:
            super().__init__(modelgrid, faobj._fdir, faobj._facc, faobj._shape)

    def get_stream_conectivity(self, stream_array=None):
        """

        stream_array :

        :return:
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
        segstrts = []
        for node in nodesup:
            if node not in nodesdn:
                segstrts.append(node)

        for node in nodesdn:
            x = np.where(nodesdn == node)[0]
            if len(x) > 1:
                if node not in segstrts:
                    segstrts.append(node)

        # node, seg, rch --->??
        # segmap --->?
        nd_seg_rch = []
        seg_graph = {}
        # todo: assign a seg number to segstrts... this should be a dict...
        seg = 0
        for node in segstrts:
            rch = 1
            seg += 1
            while True:
                nd_seg_rch.append((node, seg, rch))
                rch += 1

                dnnode = graph[node]
                if dnnode in segstrts or dnnode is None:
                    if dnnode is not None:
                        seg_graph[seg] = seg + 1
                    else:
                        seg_graph[seg] = 0
                    break

                node = dnnode

        print('break')

        # todo: create a graph of node numbers and then find individual segments
        #  after set the segment numbers and then number the reaches....
        #  create a reachno array and segno array....



        # 1b) find headwater nodes and trace downstream...

    def reach_data(self, **kwargs):
        # start with KRCH IRCH JRCH ISEG IREACH RCHLEN [] and add complexity
        # todo: need connectivity graph to get rchlen...
        i_idx, j_idx = np.where(self._stream_array != 0)
        xcenters = self._modelgrid.xcellcenters
        ycenters = self._modelgrid.ycellcenters

        basic_rec = []
        for ix, i in enumerate(i_idx):
            krch = 0
            irch = i
            jrch = j_idx[ix]
            iseg = self._stream_array[irch, jrch]
            dwn = self._graph[iseg]
            if dwn != 0:
                dwn_irch, dwn_jrch = np.where(self._stream_array == dwn)

                a2 = (xcenters[irch, jrch] - xcenters[dwn_irch, dwn_jrch]) ** 2
                b2 = (ycenters[irch, jrch] - ycenters[dwn_irch, dwn_jrch]) ** 2
                dist = np.sqrt(a2 + b2)
                rchlen = dist * 1.5
            else:
                rchlen = 1.5 * self._modelgrid.delr[jrch]

            basic_rec.append((krch, irch, jrch))



        pass

    def segment_data(self):
        pass


class PrmsStrms(SfrBase):

    def __init__(self):
        super().__init__()



class Topology(object):
    """
    A topological sort method that uses a modified Khan algorithm to sort the
    SFR network

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