Introduction
============

The D-Any package (dany) consists of a set of Python modules for performing
hydrologic conditioning, flow accumulation, and stream and watershed delineation
on standard Digital Elevation Model raster data and resampled raster data represented
in complex unstructured meshes. The D-Any code was developed to support both
"structured" and "unstructured" grid representations of stream networks in hydrologic
models like MODFLOW-6, MODFLOW-2005, MODFLOW-USG, GSFLOW, PRMS, and other modeling
software. Because the code was developed with unstructured (any number of
connection directions) in mind, D-Any flow direction numbers look a little different
than traditional D8 or D-inf flow direction calculations. D-Any uses adjacency graphs to
map node to node connections based on queen (shared vertices) or rook (shared faces)
neighbors and create flow directions. This difference in flow direction representation
allows for grid agnostic flow accumulation and surface water delineation.

D-Any was developed to be fully compatible with the
`FloPy <https://github.com/modflowpy/flopy>`_ python package for MODFLOW models
and the pyGSFLOW `pyGSFLOW <https://github.com/pygsflow/pygsflow>`_ package for GSFLOW
and PRMS models. Input data and grid representations are based on FloPy's model grid
objects. And stream delineation utilities support developing input for MODFLOW-6,
MODFLOW-2005/NWT, GSFLOW, and PRMS through FloPy and pyGSLFOW.

D-Any is an open-source project and collaboration is welcomed. Please email the
development team or open a pull request if you want to contribute.

Return to the Github `D-Any <https://github.com/jlarsen-usgs/dany>`_ website


Installation
------------

D-Any can be installed using pip.

To install with pip:

.. code-block:: bash

    pip install dany


To install the bleeding edge version of D-Any from the git repository type:

.. code-block:: bash

    pip install git+https://github.com/jlarsen-usgs/dany.git


Development Team
----------------

D-Any is currently being developed and supported by:

 * Joshua D. Larsen |orcid_Joshua_D_Larsen|

.. |orcid_Joshua_D_Larsen| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0002-1218-800X


How to Cite
-----------

* citation information coming soon
