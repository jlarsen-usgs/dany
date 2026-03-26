API Reference
=============

D-Any
---------

D-Any supports hydrologic conditioning, flow direction calculations, flow accumulation,
watershed and sub-watershed delineation, and stream network creation from structured
and unstructured Digital Elevation Model data.


Hydrologic Conditioning
^^^^^^^^^^^^^^^^^^^^^^^

D-Any supports multiple methods for hydrologic conditioning including: Improved-Epsilon
priority flood conditioning, complete fill conditioning, and flood and drain
conditioning. The API documentation for conditioning is documented here:


Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/dany.dem_conditioning


Flow directions and Flow Accumulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Flow direction calculations, flow accumulation, watershed delination, and sub-watershed
identification/delineation processes are all hosted within the flow directions module
and is documented here:

Contents:

.. toctree::
   :glob:
   :maxdepth: 4

   ./source/dany.flow_directions


Stream network utilities
^^^^^^^^^^^^^^^^^^^^^^^^

D-Any's stream network utilities provide support for delineating stream networks
from flow directions and accumulation data and for building model inputs for MODFLOW-6,
MODFLOW-2005/NWT, GSFLOW.

Contents:

.. toctree::
   :maxdepth: 4

   ./source/dany.stream_util
