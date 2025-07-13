.. _api_ref:

.. currentmodule:: spopt

API reference
=============

.. _data_api:

Region Methods
--------------
Model based approaches for aggregating a large set of geographic units (with small footprints) into a smaller number of regions (with large footprints).

.. autosummary::
   :toctree: generated/

    region.AZP
    region.MaxPHeuristic
    region.RandomRegion
    region.RandomRegions
    region.RegionKMeansHeuristic
    region.SA3
    region.extract_clusters
    region.Skater
    region.Spenc
    region.WardSpatial

Locate Methods
--------------

Exact solution approaches to facility location modeling problems.

.. autosummary::
   :toctree: generated/

    locate.LSCP
    locate.LSCPB
    locate.MCLP
    locate.PMedian
    locate.KNearestPMedian
    locate.PCenter
    locate.PDispersion
