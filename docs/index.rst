.. documentation master file

spopt: Spatial Optimization
===========================



Regionalization, facility location, and transportation-oriented modeling
------------------------------------------------------------------------

**spopt** is an open-source Python library for solving optimization problems with spatial data. Originating from the **region** module in `PySAL (Python Spatial Analysis Library) <http://pysal.org>`_, it is under active development for the inclusion of newly proposed models and methods for regionalization, facility location, and transportation-oriented solutions.


.. raw:: html

    <div class="container-fluid">
      <div class="row equal-height">
        <div class="col-sm-1 col-xs-hidden">
        </div>
        <div class="col-md-3 col-xs-12">
            <a 
            href="https://pysal.org/spopt/notebooks/maxp.html" class="thumbnail">
                <img src="_static/images/notebooks_maxp_12_1.png" class="img-responsive center-block">
                <div class="caption text-center">
                <h6>Mexican State Regional Income Clustering</h6>
                </div>
            </a>
        </div>
        <div class="col-sm-3 col-xs-12">
            <a href="https://pysal.org/spopt/notebooks/skater.html" class="thumbnail">
                <img src="_static/images/notebooks_skater_12_1.png" class="img-responsive center-block">
                <div class="caption text-center">
                <h6>Airbnb Spots Clustering in Chicago</h6>
                </div>
            </a>
        </div>
        <div class="col-sm-3 col-xs-12">
            <a href="https://pysal.org/spopt/notebooks/reg-k-means.html" class="thumbnail">
                <img src="_static/images/notebooks_reg-k-means_15_1.png"
                class="img-responsive center-block">
                <div class="caption text-center">
                <h6>Grid Clustering
                </h6>
                </div>
            </a>
        </div>
        <div class="col-sm-2 col-xs-hidden">
        </div>
      </div>
    </div>
    



Development
-----------

**spopt** development is hosted on github_.



Citing **spopt**
----------------

If you use **PySAL-spopt** in a scientific publication, we would appreciate citations to the following paper::

  @misc{spopt2021,
    author    = {Feng, Xin, and Gaboardi, James D. and Knaap, Elijah and Rey, Sergio J. and Wei, Ran},
    month     = {jan},
    year      = {2021},
    title     = {pysal/spopt},
    url       = {https://github.com/pysal/spopt},
    doi       = {10.5281/zenodo.4444156},
    keywords  = {python,regionalization,spatial-optimization,location-modeling}
  }



Funding
-------

This project is/was partially funded through:


.. figure:: _static/images/nsf_logo.png
    :target: https://www.nsf.gov/index.jsp
    :width: 100
    :align: left

    National Science Foundation Award #1831615: `RIDIR: Scalable Geospatial Analytics for Social Science Research <https://www.nsf.gov/awardsearch/showAward?AWD_ID=1831615>`_


.. raw:: html

    <img 
        src="_static/images/pysal_banner.svg" 
        class="img-responsive center-block" 
        alt="PySAL Logo" 
        width="370" 
        height="200"
    >



.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Contents:

   Installation <installation>
   Tutorials <tutorials>
   API <api>
   References <references>



.. _PySAL: https://github.com/pysal/pysal
.. _github : https://github.com/pysal/spopt
