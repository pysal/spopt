
<p align="center">
<img src="docs/_static/images/pysal_banner.svg" width="370" height="200" />
</p>

# `spopt`: Spatial Optimization

#### Regionalization, facility location, and transportation-oriented modeling

![tag](https://img.shields.io/github/v/release/pysal/spopt?include_prereleases&sort=semver)
[![unittests](https://github.com/pysal/spopt/workflows/.github/workflows/unittests.yml/badge.svg)](https://github.com/pysal/spopt/actions?query=workflow%3A.github%2Fworkflows%2Funittests.yml)
[![codecov](https://codecov.io/gh/pysal/spopt/branch/main/graph/badge.svg)](https://codecov.io/gh/pysal/spopt)
[![Documentation](https://img.shields.io/static/v1.svg?label=docs&message=current&color=9cf)](http://pysal.org/spopt/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Spopt is an open-source Python library for solving optimization problem with spatial data. Originating from the `region` module in [PySAL (Python Spatial Analysis Library)](http://pysal.org), it is under active development for the inclusion of newly proposed models and methods for regionalization, facility location, and transportation-oriented solutions. 

### Regionalization

```python
import spopt, libpysal, geopandas, numpy
mexico = geopandas.read_file(libpysal.examples.get_path("mexicojoin.shp"))
mexico["count"] = 1
attrs = [f"PCGDP{year}" for year in range(1950, 2010, 10)]
w = libpysal.weights.Queen.from_dataframe(mexico)
mexico["count"], threshold_name, threshold, top_n  = 1, "count", 4, 2
numpy.random.seed(123456)
model = spopt.MaxPHeuristic(mexico, w, attrs, threshold_name, threshold, top_n)
model.solve()
mexico["maxp_new"] = model.labels_
mexico.plot(column="maxp_new", categorical=True, figsize=(12,8), ec="w");
```
<p align="center">
<img src="docs/_static/images/maxp.svg" height="350" />
</p>

### Facility Location

Coming Soon.

### Transportation & Routing

Coming Soon.


## Examples
More examples can be found in the [Tutorials](https://pysal.org/spopt/tutorial.html) section of the documentation. 
- [Max-p-regions problem](https://pysal.org/spopt/notebooks/maxp.html)
- [Skater](https://pysal.org/spopt/notebooks/skater.html)
- [Region K means](https://pysal.org/spopt/notebooks/reg-k-means.html)


## Requirements
- [scipy](http://scipy.github.io/devdocs/)
- [numpy](https://numpy.org/devdocs/)
- [pandas](https://pysal.org/libpysal/)
- [networkx](https://networkx.org/)
- [libpysal](https://pysal.org/spaghetti/)
- [scilit-learn](https://scikit-learn.org/stable/)
- [geopandas](https://geopandas.org/)

## Installation


## Contribute

PySAL-spopt is under active development and contributors are welcome.

If you have any suggestions, feature requests, or bug reports, please open new [issues](https://github.com/pysal/spopt/issues) on GitHub. To submit patches, please review [PySAL: Getting Started](http://pysal.org/getting_started#for-developers), the PySAL [development guidelines](https://github.com/pysal/pysal/wiki), the `spopt` [contributing guidelines](https://github.com/pysal/spopt/blob/main/.github/CONTRIBUTING.md) before  opening a [pull request](https://github.com/pysal/spopt/pulls). Once your changes get merged, you’ll automatically be added to the [Contributors List](https://github.com/pysal/spopt/graphs/contributors).


## Support
If you are having trouble, please [create an issue](https://github.com/pysal/spopt/issues), [start a discussion](https://github.com/pysal/spopt/discussions), or talk to us in the [gitter room](https://gitter.im/pysal/spopt).

## Code of Conduct

As a PySAL-federated project, `spopt` follows the [Code of Conduct](https://github.com/pysal/governance/blob/master/conduct/code_of_conduct.rst) under the [PySAL governance model](https://github.com/pysal/governance).


## License

The project is licensed under the [BSD 3-Clause license](https://github.com/pysal/spopt/blob/main/LICENSE.txt).


## Funding

This project is/was partially funded through:

[<img align="middle" src="docs/_static/images/nsf_logo.png" width="150">](https://www.nsf.gov/index.jsp) National Science Foundation Award #1831615: [RIDIR: Scalable Geospatial Analytics for Social Science Research](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1831615)

<!-- [<img align="middle" src="docs/_static/image/IMAGE2.png" width="150">](link2) Some text2: [Project title 2](another_link2) -->
