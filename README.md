# `spopt`: Spatial Optimization

#### Regionalization, facility location, and transportation-oriented modeling

![tag](https://img.shields.io/github/v/release/pysal/spopt?include_prereleases&sort=semver)
[![unittests](https://github.com/pysal/spopt/workflows/.github/workflows/unittests.yml/badge.svg)](https://github.com/pysal/spopt/actions?query=workflow%3A.github%2Fworkflows%2Funittests.yml)
[![codecov](https://codecov.io/gh/pysal/spopt/branch/master/graph/badge.svg)](https://codecov.io/gh/pysal/spopt)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Blurb of several sentences for high-level description.

### Regionalization

```python
import spopt, libpysal, geopandas, numpy
mexico = geopandas.read_file(libpysal.examples.get_path("mexicojoin.shp"))
mexico["count"] = 1
attrs = [f"PCGDP{year}" for year in range(1950,2010, 10)]
w = libpysal.weights.Queen.from_dataframe(mexico)
mexico["count"], threshold_name, threshold, top_n  = 1, "count", 4, 2
numpy.random.seed(123456)
model = spopt.MaxPHeuristic(mexico, w, attrs, threshold_name, threshold, top_n)
model.solve()
mexico["maxp_new"] = model.labels_
mexico.plot(column="maxp_new", categorical=True, figsize=(12,8), ec="w");
```
<p align="center">
<img src="docs/_static/images/maxp.svg" height="400" />
</p>

### Facility Location

Coming Soon.

### Transportation & Routing

Coming Soon.


## Examples


## Requirements


## Installation


## Contribute


## Support


## Code of Conduct

As a PySAL-federated project, `spopt` follows the [Code of Conduct](https://github.com/pysal/governance/blob/master/conduct/code_of_conduct.rst) under the [PySAL governance model](https://github.com/pysal/governance).


## License

The project is licensed under the [BSD 3-Clause license](https://github.com/pysal/spopt/blob/master/LICENSE.txt).


## Funding

This project is/was partially funded through:

<!-- [<img align="middle" src="docs/_static/image/IMAGE1.png" width="150">](link1) Some text1: [Project title 1](another_link1) -->

<!-- [<img align="middle" src="docs/_static/image/IMAGE2.png" width="150">](link2) Some text2: [Project title 2](another_link2) -->
