---
title: 'spopt: a python package for solving spatial optimization problems in PySAL'
tags:
  - Python
  - spatial optimization
  - PySAL
authors:
  - name: Xin Feng
    orcid: 0000-0001-7253-3154
    affiliation: "1, 2" 
  - name: Germano Barcelos
    orcid: 0000-0002-4758-1776
    affiliation: 3
  - name: James D. Gaboardi
    orcid: 0000-0002-4776-6826
    affiliation: "4, 5"
  - name: Elijah Knaap
    orcid: 0000-0001-7520-2238
    affiliation: 1
  - name: Ran Wei
    affiliation: 1
  - name: Levi J. Wolf
    orcid: 0000-0003-0274-599X
    affiliation: 6
  - name: Qunshan Zhao
    orcid: 0000-0002-5549-9457
    affiliation: 7
  - name: Sergio J. Rey
    orcid: 0000-0001-5857-9762
    affiliation: 1
affiliations:
 - name: Center for Geospatial Sciences, University of California Riverside
   index: 1
 - name: Department of Geography and Environmental Sustainability, University of Oklahoma
   index: 2
 - name: Federal University of Vi\c{c}osa
   index: 3
 - name: Oak Ridge National Laboratory
   index: 4
 - name: The Peter R. Gould Center for Geography Education and Outreach, Penn State
   index: 5 
 - name: University of Bristol
   index: 6
 - name: Urban Big Data Centre, School of Social & Political Sciences, University of Glasgow
   index: 7
date: 09 June 2022
bibliography: paper.bib
---


# Summary

Spatial optimization is a major spatial analytical tool in management and planning, the significance of which cannot be overstated. Spatial optimization models play an important role in designing and managing effective and efficient service systems such as transportation, education, public health, environmental protection, and commercial investment among others. To this end, **spopt** (\textbf{sp}atial \textbf{opt}imization) is under active development for the inclusion of newly proposed models and methods for regionalization, facility location, and transportation-oriented solutions [@spopt2021].  **Spopt** is a submodule in the open-source spatial analysis library PySAL (Python Spatial Analysis Library) founded by Dr. Sergio J. Rey and Dr. Luc Anselin in 2005 [@pysal2007;@rey2015open;@Rey2021]. The goal of developing **spopt** is to provide management and decision-making support to all relevant practitioners and to further promote the appropriate and meaningful application of spatial optimization models in practice.

# Statement of need

Spatial optimization methods and algorithms can be accessed in many ways. ArcGIS[^arc] and TransCAD[^transcad] are two well-known commercial GIS software packages that provide modules designed for structuring and solving spatial optimization problems. The optimization functions they offer focus on a set classical single facility location methods (e.g., Weber, Median, Centroid, 1-center), routing and shortest path methods (e.g., shortest path on the network, least cost path over the terrain), and multi-facility location-allocation methods (e.g., coverage models, p-median problem). They are user-friendly and visually appealing, but the cost is relatively high [@murray2021contemporary]. 

[^arc]: [https://www.esri.com/en-us/home](https://www.esri.com/en-us/home)
[^transcad]: [https://www.caliper.com/](https://www.caliper.com/)

Open-source software is another option to access spatial optimization. Although it may require users to have a certain level of programming experience, open-source software provides relatively novel and comprehensive methods, and more importantly, it is free and can be easily replicated. This is particularly true for regionalization and facility-location methods. Regionalization methods are limited in commercial GIS software, and may only have grouping analysis for vector data and region identification for raster data. On the contrary, there are many application-oriented open-source packages that facilitate the implementation of regionalization methods in various fields, including climate (e.g., HiClimR [@badr2015tool], synoptReg [@LEMUSCANOVAS2019114]), biology (e.g., Phyloregion [@daru2020phyloregion], regioneR [@regioneR2015]), hydrology (e.g., nsRFA [@nsRFA]), agriculture (e.g., OpenLCA[^openlca]), and so on. The functions of graph regionalization with clustering and partitioning have been provided by several packages in R[^R] such as Rgeoda [@rgeoda], maxcut: Max-Cut Problem in sdpt3r [@rahman2018sdpt3r], and RBGL: R Boost Graph Library [@RBGL], and grPartition within MATLAB[^matlab]. They are probably the most closely related projects to the regionalization functionality of **spopt**, however, they are written in R and MATLAB. For facility-location methods, commercial software such as ArcGIS and TransCAD implement models using heuristic approaches. However, they don't provide details about the solution found, which limits the interpretability of the results [@Chen2021]. On the other hand, existing open-source packages mostly aim at solving coverage problems such as pyspatialopt [@pyspatialopt], allagash [@allagash] and maxcovr [@maxcovr], but the available models, solvers, and overall accessibility vary significantly.  Therefore, it is necessary to develop an open-source optimization package written in Python that includes various types of classic facility-location methods with a wide range of supported optimization solvers.

[^openlca]: [https://www.openlca.org](https://www.openlca.org)
[^R]: [https://www.r-project.org](https://www.r-project.org)
[^matlab]: [https://www.mathworks.com/products/matlab.html](https://www.mathworks.com/products/matlab.html)

# Current functionality 

Originating from the region module in PySAL, **spopt** is under active development for the inclusion of newly proposed models and methods for regionalization and facility location. Regarding regionalization, six models are developed for aggregating a large set of geographic units (with small footprints) into a smaller number of regions (with large footprints). They are:

1. Max-p-regions: the clustering of a set of geographic areas into the maximum number of homogeneous and spatially contiguous regions such that the value of a spatially extensive regional attribute is above a predefined threshold [@duque2012max;@wei2020efficient].
2. Spatially-encouraged spectral clustering (spenc): an algorithm to balance spatial and feature coherence using kernel combination in spectral clustering [@wolf2020].
3. Region-K-means: K-means clustering for regions with the constraint that each cluster forms a spatially connected component.
4. Automatic Zoning Procedure (AZP): the aggregation of data for a larger number of zones into a prespecified smaller number of regions based on a predefined type of objective function [@openshaw1977geographical;@openshaw1995algorithms].
5. Skater: a constrained spatial regionalization algorithm based on spanning tree pruning. Specifically, the number of edges is prespecified to be cut in a continuous tree to group spatial units into contiguous regions [@assunccao2006efficient].
6. WardSpatial: an agglomerative clustering (each observation starts in its own cluster, and pairs of clusters are chosen to merge at each step) using ward linkage (the goal is to minimize the variance of the clusters) with a spatial connectivity constraint ([sklearn.cluster.AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) [@scikitlearn]).

Take the functionality of Max-p-regions as an example. Other methods can be applied in a similar process, including importing the needed packages, imputing and reading data, defining the parameters, solving the model, and plotting the solution. 

```python
from spopt.region import MaxPHeuristic as MaxP
import geopandas, libpysal
# read in the data on regional incomes for Mexican states
mexico = geopandas.read_file(libpysal.examples.get_path("mexicojoin.shp"))
# specify parameters for the Max-p-regions model
# setails can be found at https://pysal.org/spopt/notebooks/maxp.html.
attrs_name = [f"PCGDP{2000}"]
w = libpysal.weights.Queen.from_dataframe(mexico)
threshold_name, threshold, top_n, mexico["count"] = "count", 6, 2, 1
# solve the Max-p-regions model
model = MaxP(mexico, w, attrs_name, threshold_name, threshold, top_n)
model.solve()
# plot the model solution
mexico["maxp_new"] = model.labels_
mexico.plot(column="maxp_new", categorical=True, edgecolor="w");
```

The corresponding solution of Max-p-regions running the above code is shown in \autoref{fig: maxp}.
It results in five regions, three of which have six states, and two with seven states each. Each region is a spatially connected component, as required by the Max-p-regions problem.

![The solution of Max-p-regions when 32 Mexican states are clustered into the maximum number of regions such that each region has at least 6 states and homogeneity in per capita gross domestic product in 2000 is maximized.\label{fig: maxp}](figs/mexico_maxp.png)

For facility-location, four models, including two coverage models and two location-allocation models based on median and center problems, are developed using an exact approach with pulp [@mitchell2011pulp] providing an interface to installed solvers.

1. Location Set Covering Problem (LSCP): Finding the minimum number of facilities and their locations such that all demands are covered within the maximal threshold of service distance or time [@Toregas1971].
2. Maximal Covering Location Problem (MCLP): Locating a prespecified number of facilities such that demand coverage within a maximal threshold of service distance or time is maximized [@Church1974].
3. P-Median Problem: Locating \textit{p} facilities and allocating the demand served by these facilities so that the total weighted assignment distance or time is minimized [@ReVelle1970]. 
4. P-Center Problem: Locating \textit{p} facilities and allocating the demand served by these facilities to minimize the maximum assignment distance or time between demands and their allocated facilities [@Hakimi1964].

For example, Maximal Covering Location Model functionality is used to select 4 out of 16 store sites in the San Francisco area to maximize demand coverage, as shown in \autoref{fig: mclp}. Other facility-location methods can be applied in a similar way. Moreover, we have included functionality to place pre-determined site locations within the facility selection pool, allowing for realistic scenarios whereby 1 or more *new* facilities can be added to an *existing* set of locations. We believe this feature of intermingled new and existing facility location siting to be the first implementation in open-source optimization software.

![The solution of MCLP while siting 4 facilities using 5 kilometers as the maximum service distance between facilities and demand locations. See the "Real World Facility Location" tutorial ([https://pysal.org/spopt/notebooks/facloc-real-world.html](https://pysal.org/spopt/notebooks/facloc-real-world.html)) for more details.\label{fig: mclp}](figs/mclp.png)

# Planned Enhancements

**Spopt** is under active development and the developers look forward to your extensive attention and participation. In the near future, there are three major enhancements we plan to pursue for **spopt**:

1. The first stream will be on the enhancement of regionalization algorithms by including several novel extensions of the classical regionalization models, such as the integration of spatial data uncertainty and the shape of identified regions in the Max-p-regions problem.
2. The second direction involves adding capacity constraints and includes polygon partial coverage for facility location models. No commercial and open-source software has provided these features to date.
3. We anticipate adding functionality for solving traditional routing and transportation-oriented optimization problems. Initially, this will come in the form of integer programming formulations of the Travelling Salesperson Problem [@miller1960integer] and the Transportation Problem [@koopmans1949optimum].

# Acknowledgements
We would like to thank all the contributors to this package. Besides, we would like to extend our gratitude to all the users for inspiring and questioning this package to make it better. **Spopt** development was partially supported by National Science Foundation Award #1831615 RIDIR: Scalable Geospatial Analytics for Social Science Research.

The following acknowledgement applies to James D. Gaboardi (affiliation 4) *only*:

> This manuscript has been authored in part by UT-Battelle LLC under contract DE-AC05-00OR22725 with the US Department of Energy (DOE). The US government retains and the publisher, by accepting the article for publication, acknowledges that the US government retains a nonexclusive, paid-up, irrevocable worldwide license to publish or reproduce the published form of this manuscript, or allow others to do so, for US government purposes. DOE will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan).

Dr. Qunshan Zhao has received UK ESRC’s ongoing support for the Urban Big Data Centre (UBDC) (ES/L011921/1 and ES/S007105/1).

# References
