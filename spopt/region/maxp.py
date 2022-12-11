"""Max-p regions algorithm.

Source: Wei, Ran, Sergio J. Rey, and Elijah Knaap (2020) "Efficient
regionalization for spatially explicit neighborhood delineation." International
Journal of Geographical Information Science. Accepted 2020-04-12.
"""

__author__ = ["Ran Wei", "Serge Rey", "Elijah Knaap"]
__email__ = "sjsrey@gmail.com"

from ..BaseClass import BaseSpOptHeuristicSolver

from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree
from scipy.sparse.csgraph import connected_components
import libpysal
import numpy as np
from copy import deepcopy
from .base import infeasible_components, plot_components, modify_components

ITERCONSTRUCT = 999
ITERSA = 10


def maxp(
    gdf,
    w,
    attrs_name,
    threshold_name,
    threshold,
    top_n,
    max_iterations_construction=ITERCONSTRUCT,
    max_iterations_sa=ITERSA,
    verbose=False,
    policy="single",
):
    """The max-p-regions involves the aggregation of n areas into an unknown maximum number of
    homogeneous regions, while ensuring that each region is contiguous and satisfies a minimum
    threshold value imposed on a predefined spatially extensive attribute.

    Parameters
    ----------

    gdf : geopandas.GeoDataFrame, required
        Geodataframe containing original data

    w : libpysal.weights.W, required
        Weights object created from given data

    attrs_name : list, required
        Strings for attribute names to measure similarity (cols of ``geopandas.GeoDataFrame``).

    threshold_name : string, requied
        The name of the spatial extensive attribute variable.

    threshold : {int, float}, required
        The threshold value.

    top_n : int
        Max number of candidate regions for enclave assignment.

    max_iterations_construction : int
        Max number of iterations for construction phase.

    max_iterations_SA: int
        Max number of iterations for customized simulated annealing.

    verbose : boolean
        Set to ``True`` for reporting solution progress/debugging.
        Default is ``False``.
    policy : str
        Defaults to ``single`` to attach infeasible components using a
        single linkage between the area in the infeasible component
        with the smallest nearest neighbor distance to an area in a
        feasible component. ``multiple`` adds joins for each area
        in an infeasible component and their nearest neighbor area in a
        feasible component. ``keep`` attempts to solve without
        modification (useful for debugging). ``drop`` removes areas in
        infeasible components before solving.

    Returns
    -------

    max_p : int
        The number of regions.

    labels : numpy.array
        Region IDs for observations.

    """
    gdf, w = modify_components(gdf, w, threshold_name, threshold, policy=policy)
    attr = np.atleast_2d(gdf[attrs_name].values)
    if attr.shape[0] == 1:
        attr = attr.T
    threshold_array = gdf[threshold_name].values
    distance_matrix = squareform(pdist(attr, metric="cityblock"))
    n, k = attr.shape
    arr = np.arange(n)

    max_p, rl_list = construction_phase(
        arr,
        attr,
        threshold_array,
        distance_matrix,
        w,
        threshold,
        top_n,
        max_iterations_construction,
    )

    if verbose:
        print("max_p: ", max_p)
        print("number of good partitions:", len(rl_list))

    alpha = 0.998
    tabuLength = 10
    max_no_move = n
    best_obj_value = np.inf
    best_label = None

    for irl, rl in enumerate(rl_list):
        label, regionList, regionSpatialAttr = rl
        if verbose:
            print(irl)
        for saiter in range(max_iterations_sa):
            finalLabel, finalRegionList, finalRegionSpatialAttr = performSA(
                label,
                regionList,
                regionSpatialAttr,
                threshold_array,
                w,
                distance_matrix,
                threshold,
                alpha,
                tabuLength,
                max_no_move,
            )
            totalWithinRegionDistance = calculateWithinRegionDistance(
                finalRegionList, distance_matrix
            )
            if verbose:
                print("totalWithinRegionDistance after SA: ")
                print(totalWithinRegionDistance)
            if totalWithinRegionDistance < best_obj_value:
                best_obj_value = totalWithinRegionDistance
                best_label = finalLabel
    if verbose:
        print("best objective value:")
        print(best_obj_value)

    return max_p, best_label


def construction_phase(
    arr,
    attr,
    threshold_array,
    distance_matrix,
    weight,
    spatialThre,
    random_assign_choice,
    max_it=999,
):
    """Construct feasible solutions for max-p-regions.

    Parameters
    ----------

    arr : array, required
        An array of index of area units.

    attr : array, required
        An array of the values of the attributes.

    threshold_array : array, required
        An array of the values of the spatial extensive attribute.

    distance_matrix : array, required
        A square-form distance matrix for the attributes.

    weight : libpysal.weights.W, required
        Weights object created from given data.

    spatialThre : {int, float}, required
        The threshold value.

    random_assign_choice : int, required
        The number of top candidate regions to consider for enclave assignment.

    max_it : int
        Maximum number of iterations. Default is 999.

    Returns
    -------

    real_values : list
        ``realmaxpv``, ``realLabelsList``

    """
    labels_list = []
    pv_list = []
    max_p = 0
    maxp_labels = None
    maxp_regionList = None
    maxp_regionSpatialAttr = None

    for _ in range(max_it):
        labels = [0] * len(threshold_array)
        C = 0
        regionSpatialAttr = {}
        enclave = []
        regionList = {}
        np.random.shuffle(arr)

        labeledID = []

        for arr_index in range(0, len(threshold_array)):
            P = arr[arr_index]
            if not (labels[P] == 0):
                continue

            NeighborPolys = deepcopy(weight.neighbors[P])

            if len(NeighborPolys) == 0:
                labels[P] = -1
            else:
                C += 1
                labeledID, spatialAttrTotal = growClusterForPoly(
                    labels, threshold_array, P, NeighborPolys, C, weight, spatialThre
                )

                if spatialAttrTotal < spatialThre:
                    C -= 1
                    enclave.extend(labeledID)
                else:
                    regionList[C] = labeledID
                    regionSpatialAttr[C] = spatialAttrTotal
        num_regions = len(regionList)

        for i, l in enumerate(labels):
            if l == -1:
                enclave.append(i)

        if num_regions < max_p:
            continue
        else:
            max_p = num_regions
            maxp_labels, maxp_regionList, maxp_regionSpatialAttr = assignEnclave(
                enclave,
                labels,
                regionList,
                regionSpatialAttr,
                threshold_array,
                weight,
                distance_matrix,
                random_assign=random_assign_choice,
            )
            pv_list.append(max_p)
            labels_list.append([maxp_labels, maxp_regionList, maxp_regionSpatialAttr])
    realLabelsList = []
    realmaxpv = max(pv_list)
    for ipv, pv in enumerate(pv_list):
        if pv == realmaxpv:
            realLabelsList.append(labels_list[ipv])

    real_values = [realmaxpv, realLabelsList]
    return real_values


def growClusterForPoly(
    labels, threshold_array, P, NeighborPolys, C, weight, spatialThre
):
    """Grow one region until threshold constraint is satisfied.

    Parameters
    ----------

    labels : list, required
        A list of current region labels

    threshold_array : array, required
        An array of the values of the spatial extensive attribute.

    P : int, required
        The index of current area unit

    NeighborPolys : list, required
        The neighbors of current area unit

    C : int, required
        The index of current region

    weight : libpysal.weights.W, required
        Weights object created from given data

    spatialThre : {int, float}, required
        The threshold value.

    Returns
    -------

    cluster_info : tuple
        ``labeledID``, ``spatialAttrTotal``

    """
    labels[P] = C
    labeledID = [P]
    spatialAttrTotal = threshold_array[P]

    i = 0

    while i < len(NeighborPolys):

        if spatialAttrTotal >= spatialThre:
            break
        Pn = NeighborPolys[i]

        if labels[Pn] == 0:
            labels[Pn] = C
            labeledID.append(Pn)
            spatialAttrTotal += threshold_array[Pn]
            if spatialAttrTotal < spatialThre:
                PnNeighborPolys = weight.neighbors[Pn]
                for pnn in PnNeighborPolys:
                    if pnn not in NeighborPolys:
                        NeighborPolys.append(pnn)
        i += 1

    cluster_info = labeledID, spatialAttrTotal
    return cluster_info


def assignEnclave(
    enclave,
    labels,
    regionList,
    regionSpatialAttr,
    threshold_array,
    weight,
    distance_matrix,
    random_assign=1,
):
    """Assign the enclaves to the regions identified in the region growth phase.

    Parameters
    ----------

    enclave : list, required
        A list of enclaves.

    labels : list, required
        A list of region labels for area units.

    regionList : dict, required
        A dictionary with key as region ID and value as a list of area
        units assigned to the region.

    regionSpatialAttr : dict, required
        A dictionary with key as region ID and value as the total
        spatial extensive attribute of the region.

    threshold_array : array, required
        An array of the values of the spatial extensive attribute.

    weight : libpysal.weights.W, required
        Weights object created from given data

    distance_matrix : array, required
        A square-form distance matrix for the attributes.

    random_assign : int, required
        The number of top candidate regions to consider for enclave assignment.

    Returns
    -------

    region_info : list
        Deep copies of ``labels``, ``regionList``, and ``regionSpatialAttr``

    """
    enclave_index = 0
    while len(enclave) > 0:
        ec = enclave[enclave_index]
        ecNeighbors = weight.neighbors[ec]
        assignedRegion = 0
        ecNeighborsList = []

        for ecn in ecNeighbors:
            if ecn in enclave:
                continue
            rm = np.array(regionList[labels[ecn]])
            totalDistance = distance_matrix[ec, rm].sum()
            ecNeighborsList.append((ecn, totalDistance))
        ecNeighborsList = sorted(ecNeighborsList, key=lambda tup: tup[1])
        top_num = min([len(ecNeighborsList), random_assign])
        if top_num > 0:
            ecn_index = np.random.randint(top_num)
            assignedRegion = labels[ecNeighborsList[ecn_index][0]]

        if assignedRegion == 0:
            enclave_index += 1
        else:
            labels[ec] = assignedRegion
            regionList[assignedRegion].append(ec)
            regionSpatialAttr[assignedRegion] += threshold_array[ec]
            del enclave[enclave_index]
            enclave_index = 0

    region_info = [deepcopy(labels), deepcopy(regionList), deepcopy(regionSpatialAttr)]
    return region_info


def calculateWithinRegionDistance(regionList, distance_matrix):
    """Calculate total wthin-region distance/dissimilarity.

    Parameters
    ----------

    regionList : dict, required
        A dictionary with key as region ID and value as a list of area
        units assigned to the region.

    distance_matrix : array, required
        A square-form distance matrix for the attributes.

    Returns
    -------

    totalWithinRegionDistance : {int, float}
        the total within-region distance

    """
    totalWithinRegionDistance = 0
    for k, v in regionList.items():
        nv = np.array(v)
        regionDistance = distance_matrix[nv, :][:, nv].sum() / 2
        totalWithinRegionDistance += regionDistance

    return totalWithinRegionDistance


def pickMoveArea(
    labels,
    regionLists,
    regionSpatialAttrs,
    threshold_array,
    weight,
    distance_matrix,
    threshold,
):
    """Pick a spatial unit that can move from one region to another.

    Parameters
    ----------

    labels : list, required
        A list of current region labels

    regionLists : dict, required
        A dictionary with key as region ID and value as a list of area
        units assigned to the region.

    regionSpatialAttrs : dict, required
        A dictionary with key as region ID and value as the total
        spatial extensive attribute of the region.

    threshold_array : array, required
        An array of the values of the spatial extensive attribute.

    weight :libpysal.weights.W, required
        Weights object created from given data

    threshold : {int, float}, required
        The threshold value.

    Returns
    -------

    potentialAreas : list
        a list of area units that can move without violating
        contiguity and threshold constraints

    """
    potentialAreas = []
    for k, v in regionSpatialAttrs.items():
        rla = np.array(regionLists[k])
        rasa = threshold_array[rla]
        lostSA = v - rasa
        pas_indices = np.where(lostSA > threshold)[0]
        if pas_indices.size > 0:
            for pasi in pas_indices:
                leftAreas = np.delete(rla, pasi)
                ws = weight.sparse
                cc = connected_components(ws[leftAreas, :][:, leftAreas])
                if cc[0] == 1:
                    potentialAreas.append(rla[pasi])
        else:
            continue

    return potentialAreas


def checkMove(
    poa, labels, regionLists, threshold_array, weight, distance_matrix, threshold
):
    """Calculate the dissimilarity increase/decrease from one potential move.

    Parameters
    ----------

    poa : int, required
        The index of current area unit that can potentially move

    labels : list, required
        A list of current region labels

    regionLists : dict, required
        A dictionary with key as region ID and value as a list of area
        units assigned to the region.

    threshold_array : array, required
        An array of the values of the spatial extensive attribute.

    weight : libpysal.weights.W, required
        Weights object created from given data

    distance_matrix : array, required
        A square-form distance matrix for the attributes.

    threshold : {int, float}, required
        The threshold value.

    Returns
    -------

    move_info : list
        ``lostDistance``, ``minAddedDistance``, and ``potentialMove``.

    """
    poaNeighbor = weight.neighbors[poa]
    donorRegion = labels[poa]

    rm = np.array(regionLists[donorRegion])
    lostDistance = distance_matrix[poa, rm].sum()
    potentialMove = None

    minAddedDistance = np.Inf
    for poan in poaNeighbor:
        recipientRegion = labels[poan]
        if donorRegion != recipientRegion:
            rm = np.array(regionLists[recipientRegion])
            addedDistance = distance_matrix[poa, rm].sum()

            if addedDistance < minAddedDistance:
                minAddedDistance = addedDistance
                potentialMove = (poa, donorRegion, recipientRegion)

    move_info = [lostDistance, minAddedDistance, potentialMove]
    return move_info


def performSA(
    initLabels,
    initRegionList,
    initRegionSpatialAttr,
    threshold_array,
    weight,
    distance_matrix,
    threshold,
    alpha,
    tabuLength,
    max_no_move,
):
    """Perform the tabu list integrated simulated annealing algorithm.

    Parameters
    ----------

    initLabels : list, required
        A list of initial region labels before SA

    initRegionList : dict, required
        A dictionary with key as region ID and value as a list of area
        units assigned to the region before SA.

    initRegionSpatialAttr : dict, required
        A dictionary with key as region ID and value as the total
        spatial extensive attribute of the region before SA.

    threshold_array : array, required
        An array of the values of the spatial extensive attribute.

    weight : libpysal.weights.W, required
        Weights object created from given data.

    distance_matrix : array, required
        A square-form distance matrix for the attributes.

    threshold : {int, float}, required
        The threshold value.

    alpha : float between 0 and 1, required
        Temperature cooling rate

    tabuLength : int, required
        Max length of a tabuList

    max_no_move : int, required
        Max number of none improving movements

    Returns
    -------

    sa_res : list
        The results from simulated annealing including ``labels``,
        ``regionLists``, and ``regionSpatialAttrs``.

    """
    t = 1
    ni_move_ct = 0
    make_move_flag = False
    tabuList = []
    potentialAreas = []

    labels = deepcopy(initLabels)
    regionLists = deepcopy(initRegionList)
    regionSpatialAttrs = deepcopy(initRegionSpatialAttr)

    while ni_move_ct <= max_no_move:
        if len(potentialAreas) == 0:
            potentialAreas = pickMoveArea(
                labels,
                regionLists,
                regionSpatialAttrs,
                threshold_array,
                weight,
                distance_matrix,
                threshold,
            )

        if len(potentialAreas) == 0:
            break
        poa = potentialAreas[np.random.randint(len(potentialAreas))]
        lostDistance, minAddedDistance, potentialMove = checkMove(
            poa,
            labels,
            regionLists,
            threshold_array,
            weight,
            distance_matrix,
            threshold,
        )

        if potentialMove is None:
            potentialAreas.remove(poa)
            continue

        diff = lostDistance - minAddedDistance
        donorRegion = potentialMove[1]
        recipientRegion = potentialMove[2]

        if diff > 0:
            make_move_flag = True
            if (poa, recipientRegion, donorRegion) not in tabuList:
                if len(tabuList) == tabuLength:
                    tabuList.pop(0)
                tabuList.append((poa, recipientRegion, donorRegion))

            ni_move_ct = 0
        else:
            ni_move_ct += 1
            prob = np.exp(diff / t)
            if prob > np.random.random() and potentialMove not in tabuList:
                make_move_flag = True
            else:
                make_move_flag = False

        potentialAreas.remove(poa)
        if make_move_flag:
            labels[poa] = recipientRegion
            regionLists[donorRegion].remove(poa)
            regionLists[recipientRegion].append(poa)
            regionSpatialAttrs[donorRegion] -= threshold_array[poa]
            regionSpatialAttrs[recipientRegion] += threshold_array[poa]

            impactedAreas = []
            for pa in potentialAreas:
                if labels[pa] == recipientRegion or labels[pa] == donorRegion:
                    impactedAreas.append(pa)
            for pa in impactedAreas:
                potentialAreas.remove(pa)

        t = t * alpha
    return [labels, regionLists, regionSpatialAttrs]


class MaxPHeuristic(BaseSpOptHeuristicSolver):
    """The max-p-regions involves the aggregation of n areas into an
    unknown maximum number of homogeneous regions, while ensuring that
    each region is contiguious and satisfies a minimum threshold value
    imposed on a predefined spatially extensive attribute.

    Parameters
    ----------

    gdf : geopandas.GeoDataFrame, required
        Geodataframe containing original data.

    w : libpysal.weights.W, required
        Weights object created from given data.

    attrs_name : list, required
        Strings for attribute names (cols of ``geopandas.GeoDataFrame``).

    threshold_name : string, required
        The name of the spatial extensive attribute variable.

    threshold : {int, float}, required
        The threshold value.

    top_n : int, required
        The number of top candidate regions to consider for enclave assignment.

    max_iterations_construction : int
        Max number of iterations for construction phase.

    max_iterations_SA : int
        Max number of iterations for customized simulated annealing.

    verbose : boolean
        Set to ``True`` for reporting solution progress/debugging.
        Default is ``False``.

    policy : str
        Defaults to ``'single'`` to attach infeasible components using a
        single linkage between the area in the infeasible component
        with the smallest nearest neighbor distance to an area in a
        feasible component. ``'multiple'`` adds joins for each area
        in an infeasible component and their nearest neighbor area in a
        feasible component. ``'keep'`` attempts to solve without
        modification (useful for debugging). ``'drop'`` removes areas in
        infeasible components before solving.

    Attributes
    ----------

    max_p : int
        The number of regions.
    labels_ : numpy.array
        Region IDs for observations.

    Examples
    --------

    >>> import numpy
    >>> import libpysal
    >>> import geopandas as gpd
    >>> from spopt.region.maxp import MaxPHeuristic

    Read the data.

    >>> pth = libpysal.examples.get_path("mexicojoin.shp")
    >>> mexico = gpd.read_file(pth)
    >>> mexico["count"] = 1

    Create the weight.

    >>> w = libpysal.weights.Queen.from_dataframe(mexico)

    Define the columns of ``geopandas.GeoDataFrame`` to be spatially
    extensive attribute.

    >>> attrs_name = [f"PCGDP{year}" for year in range(1950, 2010, 10)]

    Define the spatial extensive attribute variable and the threshold value.

    >>> threshold_name = "count"
    >>> threshold = 4

    Run the max-p-regions algorithm.

    >>> model = MaxPHeuristic(mexico, w, attrs_name, threshold_name, threshold)
    >>> model.solve()

    Get the number of regions and region IDs for unit areas.

    >>> model.p
    >>> model.labels_

    """

    def __init__(
        self,
        gdf,
        w,
        attrs_name,
        threshold_name,
        threshold,
        top_n=2,
        max_iterations_construction=99,
        max_iterations_sa=ITERSA,
        verbose=False,
        policy="single",
    ):

        self.gdf = gdf
        self.w = w
        self.attrs_name = attrs_name
        self.threshold_name = threshold_name
        self.threshold = threshold
        self.top_n = top_n
        self.max_iterations_construction = max_iterations_construction
        self.max_iterations_sa = max_iterations_sa
        self.verbose = verbose
        self.policy = policy

    def solve(self):
        """Solve a max-p-regions problem and get back the results."""
        max_p, label = maxp(
            self.gdf,
            self.w,
            self.attrs_name,
            self.threshold_name,
            self.threshold,
            self.top_n,
            self.max_iterations_construction,
            self.max_iterations_sa,
            verbose=self.verbose,
            policy=self.policy,
        )
        self.labels_ = label
        self.p = max_p
