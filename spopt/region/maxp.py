"""Max-p regions algorithm.

Source: Wei, Ran, Sergio J. Rey, and Elijah Knaap (2020) "Efficient
regionalization for spatially explicit neighborhood delineation." International
Journal of Geographical Information Science. Accepted 2020-04-12.
"""

__author__ = ["Ran Wei", "Serge Rey", "Elijah Knaap"]
__email__ = "sjsrey@gmail.com"

from copy import deepcopy

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

from ..BaseClass import BaseSpOptHeuristicSolver
from .base import modify_components

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
    """The max-p-regions involves the aggregation of n areas into an unknown maximum
     number of homogeneous regions, while ensuring that each region is contiguous and
     satisfies a minimum threshold value imposed on a predefined spatially extensive
    attribute.

    Parameters
    ----------

    gdf : geopandas.GeoDataFrame, required
        Geodataframe containing original data

    w : libpysal.weights.W, required
        Weights object created from given data

    attrs_name : list, required
        Strings for attribute names to measure similarity
        (cols of ``geopandas.GeoDataFrame``).

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

    max_p, rl_list = _construction_phase(
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
    tabu_length = 10
    max_no_move = n
    best_obj_value = np.inf
    best_label = None

    for irl, rl in enumerate(rl_list):
        label, region_list, region_spatial_attr = rl
        if verbose:
            print(irl)
        for _saiter in range(max_iterations_sa):
            final_label, final_region_list, final_region_spatial_attr = _perform_sa(
                label,
                region_list,
                region_spatial_attr,
                threshold_array,
                w,
                distance_matrix,
                threshold,
                alpha,
                tabu_length,
                max_no_move,
            )
            total_within_region_distance = _calculate_within_region_distance(
                final_region_list, distance_matrix
            )
            if verbose:
                print("total_within_region_distance after SA: ")
                print(total_within_region_distance)
            if total_within_region_distance < best_obj_value:
                best_obj_value = total_within_region_distance
                best_label = final_label
    if verbose:
        print("best objective value:")
        print(best_obj_value)

    return max_p, best_label


def _construction_phase(
    arr,
    attr,  # noqa: ARG001
    threshold_array,
    distance_matrix,
    weight,
    spatial_thre,
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

    spatial_thre : {int, float}, required
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
    maxp_region_list = None
    maxp_region_spatial_attr = None

    for _ in range(max_it):
        labels = [0] * len(threshold_array)
        c = 0
        region_spatial_attr = {}
        enclave = []
        region_list = {}
        np.random.shuffle(arr)

        labeled_id = []

        for arr_index in range(0, len(threshold_array)):
            p = arr[arr_index]
            if labels[p] != 0:
                continue

            neighbor_polys = deepcopy(weight.neighbors[p])

            if len(neighbor_polys) == 0:
                labels[p] = -1
            else:
                c += 1
                labeled_id, spatial_attr_total = _grow_cluster_for_poly(
                    labels,
                    threshold_array,
                    p,
                    neighbor_polys,
                    c,
                    weight,
                    spatial_thre,
                )

                if spatial_attr_total < spatial_thre:
                    c -= 1
                    enclave.extend(labeled_id)
                else:
                    region_list[c] = labeled_id
                    region_spatial_attr[c] = spatial_attr_total
        num_regions = len(region_list)

        for i, _l in enumerate(labels):
            if _l == -1:
                enclave.append(i)

        if num_regions < max_p:
            continue
        else:
            max_p = num_regions
            maxp_labels, maxp_region_list, maxp_region_spatial_attr = _assign_enclave(
                enclave,
                labels,
                region_list,
                region_spatial_attr,
                threshold_array,
                weight,
                distance_matrix,
                random_assign=random_assign_choice,
            )
            pv_list.append(max_p)
            labels_list.append(
                [maxp_labels, maxp_region_list, maxp_region_spatial_attr]
            )
    real_labels_list = []
    realmaxpv = max(pv_list)
    for ipv, pv in enumerate(pv_list):
        if pv == realmaxpv:
            real_labels_list.append(labels_list[ipv])

    real_values = [realmaxpv, real_labels_list]
    return real_values


def _grow_cluster_for_poly(
    labels, threshold_array, p, neighbor_polys, c, weight, spatial_thre
):
    """Grow one region until threshold constraint is satisfied.

    Parameters
    ----------

    labels : list, required
        A list of current region labels

    threshold_array : array, required
        An array of the values of the spatial extensive attribute.

    p : int, required
        The index of current area unit

    neighbor_polys : list, required
        The neighbors of current area unit

    c : int, required
        The index of current region

    weight : libpysal.weights.W, required
        Weights object created from given data

    spatial_thre : {int, float}, required
        The threshold value.

    Returns
    -------

    cluster_info : tuple
        ``labeled_id``, ``spatial_attr_total``

    """
    labels[p] = c
    labeled_id = [p]
    spatial_attr_total = threshold_array[p]

    i = 0

    while i < len(neighbor_polys):
        if spatial_attr_total >= spatial_thre:
            break
        p_n = neighbor_polys[i]

        if labels[p_n] == 0:
            labels[p_n] = c
            labeled_id.append(p_n)
            spatial_attr_total += threshold_array[p_n]
            if spatial_attr_total < spatial_thre:
                p_n_neighbor_polys = weight.neighbors[p_n]
                for pnn in p_n_neighbor_polys:
                    if pnn not in neighbor_polys:
                        neighbor_polys.append(pnn)
        i += 1

    cluster_info = labeled_id, spatial_attr_total
    return cluster_info


def _assign_enclave(
    enclave,
    labels,
    region_list,
    region_spatial_attr,
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

    region_list : dict, required
        A dictionary with key as region ID and value as a list of area
        units assigned to the region.

    region_spatial_attr : dict, required
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
        Deep copies of ``labels``, ``region_list``, and ``region_spatial_attr``

    """
    enclave_index = 0
    while len(enclave) > 0:
        ec = enclave[enclave_index]
        ec_neighbors = weight.neighbors[ec]
        assigned_region = 0
        ec_neighbors_list = []

        for ecn in ec_neighbors:
            if ecn in enclave:
                continue
            rm = np.array(region_list[labels[ecn]])
            total_distance = distance_matrix[ec, rm].sum()
            ec_neighbors_list.append((ecn, total_distance))
        ec_neighbors_list = sorted(ec_neighbors_list, key=lambda tup: tup[1])
        top_num = min([len(ec_neighbors_list), random_assign])
        if top_num > 0:
            ecn_index = np.random.randint(top_num)
            assigned_region = labels[ec_neighbors_list[ecn_index][0]]

        if assigned_region == 0:
            enclave_index += 1
        else:
            labels[ec] = assigned_region
            region_list[assigned_region].append(ec)
            region_spatial_attr[assigned_region] += threshold_array[ec]
            del enclave[enclave_index]
            enclave_index = 0

    region_info = [
        deepcopy(labels),
        deepcopy(region_list),
        deepcopy(region_spatial_attr),
    ]
    return region_info


def _calculate_within_region_distance(region_list, distance_matrix):
    """Calculate total wthin-region distance/dissimilarity.

    Parameters
    ----------

    region_list : dict, required
        A dictionary with key as region ID and value as a list of area
        units assigned to the region.

    distance_matrix : array, required
        A square-form distance matrix for the attributes.

    Returns
    -------

    total_within_region_distance : {int, float}
        the total within-region distance

    """
    total_within_region_distance = 0
    for _k, v in region_list.items():
        nv = np.array(v)
        region_distance = distance_matrix[nv, :][:, nv].sum() / 2
        total_within_region_distance += region_distance

    return total_within_region_distance


def _pick_move_area(
    labels,  # noqa: ARG001
    region_lists,
    region_spatial_attrs,
    threshold_array,
    weight,
    distance_matrix,  # noqa: ARG001
    threshold,
):
    """Pick a spatial unit that can move from one region to another.

    Parameters
    ----------

    labels : list, required
        A list of current region labels

    region_lists : dict, required
        A dictionary with key as region ID and value as a list of area
        units assigned to the region.

    region_spatial_attrs : dict, required
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

    potential_areas : list
        a list of area units that can move without violating
        contiguity and threshold constraints

    """
    potential_areas = []
    for k, v in region_spatial_attrs.items():
        rla = np.array(region_lists[k])
        rasa = threshold_array[rla]
        lost_sa = v - rasa
        pas_indices = np.where(lost_sa > threshold)[0]
        if pas_indices.size > 0:
            for pasi in pas_indices:
                left_areas = np.delete(rla, pasi)
                ws = weight.sparse
                cc = connected_components(ws[left_areas, :][:, left_areas])
                if cc[0] == 1:
                    potential_areas.append(rla[pasi])
        else:
            continue

    return potential_areas


def _check_move(
    poa,
    labels,
    region_lists,
    threshold_array,  # noqa: ARG001
    weight,
    distance_matrix,
    threshold,  # noqa: ARG001
):
    """Calculate the dissimilarity increase/decrease from one potential move.

    Parameters
    ----------

    poa : int, required
        The index of current area unit that can potentially move

    labels : list, required
        A list of current region labels

    region_lists : dict, required
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
        ``lost_distance``, ``min_added_distance``, and ``potential_move``.

    """
    poa_neighbor = weight.neighbors[poa]
    donor_region = labels[poa]

    rm = np.array(region_lists[donor_region])
    lost_distance = distance_matrix[poa, rm].sum()
    potential_move = None

    min_added_distance = np.inf
    for poan in poa_neighbor:
        recipient_region = labels[poan]
        if donor_region != recipient_region:
            rm = np.array(region_lists[recipient_region])
            added_distance = distance_matrix[poa, rm].sum()

            if added_distance < min_added_distance:
                min_added_distance = added_distance
                potential_move = (poa, donor_region, recipient_region)

    move_info = [lost_distance, min_added_distance, potential_move]
    return move_info


def _perform_sa(
    init_labels,
    init_region_list,
    init_region_spatial_attr,
    threshold_array,
    weight,
    distance_matrix,
    threshold,
    alpha,
    tabu_length,
    max_no_move,
):
    """Perform the tabu list integrated simulated annealing algorithm.

    Parameters
    ----------

    init_labels : list, required
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

    tabu_length : int, required
        Max length of a tabuList

    max_no_move : int, required
        Max number of none improving movements

    Returns
    -------

    sa_res : list
        The results from simulated annealing including ``labels``,
        ``region_lists``, and ``region_spatial_attrs``.

    """
    t = 1
    ni_move_ct = 0
    make_move_flag = False
    tabu_list = []
    potential_areas = []

    labels = deepcopy(init_labels)
    region_lists = deepcopy(init_region_list)
    region_spatial_attrs = deepcopy(init_region_spatial_attr)

    while ni_move_ct <= max_no_move:
        if len(potential_areas) == 0:
            potential_areas = _pick_move_area(
                labels,
                region_lists,
                region_spatial_attrs,
                threshold_array,
                weight,
                distance_matrix,
                threshold,
            )

        if len(potential_areas) == 0:
            break
        poa = potential_areas[np.random.randint(len(potential_areas))]
        lost_distance, min_added_distance, potential_move = _check_move(
            poa,
            labels,
            region_lists,
            threshold_array,
            weight,
            distance_matrix,
            threshold,
        )

        if potential_move is None:
            potential_areas.remove(poa)
            continue

        diff = lost_distance - min_added_distance
        donor_region = potential_move[1]
        recipient_region = potential_move[2]

        if diff > 0:
            make_move_flag = True
            if (poa, recipient_region, donor_region) not in tabu_list:
                if len(tabu_list) == tabu_length:
                    tabu_list.pop(0)
                tabu_list.append((poa, recipient_region, donor_region))

            ni_move_ct = 0
        else:
            ni_move_ct += 1
            prob = np.exp(diff / t)
            if prob > np.random.random() and potential_move not in tabu_list:
                make_move_flag = True
            else:
                make_move_flag = False

        potential_areas.remove(poa)
        if make_move_flag:
            labels[poa] = recipient_region
            region_lists[donor_region].remove(poa)
            region_lists[recipient_region].append(poa)
            region_spatial_attrs[donor_region] -= threshold_array[poa]
            region_spatial_attrs[recipient_region] += threshold_array[poa]

            impacted_areas = []
            for pa in potential_areas:
                if labels[pa] == recipient_region or labels[pa] == donor_region:
                    impacted_areas.append(pa)
            for pa in impacted_areas:
                potential_areas.remove(pa)

        t = t * alpha
    return [labels, region_lists, region_spatial_attrs]


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
