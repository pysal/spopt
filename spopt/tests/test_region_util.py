import libpysal
import geopandas
import networkx
import numpy
import pulp
import pytest

import spopt.region.util as util


RANDOM_STATE = 123456

# Mexican states
pth = libpysal.examples.get_path("mexicojoin.shp")
MEXICO = geopandas.read_file(pth)


class TestRegionUtil:
    def setup_method(self):
        pass

    def test_array_from_dict_values(self):
        dict_flat = {0: 0, 1: 10}
        dict_it = {0: [0], 1: [10]}
        desired_flat = numpy.array([0, 10])
        desired_2d = numpy.array([[0], [10]])

        observed = util.array_from_dict_values(dict_flat, flat_output=True)
        numpy.testing.assert_array_equal(observed, desired_flat)

        observed = util.array_from_dict_values(dict_flat)
        numpy.testing.assert_array_equal(observed, desired_2d)

        observed = util.array_from_dict_values(dict_it, flat_output=True)
        numpy.testing.assert_array_equal(observed, desired_flat)

        observed = util.array_from_dict_values(dict_it)
        numpy.testing.assert_array_equal(observed, desired_2d)

    def test_scipy_sparse_matrix_from_dict(self):
        neighbors = {
            0: {1, 3},
            1: {0, 2, 4},
            2: {1, 5},
            3: {0, 4},
            4: {1, 3, 5},
            5: {2, 4},
        }
        desired = numpy.array(
            [
                [0, 1, 0, 1, 0, 0],
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 1, 0, 1],
                [0, 0, 1, 0, 1, 0],
            ]
        )
        observed = util.scipy_sparse_matrix_from_dict(neighbors)
        numpy.testing.assert_array_equal(observed.todense(), desired)

        neighbors = {
            "left": {"middle"},
            "middle": {"left", "right"},
            "right": {"middle"},
        }
        desired = numpy.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        observed = util.scipy_sparse_matrix_from_dict(neighbors)
        numpy.testing.assert_array_equal(observed.todense(), desired)

    def test_dict_from_graph_attr(self):
        desired = {0: [1]}
        observed = util.dict_from_graph_attr(None, desired)
        assert observed == desired

        edges = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5)]
        graph = networkx.Graph(edges)
        data_dict = {node: 10 * node for node in graph}
        networkx.set_node_attributes(graph, data_dict, "test_data")

        desired = {key: [value] for key, value in data_dict.items()}
        observed = util.dict_from_graph_attr(graph, "test_data")
        assert observed == desired

        desired_array = dict()
        for node in data_dict:
            desired_array[node] = numpy.array(data_dict[node])
        observed = util.dict_from_graph_attr(graph, "test_data", array_values=True)
        assert observed == desired

    def test_check_solver(self):
        util.check_solver("cbc")
        with pytest.raises(ValueError, match="The solver must be one of"):
            util.check_solver("bcb")

    def test_get_solver_instance(self):
        known_name = "PULP_CBC_CMD"
        observed_name = util.get_solver_instance("cbc").name
        assert known_name == observed_name
