import numpy as np
import pandas as pd
import pulp
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix
from spopt.locate.flow import FRLM
import os
import pickle


def load_grid_test_data():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    filepath = os.path.join(data_dir, "flow_grid_network.pkl")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["network"], data["flows"]


class TestFRLMBasicFunctionality:

    @pytest.fixture
    def setup_simple_network(self):
        network, flows = load_grid_test_data()
        simple_flows = flows.iloc[:2].copy()
        return network, simple_flows

    @pytest.fixture
    def setup_grid_network(self):
        return load_grid_test_data()

    def test_basic_initialization(self):
        model = FRLM()
        assert model.vehicle_range == 200000
        assert model.p_facilities == 5
        assert model.capacity is None
        assert model.threshold == 0.0
        assert model.weight == 0.99
        assert model.objective == "flow"
        assert model.include_destination is False

    def test_vehicle_range_percentage(self, setup_simple_network):
        network, flows = setup_simple_network

        model = FRLM.from_flow_dataframe(
            network=network, flows=flows, p_facilities=3, vehicle_range=0.5
        )

        assert model.vehicle_range == 30
        result = model.solve(solver=pulp.PULP_CBC_CMD(msg=0))
        assert result["status"] == "Optimal"
        assert len(model.selected_facilities) == 3
        coverage = model.get_flow_coverage()
        assert coverage["covered_proportion"] == 1.0

    def test_ac_pc_approach(self, setup_grid_network):
        network, flows = setup_grid_network
        model = FRLM.from_flow_dataframe(
            network=network, flows=flows, p_facilities=3, vehicle_range=50
        )

        model.generate_path_refueling_combinations(method="ac_pc")
        assert model.use_ac_pc is True

        expected_K = {
            (1, 1): [0, 1, 2, 3, 4, 5, 6, 8, 9],
            (1, 2): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13],
            (1, 3): [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            (1, 4): [1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
            (1, 5): [5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
            (1, 6): [6, 7, 9, 10, 11, 12, 13, 14, 15],
            (2, 1): [0, 1, 2, 3, 5, 6, 7, 10, 11],
            (2, 2): [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14],
            (2, 3): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15],
            (2, 4): [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            (2, 5): [5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
            (2, 6): [4, 5, 8, 9, 10, 12, 13, 14, 15],
            (3, 1): [0, 4, 5, 8, 9, 10, 12, 13, 14],
            (3, 2): [0, 1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14],
            (3, 3): [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            (3, 4): [2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15],
            (3, 5): [2, 3, 5, 6, 7, 9, 10, 11, 14, 15],
            (3, 6): [1, 2, 3, 5, 6, 7, 10, 11, 15],
            (4, 1): [3, 6, 7, 9, 10, 11, 13, 14, 15],
            (4, 2): [2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15],
            (4, 3): [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            (4, 4): [0, 1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14],
            (4, 5): [0, 1, 4, 5, 6, 8, 9, 10, 12, 13],
            (4, 6): [0, 1, 2, 4, 5, 6, 8, 9, 12],
            (5, 1): [0, 1, 2, 4, 5, 6, 8, 9, 12],
            (5, 2): [0, 1, 4, 5, 6, 8, 9, 10, 12, 13],
            (5, 3): [0, 4, 5, 8, 9, 10, 12, 13, 14],
            (6, 1): [3, 6, 7, 9, 10, 11, 13, 14, 15],
            (6, 2): [2, 3, 5, 6, 7, 9, 10, 11, 14, 15],
            (6, 3): [1, 2, 3, 5, 6, 7, 10, 11, 15]
        }

        expected_a = {
            1: [1, 2, 3, 4, 5, 6],
            2: [1, 2, 3, 4, 5, 6],
            3: [1, 2, 3, 4, 5, 6],
            4: [1, 2, 3, 4, 5, 6],
            5: [1, 2, 3],
            6: [1, 2, 3]
        }
        
        assert model.K == expected_K
        assert model.a == expected_a

        result = model.solve(solver=pulp.PULP_CBC_CMD(msg=0))
        assert result["status"] == "Optimal"

    def test_combination_approach(self, setup_grid_network):
        network, flows = setup_grid_network
        model = FRLM.from_flow_dataframe(
            network=network, flows=flows, p_facilities=3, vehicle_range=50, capacity=500
        )

        assert model.use_ac_pc is False

        expected_combinations = {
            (0, 15): [[0, 13], [0, 14], [1, 13], [1, 14], [1, 15], [5, 13], [5, 14], [5, 15], [0, 1, 13], [0, 1, 14], [0, 1, 15], [0, 2, 13], [0, 2, 14], [0, 3, 13], [0, 3, 14], [0, 4, 13], [0, 4, 14], [0, 5, 13], [0, 5, 14], [0, 5, 15], [0, 6, 13], [0, 6, 14], [0, 7, 13], [0, 7, 14], [0, 8, 13], [0, 8, 14], [0, 9, 13], [0, 9, 14], [0, 9, 15], [0, 10, 13], [0, 10, 14], [0, 11, 13], [0, 11, 14], [0, 12, 13], [0, 12, 14], [0, 13, 14], [0, 13, 15], [0, 14, 15], [1, 2, 13], [1, 2, 14], [1, 2, 15], [1, 3, 13], [1, 3, 14], [1, 3, 15], [1, 4, 13], [1, 4, 14], [1, 4, 15], [1, 5, 13], [1, 5, 14], [1, 5, 15], [1, 6, 13], [1, 6, 14], [1, 6, 15], [1, 7, 13], [1, 7, 14], [1, 7, 15], [1, 8, 13], [1, 8, 14], [1, 8, 15], [1, 9, 13], [1, 9, 14], [1, 9, 15], [1, 10, 13], [1, 10, 14], [1, 10, 15], [1, 11, 13], [1, 11, 14], [1, 11, 15], [1, 12, 13], [1, 12, 14], [1, 12, 15], [1, 13, 14], [1, 13, 15], [1, 14, 15], [2, 5, 13], [2, 5, 14], [2, 5, 15], [3, 5, 13], [3, 5, 14], [3, 5, 15], [4, 5, 13], [4, 5, 14], [4, 5, 15], [5, 6, 13], [5, 6, 14], [5, 6, 15], [5, 7, 13], [5, 7, 14], [5, 7, 15], [5, 8, 13], [5, 8, 14], [5, 8, 15], [5, 9, 13], [5, 9, 14], [5, 9, 15], [5, 10, 13], [5, 10, 14], [5, 10, 15], [5, 11, 13], [5, 11, 14], [5, 11, 15], [5, 12, 13], [5, 12, 14], [5, 12, 15], [5, 13, 14], [5, 13, 15], [5, 14, 15]],
            (3, 12): [[2, 12], [2, 13], [2, 14], [3, 13], [3, 14], [6, 12], [6, 13], [6, 14], [0, 2, 12], [0, 2, 13], [0, 2, 14], [0, 3, 13], [0, 3, 14], [0, 6, 12], [0, 6, 13], [0, 6, 14], [1, 2, 12], [1, 2, 13], [1, 2, 14], [1, 3, 13], [1, 3, 14], [1, 6, 12], [1, 6, 13], [1, 6, 14], [2, 3, 12], [2, 3, 13], [2, 3, 14], [2, 4, 12], [2, 4, 13], [2, 4, 14], [2, 5, 12], [2, 5, 13], [2, 5, 14], [2, 6, 12], [2, 6, 13], [2, 6, 14], [2, 7, 12], [2, 7, 13], [2, 7, 14], [2, 8, 12], [2, 8, 13], [2, 8, 14], [2, 9, 12], [2, 9, 13], [2, 9, 14], [2, 10, 12], [2, 10, 13], [2, 10, 14], [2, 11, 12], [2, 11, 13], [2, 11, 14], [2, 12, 13], [2, 12, 14], [2, 12, 15], [2, 13, 14], [2, 13, 15], [2, 14, 15], [3, 4, 13], [3, 4, 14], [3, 5, 13], [3, 5, 14], [3, 6, 12], [3, 6, 13], [3, 6, 14], [3, 7, 13], [3, 7, 14], [3, 8, 13], [3, 8, 14], [3, 9, 13], [3, 9, 14], [3, 10, 12], [3, 10, 13], [3, 10, 14], [3, 11, 13], [3, 11, 14], [3, 12, 13], [3, 12, 14], [3, 13, 14], [3, 13, 15], [3, 14, 15], [4, 6, 12], [4, 6, 13], [4, 6, 14], [5, 6, 12], [5, 6, 13], [5, 6, 14], [6, 7, 12], [6, 7, 13], [6, 7, 14], [6, 8, 12], [6, 8, 13], [6, 8, 14], [6, 9, 12], [6, 9, 13], [6, 9, 14], [6, 10, 12], [6, 10, 13], [6, 10, 14], [6, 11, 12], [6, 11, 13], [6, 11, 14], [6, 12, 13], [6, 12, 14], [6, 12, 15], [6, 13, 14], [6, 13, 15], [6, 14, 15]],
            (12, 3): [[3, 8], [3, 9], [7, 8], [7, 9], [7, 12], [8, 11], [9, 11], [11, 12], [0, 3, 8], [0, 3, 9], [0, 7, 8], [0, 7, 9], [0, 7, 12], [0, 8, 11], [0, 9, 11], [0, 11, 12], [1, 3, 8], [1, 3, 9], [1, 7, 8], [1, 7, 9], [1, 7, 12], [1, 8, 11], [1, 9, 11], [1, 11, 12], [2, 3, 8], [2, 3, 9], [2, 7, 8], [2, 7, 9], [2, 7, 12], [2, 8, 11], [2, 9, 11], [2, 11, 12], [3, 4, 8], [3, 4, 9], [3, 5, 8], [3, 5, 9], [3, 6, 8], [3, 6, 9], [3, 7, 8], [3, 7, 9], [3, 7, 12], [3, 8, 9], [3, 8, 10], [3, 8, 11], [3, 8, 12], [3, 8, 13], [3, 8, 14], [3, 8, 15], [3, 9, 10], [3, 9, 11], [3, 9, 12], [3, 9, 13], [3, 9, 14], [3, 9, 15], [3, 10, 12], [3, 11, 12], [4, 7, 8], [4, 7, 9], [4, 7, 12], [4, 8, 11], [4, 9, 11], [4, 11, 12], [5, 7, 8], [5, 7, 9], [5, 7, 12], [5, 8, 11], [5, 9, 11], [5, 11, 12], [6, 7, 8], [6, 7, 9], [6, 7, 12], [6, 8, 11], [6, 9, 11], [6, 11, 12], [7, 8, 9], [7, 8, 10], [7, 8, 11], [7, 8, 12], [7, 8, 13], [7, 8, 14], [7, 8, 15], [7, 9, 10], [7, 9, 11], [7, 9, 12], [7, 9, 13], [7, 9, 14], [7, 9, 15], [7, 10, 12], [7, 11, 12], [7, 12, 13], [7, 12, 14], [7, 12, 15], [8, 9, 11], [8, 10, 11], [8, 11, 12], [8, 11, 13], [8, 11, 14], [8, 11, 15], [9, 10, 11], [9, 11, 12], [9, 11, 13], [9, 11, 14], [9, 11, 15], [10, 11, 12], [11, 12, 13], [11, 12, 14], [11, 12, 15]],
            (15, 0): [[0, 10], [0, 11], [4, 10], [4, 11], [4, 15], [8, 10], [8, 11], [8, 15], [0, 1, 10], [0, 1, 11], [0, 2, 10], [0, 2, 11], [0, 3, 10], [0, 3, 11], [0, 4, 10], [0, 4, 11], [0, 4, 15], [0, 5, 10], [0, 5, 11], [0, 6, 10], [0, 6, 11], [0, 7, 10], [0, 7, 11], [0, 8, 10], [0, 8, 11], [0, 8, 15], [0, 9, 10], [0, 9, 11], [0, 9, 15], [0, 10, 11], [0, 10, 12], [0, 10, 13], [0, 10, 14], [0, 10, 15], [0, 11, 12], [0, 11, 13], [0, 11, 14], [0, 11, 15], [1, 4, 10], [1, 4, 11], [1, 4, 15], [1, 8, 10], [1, 8, 11], [1, 8, 15], [2, 4, 10], [2, 4, 11], [2, 4, 15], [2, 8, 10], [2, 8, 11], [2, 8, 15], [3, 4, 10], [3, 4, 11], [3, 4, 15], [3, 8, 10], [3, 8, 11], [3, 8, 15], [4, 5, 10], [4, 5, 11], [4, 5, 15], [4, 6, 10], [4, 6, 11], [4, 6, 15], [4, 7, 10], [4, 7, 11], [4, 7, 15], [4, 8, 10], [4, 8, 11], [4, 8, 15], [4, 9, 10], [4, 9, 11], [4, 9, 15], [4, 10, 11], [4, 10, 12], [4, 10, 13], [4, 10, 14], [4, 10, 15], [4, 11, 12], [4, 11, 13], [4, 11, 14], [4, 11, 15], [4, 12, 15], [4, 13, 15], [4, 14, 15], [5, 8, 10], [5, 8, 11], [5, 8, 15], [6, 8, 10], [6, 8, 11], [6, 8, 15], [7, 8, 10], [7, 8, 11], [7, 8, 15], [8, 9, 10], [8, 9, 11], [8, 9, 15], [8, 10, 11], [8, 10, 12], [8, 10, 13], [8, 10, 14], [8, 10, 15], [8, 11, 12], [8, 11, 13], [8, 11, 14], [8, 11, 15], [8, 12, 15], [8, 13, 15], [8, 14, 15]],
            (0, 12): [[4], [8], [0, 4], [0, 8], [0, 12], [1, 4], [1, 8], [2, 4], [2, 8], [3, 4], [3, 8], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [5, 8], [6, 8], [7, 8], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [8, 15], [0, 1, 4], [0, 1, 8], [0, 1, 12], [0, 2, 4], [0, 2, 8], [0, 2, 12], [0, 3, 4], [0, 3, 8], [0, 3, 12], [0, 4, 5], [0, 4, 6], [0, 4, 7], [0, 4, 8], [0, 4, 9], [0, 4, 10], [0, 4, 11], [0, 4, 12], [0, 4, 13], [0, 4, 14], [0, 4, 15], [0, 5, 8], [0, 5, 12], [0, 6, 8], [0, 6, 12], [0, 7, 8], [0, 7, 12], [0, 8, 9], [0, 8, 10], [0, 8, 11], [0, 8, 12], [0, 8, 13], [0, 8, 14], [0, 8, 15], [0, 9, 12], [0, 10, 12], [0, 11, 12], [0, 12, 13], [0, 12, 14], [0, 12, 15], [1, 2, 4], [1, 2, 8], [1, 3, 4], [1, 3, 8], [1, 4, 5], [1, 4, 6], [1, 4, 7], [1, 4, 8], [1, 4, 9], [1, 4, 10], [1, 4, 11], [1, 4, 12], [1, 4, 13], [1, 4, 14], [1, 4, 15], [1, 5, 8], [1, 6, 8], [1, 7, 8], [1, 8, 9], [1, 8, 10], [1, 8, 11], [1, 8, 12], [1, 8, 13], [1, 8, 14], [1, 8, 15], [2, 3, 4], [2, 3, 8], [2, 4, 5], [2, 4, 6], [2, 4, 7], [2, 4, 8], [2, 4, 9], [2, 4, 10], [2, 4, 11], [2, 4, 12], [2, 4, 13], [2, 4, 14], [2, 4, 15], [2, 5, 8], [2, 6, 8], [2, 7, 8], [2, 8, 9], [2, 8, 10], [2, 8, 11], [2, 8, 12], [2, 8, 13], [2, 8, 14], [2, 8, 15], [3, 4, 5], [3, 4, 6], [3, 4, 7], [3, 4, 8], [3, 4, 9], [3, 4, 10], [3, 4, 11], [3, 4, 12], [3, 4, 13], [3, 4, 14], [3, 4, 15], [3, 5, 8], [3, 6, 8], [3, 7, 8], [3, 8, 9], [3, 8, 10], [3, 8, 11], [3, 8, 12], [3, 8, 13], [3, 8, 14], [3, 8, 15], [4, 5, 6], [4, 5, 7], [4, 5, 8], [4, 5, 9], [4, 5, 10], [4, 5, 11], [4, 5, 12], [4, 5, 13], [4, 5, 14], [4, 5, 15], [4, 6, 7], [4, 6, 8], [4, 6, 9], [4, 6, 10], [4, 6, 11], [4, 6, 12], [4, 6, 13], [4, 6, 14], [4, 6, 15], [4, 7, 8], [4, 7, 9], [4, 7, 10], [4, 7, 11], [4, 7, 12], [4, 7, 13], [4, 7, 14], [4, 7, 15], [4, 8, 9], [4, 8, 10], [4, 8, 11], [4, 8, 12], [4, 8, 13], [4, 8, 14], [4, 8, 15], [4, 9, 10], [4, 9, 11], [4, 9, 12], [4, 9, 13], [4, 9, 14], [4, 9, 15], [4, 10, 11], [4, 10, 12], [4, 10, 13], [4, 10, 14], [4, 10, 15], [4, 11, 12], [4, 11, 13], [4, 11, 14], [4, 11, 15], [4, 12, 13], [4, 12, 14], [4, 12, 15], [4, 13, 14], [4, 13, 15], [4, 14, 15], [5, 6, 8], [5, 7, 8], [5, 8, 9], [5, 8, 10], [5, 8, 11], [5, 8, 12], [5, 8, 13], [5, 8, 14], [5, 8, 15], [6, 7, 8], [6, 8, 9], [6, 8, 10], [6, 8, 11], [6, 8, 12], [6, 8, 13], [6, 8, 14], [6, 8, 15], [7, 8, 9], [7, 8, 10], [7, 8, 11], [7, 8, 12], [7, 8, 13], [7, 8, 14], [7, 8, 15], [8, 9, 10], [8, 9, 11], [8, 9, 12], [8, 9, 13], [8, 9, 14], [8, 9, 15], [8, 10, 11], [8, 10, 12], [8, 10, 13], [8, 10, 14], [8, 10, 15], [8, 11, 12], [8, 11, 13], [8, 11, 14], [8, 11, 15], [8, 12, 13], [8, 12, 14], [8, 12, 15], [8, 13, 14], [8, 13, 15], [8, 14, 15]],
            (15, 3): [[7], [11], [0, 7], [0, 11], [1, 7], [1, 11], [2, 7], [2, 11], [3, 7], [3, 11], [3, 15], [4, 7], [4, 11], [5, 7], [5, 11], [6, 7], [6, 11], [7, 8], [7, 9], [7, 10], [7, 11], [7, 12], [7, 13], [7, 14], [7, 15], [8, 11], [9, 11], [10, 11], [11, 12], [11, 13], [11, 14], [11, 15], [0, 1, 7], [0, 1, 11], [0, 2, 7], [0, 2, 11], [0, 3, 7], [0, 3, 11], [0, 3, 15], [0, 4, 7], [0, 4, 11], [0, 5, 7], [0, 5, 11], [0, 6, 7], [0, 6, 11], [0, 7, 8], [0, 7, 9], [0, 7, 10], [0, 7, 11], [0, 7, 12], [0, 7, 13], [0, 7, 14], [0, 7, 15], [0, 8, 11], [0, 9, 11], [0, 10, 11], [0, 11, 12], [0, 11, 13], [0, 11, 14], [0, 11, 15], [1, 2, 7], [1, 2, 11], [1, 3, 7], [1, 3, 11], [1, 3, 15], [1, 4, 7], [1, 4, 11], [1, 5, 7], [1, 5, 11], [1, 6, 7], [1, 6, 11], [1, 7, 8], [1, 7, 9], [1, 7, 10], [1, 7, 11], [1, 7, 12], [1, 7, 13], [1, 7, 14], [1, 7, 15], [1, 8, 11], [1, 9, 11], [1, 10, 11], [1, 11, 12], [1, 11, 13], [1, 11, 14], [1, 11, 15], [2, 3, 7], [2, 3, 11], [2, 3, 15], [2, 4, 7], [2, 4, 11], [2, 5, 7], [2, 5, 11], [2, 6, 7], [2, 6, 11], [2, 7, 8], [2, 7, 9], [2, 7, 10], [2, 7, 11], [2, 7, 12], [2, 7, 13], [2, 7, 14], [2, 7, 15], [2, 8, 11], [2, 9, 11], [2, 10, 11], [2, 11, 12], [2, 11, 13], [2, 11, 14], [2, 11, 15], [3, 4, 7], [3, 4, 11], [3, 4, 15], [3, 5, 7], [3, 5, 11], [3, 5, 15], [3, 6, 7], [3, 6, 11], [3, 6, 15], [3, 7, 8], [3, 7, 9], [3, 7, 10], [3, 7, 11], [3, 7, 12], [3, 7, 13], [3, 7, 14], [3, 7, 15], [3, 8, 11], [3, 8, 15], [3, 9, 11], [3, 9, 15], [3, 10, 11], [3, 10, 15], [3, 11, 12], [3, 11, 13], [3, 11, 14], [3, 11, 15], [3, 12, 15], [3, 13, 15], [3, 14, 15], [4, 5, 7], [4, 5, 11], [4, 6, 7], [4, 6, 11], [4, 7, 8], [4, 7, 9], [4, 7, 10], [4, 7, 11], [4, 7, 12], [4, 7, 13], [4, 7, 14], [4, 7, 15], [4, 8, 11], [4, 9, 11], [4, 10, 11], [4, 11, 12], [4, 11, 13], [4, 11, 14], [4, 11, 15], [5, 6, 7], [5, 6, 11], [5, 7, 8], [5, 7, 9], [5, 7, 10], [5, 7, 11], [5, 7, 12], [5, 7, 13], [5, 7, 14], [5, 7, 15], [5, 8, 11], [5, 9, 11], [5, 10, 11], [5, 11, 12], [5, 11, 13], [5, 11, 14], [5, 11, 15], [6, 7, 8], [6, 7, 9], [6, 7, 10], [6, 7, 11], [6, 7, 12], [6, 7, 13], [6, 7, 14], [6, 7, 15], [6, 8, 11], [6, 9, 11], [6, 10, 11], [6, 11, 12], [6, 11, 13], [6, 11, 14], [6, 11, 15], [7, 8, 9], [7, 8, 10], [7, 8, 11], [7, 8, 12], [7, 8, 13], [7, 8, 14], [7, 8, 15], [7, 9, 10], [7, 9, 11], [7, 9, 12], [7, 9, 13], [7, 9, 14], [7, 9, 15], [7, 10, 11], [7, 10, 12], [7, 10, 13], [7, 10, 14], [7, 10, 15], [7, 11, 12], [7, 11, 13], [7, 11, 14], [7, 11, 15], [7, 12, 13], [7, 12, 14], [7, 12, 15], [7, 13, 14], [7, 13, 15], [7, 14, 15], [8, 9, 11], [8, 10, 11], [8, 11, 12], [8, 11, 13], [8, 11, 14], [8, 11, 15], [9, 10, 11], [9, 11, 12], [9, 11, 13], [9, 11, 14], [9, 11, 15], [10, 11, 12], [10, 11, 13], [10, 11, 14], [10, 11, 15], [11, 12, 13], [11, 12, 14], [11, 12, 15], [11, 13, 14], [11, 13, 15], [11, 14, 15]]
        }
        
        assert model.path_refueling_combinations == expected_combinations

        result = model.solve(solver=pulp.PULP_CBC_CMD(msg=0))
        assert result["status"] == "Optimal"


class TestFRLMObjectives:
    @pytest.fixture
    def setup_network_with_distances(self):
        network, flows = load_grid_test_data()
        network = network.tolil()
        network[0, 15] = 100
        network[15, 0] = 100
        network = network.tocsr()
        test_flows = pd.DataFrame(
            {"origin": [0, 0], "destination": [15, 15], "volume": [100, 50]}
        )

        return network, test_flows

    def test_flow_objective(self, setup_network_with_distances):
        network, flows = setup_network_with_distances
        model = FRLM.from_flow_dataframe(
            network=network,
            flows=flows,
            p_facilities=2,
            vehicle_range=50,
            objective="flow",
        )
        result = model.solve(solver=pulp.PULP_CBC_CMD(msg=0))
        assert result["status"] == "Optimal"
        coverage = model.get_flow_coverage()
        assert coverage["covered_volume"] == 50 
        

    def test_vmt_objective(self, setup_network_with_distances):
        network, flows = setup_network_with_distances

        model = FRLM.from_flow_dataframe(
            network=network,
            flows=flows,
            p_facilities=2,
            vehicle_range=50,
            objective="vmt",
        )

        result = model.solve(solver=pulp.PULP_CBC_CMD(msg=0))
        assert result["status"] == "Optimal"
        vmt_coverage = model.get_vmt_coverage()
        assert "total_vmt" in vmt_coverage
        assert "covered_vmt" in vmt_coverage
        assert "vmt_coverage_percentage" in vmt_coverage
        assert vmt_coverage["covered_vmt"] > 0


class TestFRLMThresholdExtension:
    @pytest.fixture
    def setup_threshold_network(self):
        network, flows = load_grid_test_data()
        return network, flows

    def test_threshold_basic(self, setup_threshold_network):
        network, flows = setup_threshold_network
        model = FRLM.from_flow_dataframe(
            network=network,
            flows=flows,
            p_facilities=2,
            vehicle_range=30,
            threshold=0.8,
            weight=0.5,
        )
        result = model.solve(solver=pulp.PULP_CBC_CMD(msg=0))
        assert result["status"] == "Optimal"
        model.calculate_covered_nodes()
        for node in model.covered_nodes:
            origin_flows = [(o, d, v) for (o, d), v in model.flows.items() if o == node]
            if origin_flows:
                total_flow = sum(v for _, _, v in origin_flows)
                covered_flow = sum(
                    v * model.flow_coverage.get((o, d), {}).get("covered_proportion", 0)
                    for o, d, v in origin_flows
                )
                assert covered_flow >= 0.8 * total_flow
        assert hasattr(model, "covered_nodes")
        node_coverage_pct = model.get_node_coverage_percentage()
        assert 0 <= node_coverage_pct <= 1

    def test_threshold_weight_impact(self, setup_threshold_network):

        network, flows = setup_threshold_network
        model_high_weight = FRLM.from_flow_dataframe(
            network=network,
            flows=flows,
            p_facilities=2,
            vehicle_range=30,
            threshold=0.8,
            weight=0.99,
        )
        result_high = model_high_weight.solve(solver=pulp.PULP_CBC_CMD(msg=0))

        model_low_weight = FRLM.from_flow_dataframe(
            network=network,
            flows=flows,
            p_facilities=2,
            vehicle_range=30,
            threshold=0.8,
            weight=0.01,
        )
        result_low = model_low_weight.solve(solver=pulp.PULP_CBC_CMD(msg=0))

        assert result_high["status"] == "Optimal"
        assert result_low["status"] == "Optimal"

    def test_include_destination_parameter(self, setup_threshold_network):
        network, flows = setup_threshold_network
        model_no_dest = FRLM.from_flow_dataframe(
            network=network,
            flows=flows,
            p_facilities=2,
            vehicle_range=30,
            threshold=0.8,
            include_destination=False,
        )
        weights_no_dest = model_no_dest.calculate_node_weights(
            include_destination=False
        )

        model_with_dest = FRLM.from_flow_dataframe(
            network=network,
            flows=flows,
            p_facilities=2,
            vehicle_range=30,
            threshold=0.8,
            include_destination=True,
        )
        weights_with_dest = model_with_dest.calculate_node_weights(
            include_destination=True
        )

        assert not np.array_equal(weights_no_dest, weights_with_dest)

        assert_allclose(weights_no_dest.sum(), 1.0, rtol=1e-6)
        assert_allclose(weights_with_dest.sum(), 1.0, rtol=1e-6)


class TestFRLMGreedySolver:

    @pytest.fixture
    def setup_greedy_network(self):
        network, flows = load_grid_test_data()
        greedy_flows = flows.iloc[:3].copy()
        return network, greedy_flows

    def test_greedy_initialization_methods(self, setup_greedy_network):
        network, flows = setup_greedy_network
        initialization_methods = ["empty", "random", "central", "high_flow"]

        for method in initialization_methods:
            model = FRLM.from_flow_dataframe(
                network=network, flows=flows, p_facilities=2, vehicle_range=30
            )

            result = model.solve(
                solver="greedy",
                seed=42,
                initialization_method=method,
                max_iterations=100,
            )

            assert result["status"] == "Heuristic"
            assert len(model.selected_facilities) <= 2
            assert model.objective_value >= 0

    def test_greedy_iterations_tracking(self, setup_greedy_network):
        network, flows = setup_greedy_network
        model = FRLM.from_flow_dataframe(
            network=network, flows=flows, p_facilities=3, vehicle_range=30
        )

        result = model.solve(solver="greedy", seed=42, max_iterations=10)

        assert result["status"] == "Heuristic"
        assert hasattr(model, "greedy_iterations")
        assert len(model.greedy_iterations) > 0

        for iteration in model.greedy_iterations:
            assert "iteration" in iteration
            assert "selected_facility" in iteration
            assert "objective_value" in iteration
            assert "marginal_benefit" in iteration
            assert "candidates_evaluated" in iteration

    def test_greedy_capacitated(self, setup_greedy_network):
        network, flows = setup_greedy_network
        model = FRLM.from_flow_dataframe(
            network=network, flows=flows, p_facilities=4, vehicle_range=30, capacity=100
        )

        result = model.solve(solver="greedy", seed=42)

        assert result["status"] == "Heuristic"
        total_modules = sum(model.selected_facilities.values())
        assert total_modules <= 4
        for facility, modules in model.selected_facilities.items():
            assert modules >= 1

    def test_greedy_no_feasible_solution(self, setup_greedy_network):
        network, flows = setup_greedy_network

        model = FRLM.from_flow_dataframe(
            network=network, flows=flows, p_facilities=1, vehicle_range=5
        )

        result = model.solve(solver="greedy", seed=42)
        assert result["status"] == "Heuristic"
        assert model.objective_value == 0


class TestFRLMErrorHandling:
    def test_no_network_error(self):
        model = FRLM()
        flows = pd.DataFrame({"origin": [0], "destination": [1], "volume": [100]})

        with pytest.raises(ValueError, match="Network must be added"):
            model.add_flows(flows)

    def test_no_flows_error(self):
        network = csr_matrix(([], ([], [])), shape=(3, 3))
        model = FRLM()
        model.add_network(network)

        with pytest.raises(ValueError, match="No flows loaded"):
            model._build_model()

    def test_invalid_node_ids(self):
        network = csr_matrix(([], ([], [])), shape=(3, 3))
        model = FRLM()
        model.add_network(network)

        with pytest.raises(KeyError):
            model.add_flow(origin=0, destination=5, volume=100)

    def test_disconnected_path(self):
        network = csr_matrix(([], ([], [])), shape=(4, 4))
        model = FRLM()
        model.add_network(network)

        with pytest.raises((KeyError, ValueError)):
            model.add_flow(origin=0, destination=3, volume=100)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            model = FRLM(vehicle_range=-10)

        with pytest.raises(ValueError):
            model = FRLM(vehicle_range=0)

        model = FRLM()
        model.add_network(csr_matrix(([], ([], [])), shape=(3, 3)))
        model._model_built = True

        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            model.solve(threshold=1.5)

        with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
            model.solve(weight=-0.5)

    def test_solve_before_build(self):
        model = FRLM()
        with pytest.raises(ValueError, match="Model must be built first"):
            model.solve()


class TestFRLMOutputsAndReporting:
    @pytest.fixture
    def setup_solved_model(self):
        network, flows = load_grid_test_data()
        simple_flows = flows.iloc[:2].copy()

        model = FRLM.from_flow_dataframe(
            network=network, flows=simple_flows, p_facilities=2, vehicle_range=25
        )

        model.solve(solver=pulp.PULP_CBC_CMD(msg=0))
        return model

    def test_summary_output(self, setup_solved_model):

        model = setup_solved_model
        summary = model.summary()

        assert "model" in summary
        assert "solution" in summary
        assert "facilities" in summary
        assert "flow_coverage" in summary
        assert "solver_information" in summary

        assert summary["model"]["type"] == "Basic"
        assert summary["model"]["facilities_to_locate"] == 2

        assert summary["solution"]["status"] == "Optimal"
        assert summary["solution"]["objective_value"] > 0

        assert summary["facilities"]["total_modules"] == 2
        assert len(summary["facilities"]["locations"]) == 2

    def test_dataframe_export(self, setup_solved_model):

        model = setup_solved_model
        dfs = model.to_dataframes(include_iterations=True)

        assert "facilities" in dfs
        assert "coverage" in dfs
        assert "summary" in dfs

        assert len(dfs["facilities"]) == 2
        assert "facility_id" in dfs["facilities"].columns
        assert "modules" in dfs["facilities"].columns

        assert len(dfs["coverage"]) == len(model.flows)
        assert "origin" in dfs["coverage"].columns
        assert "destination" in dfs["coverage"].columns
        assert "covered_proportion" in dfs["coverage"].columns

    def test_solver_details(self, setup_solved_model):
        model = setup_solved_model
        details = model.get_solver_details(verbose=False)

        assert "solver_type" in details
        assert "solver_status" in details
        assert details["solver_type"] == "pulp"
        assert details["solver_status"] == "Optimal"

    def test_shadow_prices_and_reduced_costs(self, setup_solved_model):
        model = setup_solved_model

        shadow_prices = model.get_shadow_prices()
        reduced_costs = model.get_reduced_costs()

        assert isinstance(shadow_prices, dict)
        assert isinstance(reduced_costs, dict)

        for price in shadow_prices.values():
            assert isinstance(price, (int, float))

    def test_detailed_results(self, setup_solved_model):

        model = setup_solved_model
        model.extract_solver_statistics()
        results = model.get_detailed_results()

        assert "model_parameters" in results
        assert "solution" in results
        assert "solver_statistics" in results

        params = results["model_parameters"]
        assert params["vehicle_range"] == 25
        assert params["p_facilities"] == 2
        assert params["objective_type"] == "flow"


class TestFRLMCustomPaths:
    def test_custom_paths_basic(self):
        network, _ = load_grid_test_data()
        flows_data = []

        flows_data.append(
            {
                "origin": 0,
                "destination": 15,
                "volume": 100,
                "path": [0, 1, 2, 3, 7, 11, 15],
            }
        )

        flows_data.append(
            {
                "origin": 0,
                "destination": 15,
                "volume": 80,
                "path": [0, 4, 8, 12, 13, 14, 15],
            }
        )

        flows = pd.DataFrame(flows_data)

        model = FRLM.from_flow_dataframe(
            network=network, flows=flows, p_facilities=2, vehicle_range=40
        )

        assert model.flow_paths[(0, 15)] == [0, 4, 8, 12, 13, 14, 15]

        result = model.solve(solver=pulp.PULP_CBC_CMD(msg=0))
        assert result["status"] == "Optimal"

    def test_invalid_custom_paths(self):
        network = csr_matrix(([], ([], [])), shape=(4, 4))
        flows = pd.DataFrame(
            {"origin": [0], "destination": [3], "volume": [100], "path": [[1, 2, 3]]}
        )

        model = FRLM()
        model.add_network(network)

        with pytest.raises(ValueError, match="Path.*is inconsistent"):
            model._load_flows_from_dataframe(flows, path_col="path")
