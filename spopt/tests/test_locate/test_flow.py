import os
import pickle

import numpy as np
import pandas as pd
import pulp
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from spopt.locate.flow import FRLM


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

        assert model.k is not None
        assert model.a is not None

        result = model.solve(solver=pulp.PULP_CBC_CMD(msg=0))
        assert result["status"] == "Optimal"

    def test_combination_approach(self, setup_grid_network):
        network, flows = setup_grid_network
        model = FRLM.from_flow_dataframe(
            network=network, flows=flows, p_facilities=3, vehicle_range=50, capacity=500
        )

        assert model.use_ac_pc is False

        assert model.path_refueling_combinations is not None

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
        for _facility, modules in model.selected_facilities.items():
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
        details = model.get_solver_details()

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
            assert isinstance(price, int | float)

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
