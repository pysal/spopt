# ruff: noqa: N999
import numpy
import pulp
import pytest

from spopt.locate import LSCP


class TestSyntheticLocate:
    @pytest.fixture(autouse=True)
    def setup_method(self, network_instance) -> None:
        client_count, facility_count = 5, 2
        (
            self.clients_snapped,
            self.facilities_snapped,
            self.cost_matrix,
        ) = network_instance(client_count, facility_count)

    def test_clscpso_y1_lt_y2(self):
        service_radius = 8
        facility_capacity = numpy.array([5, 15])
        demand_quantity = numpy.arange(1, 6)
        clscpso = LSCP.from_cost_matrix(
            self.cost_matrix,
            service_radius,
            facility_capacity_arr=facility_capacity,
            demand_quantity_arr=demand_quantity,
        )
        result = clscpso.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, LSCP)

        known = [[1], [1], [1], [1], [1]]
        observed = clscpso.cli2fac
        assert known == observed

        known = [[], [0, 1, 2, 3, 4]]
        observed = clscpso.fac2cli
        assert known == observed

    def test_clscpso_y1_gt_y2(self):
        service_radius = 8
        facility_capacity = numpy.array([15, 5])
        demand_quantity = numpy.arange(1, 6)
        clscpso = LSCP.from_cost_matrix(
            self.cost_matrix,
            service_radius,
            facility_capacity_arr=facility_capacity,
            demand_quantity_arr=demand_quantity,
        )
        result = clscpso.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, LSCP)

        known = [[1], [1], [0, 1], [0, 1], [0, 1]]
        observed = clscpso.cli2fac
        assert known == observed

        known = [[2, 3, 4], [0, 1, 2, 3, 4]]
        observed = clscpso.fac2cli
        assert known == observed

    def test_clscpso_y1_eq_y2(self):
        service_radius = 7
        facility_capacity = numpy.array([8, 8])
        demand_quantity = numpy.arange(1, 6)
        clscpso = LSCP.from_cost_matrix(
            self.cost_matrix,
            service_radius,
            facility_capacity_arr=facility_capacity,
            demand_quantity_arr=demand_quantity,
        )
        result = clscpso.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, LSCP)

        known = [[1], [1], [0, 1], [0], [1]]
        observed = clscpso.cli2fac
        assert known == observed

        known = [[2, 3], [0, 1, 2, 4]]
        observed = clscpso.fac2cli
        assert known == observed

    def test_clscpso_dem_gt_cap_error(self):
        service_radius = 8
        facility_capacity = numpy.array([8, 8])
        demand_quantity = numpy.arange(5, 10)
        with pytest.raises(
            ValueError, match="Infeasible model. Demand greater than capacity"
        ):
            LSCP.from_cost_matrix(
                self.cost_matrix,
                service_radius,
                facility_capacity_arr=facility_capacity,
                demand_quantity_arr=demand_quantity,
            )

    def test_clscpso_infease_error(self, loc_raises_infeasible):
        service_radius = 1
        facility_capacity = numpy.array([5, 15])
        demand_quantity = numpy.arange(1, 6)
        clscpso = LSCP.from_cost_matrix(
            self.cost_matrix,
            service_radius,
            facility_capacity_arr=facility_capacity,
            demand_quantity_arr=demand_quantity,
        )
        with loc_raises_infeasible:
            clscpso.solve(pulp.PULP_CBC_CMD(msg=False))
