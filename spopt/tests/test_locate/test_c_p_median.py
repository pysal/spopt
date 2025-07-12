import numpy
import pulp
import pytest

from spopt.locate import PMedian
from spopt.locate.base import SpecificationError


class TestSyntheticLocate:
    @pytest.fixture(autouse=True)
    def setup_method(self, network_instance) -> None:
        client_count, facility_count = 2, 3
        (
            self.clients_snapped,
            self.facilities_snapped,
            self.cost_matrix,
        ) = network_instance(client_count, facility_count)

    def test_c_p_median_from_cost_matrix(self):
        facility_capacity = numpy.array([5, 7, 10])
        demand_quantity = numpy.array([4, 10])
        p_median = PMedian.from_cost_matrix(
            self.cost_matrix,
            demand_quantity,
            p_facilities=2,
            facility_capacities=facility_capacity,
        )
        result = p_median.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, PMedian)

        known = [[1], [2]]
        observed = result.cli2fac
        assert known == observed

        known = [[], [0], [1]]
        observed = result.fac2cli
        assert known == observed

    def test_c_p_median_with_predefined_facilities_from_cost_matrix(self):
        facility_capacity = numpy.array([5, 7, 10])
        demand_quantity = numpy.array([4, 10])
        predefine = numpy.array([2])
        p_median = PMedian.from_cost_matrix(
            self.cost_matrix,
            demand_quantity,
            p_facilities=2,
            facility_capacities=facility_capacity,
            predefined_facilities_arr=predefine,
            fulfill_predefined_fac=True,
        )
        result = p_median.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, PMedian)

        known = [[1], [2]]
        observed = result.cli2fac
        assert known == observed

        known = [[], [0], [1]]
        observed = result.fac2cli
        assert known == observed

    def test_c_p_median_with_predefined_facilities_infeasible(
        self, loc_raises_infeasible
    ):
        facility_capacity = numpy.array([5, 7, 10])
        demand_quantity = numpy.array([4, 10])
        predefine = numpy.array([0])
        p_median = PMedian.from_cost_matrix(
            self.cost_matrix,
            demand_quantity,
            p_facilities=2,
            facility_capacities=facility_capacity,
            predefined_facilities_arr=predefine,
            fulfill_predefined_fac=True,
        )
        with loc_raises_infeasible:
            p_median.solve(pulp.PULP_CBC_CMD(msg=False))


class TestRealWorldLocate:
    @pytest.fixture(autouse=True)
    def setup_method(self, load_locate_test_data) -> None:
        time_table = load_locate_test_data(
            "example_subject_student_school_journeys.csv"
        )

        self.cost_matrix = (
            time_table.pivot_table(
                columns="school",
                fill_value=10000,
                index="student",
                sort=False,
                values="time",
            )
            .astype(int)
            .values
        )

        self.demand_points = load_locate_test_data("example_subject_students.csv")
        self.facility_points = load_locate_test_data("example_subject_schools.csv")

        self.p_facility = 10
        self.demand = numpy.ones(len(self.demand_points))
        self.capacities_arr = numpy.array(self.facility_points["Count"])
        schools_priority_1 = self.facility_points[
            self.facility_points["priority"] == 1
        ].index.tolist()
        self.schools_priority_1_arr = numpy.array(schools_priority_1)

    def test_optimality_capacitated_pmedian_with_predefined_facilities(self):
        pmedian = PMedian.from_cost_matrix(
            self.cost_matrix,
            self.demand,
            self.p_facility,
            predefined_facilities_arr=self.schools_priority_1_arr,
            facility_capacities=self.capacities_arr,
            fulfill_predefined_fac=True,
        )
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD(msg=False))
        assert pmedian.problem.status == pulp.LpStatusOptimal

    def test_infeasibility_capacitated_pmedian(self, loc_raises_infeasible):
        pmedian = PMedian.from_cost_matrix(
            self.cost_matrix, self.demand, 0, facility_capacities=self.capacities_arr
        )
        with loc_raises_infeasible:
            pmedian.solve(pulp.PULP_CBC_CMD(msg=False))

    def test_mixin_mean_time(self):
        mean_time_expected = 87.2
        pmedian = PMedian.from_cost_matrix(
            self.cost_matrix,
            self.demand,
            self.p_facility,
            predefined_facilities_arr=self.schools_priority_1_arr,
            facility_capacities=self.capacities_arr,
            fulfill_predefined_fac=True,
        )
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD(msg=False))
        assert pmedian.mean_dist == mean_time_expected

    def test_infeasibility_predefined_facilities_fulfillment_error(self):
        schools_priority_3 = self.facility_points[
            self.facility_points["priority"] == 3
        ].index.tolist()
        schools_priority_3_arr = numpy.array(schools_priority_3)
        with pytest.raises(
            SpecificationError,
            match=(
                "Problem is infeasible. The predefined facilities can't be fulfilled, "
            ),
        ):
            PMedian.from_cost_matrix(
                self.cost_matrix,
                self.demand,
                self.p_facility,
                predefined_facilities_arr=schools_priority_3_arr,
                facility_capacities=self.capacities_arr,
                fulfill_predefined_fac=True,
            )

    def test_no_capacity_data_predefined_facilities_error(self):
        with pytest.raises(
            SpecificationError,
            match=(
                "Data on the capacity of the facility is missing, "
                "so the model cannot be calculated."
            ),
        ):
            PMedian.from_cost_matrix(
                self.cost_matrix,
                self.demand,
                self.p_facility,
                predefined_facilities_arr=self.schools_priority_1_arr,
                fulfill_predefined_fac=True,
            )

    def test_infeasibility_capacity_smaller_than_demand_error(self):
        demand_test = numpy.full(len(self.demand_points), 5)
        with pytest.raises(
            SpecificationError,
            match="Problem is infeasible. The highest possible capacity",
        ):
            PMedian.from_cost_matrix(
                self.cost_matrix,
                demand_test,
                self.p_facility,
                predefined_facilities_arr=self.schools_priority_1_arr,
                facility_capacities=self.capacities_arr,
                fulfill_predefined_fac=True,
            )
