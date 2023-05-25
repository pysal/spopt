from spopt.locate.base import FacilityModelBuilder, LocateSolver, T_FacModel,SpecificationError
import numpy
import pandas
import pulp

from spopt.locate import PMedian
import pytest
import os

class TestRealWorldLocate:
    def setup_method(self) -> None:
        self.dirpath = os.path.join(os.path.dirname(__file__), "./data/")

        time_table = pandas.read_csv(
            self.dirpath+ "example_subject_student_school_journeys.csv"
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

        self.demand_points = pandas.read_csv(self.dirpath + "example_subject_students.csv")
        self.facility_points = pandas.read_csv(self.dirpath + "example_subject_schools.csv")

        self.p_facility = 10
        self.demand = numpy.ones(len(self.demand_points))
        self.capacities_arr = numpy.array(self.facility_points['Count'])
        schools_priority_1 = self.facility_points[self.facility_points['priority'] == 1].index.tolist()
        self.schools_priority_1_arr = numpy.array(schools_priority_1)

    def test_optimality_capacitated_pmedian_with_predefined_facilities(self):
        pmedian = PMedian.from_cost_matrix(
            self.cost_matrix,
            self.demand,
            self.p_facility,
            predefined_facilities_arr = self.schools_priority_1_arr,
            facility_capacities = self.capacities_arr,
            fulfill_predefined_fac = True
        )
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD(msg=False))
        assert pmedian.problem.status == pulp.LpStatusOptimal

    def test_infeasibility_capacitated_pmedian(self):
        pmedian = PMedian.from_cost_matrix(
            self.cost_matrix, self.demand, 0, facility_capacities = self.capacities_arr)
        with pytest.raises(RuntimeError, match="Model is not solved: Infeasible."):
            pmedian.solve(pulp.PULP_CBC_CMD(msg=False))

    def test_mixin_mean_time(self):
        mean_time_expected = 87.2
        pmedian = PMedian.from_cost_matrix(
            self.cost_matrix,
            self.demand,
            self.p_facility,
            predefined_facilities_arr = self.schools_priority_1_arr,
            facility_capacities = self.capacities_arr,
            fulfill_predefined_fac = True
        )        
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD(msg=False))
        assert pmedian.mean_dist == mean_time_expected
    
    def test_infeasibility_predefined_facilities_fulfillment_error(self):
        schools_priority_3 = self.facility_points[self.facility_points['priority'] == 3].index.tolist()
        schools_priority_3_arr = numpy.array(schools_priority_3)
        with pytest.raises(
            SpecificationError, 
            match="Problem is infeasible. The predefined facilities can't be fulfilled, "):
            pmedian = PMedian.from_cost_matrix(
                self.cost_matrix,
                self.demand,
                self.p_facility,
                predefined_facilities_arr = schools_priority_3_arr,
                facility_capacities = self.capacities_arr,
                fulfill_predefined_fac = True
            )
    
    def test_no_capacity_data_predefined_facilities_error(self):
        with pytest.raises(
            SpecificationError, 
            match="Data on the capacity of the facility is missing, so the model cannot be calculated."):
            pmedian = PMedian.from_cost_matrix(
                self.cost_matrix,
                self.demand,
                self.p_facility,
                predefined_facilities_arr = self.schools_priority_1_arr,
                fulfill_predefined_fac = True
            )
    
    def test_infeasibility_capacity_smaller_than_demand_error(self):
        demand_test = numpy.full(len(self.demand_points), 5)
        with pytest.raises(
            SpecificationError, 
            match="Problem is infeasible. The highest possible capacity"):
            pmedian = PMedian.from_cost_matrix(
                self.cost_matrix,
                demand_test,
                self.p_facility,
                predefined_facilities_arr = self.schools_priority_1_arr,
                facility_capacities = self.capacities_arr,
                fulfill_predefined_fac = True
            )

