import geopandas
import numpy
import pulp
import pytest

from spopt.locate import PDispersion
from spopt.locate.base import FacilityModelBuilder


class TestSyntheticLocate:
    @pytest.fixture(autouse=True)
    def setup_method(self, network_instance) -> None:
        client_count, facility_count = None, 5
        _, self.facilities_snapped, self.cost_matrix = network_instance(
            client_count, facility_count
        )

    def test_p_dispersion_from_cost_matrix(self):
        pdispersion = PDispersion.from_cost_matrix(self.cost_matrix, 2)
        result = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, PDispersion)

    def test_p_dispersion_from_cost_matrix_no_results(self):
        pdispersion = PDispersion.from_cost_matrix(self.cost_matrix, 2)
        result = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, PDispersion)

        with pytest.raises(AttributeError):
            result.cli2fac  # noqa: B018
        with pytest.raises(AttributeError):
            result.fac2clif  # noqa: B018

    def test_p_dispersion_from_geodataframe(self):
        pdispersion = PDispersion.from_geodataframe(
            self.facilities_snapped,
            "geometry",
            2,
        )
        result = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, PDispersion)

    def test_p_dispersion_preselected_facility_array_from_geodataframe(self):
        known_objval = 4.213464
        known_solution_set = ["y_1_", "y_3_", "y_4_"]

        fac_snapped = self.facilities_snapped.copy()
        fac_snapped["predefined_loc"] = numpy.array([0, 0, 0, 1, 1])

        pdispersion = PDispersion.from_geodataframe(
            fac_snapped,
            "geometry",
            3,
            predefined_facility_col="predefined_loc",
        )
        result = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, PDispersion)

        observed_objval = pdispersion.problem.objective.value()
        assert known_objval == pytest.approx(observed_objval)

        observed_solution_set = [
            dv.name for dv in pdispersion.fac_vars if dv.varValue == 1
        ]
        numpy.testing.assert_array_equal(
            numpy.array(known_solution_set, dtype=object),
            numpy.array(observed_solution_set, dtype=object),
        )


class TestRealWorldLocate:
    @pytest.fixture(autouse=True)
    def setup_method(self, load_locate_test_data) -> None:
        network_distance = load_locate_test_data(
            "SF_network_distance_candidateStore_16_censusTract_205_new.csv"
        )

        ntw_dist_piv = network_distance.pivot_table(
            values="distance", index="DestinationName", columns="name"
        )

        self.cost_matrix = ntw_dist_piv.to_numpy()

        facility_points = load_locate_test_data("SF_store_site_16_longlat.csv")

        self.facility_points_gdf = (
            geopandas.GeoDataFrame(
                facility_points,
                geometry=geopandas.points_from_xy(
                    facility_points.long, facility_points.lat
                ),
            )
            .sort_values(by=["NAME"])
            .reset_index()
        )

        self.service_dist = 5000.0
        self.p_facility = 4

    def test_optimality_p_dispersion_from_cost_matrix(self):
        pdispersion = PDispersion.from_cost_matrix(self.cost_matrix, self.p_facility)
        pdispersion = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        assert pdispersion.problem.status == pulp.LpStatusOptimal

        known_solution_set = ["y_0_", "y_1_", "y_14_", "y_15_"]
        observed_solution_set = [
            dv.name for dv in pdispersion.fac_vars if dv.varValue == 1
        ]
        assert known_solution_set == observed_solution_set

    def test_infeasibility_p_dispersion_from_cost_matrix(self, loc_raises_infeasible):
        pdispersion = PDispersion.from_cost_matrix(self.cost_matrix, 17)
        with loc_raises_infeasible:
            pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))

    def test_optimality_p_dispersion_from_geodataframe(self):
        pdispersion = PDispersion.from_geodataframe(
            self.facility_points_gdf,
            "geometry",
            self.p_facility,
        )
        pdispersion = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        assert pdispersion.problem.status == pulp.LpStatusOptimal

        known_solution_set = ["y_0_", "y_2_", "y_8_", "y_14_"]
        observed_solution_set = [
            dv.name for dv in pdispersion.fac_vars if dv.varValue == 1
        ]
        assert known_solution_set == observed_solution_set

    def test_infeasibility_p_dispersion_from_geodataframe(self, loc_raises_infeasible):
        pdispersion = PDispersion.from_geodataframe(
            self.facility_points_gdf,
            "geometry",
            17,
        )
        with loc_raises_infeasible:
            pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))


class TestErrorsWarnings:
    @pytest.fixture(autouse=True)
    def setup_method(self, toy_fac_data) -> None:
        self.gdf_fac = toy_fac_data

    def test_attribute_error_add_facility_constraint(self, loc_raises_fac_constr):
        with loc_raises_fac_constr:
            dummy_p_facility = 1
            dummy_class = PDispersion("dummy", pulp.LpProblem("name"), dummy_p_facility)
            FacilityModelBuilder.add_facility_constraint(dummy_class, dummy_p_facility)

    def test_attribute_error_add_p_dispersion_interfacility_constraint(self):
        with pytest.raises(
            AttributeError, match="Before setting interfacility distance constraints"
        ):
            dummy_p_facility = 1
            dummy_matrix = numpy.array([])
            dummy_range = range(1)
            dummy_class = PDispersion("dummy", pulp.LpProblem("name"), dummy_p_facility)
            FacilityModelBuilder.add_p_dispersion_interfacility_constraint(
                dummy_class,
                dummy_matrix,
                dummy_range,
            )

    def test_attribute_error_add_predefined_facility_constraint(
        self, loc_raises_fac_constr
    ):
        with loc_raises_fac_constr:
            dummy_p_facility = 1
            dummy_matrix = numpy.array([])
            dummy_class = PDispersion("dummy", pulp.LpProblem("name"), dummy_p_facility)
            FacilityModelBuilder.add_facility_constraint(dummy_class, dummy_matrix)

    def test_warning_facility_geodataframe(
        self, loc_warns_mixed_type_fac, loc_warns_geo_crs
    ):
        with loc_warns_mixed_type_fac, loc_warns_geo_crs:
            PDispersion.from_geodataframe(self.gdf_fac, "geometry", 1)
