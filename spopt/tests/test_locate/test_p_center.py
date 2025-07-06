import geopandas
import numpy
import pulp
import pytest

from spopt.locate import PCenter
from spopt.locate.base import FacilityModelBuilder


class TestSyntheticLocate:
    @pytest.fixture(autouse=True)
    def setup_method(self, network_instance) -> None:
        client_count, facility_count = 100, 5
        (
            self.clients_snapped,
            self.facilities_snapped,
            self.cost_matrix,
        ) = network_instance(client_count, facility_count)

    def test_p_center_from_cost_matrix(self):
        p_center = PCenter.from_cost_matrix(self.cost_matrix, p_facilities=4)
        result = p_center.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, PCenter)

    def test_p_center_from_cost_matrix_no_results(self):
        p_center = PCenter.from_cost_matrix(self.cost_matrix, p_facilities=4)
        result = p_center.solve(pulp.PULP_CBC_CMD(msg=False), results=False)
        assert isinstance(result, PCenter)

        with pytest.raises(AttributeError):
            result.cli2fac  # noqa: B018
        with pytest.raises(AttributeError):
            result.fac2cli  # noqa: B018

    def test_pcenter_facility_client_array_from_cost_matrix(
        self, load_locate_test_data
    ):
        pcenter_objective = load_locate_test_data("pcenter_fac2cli.pkl")

        pcenter = PCenter.from_cost_matrix(self.cost_matrix, p_facilities=4)
        pcenter = pcenter.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(pcenter.fac2cli, dtype=object),
            numpy.array(pcenter_objective, dtype=object),
        )

    def test_p_center_from_geodataframe(self):
        p_center = PCenter.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            p_facilities=4,
        )
        result = p_center.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, PCenter)

    def test_pcenter_facility_client_array_from_geodataframe(
        self, load_locate_test_data
    ):
        pcenter_objective = load_locate_test_data("pcenter_geodataframe_fac2cli.pkl")

        pcenter = PCenter.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            p_facilities=4,
        )
        pcenter = pcenter.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(pcenter.fac2cli, dtype=object),
            numpy.array(pcenter_objective, dtype=object),
        )

    def test_pcenter_client_facility_array_from_geodataframe(
        self, load_locate_test_data
    ):
        pcenter_objective = load_locate_test_data("pcenter_geodataframe_cli2fac.pkl")

        pcenter = PCenter.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            4,
        )
        pcenter = pcenter.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(pcenter.cli2fac, dtype=object),
            numpy.array(pcenter_objective, dtype=object),
        )

    def test_pcenter_preselected_facility_client_array_from_geodataframe(self):
        known_objval = 6.2520432
        known_solution_set = ["y_2_", "y_3_", "y_4_"]

        fac_snapped = self.facilities_snapped.copy()
        fac_snapped["predefined_loc"] = numpy.array([0, 0, 0, 1, 1])

        pcenter = PCenter.from_geodataframe(
            self.clients_snapped,
            fac_snapped,
            "geometry",
            "geometry",
            3,
            predefined_facility_col="predefined_loc",
        )
        pcenter = pcenter.solve(pulp.PULP_CBC_CMD(msg=False, warmStart=True))

        observed_objval = pcenter.problem.objective.value()
        assert known_objval == pytest.approx(observed_objval)

        observed_solution_set = [dv.name for dv in pcenter.fac_vars if dv.varValue == 1]
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

        demand_points = load_locate_test_data(
            "SF_demand_205_centroid_uniform_weight.csv"
        )
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

        self.demand_points_gdf = (
            geopandas.GeoDataFrame(
                demand_points,
                geometry=geopandas.points_from_xy(
                    demand_points.long, demand_points.lat
                ),
            )
            .sort_values(by=["NAME"])
            .reset_index()
        )

        self.service_dist = 5000.0
        self.p_facility = 4
        self.ai = self.demand_points_gdf["POP2000"].to_numpy()

    def test_optimality_pcenter_from_cost_matrix(self):
        pcenter = PCenter.from_cost_matrix(self.cost_matrix, self.p_facility)
        pcenter = pcenter.solve(pulp.PULP_CBC_CMD(msg=False))
        assert pcenter.problem.status == pulp.LpStatusOptimal

    def test_infeasibility_pcenter_from_cost_matrix(self, loc_raises_infeasible):
        pcenter = PCenter.from_cost_matrix(self.cost_matrix, 0)
        with loc_raises_infeasible:
            pcenter.solve(pulp.PULP_CBC_CMD(msg=False))

    def test_optimality_pcenter_from_geodataframe(self):
        pcenter = PCenter.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            self.p_facility,
        )
        pcenter = pcenter.solve(pulp.PULP_CBC_CMD(msg=False))
        assert pcenter.problem.status == pulp.LpStatusOptimal

    def test_infeasibility_pcenter_from_geodataframe(self, loc_raises_infeasible):
        pcenter = PCenter.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            0,
        )
        with loc_raises_infeasible:
            pcenter.solve(pulp.PULP_CBC_CMD(msg=False))


class TestErrorsWarnings:
    @pytest.fixture(autouse=True)
    def setup_method(self, toy_fac_data, toy_dem_data) -> None:
        self.gdf_fac = toy_fac_data

        gdf_dem, gdf_dem_crs, gdf_dem_buffered = toy_dem_data
        self.gdf_dem = gdf_dem
        self.gdf_dem_crs = gdf_dem_crs
        self.gdf_dem_buffered = gdf_dem_buffered

    def test_attribute_error_add_minimized_maximum_constraint(self):
        with pytest.raises(
            AttributeError, match="Before setting minimized maximum constraints"
        ):
            dummy_matrix = numpy.array([])
            dummy_class = PCenter("dummy", pulp.LpProblem("name"), dummy_matrix)
            dummy_range = range(1)
            FacilityModelBuilder.add_minimized_maximum_constraint(
                dummy_class, dummy_matrix, dummy_range, dummy_range
            )

    def test_error_pcenter_different_crs(
        self, loc_warns_mixed_type_fac, loc_warns_geo_crs, loc_raises_diff_crs
    ):
        with loc_warns_mixed_type_fac, loc_warns_geo_crs, loc_raises_diff_crs:
            PCenter.from_geodataframe(
                self.gdf_dem_crs, self.gdf_fac, "geometry", "geometry", 2
            )

    def test_warning_pcenter_demand_geodataframe(
        self, loc_warns_mixed_type_dem, loc_warns_mixed_type_fac, loc_warns_geo_crs
    ):
        with loc_warns_mixed_type_dem, loc_warns_mixed_type_fac, loc_warns_geo_crs:
            PCenter.from_geodataframe(
                self.gdf_dem_buffered, self.gdf_fac, "geometry", "geometry", 2
            )
