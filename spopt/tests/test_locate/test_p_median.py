import geopandas
import numpy
import pulp
import pytest

from spopt.locate import PMedian
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

        self.ai = numpy.random.randint(1, 12, client_count)

        self.clients_snapped["weights"] = self.ai

    def test_p_median_from_cost_matrix(self):
        p_median = PMedian.from_cost_matrix(self.cost_matrix, self.ai, p_facilities=4)
        result = p_median.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, PMedian)

    def test_p_median_from_cost_matrix_no_results(self):
        p_median = PMedian.from_cost_matrix(self.cost_matrix, self.ai, p_facilities=4)
        result = p_median.solve(pulp.PULP_CBC_CMD(msg=False), results=False)
        assert isinstance(result, PMedian)

        with pytest.raises(AttributeError):
            result.cli2fac  # noqa: B018
        with pytest.raises(AttributeError):
            result.fac2cli  # noqa: B018
        with pytest.raises(AttributeError):
            result.mean_dist  # noqa: B018

    def test_pmedian_facility_client_array_from_cost_matrix(
        self, load_locate_test_data
    ):
        pmedian_objective = load_locate_test_data("pmedian_fac2cli.pkl")

        pmedian = PMedian.from_cost_matrix(self.cost_matrix, self.ai, p_facilities=4)
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(pmedian.fac2cli, dtype=object),
            numpy.array(pmedian_objective, dtype=object),
        )

    def test_pmedian_client_facility_array_from_cost_matrix(
        self, load_locate_test_data
    ):
        pmedian_objective = load_locate_test_data("pmedian_cli2fac.pkl")

        pmedian = PMedian.from_cost_matrix(self.cost_matrix, self.ai, p_facilities=4)
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(pmedian.cli2fac, dtype=object),
            numpy.array(pmedian_objective, dtype=object),
        )

    def test_p_median_from_geodataframe(self):
        p_median = PMedian.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            "weights",
            p_facilities=4,
        )
        result = p_median.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, PMedian)

    def test_pmedian_facility_client_array_from_geodataframe(
        self, load_locate_test_data
    ):
        pmedian_objective = load_locate_test_data("pmedian_geodataframe_fac2cli.pkl")

        pmedian = PMedian.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            "weights",
            p_facilities=4,
        )
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(pmedian.fac2cli, dtype=object),
            numpy.array(pmedian_objective, dtype=object),
        )

    def test_pmedian_client_facility_array_from_geodataframe(
        self, load_locate_test_data
    ):
        pmedian_objective = load_locate_test_data("pmedian_geodataframe_cli2fac.pkl")

        pmedian = PMedian.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            "weights",
            p_facilities=4,
        )
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(pmedian.cli2fac, dtype=object),
            numpy.array(pmedian_objective, dtype=object),
        )

    def test_pmedian_preselected_facility_client_array_from_geodataframe(self):
        known_objval = 1872.5093264630818
        known_mean = 3.0299503664451164
        known_solution_set = ["y_1_", "y_3_", "y_4_"]

        fac_snapped = self.facilities_snapped.copy()
        fac_snapped["predefined_loc"] = numpy.array([0, 0, 0, 1, 1])

        pmedian = PMedian.from_geodataframe(
            self.clients_snapped,
            fac_snapped,
            "geometry",
            "geometry",
            "weights",
            predefined_facility_col="predefined_loc",
            p_facilities=3,
        )
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD(msg=False, warmStart=True))

        observed_objval = pmedian.problem.objective.value()
        assert known_objval == pytest.approx(observed_objval)

        observed_mean = pmedian.mean_dist
        assert known_mean == pytest.approx(observed_mean)

        observed_solution_set = [dv.name for dv in pmedian.fac_vars if dv.varValue == 1]
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

    def test_optimality_pmedian_from_cost_matrix(self):
        pmedian = PMedian.from_cost_matrix(self.cost_matrix, self.ai, self.p_facility)
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD(msg=False))
        assert pmedian.problem.status == pulp.LpStatusOptimal

    def test_infeasibility_pmedian_from_cost_matrix(self, loc_raises_infeasible):
        pmedian = PMedian.from_cost_matrix(self.cost_matrix, self.ai, 0)
        with loc_raises_infeasible:
            pmedian.solve(pulp.PULP_CBC_CMD(msg=False))

    def test_mixin_mean_distance(self):
        mean_distance_expected = 2982.1268579890657
        pmedian = PMedian.from_cost_matrix(self.cost_matrix, self.ai, self.p_facility)
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD(msg=False))

        assert pmedian.mean_dist == mean_distance_expected

    def test_optimality_pmedian_from_geodataframe(self):
        pmedian = PMedian.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            self.p_facility,
        )
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD(msg=False))
        assert pmedian.problem.status == pulp.LpStatusOptimal

    def test_infeasibility_pmedian_from_geodataframe(self, loc_raises_infeasible):
        pmedian = PMedian.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            0,
        )
        with loc_raises_infeasible:
            pmedian.solve(pulp.PULP_CBC_CMD(msg=False))


class TestErrorsWarnings:
    @pytest.fixture(autouse=True)
    def setup_method(self, toy_fac_data, toy_dem_data) -> None:
        self.gdf_fac = toy_fac_data

        gdf_dem, gdf_dem_crs, gdf_dem_buffered = toy_dem_data
        self.gdf_dem = gdf_dem
        self.gdf_dem_crs = gdf_dem_crs
        self.gdf_dem_buffered = gdf_dem_buffered

    def test_attribute_error_add_assignment_constraint(self):
        with pytest.raises(
            AttributeError, match="Before setting assignment constraints"
        ):
            dummy_class = PMedian("dummy", pulp.LpProblem("name"), numpy.array([]), 1)
            dummy_range = range(1)
            FacilityModelBuilder.add_assignment_constraint(
                dummy_class, dummy_range, dummy_range
            )

    def test_attribute_error_add_opening_constraint(self):
        with pytest.raises(AttributeError, match="Before setting opening constraints"):
            dummy_class = PMedian("dummy", pulp.LpProblem("name"), numpy.array([]), 1)
            dummy_range = range(1)
            FacilityModelBuilder.add_opening_constraint(
                dummy_class, dummy_range, dummy_range
            )

    def test_error_pmedian_different_crs(
        self, loc_warns_mixed_type_fac, loc_warns_geo_crs, loc_raises_diff_crs
    ):
        with loc_warns_mixed_type_fac, loc_warns_geo_crs, loc_raises_diff_crs:
            PMedian.from_geodataframe(
                self.gdf_dem_crs, self.gdf_fac, "geometry", "geometry", "weight", 2
            )

    def test_warning_pmedian_demand_geodataframe(
        self, loc_warns_mixed_type_dem, loc_warns_mixed_type_fac, loc_warns_geo_crs
    ):
        with loc_warns_mixed_type_dem, loc_warns_mixed_type_fac, loc_warns_geo_crs:
            PMedian.from_geodataframe(
                self.gdf_dem_buffered, self.gdf_fac, "geometry", "geometry", "weight", 2
            )
