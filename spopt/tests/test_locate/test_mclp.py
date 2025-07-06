import geopandas
import numpy
import pulp
import pytest

from spopt.locate import MCLP
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

    def test_mclp_from_cost_matrix(self):
        mclp = MCLP.from_cost_matrix(
            self.cost_matrix, self.ai, service_radius=7, p_facilities=4
        )
        result = mclp.solve(pulp.PULP_CBC_CMD(msg=False), results=True)

        assert isinstance(result, MCLP)
        assert result.n_cli_uncov == 1
        assert result.perc_cov == 99.0

    def test_mclp_from_cost_matrix_no_results(self):
        mclp = MCLP.from_cost_matrix(
            self.cost_matrix, self.ai, service_radius=7, p_facilities=4
        )
        result = mclp.solve(pulp.PULP_CBC_CMD(msg=False), results=False)

        assert isinstance(result, MCLP)

        with pytest.raises(AttributeError):
            result.cli2fac  # noqa: B018
        with pytest.raises(AttributeError):
            result.fac2cli  # noqa: B018
        with pytest.raises(AttributeError):
            result.n_cli_uncov  # noqa: B018
        with pytest.raises(AttributeError):
            result.perc_cov  # noqa: B018

    def test_mclp_facility_client_array_from_cost_matrix(self, load_locate_test_data):
        mclp_objective = load_locate_test_data("mclp_fac2cli.pkl")

        mclp = MCLP.from_cost_matrix(
            self.cost_matrix,
            self.ai,
            service_radius=7,
            p_facilities=4,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(mclp.fac2cli, dtype=object),
            numpy.array(mclp_objective, dtype=object),
        )

    def test_mclp_client_facility_array_from_cost_matrix(self, load_locate_test_data):
        mclp_objective = load_locate_test_data("mclp_cli2fac.pkl")

        mclp = MCLP.from_cost_matrix(
            self.cost_matrix,
            self.ai,
            service_radius=7,
            p_facilities=4,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(mclp.cli2fac, dtype=object),
            numpy.array(mclp_objective, dtype=object),
        )

    def test_mclp_from_geodataframe(self):
        mclp = MCLP.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            "weights",
            service_radius=7,
            p_facilities=4,
        )
        result = mclp.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, MCLP)

    def test_mclp_facility_client_array_from_geodataframe(self, load_locate_test_data):
        mclp_objective = load_locate_test_data("mclp_geodataframe_fac2cli.pkl")

        mclp = MCLP.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            "weights",
            service_radius=7,
            p_facilities=4,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(mclp.fac2cli, dtype=object),
            numpy.array(mclp_objective, dtype=object),
        )

    def test_mclp_preselected_facility_client_array_from_geodataframe(
        self, load_locate_test_data
    ):
        mclp_objective = load_locate_test_data(
            "mclp_preselected_loc_geodataframe_fac2cli.pkl"
        )

        fac_snapped = self.facilities_snapped.copy()

        fac_snapped["predefined_loc"] = numpy.array([1, 1, 0, 1, 0])

        mclp = MCLP.from_geodataframe(
            self.clients_snapped,
            fac_snapped,
            "geometry",
            "geometry",
            "weights",
            predefined_facility_col="predefined_loc",
            service_radius=7,
            p_facilities=4,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=False, warmStart=True))

        numpy.testing.assert_array_equal(
            numpy.array(mclp.fac2cli, dtype=object),
            numpy.array(mclp_objective, dtype=object),
        )

    def test_mclp_client_facility_array_from_geodataframe(self, load_locate_test_data):
        mclp_objective = load_locate_test_data("mclp_geodataframe_cli2fac.pkl")

        mclp = MCLP.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            "weights",
            service_radius=7,
            p_facilities=4,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(mclp.cli2fac, dtype=object),
            numpy.array(mclp_objective, dtype=object),
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

    def test_optimality_mclp_from_cost_matrix(self):
        mclp = MCLP.from_cost_matrix(
            self.cost_matrix,
            self.ai,
            self.service_dist,
            self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=False))
        assert mclp.problem.status == pulp.LpStatusOptimal

    def test_infeasibility_mclp_from_cost_matrix(self, loc_raises_infeasible):
        mclp = MCLP.from_cost_matrix(
            self.cost_matrix,
            self.ai,
            self.service_dist,
            1000,
        )
        with loc_raises_infeasible:
            mclp.solve(pulp.PULP_CBC_CMD(msg=False))

    def test_mixin_mclp_get_uncovered_clients(self):
        uncovered_clients_expected = 21
        mclp = MCLP.from_cost_matrix(
            self.cost_matrix,
            self.ai,
            self.service_dist,
            self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=False))

        assert mclp.n_cli_uncov == uncovered_clients_expected

    def test_mixin_mclp_get_percentage(self):
        percentage_expected = 89.75609756097561
        mclp = MCLP.from_cost_matrix(
            self.cost_matrix,
            self.ai,
            self.service_dist,
            self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=False))

        assert mclp.perc_cov == pytest.approx(percentage_expected)

    def test_optimality_mclp_from_geodataframe(self):
        mclp = MCLP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            self.service_dist,
            self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=False))
        assert mclp.problem.status == pulp.LpStatusOptimal

    def test_infeasibility_mclp_from_geodataframe(self, loc_raises_infeasible):
        mclp = MCLP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            self.service_dist,
            1000,
        )
        with loc_raises_infeasible:
            mclp.solve(pulp.PULP_CBC_CMD(msg=False))

    def test_attribute_error_fac2cli_mclp_facility_client_array(self):
        mclp = MCLP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            self.service_dist,
            self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=False), results=False)

        with pytest.raises(
            AttributeError, match="'MCLP' object has no attribute 'fac2cli'"
        ):
            mclp.fac2cli  # noqa: B018

    def test_attribute_error_cli2fac_mclp_facility_client_array(self):
        mclp = MCLP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            self.service_dist,
            self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=False), results=False)

        with pytest.raises(
            AttributeError, match="'MCLP' object has no attribute 'cli2fac'"
        ):
            mclp.cli2fac  # noqa: B018

    def test_attribute_error_ncliuncov_mclp_facility_client_array(self):
        mclp = MCLP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            self.service_dist,
            self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=False), results=False)

        with pytest.raises(
            AttributeError, match="'MCLP' object has no attribute 'n_cli_uncov'"
        ):
            mclp.n_cli_uncov  # noqa: B018

    def test_attribute_error_percentage_mclp_facility_client_array(self):
        mclp = MCLP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            self.service_dist,
            self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=False), results=False)

        with pytest.raises(
            AttributeError, match="The attribute `n_cli_uncov` is not set."
        ):
            mclp.get_percentage()


class TestErrorsWarnings:
    @pytest.fixture(autouse=True)
    def setup_method(self, toy_fac_data, toy_dem_data) -> None:
        self.gdf_fac = toy_fac_data

        gdf_dem, gdf_dem_crs, gdf_dem_buffered = toy_dem_data
        self.gdf_dem = gdf_dem
        self.gdf_dem_crs = gdf_dem_crs
        self.gdf_dem_buffered = gdf_dem_buffered

    def test_attribute_error_add_facility_constraint(self, loc_raises_fac_constr):
        with loc_raises_fac_constr:
            dummy_class = MCLP("dummy", pulp.LpProblem("name"))
            dummy_p_facility = 1
            FacilityModelBuilder.add_facility_constraint(dummy_class, dummy_p_facility)

    def test_attribute_error_add_maximal_coverage_constraint(self):
        with pytest.raises(
            AttributeError, match="Before setting maximal coverage constraints"
        ):
            dummy_class = MCLP("dummy", pulp.LpProblem("name"))
            dummy_class.aij = numpy.array([])
            dummy_range = range(1)
            FacilityModelBuilder.add_maximal_coverage_constraint(
                dummy_class, dummy_range, dummy_range
            )

    def test_error_mclp_different_crs(
        self, loc_warns_mixed_type_fac, loc_raises_diff_crs, loc_warns_geo_crs
    ):
        with loc_warns_mixed_type_fac, loc_raises_diff_crs, loc_warns_geo_crs:
            MCLP.from_geodataframe(
                self.gdf_dem_crs,
                self.gdf_fac,
                "geometry",
                "geometry",
                "weight",
                10,
                2,
            )

    def test_warning_mclp_demand_geodataframe(
        self, loc_warns_mixed_type_dem, loc_warns_mixed_type_fac, loc_warns_geo_crs
    ):
        with loc_warns_mixed_type_dem, loc_warns_mixed_type_fac, loc_warns_geo_crs:
            MCLP.from_geodataframe(
                self.gdf_dem_buffered,
                self.gdf_fac,
                "geometry",
                "geometry",
                "weight",
                10,
                2,
            )
