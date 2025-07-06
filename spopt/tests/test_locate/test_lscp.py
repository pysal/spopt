import geopandas
import numpy
import pulp
import pytest

from spopt.locate import LSCP
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

    def test_lscp_from_cost_matrix(self):
        lscp = LSCP.from_cost_matrix(self.cost_matrix, 10)
        result = lscp.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, LSCP)

    def test_lscp_from_cost_matrix_no_results(self):
        lscp = LSCP.from_cost_matrix(self.cost_matrix, 10)
        result = lscp.solve(pulp.PULP_CBC_CMD(msg=False), results=False)
        assert isinstance(result, LSCP)

        with pytest.raises(AttributeError):
            result.cli2fac  # noqa: B018
        with pytest.raises(AttributeError):
            result.fac2cli  # noqa: B018

    def test_lscp_facility_client_array_from_cost_matrix(self, load_locate_test_data):
        lscp_objective = load_locate_test_data("lscp_fac2cli.pkl")

        lscp = LSCP.from_cost_matrix(self.cost_matrix, 8)
        lscp = lscp.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(lscp.fac2cli, dtype=object),
            numpy.array(lscp_objective, dtype=object),
        )

    def test_lscp_client_facility_array_from_cost_matrix(self, load_locate_test_data):
        lscp_objective = load_locate_test_data("lscp_cli2fac.pkl")

        lscp = LSCP.from_cost_matrix(self.cost_matrix, 8)
        lscp = lscp.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(lscp.cli2fac, dtype=object),
            numpy.array(lscp_objective, dtype=object),
        )

    def test_lscp_from_geodataframe(self):
        lscp = LSCP.from_geodataframe(
            self.clients_snapped, self.facilities_snapped, "geometry", "geometry", 10
        )
        result = lscp.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, LSCP)

    def test_lscp_facility_client_array_from_geodataframe(self, load_locate_test_data):
        lscp_objective = load_locate_test_data("lscp_geodataframe_fac2cli.pkl")

        lscp = LSCP.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            8,
        )
        lscp = lscp.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(lscp.fac2cli, dtype=object),
            numpy.array(lscp_objective, dtype=object),
        )

    def test_lscp_client_facility_array_from_geodataframe(self, load_locate_test_data):
        lscp_objective = load_locate_test_data("lscp_geodataframe_cli2fac.pkl")

        lscp = LSCP.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            8,
        )
        lscp = lscp.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(lscp.cli2fac, dtype=object),
            numpy.array(lscp_objective, dtype=object),
        )

    def test_lscp_preselected_facility_client_array_from_geodataframe(
        self, load_locate_test_data
    ):
        lscp_objective = load_locate_test_data(
            "lscp_preselected_loc_geodataframe_fac2cli.pkl"
        )

        fac_snapped = self.facilities_snapped.copy()

        fac_snapped["predefined_loc"] = numpy.array([0, 0, 0, 0, 1])

        lscp = LSCP.from_geodataframe(
            self.clients_snapped,
            fac_snapped,
            "geometry",
            "geometry",
            predefined_facility_col="predefined_loc",
            service_radius=8,
        )
        lscp = lscp.solve(pulp.PULP_CBC_CMD(msg=False, warmStart=True))

        numpy.testing.assert_array_equal(
            numpy.array(lscp.fac2cli, dtype=object),
            numpy.array(lscp_objective, dtype=object),
        )


class TestRealWorldLSCP:
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

    def test_optimality_lscp_from_cost_matrix(self):
        lscp = LSCP.from_cost_matrix(self.cost_matrix, self.service_dist)
        lscp = lscp.solve(pulp.PULP_CBC_CMD(msg=False))

        assert lscp.problem.status == pulp.LpStatusOptimal

    def test_infeasibility_lscp_from_cost_matrix(self, loc_raises_infeasible):
        lscp = LSCP.from_cost_matrix(self.cost_matrix, 20)
        with loc_raises_infeasible:
            lscp.solve(pulp.PULP_CBC_CMD(msg=False))

    def test_optimality_lscp_from_geodataframe(self):
        lscp = LSCP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            self.service_dist,
        )
        lscp = lscp.solve(pulp.PULP_CBC_CMD(msg=False))
        assert lscp.problem.status == pulp.LpStatusOptimal

    def test_infeasibility_lscp_from_geodataframe(self, loc_raises_infeasible):
        lscp = LSCP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            0,
        )
        with loc_raises_infeasible:
            lscp.solve(pulp.PULP_CBC_CMD(msg=False))


class TestErrorsWarnings:
    @pytest.fixture(autouse=True)
    def setup_method(self, toy_fac_data, toy_dem_data) -> None:
        self.gdf_fac = toy_fac_data

        gdf_dem, gdf_dem_crs, gdf_dem_buffered = toy_dem_data
        self.gdf_dem = gdf_dem
        self.gdf_dem_crs = gdf_dem_crs
        self.gdf_dem_buffered = gdf_dem_buffered

    def test_attribute_error_add_set_covering_constraint(self):
        with pytest.raises(AttributeError, match="Before setting coverage constraints"):
            dummy_class = LSCP("dummy", pulp.LpProblem("name"))
            dummy_class.aij = numpy.array([])
            dummy_range = range(1)
            FacilityModelBuilder.add_set_covering_constraint(
                dummy_class, dummy_range, dummy_range
            )

    def test_error_lscp_different_crs(
        self, loc_warns_mixed_type_fac, loc_raises_diff_crs, loc_warns_geo_crs
    ):
        with loc_warns_mixed_type_fac, loc_raises_diff_crs, loc_warns_geo_crs:
            LSCP.from_geodataframe(
                self.gdf_dem_crs, self.gdf_fac, "geometry", "geometry", 10
            )

    def test_warning_lscp_demand_geodataframe(
        self, loc_warns_mixed_type_dem, loc_warns_mixed_type_fac, loc_warns_geo_crs
    ):
        with loc_warns_mixed_type_dem, loc_warns_mixed_type_fac, loc_warns_geo_crs:
            LSCP.from_geodataframe(
                self.gdf_dem_buffered, self.gdf_fac, "geometry", "geometry", 10
            )
