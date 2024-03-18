import os
import pickle
import platform
import warnings

import geopandas
import numpy
import pandas
import pulp
import pytest
from shapely.geometry import Point, Polygon

from spopt.locate import LSCP
from spopt.locate.base import FacilityModelBuilder

WINDOWS = platform.platform()[:7].lower() == "windows"


class TestSyntheticLocate:
    @pytest.fixture(autouse=True)
    def setup_method(self, network_instance) -> None:
        self.dirpath = os.path.join(os.path.dirname(__file__), "./data/")

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

    def test_lscp_facility_client_array_from_cost_matrix(self):
        with open(self.dirpath + "lscp_fac2cli.pkl", "rb") as f:
            lscp_objective = pickle.load(f)

        lscp = LSCP.from_cost_matrix(self.cost_matrix, 8)
        lscp = lscp.solve(pulp.PULP_CBC_CMD(msg=False))

        numpy.testing.assert_array_equal(
            numpy.array(lscp.fac2cli, dtype=object),
            numpy.array(lscp_objective, dtype=object),
        )

    def test_lscp_client_facility_array_from_cost_matrix(self):
        with open(self.dirpath + "lscp_cli2fac.pkl", "rb") as f:
            lscp_objective = pickle.load(f)

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

    def test_lscp_facility_client_array_from_geodataframe(self):
        with open(self.dirpath + "lscp_geodataframe_fac2cli.pkl", "rb") as f:
            lscp_objective = pickle.load(f)

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

    def test_lscp_client_facility_array_from_geodataframe(self):
        with open(self.dirpath + "lscp_geodataframe_cli2fac.pkl", "rb") as f:
            lscp_objective = pickle.load(f)

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

    def test_lscp_preselected_facility_client_array_from_geodataframe(self):
        with open(
            self.dirpath + "lscp_preselected_loc_geodataframe_fac2cli.pkl", "rb"
        ) as f:
            lscp_objective = pickle.load(f)

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
    def setup_method(self) -> None:
        self.dirpath = os.path.join(os.path.dirname(__file__), "./data/")
        network_distance = pandas.read_csv(
            self.dirpath
            + "SF_network_distance_candidateStore_16_censusTract_205_new.csv"
        )

        ntw_dist_piv = network_distance.pivot_table(
            values="distance", index="DestinationName", columns="name"
        )

        self.cost_matrix = ntw_dist_piv.to_numpy()

        demand_points = pandas.read_csv(
            self.dirpath + "SF_demand_205_centroid_uniform_weight.csv"
        )
        facility_points = pandas.read_csv(self.dirpath + "SF_store_site_16_longlat.csv")

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

    def test_infeasibility_lscp_from_cost_matrix(self):
        lscp = LSCP.from_cost_matrix(self.cost_matrix, 20)
        with pytest.raises(RuntimeError, match="Model is not solved: Infeasible."):
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

    def test_infeasibility_lscp_from_geodataframe(self):
        lscp = LSCP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            0,
        )
        with pytest.raises(RuntimeError, match="Model is not solved: Infeasible."):
            lscp.solve(pulp.PULP_CBC_CMD(msg=False))


class TestErrorsWarnings:
    def setup_method(self) -> None:
        pol1 = Polygon([(0, 0), (1, 0), (1, 1)])
        pol2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        pol3 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        polygon_dict = {"geometry": [pol1, pol2, pol3]}

        point = Point(10, 10)
        point_dict = {"weight": 4, "geometry": [point]}

        self.gdf_fac = geopandas.GeoDataFrame(polygon_dict, crs="EPSG:4326")
        self.gdf_dem = geopandas.GeoDataFrame(point_dict, crs="EPSG:4326")

        self.gdf_dem_crs = self.gdf_dem.to_crs("EPSG:3857")

        self.gdf_dem_buffered = self.gdf_dem.copy()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Geometry is in a geographic CRS",
            )
            self.gdf_dem_buffered["geometry"] = self.gdf_dem.buffer(2)

    def test_attribute_error_add_set_covering_constraint(self):
        with pytest.raises(AttributeError, match="Before setting coverage constraints"):
            dummy_class = LSCP("dummy", pulp.LpProblem("name"))
            dummy_class.aij = numpy.array([])
            dummy_range = range(1)
            FacilityModelBuilder.add_set_covering_constraint(
                dummy_class, dummy_range, dummy_range
            )

    def test_error_lscp_different_crs(self):
        with (
            pytest.warns(
                UserWarning, match="Facility geodataframe contains mixed type"
            ),
            pytest.raises(ValueError, match="Geodataframes crs are different: "),
        ):
            LSCP.from_geodataframe(
                self.gdf_dem_crs, self.gdf_fac, "geometry", "geometry", 10
            )

    def test_warning_lscp_demand_geodataframe(self):
        with pytest.warns(UserWarning, match="Demand geodataframe contains mixed type"):
            LSCP.from_geodataframe(
                self.gdf_dem_buffered, self.gdf_fac, "geometry", "geometry", 10
            )
