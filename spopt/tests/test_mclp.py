from pyproj import crs
from spopt.locate.base import FacilityModelBuilder, LocateSolver, T_FacModel
import numpy
import geopandas
import pandas
import pulp
import spaghetti
from shapely.geometry import Point, Polygon

from spopt.locate import MCLP
from spopt.locate.util import simulated_geo_points
import os
import pickle
import platform
import pytest
import warnings

operating_system = platform.platform()[:7].lower()
if operating_system == "windows":
    WINDOWS = True
else:
    WINDOWS = False


class TestSyntheticLocate:
    def setup_method(self) -> None:
        self.dirpath = os.path.join(os.path.dirname(__file__), "./data/")

        lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
        ntw = spaghetti.Network(in_data=lattice)
        gdf = spaghetti.element_as_gdf(ntw, arcs=True)
        street = geopandas.GeoDataFrame(
            geopandas.GeoSeries(gdf["geometry"].buffer(0.2).unary_union),
            crs=gdf.crs,
            columns=["geometry"],
        )

        client_count = 100
        facility_count = 5

        self.client_points = simulated_geo_points(street, needed=client_count, seed=5)
        self.facility_points = simulated_geo_points(
            street, needed=facility_count, seed=6
        )

        ntw = spaghetti.Network(in_data=lattice)

        ntw.snapobservations(self.client_points, "clients", attribute=True)
        ntw.snapobservations(self.facility_points, "facilities", attribute=True)

        self.clients_snapped = spaghetti.element_as_gdf(
            ntw, pp_name="clients", snapped=True
        )

        self.facilities_snapped = spaghetti.element_as_gdf(
            ntw, pp_name="facilities", snapped=True
        )

        self.cost_matrix = ntw.allneighbordistances(
            sourcepattern=ntw.pointpatterns["clients"],
            destpattern=ntw.pointpatterns["facilities"],
        )

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
            result.cli2fac
        with pytest.raises(AttributeError):
            result.fac2cli
        with pytest.raises(AttributeError):
            result.n_cli_uncov
        with pytest.raises(AttributeError):
            result.perc_cov

    def test_mclp_facility_client_array_from_cost_matrix(self):
        with open(self.dirpath + "mclp_fac2cli.pkl", "rb") as f:
            mclp_objective = pickle.load(f)

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

    def test_mclp_client_facility_array_from_cost_matrix(self):
        with open(self.dirpath + "mclp_cli2fac.pkl", "rb") as f:
            mclp_objective = pickle.load(f)

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

    def test_mclp_facility_client_array_from_geodataframe(self):
        with open(self.dirpath + "mclp_geodataframe_fac2cli.pkl", "rb") as f:
            mclp_objective = pickle.load(f)

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

    def test_mclp_preselected_facility_client_array_from_geodataframe(self):
        with open(
            self.dirpath + "mclp_preselected_loc_geodataframe_fac2cli.pkl", "rb"
        ) as f:
            mclp_objective = pickle.load(f)

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

    def test_mclp_client_facility_array_from_geodataframe(self):
        with open(self.dirpath + "mclp_geodataframe_cli2fac.pkl", "rb") as f:
            mclp_objective = pickle.load(f)

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

    def test_optimality_mclp_from_cost_matrix(self):
        mclp = MCLP.from_cost_matrix(
            self.cost_matrix,
            self.ai,
            self.service_dist,
            self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD(msg=False))
        assert mclp.problem.status == pulp.LpStatusOptimal

    def test_infeasibility_mclp_from_cost_matrix(self):
        mclp = MCLP.from_cost_matrix(
            self.cost_matrix,
            self.ai,
            self.service_dist,
            1000,
        )
        with pytest.raises(RuntimeError, match="Model is not solved: Infeasible."):
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

    def test_infeasibility_mclp_from_geodataframe(self):
        mclp = MCLP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            self.service_dist,
            1000,
        )
        with pytest.raises(RuntimeError, match="Model is not solved: Infeasible."):
            mclp.solve(pulp.PULP_CBC_CMD(msg=False))

    def test_attribute_error_fac2cli_MCLP_facility_client_array(self):
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
            mclp.fac2cli

    def test_attribute_error_cli2fac_MCLP_facility_client_array(self):
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
            mclp.cli2fac

    def test_attribute_error_ncliuncov_MCLP_facility_client_array(self):
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
            mclp.n_cli_uncov

    def test_attribute_error_percentage_MCLP_facility_client_array(self):
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

    def test_attribute_error_add_facility_constraint(self):
        with pytest.raises(AttributeError, match="Before setting facility constraint"):
            dummy_class = MCLP("dummy", pulp.LpProblem("name"))
            dummy_p_facility = 1
            FacilityModelBuilder.add_facility_constraint(dummy_class, 1)

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

    def test_error_mclp_different_crs(self):
        with pytest.warns(
            UserWarning, match="Facility geodataframe contains mixed type"
        ):
            with pytest.raises(ValueError, match="Geodataframes crs are different: "):
                MCLP.from_geodataframe(
                    self.gdf_dem_crs,
                    self.gdf_fac,
                    "geometry",
                    "geometry",
                    "weight",
                    10,
                    2,
                )

    def test_warning_mclp_demand_geodataframe(self):
        with pytest.warns(UserWarning, match="Demand geodataframe contains mixed type"):
            MCLP.from_geodataframe(
                self.gdf_dem_buffered,
                self.gdf_fac,
                "geometry",
                "geometry",
                "weight",
                10,
                2,
            )
