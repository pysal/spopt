from spopt.locate.base import FacilityModelBuilder, LocateSolver, T_FacModel
import numpy
import geopandas
import pandas
import pulp
import spaghetti
from shapely.geometry import Point

from spopt.locate import LSCP, MCLP, PCenter, PMedian
from spopt.locate.util import simulated_geo_points
import unittest
import os
import pickle
import platform

if os == "windows":
    WINDOWS = True
else:
    WINDOWS = False


class TestGlobalLocate(unittest.TestCase):
    def setUp(self) -> None:
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
            ntw, pp_name="clients", snapped=True
        )

        self.cost_matrix = ntw.allneighbordistances(
            sourcepattern=ntw.pointpatterns["clients"],
            destpattern=ntw.pointpatterns["facilities"],
        )

        self.ai = numpy.random.randint(1, 12, client_count)

    def test_lscp_from_cost_matrix(self):
        lscp = LSCP.from_cost_matrix(self.cost_matrix, 10)
        result = lscp.solve(pulp.PULP_CBC_CMD())
        self.assertIsInstance(result, LSCP)

    def test_lscp_from_geodataframe(self):
        lscp = LSCP.from_geodataframe(
            self.clients_snapped, self.facilities_snapped, "geometry", "geometry", 10
        )
        result = lscp.solve(pulp.PULP_CBC_CMD())
        self.assertIsInstance(result, LSCP)

    def test_mclp_from_cost_matrix(self):
        mclp = MCLP.from_cost_matrix(
            self.cost_matrix, self.ai, max_coverage=7, p_facilities=4
        )
        result = mclp.solve(pulp.PULP_CBC_CMD())
        self.assertIsInstance(result, MCLP)

    def test_mclp_from_geodataframe(self):
        self.clients_snapped["weights"] = self.ai
        mclp = MCLP.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            "weights",
            max_coverage=7,
            p_facilities=4,
        )
        result = mclp.solve(pulp.PULP_CBC_CMD())
        self.assertIsInstance(result, MCLP)

    def test_p_median_from_cost_matrix(self):
        p_median = PMedian.from_cost_matrix(self.cost_matrix, self.ai, p_facilities=4)
        result = p_median.solve(pulp.PULP_CBC_CMD())
        self.assertIsInstance(result, PMedian)

    def test_p_median_from_geodataframe(self):
        self.clients_snapped["weights"] = self.ai
        p_median = PMedian.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            "weights",
            p_facilities=4,
        )
        result = p_median.solve(pulp.PULP_CBC_CMD())
        self.assertIsInstance(result, PMedian)

    def test_p_center_from_cost_matrix(self):
        p_center = PCenter.from_cost_matrix(self.cost_matrix, self.ai, p_facilities=4)
        result = p_center.solve(pulp.PULP_CBC_CMD())
        self.assertIsInstance(result, PCenter)

    def test_p_center_from_geodataframe(self):
        self.clients_snapped["weights"] = self.ai
        p_center = PCenter.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            "weights",
            p_facilities=4,
        )
        result = p_center.solve(pulp.PULP_CBC_CMD())
        self.assertIsInstance(result, PCenter)


class TestOptimalLocate(unittest.TestCase):
    def setUp(self) -> None:
        self.dirpath = os.path.join(os.path.dirname(__file__), "./data/")
        network_distance = pandas.read_csv(
            self.dirpath
            + "SF_network_distance_candidateStore_16_censusTract_205_new.csv"
        )

        ntw_dist_piv = network_distance.pivot_table(
            values="distance", index="DestinationName", columns="name"
        )
        demand_name = network_distance[["DestinationName", "demand"]].drop_duplicates()

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
        lscp = lscp.solve(pulp.PULP_CBC_CMD())

        self.assertEqual(lscp.problem.status, pulp.LpStatusOptimal)

    def test_infeasibility_lscp_from_cost_matrix(self):
        lscp = LSCP.from_cost_matrix(self.cost_matrix, 20)
        lscp = lscp.solve(pulp.PULP_CBC_CMD())

        self.assertEqual(lscp.problem.status, pulp.LpStatusInfeasible)

    def test_lscp_facility_client_array_from_cost_matrix(self):
        with open(self.dirpath + "lscp_objective.pkl", "rb") as f:
            lscp_objective = pickle.load(f)

        lscp = LSCP.from_cost_matrix(self.cost_matrix, self.service_dist)
        lscp = lscp.solve(pulp.PULP_CBC_CMD())
        lscp.facility_client_array()

        numpy.testing.assert_array_equal(lscp.fac2cli, lscp_objective)

    def test_optimality_lscp_from_geodataframe(self):
        lscp = LSCP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            self.service_dist,
        )
        lscp = lscp.solve(pulp.PULP_CBC_CMD())

        self.assertEqual(lscp.problem.status, pulp.LpStatusOptimal)

    def test_infeasibility_lscp_from_geodataframe(self):
        lscp = LSCP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            0,
        )
        lscp = lscp.solve(pulp.PULP_CBC_CMD())
        self.assertEqual(lscp.problem.status, pulp.LpStatusInfeasible)

    def test_lscp_facility_client_array_from_geodataframe(self):
        with open(self.dirpath + "lscp_geodataframe_objective.pkl", "rb") as f:
            lscp_objective = pickle.load(f)

        lscp = LSCP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            self.service_dist,
        )
        lscp = lscp.solve(pulp.PULP_CBC_CMD())
        lscp.facility_client_array()

        numpy.testing.assert_array_equal(lscp.fac2cli, lscp_objective)

    def test_optimality_mclp_from_cost_matrix(self):
        mclp = MCLP.from_cost_matrix(
            self.cost_matrix,
            self.ai,
            max_coverage=self.service_dist,
            p_facilities=self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD())
        self.assertEqual(mclp.problem.status, pulp.LpStatusOptimal)

    def test_infeasibility_mclp_from_cost_matrix(self):
        mclp = MCLP.from_cost_matrix(
            self.cost_matrix,
            self.ai,
            max_coverage=self.service_dist,
            p_facilities=1000,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD())
        self.assertEqual(mclp.problem.status, pulp.LpStatusInfeasible)

    def test_mclp_facility_client_array_from_cost_matrix(self):
        with open(self.dirpath + "mclp_objective.pkl", "rb") as f:
            mclp_objective = pickle.load(f)

        mclp = MCLP.from_cost_matrix(
            self.cost_matrix,
            self.ai,
            max_coverage=self.service_dist,
            p_facilities=self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD())
        mclp.facility_client_array()

        numpy.testing.assert_array_equal(mclp.fac2cli, mclp_objective)

    def test_mixin_mclp_get_uncovered_clients(self):
        uncovered_cleints_expected = 21
        mclp = MCLP.from_cost_matrix(
            self.cost_matrix,
            self.ai,
            max_coverage=self.service_dist,
            p_facilities=self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD())
        mclp.facility_client_array()
        mclp.uncovered_clients()

        self.assertEqual(mclp.n_cli_uncov, uncovered_cleints_expected)

    def test_mixin_mclp_get_percentage(self):
        percentage_expected = 0.8975609756097561
        mclp = MCLP.from_cost_matrix(
            self.cost_matrix,
            self.ai,
            max_coverage=self.service_dist,
            p_facilities=self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD())
        mclp.facility_client_array()
        mclp.uncovered_clients()
        mclp.get_percentage()

        self.assertEqual(mclp.percentage, percentage_expected)

    def test_optimality_mclp_from_geodataframe(self):
        mclp = MCLP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            max_coverage=self.service_dist,
            p_facilities=self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD())
        self.assertEqual(mclp.problem.status, pulp.LpStatusOptimal)

    def test_infeasibility_mclp_from_geodataframe(self):
        mclp = MCLP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            max_coverage=self.service_dist,
            p_facilities=1000,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD())
        self.assertEqual(mclp.problem.status, pulp.LpStatusInfeasible)

    @unittest.skipIf(WINDOWS, "Skipping Windows")
    def test_mclp_facility_client_array_from_geodataframe(self):
        with open(self.dirpath + "mclp_geodataframe_objective.pkl", "rb") as f:
            mclp_objective = pickle.load(f)

        mclp = MCLP.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            max_coverage=self.service_dist,
            p_facilities=self.p_facility,
        )
        mclp = mclp.solve(pulp.PULP_CBC_CMD())
        mclp.facility_client_array()

        numpy.testing.assert_array_equal(mclp.fac2cli, mclp_objective)

    def test_optimality_pcenter_from_cost_matrix(self):
        pcenter = PCenter.from_cost_matrix(
            self.cost_matrix, self.ai, p_facilities=self.p_facility
        )
        pcenter = pcenter.solve(pulp.PULP_CBC_CMD())
        self.assertEqual(pcenter.problem.status, pulp.LpStatusOptimal)

    def test_infeasibility_pcenter_from_cost_matrix(self):
        pcenter = PCenter.from_cost_matrix(self.cost_matrix, self.ai, p_facilities=0)
        pcenter = pcenter.solve(pulp.PULP_CBC_CMD())
        self.assertEqual(pcenter.problem.status, pulp.LpStatusInfeasible)

    def test_pcenter_facility_client_array_from_cost_matrix(self):
        with open(self.dirpath + "pcenter_objective.pkl", "rb") as f:
            pcenter_objective = pickle.load(f)

        pcenter = PCenter.from_cost_matrix(
            self.cost_matrix, self.ai, p_facilities=self.p_facility
        )
        pcenter = pcenter.solve(pulp.PULP_CBC_CMD())
        pcenter.facility_client_array()

        numpy.testing.assert_array_equal(pcenter.fac2cli, pcenter_objective)

    def test_optimality_pcenter_from_geodataframe(self):
        pcenter = PCenter.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            p_facilities=self.p_facility,
        )
        pcenter = pcenter.solve(pulp.PULP_CBC_CMD())
        self.assertEqual(pcenter.problem.status, pulp.LpStatusOptimal)

    def test_infeasibility_pcenter_from_geodataframe(self):
        pcenter = PCenter.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            p_facilities=0,
        )
        pcenter = pcenter.solve(pulp.PULP_CBC_CMD())
        self.assertEqual(pcenter.problem.status, pulp.LpStatusInfeasible)

    @unittest.skipIf(WINDOWS, "Skipping Windows")
    def test_pcenter_facility_client_array_from_geodataframe(self):
        with open(self.dirpath + "pcenter_geodataframe_objective.pkl", "rb") as f:
            pcenter_objective = pickle.load(f)

        pcenter = PCenter.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            p_facilities=self.p_facility,
        )
        pcenter = pcenter.solve(pulp.PULP_CBC_CMD())
        pcenter.facility_client_array()

        numpy.testing.assert_array_equal(pcenter.fac2cli, pcenter_objective)

    def test_optimality_pmedian_from_cost_matrix(self):
        pmedian = PMedian.from_cost_matrix(
            self.cost_matrix, self.ai, p_facilities=self.p_facility
        )
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD())
        self.assertEqual(pmedian.problem.status, pulp.LpStatusOptimal)

    def test_infeasibility_pmedian_from_cost_matrix(self):
        pmedian = PMedian.from_cost_matrix(self.cost_matrix, self.ai, p_facilities=0)
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD())
        self.assertEqual(pmedian.problem.status, pulp.LpStatusInfeasible)

    def test_pmedian_facility_client_array_from_cost_matrix(self):
        with open(self.dirpath + "pmedian_objective.pkl", "rb") as f:
            pmedian_objective = pickle.load(f)

        pmedian = PMedian.from_cost_matrix(
            self.cost_matrix, self.ai, p_facilities=self.p_facility
        )
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD())
        pmedian.facility_client_array()

        numpy.testing.assert_array_equal(pmedian.fac2cli, pmedian_objective)

    def test_mixin_mean_distance(self):
        mean_distance_expected = 2982.1268579890657
        pmedian = PMedian.from_cost_matrix(
            self.cost_matrix, self.ai, p_facilities=self.p_facility
        )
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD())
        pmedian.get_mean_distance(self.ai)

        self.assertEqual(pmedian.mean_dist, mean_distance_expected)

    def test_optimality_pmedian_from_geodataframe(self):
        pmedian = PMedian.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            p_facilities=self.p_facility,
        )
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD())
        self.assertEqual(pmedian.problem.status, pulp.LpStatusOptimal)

    def test_infeasibility_pmedian_from_geodataframe(self):
        pmedian = PMedian.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            p_facilities=0,
        )
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD())
        self.assertEqual(pmedian.problem.status, pulp.LpStatusInfeasible)

    def test_pmedian_facility_client_array_from_geodataframe(self):
        with open(self.dirpath + "pmedian_geodataframe_objective.pkl", "rb") as f:
            pmedian_objective = pickle.load(f)

        pmedian = PMedian.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            "POP2000",
            p_facilities=self.p_facility,
        )
        pmedian = pmedian.solve(pulp.PULP_CBC_CMD())
        pmedian.facility_client_array()

        numpy.testing.assert_array_equal(pmedian.fac2cli, pmedian_objective)


class TestErrors(unittest.TestCase):
    def test_attribute_error_add_set_covering_constraint(self):
        with self.assertRaises(AttributeError):
            dummy_class = LSCP("dummy", pulp.LpProblem("name"))
            dummy_matrix = numpy.array([])
            dummy_range = range(1)
            FacilityModelBuilder.add_set_covering_constraint(
                dummy_class, dummy_class.problem, dummy_matrix, dummy_range, dummy_range
            )

    def test_attribute_error_add_facility_constraint(self):
        with self.assertRaises(AttributeError):
            dummy_class = LSCP("dummy", pulp.LpProblem("name"))
            dummy_p_facility = 1
            FacilityModelBuilder.add_facility_constraint(
                dummy_class, dummy_class.problem, 1
            )

    def test_attribute_error_add_maximal_coverage_constraint(self):
        with self.assertRaises(AttributeError):
            dummy_class = LSCP("dummy", pulp.LpProblem("name"))
            dummy_matrix = numpy.array([])
            dummy_range = range(1)
            FacilityModelBuilder.add_maximal_coverage_constraint(
                dummy_class, dummy_class.problem, dummy_matrix, dummy_range, dummy_range
            )

    def test_attribute_error_add_assignment_constraint(self):
        with self.assertRaises(AttributeError):
            dummy_class = LSCP("dummy", pulp.LpProblem("name"))
            dummy_range = range(1)
            FacilityModelBuilder.add_assignment_constraint(
                dummy_class, dummy_class.problem, dummy_range, dummy_range
            )

    def test_attribute_error_add_opening_constraint(self):
        with self.assertRaises(AttributeError):
            dummy_class = LSCP("dummy", pulp.LpProblem("name"))
            dummy_range = range(1)
            FacilityModelBuilder.add_opening_constraint(
                dummy_class, dummy_class.problem, dummy_range, dummy_range
            )

    def test_attribute_error_add_minimized_maximum_constraint(self):
        with self.assertRaises(AttributeError):
            dummy_class = LSCP("dummy", pulp.LpProblem("name"))
            dummy_matrix = numpy.array([])
            dummy_range = range(1)
            FacilityModelBuilder.add_minimized_maximum_constraint(
                dummy_class, dummy_class.problem, dummy_matrix, dummy_range, dummy_range
            )
