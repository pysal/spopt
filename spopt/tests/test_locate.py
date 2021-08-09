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
        dirpath = os.path.join(os.path.dirname(__file__), "./data/")
        network_distance = pandas.read_csv(
            dirpath + "SF_network_distance_candidateStore_16_censusTract_205_new.csv"
        )

        ntw_dist_piv = network_distance.pivot_table(
            values="distance", index="DestinationName", columns="name"
        )
        facilities_name = ntw_dist_piv.columns
        demand_name = network_distance[["DestinationName", "demand"]].drop_duplicates()

        self.cost_matrix = ntw_dist_piv.to_numpy()

        demand_points = pandas.read_csv(
            dirpath + "SF_demand_205_centroid_uniform_weight.csv"
        )
        facility_points = pandas.read_csv(dirpath + "SF_store_site_16_longlat.csv")

        self.facility_points_gdf = geopandas.GeoDataFrame(
            facility_points,
            geometry=geopandas.points_from_xy(
                facility_points.long, facility_points.lat
            ),
        )
        self.demand_points_gdf = geopandas.GeoDataFrame(
            demand_points,
            geometry=geopandas.points_from_xy(demand_points.long, demand_points.lat),
        )

        self.service_dist = 5000.0
        self.p_facility = 4
        self.ai = demand_name.sort_values(by=["DestinationName"])["demand"].to_numpy()

    def test_optimality_lscp_from_cost_matrix(self):
        lscp = LSCP.from_cost_matrix(self.cost_matrix, self.service_dist)
        lscp = lscp.solve(pulp.PULP_CBC_CMD())

        self.assertEqual(lscp.problem.status, pulp.LpStatusOptimal)

    def test_infeasibility_lscp_from_cost_matrix(self):
        lscp = LSCP.from_cost_matrix(self.cost_matrix, 20)
        lscp = lscp.solve(pulp.PULP_CBC_CMD())

        self.assertEqual(lscp.problem.status, pulp.LpStatusInfeasible)

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
