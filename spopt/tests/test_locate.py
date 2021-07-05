import numpy
import geopandas
import pulp
import spaghetti
from shapely.geometry import Point

import spopt.locate
from spopt.locate import LSCP, MCLP, PCenter, PMedian
import unittest


def simulated_geo_points(in_data, needed=20, seed=0, to_file=None):
    geoms = in_data.geometry
    area = tuple(in_data.total_bounds)
    simulated_points_list = []
    simulated_points_all = False
    numpy.random.seed(seed)
    while simulated_points_all == False:
        x = numpy.random.uniform(area[0], area[2], 1)
        y = numpy.random.uniform(area[1], area[3], 1)
        point = Point(x, y)
        if geoms.intersects(point)[0]:
            simulated_points_list.append(point)
        if len(simulated_points_list) == needed:
            simulated_points_all = True
    sim_pts = geopandas.GeoDataFrame(
        simulated_points_list, columns=["geometry"], crs=in_data.crs
    )

    return sim_pts


class TestLocate(unittest.TestCase):
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
            self.client_points, self.facility_points, "geometry", "geometry", 10
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
        self.client_points["weights"] = self.ai
        mclp = MCLP.from_geodataframe(
            self.client_points,
            self.facility_points,
            "geometry",
            "geometry",
            "weights",
            max_coverage=7,
            p_facilities=4,
        )
        result = mclp.solve(pulp.PULP_CBC_CMD())
        self.assertIsInstance(result, MCLP)

    def test_p_median_from_cost_matrix(self):
        p_median = PMedian.from_cost_matrix(
            self.cost_matrix, self.ai, max_coverage=7, p_facilities=4
        )
        result = p_median.solve(pulp.PULP_CBC_CMD())
        self.assertIsInstance(result, PMedian)

    def test_p_median_from_geodataframe(self):
        self.client_points["weights"] = self.ai
        p_median = PMedian.from_geodataframe(
            self.client_points,
            self.facility_points,
            "geometry",
            "geometry",
            "weights",
            max_coverage=7,
            p_facilities=4,
        )
        result = p_median.solve(pulp.PULP_CBC_CMD())
        self.assertIsInstance(result, PMedian)

    def test_p_center_from_cost_matrix(self):
        p_center = PCenter.from_cost_matrix(
            self.cost_matrix, self.ai, max_coverage=7, p_facilities=4
        )
        result = p_center.solve(pulp.PULP_CBC_CMD())
        self.assertIsInstance(result, PCenter)

    def test_p_center_from_geodataframe(self):
        self.client_points["weights"] = self.ai
        p_center = PCenter.from_geodataframe(
            self.client_points,
            self.facility_points,
            "geometry",
            "geometry",
            "weights",
            max_coverage=7,
            p_facilities=4,
        )
        result = p_center.solve(pulp.PULP_CBC_CMD())
        self.assertIsInstance(result, PCenter)
