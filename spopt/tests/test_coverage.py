import numpy
import geopandas
import pulp
import spaghetti
from shapely.geometry import Point
from spopt.locate import LSCP, MCLP
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


class TestLSCP(unittest.TestCase):
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

        client_points = simulated_geo_points(street, needed=client_count, seed=5)
        facility_points = simulated_geo_points(street, needed=facility_count, seed=6)

        ntw = spaghetti.Network(in_data=lattice)

        ntw.snapobservations(client_points, "clients", attribute=True)
        ntw.snapobservations(facility_points, "facilities", attribute=True)

        self.cost_matrix = ntw.allneighbordistances(
            sourcepattern=ntw.pointpatterns["clients"],
            destpattern=ntw.pointpatterns["facilities"],
        )

        self.ai = numpy.random.randint(1, 12, client_count)

    def test_lscp_solve(self):
        lscp = LSCP.from_cost_matrix(self.cost_matrix, 10)
        status = lscp.solve(pulp.GLPK())
        self.assertEqual(status, 1)

    def test_mclp_solve(self):
        mclp = MCLP.from_cost_matrix(
            self.cost_matrix, self.ai, max_coverage=7, p_facilities=4
        )
        status = mclp.solve(pulp.GLPK())
        self.assertEqual(status, 1)
