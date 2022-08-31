from pyproj import crs
from spopt.locate.base import FacilityModelBuilder, LocateSolver, T_FacModel
import numpy
import geopandas
import pandas
import pulp
import spaghetti
from shapely.geometry import Point, Polygon

from spopt.locate import PDispersion
from spopt.locate.util import simulated_geo_points
import unittest
import os
import pickle
import platform

operating_system = platform.platform()[:7].lower()
if operating_system == "windows":
    WINDOWS = True
else:
    WINDOWS = False


class TestSyntheticLocate(unittest.TestCase):
    def setUp(self) -> None:
        self.dirpath = os.path.join(os.path.dirname(__file__), "./data/")

        lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
        ntw = spaghetti.Network(in_data=lattice)
        gdf = spaghetti.element_as_gdf(ntw, arcs=True)
        street = geopandas.GeoDataFrame(
            geopandas.GeoSeries(gdf["geometry"].buffer(0.2).unary_union),
            crs=gdf.crs,
            columns=["geometry"],
        )

        facility_count = 5

        self.facility_points = simulated_geo_points(
            street, needed=facility_count, seed=6
        )

        ntw = spaghetti.Network(in_data=lattice)

        ntw.snapobservations(self.facility_points, "facilities", attribute=True)

        self.facilities_snapped = spaghetti.element_as_gdf(
            ntw, pp_name="facilities", snapped=True
        )

        self.cost_matrix = ntw.allneighbordistances(
            sourcepattern=ntw.pointpatterns["facilities"],
            destpattern=ntw.pointpatterns["facilities"],
        )

    def test_p_dispersion_from_cost_matrix(self):
        pdispersion = PDispersion.from_cost_matrix(self.cost_matrix, p_fac=2)
        result = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        self.assertIsInstance(result, PDispersion)

    def test_p_dispersion_from_geodataframe(self):
        pdispersion = PDispersion.from_geodataframe(
            self.facilities_snapped,
            "geometry",
            p_fac=2,
        )
        result = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        self.assertIsInstance(result, PDispersion)


class TestRealWorldLocate(unittest.TestCase):
    def setUp(self) -> None:
        self.dirpath = os.path.join(os.path.dirname(__file__), "./data/")
        network_distance = pandas.read_csv(
            self.dirpath
            + "SF_network_distance_candidateStore_16_censusTract_205_new.csv"
        )

        ntw_dist_piv = network_distance.pivot_table(
            values="distance", index="DestinationName", columns="name"
        )

        self.cost_matrix = ntw_dist_piv.to_numpy()

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

        self.service_dist = 5000.0
        self.p_facility = 4

    def test_optimality_p_dispersion_from_cost_matrix(self):
        pdispersion = PDispersion.from_cost_matrix(
            self.cost_matrix, p_fac=self.p_facility
        )
        pdispersion = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        self.assertEqual(pdispersion.problem.status, pulp.LpStatusOptimal)

    def test_infeasibility_p_dispersion_from_cost_matrix(self):
        pdispersion = PDispersion.from_cost_matrix(self.cost_matrix, p_fac=17)
        with self.assertRaises(RuntimeError):
            pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))

    def test_optimality_p_dispersion_from_geodataframe(self):
        pdispersion = PDispersion.from_geodataframe(
            self.facility_points_gdf,
            "geometry",
            p_fac=self.p_facility,
        )
        pdispersion = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        self.assertEqual(pdispersion.problem.status, pulp.LpStatusOptimal)

    def test_infeasibility_p_dispersion_from_geodataframe(self):
        pdispersion = PDispersion.from_geodataframe(
            self.facility_points_gdf,
            "geometry",
            p_fac=17,
        )
        with self.assertRaises(RuntimeError):
            pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))


class TestErrorsWarnings(unittest.TestCase):
    def setUp(self) -> None:

        pol1 = Polygon([(0, 0), (1, 0), (1, 1)])
        pol2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        pol3 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        polygon_dict = {"geometry": [pol1, pol2, pol3]}

        self.gdf_fac = geopandas.GeoDataFrame(polygon_dict, crs="EPSG:4326")

    def test_attribute_error_add_facility_constraint(self):
        with self.assertRaises(AttributeError):
            dummy_p_facility = 1
            dummy_class = PDispersion("dummy", pulp.LpProblem("name"), dummy_p_facility)
            FacilityModelBuilder.add_facility_constraint(
                dummy_class,
                dummy_class.problem,
                dummy_p_facility,
            )

    def test_attribute_error_add_p_dispersion_interfacility_constraint(self):
        with self.assertRaises(AttributeError):
            dummy_p_facility = 1
            dummy_matrix = numpy.array([])
            dummy_range = range(1)
            dummy_class = PDispersion("dummy", pulp.LpProblem("name"), dummy_p_facility)
            FacilityModelBuilder.add_p_dispersion_interfacility_constraint(
                dummy_class,
                dummy_class.problem,
                dummy_matrix,
                dummy_range,
            )

    def test_attribute_error_add_predefined_facility_constraint(self):
        with self.assertRaises(AttributeError):
            dummy_p_facility = 1
            dummy_matrix = numpy.array([])
            dummy_class = PDispersion("dummy", pulp.LpProblem("name"), dummy_p_facility)
            FacilityModelBuilder.add_facility_constraint(
                dummy_class, dummy_class.problem, dummy_matrix
            )

    def test_warning_facility_geodataframe(self):
        with self.assertWarns(Warning):
            dummy_class = PDispersion.from_geodataframe(
                self.gdf_fac, "geometry", 1
            )
