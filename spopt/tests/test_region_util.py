import libpysal
import geopandas
import numpy
import pulp
import unittest

import spopt.region.util as util


RANDOM_STATE = 123456

# Mexican states
pth = libpysal.examples.get_path("mexicojoin.shp")
MEXICO = geopandas.read_file(pth)


class TestRegionUtil(unittest.TestCase):
    def setUp(self):

        pass

    def test_check_solver(self):
        util.check_solver("cbc")
        with self.assertRaises(ValueError):
            util.check_solver("bcb")

    def test_get_solver_instance(self):
        known_name = "PULP_CBC_CMD"
        observed_name = util.get_solver_instance("cbc").name
        self.assertEqual(known_name, observed_name)
