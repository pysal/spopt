import numpy
import geopandas
import pandas
import pulp
from shapely.geometry import Point

from spopt.locate.p_median import KNearestPMedian
import os
import pickle
import platform
import pytest
import warnings


class TestKNearestPMedian:
    def setup_method(self) -> None:
        # Create the test data
        k_list = numpy.array([1, 1])
        demand_data = {
            "ID": [1, 2],
            "geometry": [Point(0.5, 1), Point(1.5, 1)],
            "demand": [1, 1],
        }
        facility_data = {
            "ID": [101, 102, 103],
            "geometry": [Point(1, 1), Point(0, 2), Point(2, 0)],
            "capacity": [1, 1, 1],
        }
        gdf_demand = geopandas.GeoDataFrame(demand_data, crs="EPSG:4326")
        gdf_fac = geopandas.GeoDataFrame(facility_data, crs="EPSG:4326")
        self.k_nearest_pmedian = KNearestPMedian.from_geodataframe(
            gdf_demand,
            gdf_fac,
            "geometry",
            "geometry",
            "demand",
            p_facilities=2,
            facility_capacity_col="capacity",
            k_list=k_list,
        )

    def test_knearest_p_median_from_geodataframe(self):
        result = self.k_nearest_pmedian.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, KNearestPMedian)

    def test_knearest_p_median_from_geodataframe_no_results(self):
        result = self.k_nearest_pmedian.solve(
            pulp.PULP_CBC_CMD(msg=False), results=False
        )
        assert isinstance(result, KNearestPMedian)

        with pytest.raises(AttributeError):
            result.cli2fac
        with pytest.raises(AttributeError):
            result.fac2cli
        with pytest.raises(AttributeError):
            result.mean_dist

    def test_solve(self):
        solver = pulp.PULP_CBC_CMD(msg=False)
        self.k_nearest_pmedian.solve(solver)
        assert self.k_nearest_pmedian.problem.status == pulp.LpStatusOptimal

        fac2cli_known = [[1], [0], []]
        cli2fac_known = [[1], [0]]
        mean_dist_known = 0.8090169943749475
        assert self.k_nearest_pmedian.fac2cli == fac2cli_known
        assert self.k_nearest_pmedian.cli2fac == cli2fac_known
        assert self.k_nearest_pmedian.mean_dist == mean_dist_known
