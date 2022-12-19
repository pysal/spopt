from pyproj import crs
from spopt.locate.base import FacilityModelBuilder, LocateSolver, T_FacModel
import numpy
import geopandas
import pandas
import pulp
import spaghetti
from shapely.geometry import Point, Polygon

from spopt.locate import LSCP
from spopt.locate.util import simulated_geo_points
import pytest
import warnings


class TestSyntheticLocate:
    def setup_method(self) -> None:
        lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
        ntw = spaghetti.Network(in_data=lattice)
        gdf = spaghetti.element_as_gdf(ntw, arcs=True)
        street = geopandas.GeoDataFrame(
            geopandas.GeoSeries(gdf["geometry"].buffer(0.2).unary_union),
            crs=gdf.crs,
            columns=["geometry"],
        )

        client_count = 5
        facility_count = 2

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

    def test_clscpso_y1_lt_y2(self):
        service_radius = 8
        facility_capacity = numpy.array([5, 15])
        demand_quantity = numpy.arange(1, 6)
        clscpso = LSCP.from_cost_matrix(
            self.cost_matrix,
            service_radius,
            facility_capacity_arr=facility_capacity,
            demand_quantity_arr=demand_quantity,
        )
        result = clscpso.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, LSCP)

        known = [[1], [1], [1], [1], [1]]
        observed = clscpso.cli2fac
        assert known == observed

        known = [[], [0, 1, 2, 3, 4]]
        observed = clscpso.fac2cli
        assert known == observed

    def test_clscpso_y1_gt_y2(self):
        service_radius = 8
        facility_capacity = numpy.array([15, 5])
        demand_quantity = numpy.arange(1, 6)
        clscpso = LSCP.from_cost_matrix(
            self.cost_matrix,
            service_radius,
            facility_capacity_arr=facility_capacity,
            demand_quantity_arr=demand_quantity,
        )
        result = clscpso.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, LSCP)

        known = [[1], [1], [0, 1], [0, 1], [0, 1]]
        observed = clscpso.cli2fac
        assert known == observed

        known = [[2, 3, 4], [0, 1, 2, 3, 4]]
        observed = clscpso.fac2cli
        assert known == observed

    def test_clscpso_y1_eq_y2(self):
        service_radius = 7
        facility_capacity = numpy.array([8, 8])
        demand_quantity = numpy.arange(1, 6)
        clscpso = LSCP.from_cost_matrix(
            self.cost_matrix,
            service_radius,
            facility_capacity_arr=facility_capacity,
            demand_quantity_arr=demand_quantity,
        )
        result = clscpso.solve(pulp.PULP_CBC_CMD(msg=False))
        assert isinstance(result, LSCP)

        known = [[1], [1], [0, 1], [0], [1]]
        observed = clscpso.cli2fac
        assert known == observed

        known = [[2, 3], [0, 1, 2, 4]]
        observed = clscpso.fac2cli
        assert known == observed

    def test_clscpso_dem_gt_cap_error(self):
        service_radius = 8
        facility_capacity = numpy.array([8, 8])
        demand_quantity = numpy.arange(5, 10)
        with pytest.raises(
            ValueError, match="Infeasible model. Demand greater than capacity"
        ):
            clscpso = LSCP.from_cost_matrix(
                self.cost_matrix,
                service_radius,
                facility_capacity_arr=facility_capacity,
                demand_quantity_arr=demand_quantity,
            )

    def test_clscpso_infease_error(self):
        service_radius = 1
        facility_capacity = numpy.array([5, 15])
        demand_quantity = numpy.arange(1, 6)
        clscpso = LSCP.from_cost_matrix(
            self.cost_matrix,
            service_radius,
            facility_capacity_arr=facility_capacity,
            demand_quantity_arr=demand_quantity,
        )
        with pytest.raises(RuntimeError, match="Model is not solved:"):
            clscpso.solve(pulp.PULP_CBC_CMD(msg=False))
