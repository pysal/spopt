import pytest

from spopt.locate.util import simulated_geo_points


class TestSimulatedGeoPoints:
    @pytest.fixture(autouse=True)
    def setup_method(self, network_instance) -> None:
        # case 1 - single polygon
        client_count, facility_count = None, None
        self.indata_polygons_geoseries, self.indata_polygon = network_instance(
            client_count, facility_count
        )

        # case 2 - multipolygon
        subslice = [0, 10]
        self.indata_multipolygon_geoseries, self.indata_multipolygon = network_instance(
            client_count, facility_count, loc_slice=subslice
        )

    def test_from_polygon(self):
        needed = 5
        observed = simulated_geo_points(self.indata_polygon, needed=needed, seed=5)

        assert observed.shape[0] == needed

        for point in observed.geometry:
            assert point.intersects(self.indata_polygon)

    def test_from_polygons_geoseries(self):
        needed = 10.0
        observed = simulated_geo_points(
            self.indata_polygons_geoseries, needed=needed, seed=10
        )

        assert observed.shape[0] == needed

        for point in observed.geometry:
            assert point.intersects(self.indata_polygon)

    def test_from_multipolygon(self):
        needed = 5
        observed = simulated_geo_points(self.indata_multipolygon, needed=needed, seed=5)

        assert observed.shape[0] == needed

        for point in observed.geometry:
            assert point.intersects(self.indata_multipolygon)

    def test_from_multipolygon_geoseries(self):
        needed = 10.0
        observed = simulated_geo_points(
            self.indata_multipolygon_geoseries, needed=needed, seed=10
        )

        assert observed.shape[0] == needed

        for point in observed.geometry:
            assert point.intersects(self.indata_multipolygon)

    def test_error_indata(self):
        in_data = [self.indata_polygon, self.indata_polygon]
        with pytest.raises(
            ValueError, match=f"'{type(in_data)}' not valid for ``in_data``."
        ):
            simulated_geo_points(in_data)

    def test_error_neq_needed(self):
        needed = -1
        with pytest.raises(ValueError, match=f"Cannot generate {needed} points."):
            simulated_geo_points(self.indata_polygon, needed=needed)

    def test_error_flt_needed(self):
        needed = 1.5
        with pytest.raises(ValueError, match=f"Cannot generate {needed} points."):
            simulated_geo_points(self.indata_polygon, needed=needed)

    def test_error_neg_seed(self):
        bad_seed = -1
        with pytest.raises(ValueError, match="Seed must be between"):
            simulated_geo_points(self.indata_polygon, seed=bad_seed)

    def test_error_flt_seed(self):
        bad_seed = 1.5
        with pytest.raises(TypeError, match="Cannot cast scalar from"):
            simulated_geo_points(self.indata_polygon, seed=bad_seed)
