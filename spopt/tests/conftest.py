import pytest
import warnings

import geopandas
import numpy
import shapely

from spopt.locate.util import simulated_geo_points

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # ignore deprecation warning - GH pysal/spaghetti#649
    import spaghetti


from packaging.version import Version

# see gh:spopt#437
GPD_GE_10 = Version(geopandas.__version__) >= Version("1.0")


@pytest.fixture
def network_instance():
    """Return:

    * If client_count & facility_count are both integers:
        snapped clients, snapped facilities, network cost matrix

    * If client_count is None:
        None, snapped facilities, network cost matrix

    * If client_count & facility_count are both None & loc_slice is None:
        buffered polygons of the network and unioned polygon

    * If client_count & facility_count are both None & loc_slice is list:
        buffered polygons of the network and unioned polygon or multipolygon

    """

    def _network_instance(
        client_count: None | int,
        facility_count: None | int,
        loc_slice: None | list = None,
    ) -> (
        tuple[geopandas.GeoSeries, shapely.Polygon | shapely.MultiPolygon]
        | tuple[None | geopandas.GeoDataFrame, geopandas.GeoDataFrame, numpy.ndarray]
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ignore deprecation warning - GH pysal/libpysal#468
            lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)

        ntw = spaghetti.Network(in_data=lattice)
        gdf = spaghetti.element_as_gdf(ntw, arcs=True)

        if loc_slice:
            net_buffer = gdf.loc[loc_slice, "geometry"].buffer(0.2)
        else:
            net_buffer = gdf["geometry"].buffer(0.2)
        net_space = net_buffer.union_all() if GPD_GE_10 else net_buffer.unary_union

        if not client_count and not facility_count:
            return net_buffer, net_space

        street = geopandas.GeoDataFrame(
            geopandas.GeoSeries(net_space), crs=gdf.crs, columns=["geometry"]
        )

        # prepare clients locations (except p-dispersion)
        if client_count:
            client_points = simulated_geo_points(street, needed=client_count, seed=5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ignore deprecation warning - GH pysal/libpysal#468
                ntw.snapobservations(client_points, "clients", attribute=True)
            clients_snapped = spaghetti.element_as_gdf(
                ntw, pp_name="clients", snapped=True
            )
            source_pattern = ntw.pointpatterns["clients"]

        # prepare facility locations
        facility_points = simulated_geo_points(street, needed=facility_count, seed=6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ignore deprecation warning - GH pysal/libpysal#468
            ntw.snapobservations(facility_points, "facilities", attribute=True)
        facilities_snapped = spaghetti.element_as_gdf(
            ntw, pp_name="facilities", snapped=True
        )
        dest_pattern = ntw.pointpatterns["facilities"]

        # calculate network cost matrix
        if not client_count:
            clients_snapped = None
            source_pattern = dest_pattern
        cost_matrix = ntw.allneighbordistances(
            sourcepattern=source_pattern,
            destpattern=dest_pattern,
        )

        return clients_snapped, facilities_snapped, cost_matrix

    return _network_instance
