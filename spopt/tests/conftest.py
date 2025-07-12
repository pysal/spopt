import pathlib
import pickle
import warnings

import geopandas
import numpy
import pandas
import pytest
import shapely

from spopt.locate.util import simulated_geo_points

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # ignore deprecation warning - GH pysal/spaghetti#649
    import spaghetti


from packaging.version import Version

# see gh:spopt#437
GPD_GE_10 = Version(geopandas.__version__) >= Version("1.0")


def locate_dirpath() -> pathlib.Path:
    """Path to locate test data directory"""
    return pathlib.Path(__file__).absolute().parent / "test_locate" / "data"


@pytest.fixture
def load_locate_test_data():
    """Load test data for the ``locate`` module."""

    def _load_test_data(_file: str) -> dict | pandas.DataFrame:
        if _file.endswith(".pkl"):
            with open(locate_dirpath() / _file, "rb") as f:
                test_data = pickle.load(f)
        elif _file.endswith(".csv"):
            test_data = pandas.read_csv(locate_dirpath() / _file)
        else:
            raise FileNotFoundError(f"`{_file}` does not exist.")

        return test_data

    return _load_test_data


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


_warns_geo_crs = pytest.warns(UserWarning, match="Geometry is in a geographic CRS")


@pytest.fixture
def loc_warns_geo_crs():
    """`locate` warning"""
    return _warns_geo_crs


@pytest.fixture
def loc_warns_mixed_type_dem():
    """`locate` warning"""
    return pytest.warns(UserWarning, match="Demand geodataframe contains mixed type")


@pytest.fixture
def loc_warns_mixed_type_fac():
    """`locate` warning"""
    return pytest.warns(UserWarning, match="Facility geodataframe contains mixed type")


@pytest.fixture
def loc_raises_diff_crs():
    """`locate` error"""
    return pytest.raises(ValueError, match="Geodataframes crs are different: ")


@pytest.fixture
def loc_raises_infeasible():
    """`locate` error"""
    return pytest.raises(RuntimeError, match="Model is not solved: Infeasible.")


@pytest.fixture
def loc_raises_fac_constr():
    """`locate` error"""
    return pytest.raises(AttributeError, match="Before setting facility constraint")


@pytest.fixture
def toy_fac_data() -> geopandas.GeoDataFrame:
    """Toy facility data used in ``locate`` error & warning tests."""
    pol1 = shapely.Polygon([(0, 0), (1, 0), (1, 1)])
    pol2 = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    pol3 = shapely.Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    polygon_dict = {"geometry": [pol1, pol2, pol3]}

    return geopandas.GeoDataFrame(polygon_dict, crs="EPSG:4326")


@pytest.fixture
def toy_dem_data() -> tuple[
    geopandas.GeoDataFrame, geopandas.GeoDataFrame, geopandas.GeoDataFrame
]:
    """Toy demand data used in ``locate`` error & warning tests."""

    point = shapely.Point(10, 10)
    point_dict = {"weight": 4, "geometry": [point]}

    gdf_dem = geopandas.GeoDataFrame(point_dict, crs="EPSG:4326")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Conversion of an array with ndim > 0"
        )
        gdf_dem_crs = gdf_dem.to_crs("EPSG:3857")

    gdf_dem_buffered = gdf_dem.copy()
    with _warns_geo_crs:
        gdf_dem_buffered["geometry"] = gdf_dem.buffer(2)

    return gdf_dem, gdf_dem_crs, gdf_dem_buffered
