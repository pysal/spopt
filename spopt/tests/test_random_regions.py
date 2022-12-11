import geopandas
import libpysal
import numpy
import pytest

from spopt.region import RandomRegion, RandomRegions


# Empirical tests -- Mexican states
RANDOM_STATE = 12345
pth = libpysal.examples.get_path("mexicojoin.shp")
MEXICO = geopandas.read_file(pth)

# Synthetic tests
N_REGIONS = 13
SYNTH_CARDS = list(range(2, 14)) + [10]
SYNTH_W = libpysal.weights.lat2W(10, 10, rook=True)
SYNTH_IDS = SYNTH_W.id_order


# Empirical tests ------------------------------------------------------------------------
class TestRandomRegionEmpirical:
    def setup_method(self):

        self.mexico = MEXICO.copy()
        self.cards = self.mexico.groupby(by="HANSON03").count().NAME.values.tolist()
        self.ids = self.mexico.index.values.tolist()
        self.w = libpysal.weights.Queen.from_dataframe(self.mexico)

    def test_random_region_6_card(self):
        known_regions = [
            [27, 12, 18, 3, 15, 8],
            [0, 25, 21, 20, 7, 6, 24],
            [23, 10, 13, 11, 19, 16, 26, 14, 17, 22],
            [28, 31],
            [30, 9, 4],
            [1, 29, 5, 2],
        ]
        numpy.random.seed(RANDOM_STATE)
        kwargs = {"num_regions": 6, "cardinality": self.cards}
        model = RandomRegion(self.ids, **kwargs)

        numpy.testing.assert_array_equal(
            numpy.array(known_regions, dtype=object),
            numpy.array(model.regions, dtype=object),
        )

    def test_random_region_6_card_contig_compact(self):
        known_regions = [
            [27, 29, 5, 26, 3, 24, 4, 30, 23, 2],
            [12, 31, 7, 15, 18, 10, 17],
            [8, 11],
            [21, 19, 20, 14, 13, 16],
            [0, 22, 1, 25],
            [28, 6, 9],
        ]
        numpy.random.seed(RANDOM_STATE)
        kwargs = {
            "num_regions": 6,
            "cardinality": self.cards,
            "contiguity": self.w,
            "compact": True,
        }
        model = RandomRegion(self.ids, **kwargs)

        numpy.testing.assert_array_equal(
            numpy.array(known_regions, dtype=object),
            numpy.array(model.regions, dtype=object),
        )


class TestRandomRegionsEmpirical:
    def setup_method(self):

        self.mexico = MEXICO.copy()
        self.cards = self.mexico.groupby(by="HANSON03").count().NAME.values.tolist()
        self.ids = self.mexico.index.values.tolist()

    def test_random_regions_6_card(self):
        known_regions = [
            [8, 5, 29, 22, 16, 10],
            [4, 26, 1, 2, 6, 13, 15],
            [19, 3, 20, 11, 31, 12, 0, 17, 7, 21],
            [23, 14],
            [25, 27, 28],
            [18, 9, 30, 24],
        ]
        numpy.random.seed(RANDOM_STATE)
        kwargs = {"num_regions": 6, "cardinality": self.cards, "permutations": 99}
        model = RandomRegions(self.ids, **kwargs)

        numpy.testing.assert_array_equal(
            numpy.array(known_regions, dtype=object),
            numpy.array(model.solutions_feas[2].regions, dtype=object),
        )


# Synthetic tests ------------------------------------------------------------------------
class TestRandomRegionSynthetic:
    def setup_method(self):

        self.nregs = N_REGIONS
        self.cards = SYNTH_CARDS
        self.w = SYNTH_W
        self.ids = SYNTH_W.id_order

    def test_random_region_unconstrained(self):
        known_region_0 = [19, 14, 43, 37, 66, 3, 79, 41, 38, 68, 2, 1, 60]
        numpy.random.seed(10)
        model = RandomRegion(self.ids)
        assert known_region_0 == model.regions[0]

    def test_random_region_exo_regions(self):
        known_region_0 = [37, 62, 26, 41, 35, 25, 36]
        numpy.random.seed(100)
        kwargs = {"num_regions": self.nregs}
        model = RandomRegion(self.ids, **kwargs)
        assert known_region_0 == model.regions[0]

    def test_random_region_endo_regions_constrained_card(self):
        known_region_0 = [37, 62]
        numpy.random.seed(100)
        kwargs = {"cardinality": self.cards}
        model = RandomRegion(self.ids, **kwargs)
        assert known_region_0 == model.regions[0]

    def test_random_region_exo_regions_constrained_card(self):
        known_region_0 = [37, 62]
        numpy.random.seed(100)
        kwargs = {"num_regions": self.nregs, "cardinality": self.cards}
        model = RandomRegion(self.ids, **kwargs)
        assert known_region_0 == model.regions[0]

    def test_random_region_endo_regions_constrained_contig(self):
        known_region_5 = [33, 43, 32, 31]
        numpy.random.seed(100)
        kwargs = {"contiguity": self.w}
        model = RandomRegion(self.ids, **kwargs)
        assert known_region_5 == model.regions[5]

    def test_random_region_exo_regions_constrained_contig(self):
        known_region_5 = [92, 93, 91, 81, 71, 70, 90, 80]
        numpy.random.seed(100)
        kwargs = {"num_regions": self.nregs, "contiguity": self.w}
        model = RandomRegion(self.ids, **kwargs)
        assert known_region_5 == model.regions[5]

    def test_random_region_exo_regions_constrained_card_contig(self):
        known_region_0 = [62, 61, 81, 71, 64, 90, 72, 51, 80, 63, 50, 73, 52]
        kwargs = {
            "num_regions": self.nregs,
            "cardinality": self.cards,
            "contiguity": self.w,
        }
        numpy.random.seed(60)
        model = RandomRegion(self.ids, **kwargs)
        assert known_region_0, model.regions[0]

    def test_random_region_endo_regions_constrained_card_contig(self):
        known_region_0 = [62, 61, 81, 71, 64, 90, 72, 51, 80, 63, 50, 73, 52]
        kwargs = {"cardinality": self.cards, "contiguity": self.w}
        numpy.random.seed(60)
        model = RandomRegion(self.ids, **kwargs)
        assert known_region_0 == model.regions[0]

    def test_random_regions_error_card(self):
        with pytest.raises(ValueError, match="Number of areas"):
            RandomRegion([0, 1], cardinality=[4])

    def test_random_regions_error_contig(self):
        with pytest.raises(ValueError, match="Order of `area_ids`"):

            class _shell_w_:
                def __init__(self):
                    self.id_order = [1, 0]

            RandomRegion([0, 1], contiguity=_shell_w_())

    def test_random_regions_error_nregs(self):
        with pytest.raises(ValueError, match="Number of regions"):
            RandomRegion([0, 1, 2, 3, 4, 5], num_regions=2, cardinality=[1, 2, 3])


class TestRandomRegionsSynthetic:
    def setup_method(self):

        self.nregs = N_REGIONS
        self.cards = SYNTH_CARDS
        self.w = SYNTH_W
        self.ids = SYNTH_W.id_order
        self.permutations = 5

    def test_random_region_unconstrained(self):
        known_region_0 = [19, 14, 43, 37, 66, 3, 79, 41, 38, 68, 2, 1, 60]
        numpy.random.seed(10)
        kwargs = {"permutations": self.permutations}
        model = RandomRegions(self.ids, **kwargs)
        assert known_region_0 == model.solutions[0].regions[0]

    def test_random_region_exo_regions(self):
        known_region_0 = [37, 62, 26, 41, 35, 25, 36]
        numpy.random.seed(100)
        kwargs = {"num_regions": self.nregs, "permutations": self.permutations}
        model = RandomRegions(self.ids, **kwargs)
        assert known_region_0 == model.solutions[0].regions[0]

    def test_random_region_endo_regions_constrained_card(self):
        known_region_0 = [37, 62]
        numpy.random.seed(100)
        kwargs = {"cardinality": self.cards, "permutations": self.permutations}
        model = RandomRegions(self.ids, **kwargs)
        assert known_region_0 == model.solutions[0].regions[0]

    def test_random_region_exo_regions_constrained_card(self):
        known_region_0 = [37, 62]
        numpy.random.seed(100)
        kwargs = {
            "num_regions": self.nregs,
            "cardinality": self.cards,
            "permutations": self.permutations,
        }
        model = RandomRegions(self.ids, **kwargs)
        assert known_region_0 == model.solutions[0].regions[0]

    def test_random_region_endo_regions_constrained_contig(self):
        known_region_5 = [33, 43, 32, 31]
        numpy.random.seed(100)
        kwargs = {"contiguity": self.w, "permutations": self.permutations}
        model = RandomRegions(self.ids, **kwargs)
        assert known_region_5 == model.solutions[0].regions[5]

    def test_random_region_exo_regions_constrained_contig(self):
        known_region_5 = [92, 93, 91, 81, 71, 70, 90, 80]
        numpy.random.seed(100)
        kwargs = {
            "num_regions": self.nregs,
            "contiguity": self.w,
            "permutations": self.permutations,
        }
        model = RandomRegions(self.ids, **kwargs)
        assert known_region_5 == model.solutions[0].regions[5]

    def test_random_region_exo_regions_constrained_card_contig(self):
        known_region_0 = [62, 61, 81, 71, 64, 90, 72, 51, 80, 63, 50, 73, 52]
        kwargs = {
            "num_regions": self.nregs,
            "cardinality": self.cards,
            "contiguity": self.w,
            "permutations": self.permutations,
        }
        numpy.random.seed(60)
        model = RandomRegions(self.ids, **kwargs)
        assert known_region_0 == model.solutions[0].regions[0]

    def test_random_region_endo_regions_constrained_card_contig(self):
        known_region_0 = [62, 61, 81, 71, 64, 90, 72, 51, 80, 63, 50, 73, 52]
        kwargs = {
            "cardinality": self.cards,
            "contiguity": self.w,
            "permutations": self.permutations,
        }
        numpy.random.seed(60)
        model = RandomRegions(self.ids, **kwargs)
        assert known_region_0 == model.solutions[0].regions[0]
