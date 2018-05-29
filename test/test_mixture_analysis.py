import numpy
from numpy.testing import assert_, assert_array_less, assert_equal
import pandas

from phip import mixture_analysis

numpy.random.seed(0)


def test_basic():
    counts_df = pandas.DataFrame([
        # First 3 are beads only
        [3, 0, 2, 3, 1, 3],
        [3, 0, 2, 1, 1, 1],
        [300, 0, 2, 2, 2, 2],
        [1, 1, 1, 3, 50, 6],
        [0, 2, 2, 4, 50, 60],
    ], columns=[
        "beads1", "beads2", "beads3", "pulldown1", "pulldown2", "pulldown3"
    ], index=["decoy1", "decoy2", "decoy3", "hit1", "hit2"],
    ).astype("float64")
    counts_df += numpy.random.poisson(1.0, size=counts_df.shape)

    fitted = mixture_analysis.do_mixture_analysis(
        counts_df,
        beads_only_samples=counts_df.columns[:3])

    assert_equal(fitted.index.values, counts_df.index.values)
    assert_array_less(0, fitted.gap1_fdr[["hit1", "hit2"]])
    assert_(fitted.gap1_fdr[["decoy1", "decoy2", "decoy3"]].isnull().all())


