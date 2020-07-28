import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from phip.utils import DEFAULT_FDR, DEFAULT_REFERENCE_QUANTILE


def do_hit_calling(
    counts_df,
    beads_only_samples,
    fdr=DEFAULT_FDR,
    reference_quantile=DEFAULT_REFERENCE_QUANTILE,
    normalize_to_reads_per_million=None,
    verbosity=2,
):
    """
    Call hits at the specified FDR using a heuristic.

    A hit is a (clone, sample) pair whose read count exceeds what is expected
    by chance.

    The basic assumption is that the beads-only (negative control) samples
    for some clone c give a good indication of the read count distribution
    of non-hits for that clone. Background effects that are not captured by
    beads-only samples will not be accounted for by this approach. At least
    two beads-only samples are required, but more are better.

    Overview of the algorithm:

    We first convert read counts to log read counts given by:

        log10(pseudocount + read count)

    for some pseudocount > 0. We'll show how to choose this value later.

    All further analysis is done using these transformed values.

    Now we consider *for each clone c* how high the beads-only samples are. We
    care much more about the extreme value than about the central tendency.
    For now, we are using the 99.9th percentile of the beads-only distribution
    as the quantity to consider. Since we typically only have a few beads
    only samples, this is effectively the same as the max value.

    We now compute the difference between each entry and the 99.9th
    percentile of the beads only samples for the corresponding clone. This gives
    us a (samples x clones) matrix of "rescaled" values. When computing the
    rescaled values for beads-only samples themselves, we hold out each
    beads-only sample and use 99.9% percentile of the distribution of the
    remaining beads-only samples.

    We'd like to pick a "hit threshold" value and call rescaled values above
    that value hits. But (a) how do we pick this threshold and (b) what value
    of pseudocount should we use?

    To answer (a), we consider the rescaled beads-only samples. The fraction
    of entries for which this value exceeds a threshold is an estimate of
    (1-s) where s is the specificity (proportion of actual negatives correctly
    called as negative) associated with that threshold. We are basically
    asking: if we hold out a beads only sample, how often would we have
    called it a hit using that threshold?

    The product (1-s) * (num clones) * (num non-beads-only samples) gives the
    number of false calls we expect if we apply this threshold to the real
    pull-down (non-beads-only) samples. The number of these entries actually
    exceeding the threshold gives the total number of calls. The first
    quantity divided by the second is the FDR associated with the threshold.
    We can select the hit threshold to achieve any desired FDR.

    Question (b) is simpler. We will simply choose the pseudocount that gives
    us the most hits.

    Implementation: we have a one-dimensional non-differentiable optimization
    problem to find the pseudocount value that gives the most hits. In each
    evaluation of the cost function for this optimization, we solve another
    one-dimensional optimization problem to find the hit threshold that
    achieves the desired FDR at some pseudocount value. Both optimizations are
    solved numerically using scipy.

    Parameters
    ----------
    counts_df : pandas.DataFrame of shape (clones, samples)
         read counts
    beads_only_samples : list of string
        Names of beads-only samples. These should be columns in counts_df.
    fdr : float
        False discovery rate
    reference_quantile : float
        Percentile to take of each clone's beads-only samples. The default is
        likely fine and the algorithm should not be particularly sensitive to
        this parameter.
    normalize_to_reads_per_million : boolean
        If true, first divide each column by the total number of reads for that
        sample and multiple by 1 million. If None, do this only when all values
        in the counts matrix are positive. If False, never do this.
    verbosity : int
        verbosity: no output (0), result summary only (1), or progress (2)

    Returns
    -------
    pandas.DataFrame of shape (clones x samples)

    Entries above 1.0 are the hit calls.

    Higher values indicate more evidence for a hit, but there is no simple
    interpretation of these values beyond whether they are below/above 1.0.
    """
    if len(beads_only_samples) < 2:
        raise ValueError("At least two beads-only samples are required.")

    if normalize_to_reads_per_million is None:
        normalize_to_reads_per_million = ~((counts_df < 0).any().any())

    counts_df = np.maximum(counts_df, 0)
    if normalize_to_reads_per_million:
        counts_df = counts_df * 1e6 / counts_df.sum(0)
        if verbosity > 0:
            print("Normalized to reads-per-million.")
    elif verbosity > 0:
        print("Did NOT normalize counts to reads-per-million.")

    counts_df = counts_df.astype("float32")

    start = time.time()

    all_samples = counts_df.columns.tolist()
    pull_down_samples = [s for s in all_samples if s not in beads_only_samples]

    iteration_mutable_value = [0]
    start = time.time()

    def print_summary(info):
        binary_hits_df = info["rescaled_df"][pull_down_samples] > info["hit_threshold"]

        hits_by_samples = binary_hits_df.sum(0)

        print("  Pseudocount       : %0.4f" % info["pseudocount"])
        print(
            "  Hit threshold     : %0.4f (=10**%0.4f)"
            % (info["hit_threshold"], np.log10(info["hit_threshold"]))
        )
        print(
            "  Clones w/ a hit   : %d (%0.1f%%)"
            % (binary_hits_df.any(1).sum(), binary_hits_df.any(1).mean() * 100.0)
        )
        print("  Median hits/sample: %d" % hits_by_samples.median())
        print(
            "  ... min/mean/max  : %0.4f/%0.4f/%0.4f"
            % (hits_by_samples.min(), hits_by_samples.mean(), hits_by_samples.max())
        )
        print("  Total hits        : %d" % info["total_hits"])
        print("  Elapsed           : %0.1f sec." % (time.time() - start))

    def function_to_minimize(current_log10_pseudocount):
        current_info = hits_at_specified_pseudocount(
            pseudocount=10 ** current_log10_pseudocount,
            counts_df=counts_df,
            beads_only_samples=beads_only_samples,
            pull_down_samples=pull_down_samples,
            fdr=fdr,
            reference_quantile=reference_quantile,
        )
        if verbosity > 1:
            print("*** Iteration %5d *** " % iteration_mutable_value[0])
            print_summary(current_info)
        iteration_mutable_value[0] += 1
        return -1 * current_info["total_hits"]

    result = minimize_scalar(
        function_to_minimize, bounds=(-10, 10), method="bounded", options={"xatol": 1.0}
    )

    pseudocount = 10 ** result.x

    # Call helper one more time to get final info.
    info = hits_at_specified_pseudocount(
        pseudocount=pseudocount,
        counts_df=counts_df,
        beads_only_samples=beads_only_samples,
        pull_down_samples=pull_down_samples,
        fdr=fdr,
        reference_quantile=reference_quantile,
    )

    if verbosity > 0:
        print("\n*** HIT CALLING RESULTS ***")
        print_summary(info)

    return info["rescaled_df"] / info["hit_threshold"]


def hits_at_specified_pseudocount(
    pseudocount,
    counts_df,
    beads_only_samples,
    pull_down_samples,
    fdr,
    reference_quantile=DEFAULT_REFERENCE_QUANTILE,
):
    """
    Compute the total number of hits at a particular smoothing value and FDR.

    Helper function for do_hit_calling().

    Parameters
    ----------
    pseudocount : float
        Smoothing value
    counts_df : pandas.DataFrame of shape (clones, samples)
         read counts
    beads_only_samples : list of string
        Names of beads-only samples. These names should be columns in counts_df.
    pull_down_samples : list of string
        Names of pull down samples. These names should be columns in counts_df.
    max_beads_only : pandas.Series, same index as counts_df
        Max over beads-only samples
    second_max_beads_only : pandas.Series, same index as counts_df
        Second max over beads-only samples
    fdr : float
        False discovery rate
    reference_quantile : float
        Percentile to take of each clone's beads-only samples. The default is
        likely fine and the algorithm should not be particularly sensitive to
        this parameter.

    Returns
    -------
    dict with several result values:

    total_hits : int
    hit_threshold : float
    pseudocount: float
    rescaled_df : pandas.DataFrame

    """
    all_samples = list(beads_only_samples) + list(pull_down_samples)

    log_counts_df = np.log10(pseudocount + np.maximum(counts_df, 0))

    # Compute clones x samples matrix of rescaled values.
    rescaled_df = pd.DataFrame(
        index=counts_df.index, columns=all_samples, dtype="float32"
    )

    for s in beads_only_samples:
        other_beads_only = [c for c in beads_only_samples if c != s]
        rescaled_df[s] = log_counts_df[s] - log_counts_df[other_beads_only].quantile(
            reference_quantile, axis=1
        )

    beads_only_high_quantile = log_counts_df[beads_only_samples].quantile(
        reference_quantile, axis=1
    )
    for s in pull_down_samples:
        rescaled_df[s] = log_counts_df[s] - beads_only_high_quantile

    del log_counts_df

    empirical_null = rescaled_df[beads_only_samples].values.flatten()
    empirical_null.sort()

    def fast_quantile(sorted_array, quantile):
        # Fast version of pd.Series.quantile()
        # Optimization relies on the array being sorted, low to high.
        position = (len(sorted_array) - 1) * quantile
        low_index = math.floor(position)
        residual = position - low_index
        if low_index >= len(sorted_array) - 1:
            return sorted_array[-1]
        return (
            sorted_array[low_index]
            + (sorted_array[low_index + 1] - sorted_array[low_index]) * residual
        )

    flattened_sorted_pull_down_values = rescaled_df[pull_down_samples].values.flatten()
    flattened_sorted_pull_down_values.sort()

    def fast_count_exceeding_value(sorted_array, value):
        # Fast version of (sorted_array > value).sum()
        # Optimization relies on the array being sorted, low to high.
        return len(sorted_array) - (
            np.searchsorted(sorted_array, np.float32(value), side="right")
        )

    # Pick a hit threshold. Rescaled values greater than this are hits.
    #
    # We solve a simple optimization problem to find the hit threshold that
    # most closely approximates the desired FDR.
    #
    # Instead of optimizing the hit threshold directly, we work in a
    # transformed space that makes for a better-behaved problem,
    # log-transformed quantiles of the estimated_max_beads_only distribution.
    #
    def empirical_fdr(negative_log10_1_minus_quantile):
        quantile = 1 - 10 ** (-1 * negative_log10_1_minus_quantile)

        hit_threshold = fast_quantile(empirical_null, quantile)

        # Total hits: how many ratios exceed our threshold.
        num_hits = fast_count_exceeding_value(
            flattened_sorted_pull_down_values, hit_threshold
        )

        # Num false hits: product of the quantile we are at and the total number
        # of tests we are making.
        num_false_hits = (1 - quantile) * len(pull_down_samples) * len(rescaled_df)

        # FDR is fraction of total hits that are expected to be false.
        return num_false_hits / num_hits

    transformed_quantile_result = minimize_scalar(
        lambda v: (empirical_fdr(v) - fdr) ** 2,
        method="bounded",
        bounds=(1, 10),
        options={"xatol": 0.01},
    )

    quantile = 1 - 10 ** (-1 * transformed_quantile_result.x)
    hit_threshold = fast_quantile(empirical_null, quantile)
    total_hits = fast_count_exceeding_value(
        flattened_sorted_pull_down_values, hit_threshold
    )

    return {
        "rescaled_df": rescaled_df,
        "hit_threshold": hit_threshold,
        "pseudocount": pseudocount,
        "total_hits": total_hits,
    }
