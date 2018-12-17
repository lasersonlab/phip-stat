import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def do_hit_calling(
        counts_df,
        beads_only_samples,
        fdr=0.15,
        pseudocount_bracket=(0, 500),
        normalize_to_reads_per_million=None,
        verbosity=2):
    """
    Call hits at the specified FDR using a heuristic.

    A hit is a (clone, sample) pair whose read count exceeds what is expected
    by chance.

    The basic assumption is that the beads-only (negative control) samples
    for some clone c give a good indication of the read count distribution
    of non-hits for that clone. Background effects that are not captured by
    beads-only samples will not be accounted for by this approach. At least
    three beads-only samples are required, but more are better.

    Overview of the algorithm:

    An intuitive quantity to consider for a clone c and sample s is the
    "smoothed ratio" between the read count at entry (c, s) and the maximum
    number of reads across beads-only samples for c. The smoothed ratio is
    given by (a + pseudocount) / (b + pseudocount) for some value of `pseudocount`.

    We'd like to pick a threshold value and call ratios above that value hits.
    But (a) how do we pick this threshold and (b) what value of pseudocount
    should we use?

    To answer (a), suppose we somehow have a particular pseudocount value
    selected. To select a ratio threshold, we consider the values across
    clones of the smoothed ratio of the highest beads-only sample to the
    second-highest beads-only sample. Note that these are different samples
    from one clone to another. The fraction of the clones for which this
    value exceeds a hypothetical ratio threshold is an over-estimate of (1-s)
    where s is the specificity (proportion of actual negatives correctly
    called as negative) associated with that threshold. We are basically
    asking: if we hold out a beads only sample, how often would we call it a
    hit? This is an over-estimate of (1-s) because we are always holding out
    the highest beads-only sample to be conservative.

    The product (1-s) * (num clones) * (num non-beads-only samples) gives the
    number of false calls we expect if we apply this threshold to the real
    pull-down (non-beads-only) samples. The number of these entries actually
    exceeding the threshold gives the total number of calls. The first
    quantity divided by the second is the FDR associated with the threshold.
    We can select the ratio threshold to achieve any desired FDR. See the
    helper function `hits_at_specified_pseudocount` to see how this is done.

    To summarize, for any given pseudocount value we can find the ratio
    threshold that gives us our desired FDR. Smoothed ratios exceeding this
    value are our hits.

    Question (b) is simpler. We will simply choose the pseudocount that gives
    us the most hits.

    Implementation: we have a one-dimensional non-differentiable optimization
    problem to find the pseudocount value that gives the most hits. In each
    evaluation of the cost function for this optimization, we solve another
    one-dimensional optimization problem to find the ratio threshold that
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
    pseudocount_bracket : tuple of float
        Initial estimate of pseudocount values to consider. Selected pseudocount
        value may or may not fall in this interval.
    normalize_to_reads_per_million : boolean
        If true, first divide each column by the total number of reads for that
        sample and multiple by 1 million.
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
        counts_df = (counts_df * 1e6 / counts_df.sum(0))
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

    def function_to_minimize(current_pseudocount):
        current_info = {} if verbosity > 1 else {}
        current_total_hits = hits_at_specified_pseudocount(
            pseudocount=current_pseudocount,
            counts_df=counts_df,
            beads_only_samples=beads_only_samples,
            pull_down_samples=pull_down_samples,
            fdr=fdr,
            extra_result_info_dict=current_info)
        if verbosity > 1:
            print(
                "[Iteration %5d, %5.0f sec. elapsed]: "
                "pseudocount=%0.2f\n%s\n" % (
                    iteration_mutable_value[0],
                    time.time() - start,
                    current_info['pseudocount'],
                    "\n".join(
                        "    %s=%s" % (key, value)
                        for (key, value)
                        in sorted(current_info.items())
                        if key not in ('pseudocount', "z_values_df")
            )))
        iteration_mutable_value[0] += 1
        return -1 * current_total_hits

    result = minimize_scalar(
        function_to_minimize,
        bracket=pseudocount_bracket,
        method="brent",
        tol=0.1)

    pseudocount = result.x

    # Call hits_at_specified_pseudocount one more time to populate info dict.
    info = {}
    hits_at_specified_pseudocount(
        pseudocount=pseudocount,
        counts_df=counts_df,
        beads_only_samples=beads_only_samples,
        pull_down_samples=pull_down_samples,
        fdr=fdr,
        extra_result_info_dict=info)

    if verbosity > 0:
        print("\n*** HIT CALLING RESULTS ***")
        print("  Pseudocount       : %0.4f" % pseudocount)
        print("  Hit threshold     : %0.4f" % info['hit_threshold'])
        print("  Est. false pos.   : %0.4f [fdr=%0.3f]" % (
            info['estimated_false_positives'],
            info['estimated_false_positives'] / info['total_hits']))
        print("  Clones w/ a hit   : %d" % info['clone_hits'])
        print("  Median hits/sample: %d" % info['median_hits_per_sample'])
        print("  Total hits        : %d" % info['total_hits'])
        print("  Search time       : %0.1f sec." % (time.time() - start))

    return info['z_values_df'] / info['hit_threshold']


def hits_at_specified_pseudocount(
        pseudocount,
        counts_df,
        beads_only_samples,
        pull_down_samples,
        fdr,
        extra_result_info_dict=None):
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
    extra_result_info_dict : dict or None
        Output argument. If not None, diagnostic information is added to the
        provided dict

    Returns
    -------
    int : total number of hits called

    """
    all_samples = list(beads_only_samples) + list(pull_down_samples)

    log_counts_df = np.log10(pseudocount + counts_df)

    # Compute clones x samples matrix of z values.
    z_values_df = pd.DataFrame(index=counts_df.index)

    def z_values(sample, reference_samples, std_offset=np.log10(pseudocount)):
        return ((
            log_counts_df[sample] - log_counts_df[reference_samples].mean(1)) /
                (std_offset + log_counts_df[other_beads_only].std(1)))

    for s in beads_only_samples:
        other_beads_only = [c for c in beads_only_samples if c != s]
        z_values_df[s] = z_values(s, other_beads_only)

    for s in pull_down_samples:
        z_values_df[s] = z_values(s, beads_only_samples)

    del log_counts_df

    empirical_null = pd.Series(z_values_df[beads_only_samples].values.flatten())

    # Pick a hit threshold. z-values values above this value are considered hits.
    #
    # We solve a simple optimization problem to find the ratio threshold that
    # most closely approximates the desired FDR.
    #
    # Instead of optimizing the ratio threshold directly, we work in a
    # transformed space that makes for a better-behaved problem,
    # log-transformed quantiles of the estimated_max_beads_only distribution.
    #
    def empirical_fdr(negative_log10_1_minus_quantile):
        quantile = 1 - 10 ** (-1 * negative_log10_1_minus_quantile)
        hit_threshold = empirical_null.quantile(quantile)

        # Total hits: how many ratios exceed our threshold.
        num_hits = (z_values_df[pull_down_samples] > hit_threshold).sum().sum()

        # Num false hits: product of the quantile we are at and the total number
        # of tests we are making.
        num_false_hits = (1 - quantile) * len(pull_down_samples) * len(
            z_values_df)

        # FDR is fraction of total hits that are expected to be false.
        return num_false_hits / num_hits

    transformed_quantile_result = minimize_scalar(
        lambda v: (empirical_fdr(v) - fdr) ** 2,
        method="bounded",
        bounds=(1, 10))

    quantile = 1 - 10 ** (-1 * transformed_quantile_result.x)
    hit_threshold = empirical_null.quantile(quantile)
    total_hits = (z_values_df[pull_down_samples] > hit_threshold).sum().sum()

    if extra_result_info_dict is not None:
        extra_result_info_dict.update({
            'pseudocount': pseudocount,
            'hit_threshold': hit_threshold,
            'quantile': quantile,
            'estimated_false_positives': (
                (1 - quantile) * len(pull_down_samples) * len(z_values_df)),
            'clone_hits': (z_values_df[pull_down_samples] > hit_threshold).any(1).sum(),
            'total_hits': total_hits,
            'median_hits_per_sample': (
                z_values_df[pull_down_samples] > hit_threshold).sum(0).median(),
            'z_values_df': z_values_df,
        })
    return total_hits


