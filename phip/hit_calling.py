import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def do_hit_calling(
        counts_df,
        beads_only_samples,
        fdr=0.15,
        smoothing_bracket=(0, 500),
        verbosity=2):
    """
    Call hits at the specified FDR using a heuristic.

    A hit is a (clone, sample) pair whose read count exceeds what is expected
    by chance.

    The basic assumption is that the beads-only (negative control) samples
    for some clone c give a good indication of the read count distribution
    of non-hits for that clone. Background effects that are not captured by
    beads-only samples will not be accounted for by this routine. At least
    two beads-only samples are required, but more are better.

    Overview of the algorithm:

    An intuitive quantity to consider for a clone c and sample s is the
    "smoothed ratio" between the read count at entry (c, s) and the maximum
    number of reads across beads-only samples for c. The smoothed ratio is
    given by (a + smoothing) / (b + smoothing) for some value of `smoothing`.

    We'd like to pick a threshold value and call ratios above that value hits.
    But (a) how do we pick this threshold and (b) what value of smoothing
    should we use?

    To answer (a), suppose we somehow have a particular smoothing value
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
    helper function `hits_at_specified_smoothing` to see how this is done.

    To summarize, for any given smoothing value we can find the ratio
    threshold that gives us our desired FDR. Smoothed ratios exceeding this
    value are our hits.

    Question (b) is simpler. We will simply choose the smoothing that gives
    us the most hits.

    Implementation: we have a one-dimensional non-differentiable optimization
    problem to find the smoothing value that gives the most hits. In each
    evaluation of the cost function for this optimization, we solve another
    one-dimensional optimization problem to find the ratio threshold that
    achieves the desired FDR at some smoothing value. Both optimizations are
    solved numerically using scipy.

    Parameters
    ----------
    counts_df : pandas.DataFrame of shape (clones, samples)
         read counts
    beads_only_samples : list of string
        Names of beads-only samples. These should be columns in counts_df.
    fdr : float
        False discovery rate
    smoothing_bracket : tuple of float
        Initial estimate of smoothing values to consider. Selected smoothing
        value may or may not fall in this interval.
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

    start = time.time()

    all_samples = counts_df.columns.tolist()
    pull_down_samples = [s for s in all_samples if s not in beads_only_samples]

    max_beads_only = counts_df[beads_only_samples].max(1)
    second_max_beads_only = pd.Series([
        sorted(row)[-2]
        for (_, row) in
        counts_df[beads_only_samples].iterrows()
    ], index=counts_df.index)

    iteration_mutable_value = [0]
    start = time.time()

    def function_to_minimize(current_smoothing):
        current_info = {} if verbosity > 1 else {}
        current_total_hits = hits_at_specified_smoothing(
            smoothing=current_smoothing,
            counts_df=counts_df,
            beads_only_samples=beads_only_samples,
            pull_down_samples=pull_down_samples,
            max_beads_only=max_beads_only,
            second_max_beads_only=second_max_beads_only,
            fdr=fdr,
            extra_result_info_dict=current_info)
        if verbosity > 1:
            print("[Iteration %5d, %5.0f sec. elapsed]: smoothing=%0.2f\n%s\n" % (
                iteration_mutable_value[0],
                time.time() - start,
                current_info['smoothing'],
                "\n".join(
                    "    %s=%s" % (key, value)
                    for (key, value)
                    in sorted(current_info.items())
                    if key != 'smoothing')
            ))
        iteration_mutable_value[0] += 1
        return -1 * current_total_hits

    result = minimize_scalar(
        function_to_minimize,
        bracket=smoothing_bracket,
        method="brent",
        tol=0.1)

    smoothing = result.x

    # Call hits_at_specified_smoothing one more time to populate info dict.
    info = {}
    hits_at_specified_smoothing(
        smoothing=smoothing,
        counts_df=counts_df,
        beads_only_samples=beads_only_samples,
        pull_down_samples=pull_down_samples,
        max_beads_only=max_beads_only,
        second_max_beads_only=second_max_beads_only,
        fdr=fdr,
        extra_result_info_dict=info)
    ratio_threshold = info['ratio_threshold']

    if verbosity > 0:
        print("\n*** HIT CALLING RESULTS ***")
        print("  Smoothing         : %0.4f" % smoothing)
        print("  Ratio threshold   : %0.4f" % info['ratio_threshold'])
        print("  Est. false pos.   : %0.4f [fdr=%0.3f]" % (
            info['estimated_false_positives'],
            info['estimated_false_positives'] / info['total_hits']))
        print("  Clones w/ a hit   : %d" % info['clone_hits'])
        print("  Median hits/sample: %d" % info['median_hits_per_sample'])
        print("  Total hits        : %d" % info['total_hits'])
        print("  Search time       : %0.1f sec." % (time.time() - start))

    smoothed_ratios_df = pd.DataFrame(index=counts_df.index)
    for s in all_samples:
        smoothed_ratios_df[s] = (
            (np.maximum(counts_df[s], 0) + smoothing) /
            (np.maximum(max_beads_only, 0) + smoothing)
        ).astype("float32")

    return smoothed_ratios_df / ratio_threshold


def hits_at_specified_smoothing(
        smoothing,
        counts_df,
        beads_only_samples,
        pull_down_samples,
        max_beads_only,
        second_max_beads_only,
        fdr,
        extra_result_info_dict=None):
    """
    Compute the total number of hits at a particular smoothing value and FDR.

    Helper function for do_hit_calling().

    Parameters
    ----------
    smoothing : float
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

    # Compute clones x samples matrix of smoothed ratios.
    smoothed_ratios_df = pd.DataFrame(index=counts_df.index)
    for s in all_samples:
        smoothed_ratios_df[s] = (
            (np.maximum(counts_df[s], 0) + smoothing) /
            (np.maximum(max_beads_only, 0) + smoothing))

    # Suppose, for clone c, the beads only sample with the highest read count
    # (call that value m_1) were actually a pull-down sample. Then the
    # smoothed ratio for this sample at clone c would be:
    #   (m_1 + smoothing) / (m_2 + smoothing)
    # where m_2 is the second highest beads-only read count for clone c.
    # This quantity is used the estimate the number of false positives.
    estimated_max_beads_only = (
        (np.maximum(max_beads_only, 0) + smoothing) /
        (np.maximum(second_max_beads_only, 0) + smoothing))

    # Pick a hit threshold. Ratio values above this value are considered hits.
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
        ratio_threshold = estimated_max_beads_only.quantile(quantile)

        # Total hits: how many ratios exceed our threshold.
        num_hits = (smoothed_ratios_df > ratio_threshold).sum().sum()

        # Num false hits: product of the quantile we are at and the total number
        # of tests we are making.
        num_false_hits = (1 - quantile) * len(pull_down_samples) * len(
            smoothed_ratios_df)

        # FDR is fraction of total hits that are expected to be false.
        return num_false_hits / num_hits

    transformed_quantile_result = minimize_scalar(
        lambda v: (empirical_fdr(v) - fdr) ** 2,
        method="bounded",
        bounds=(1, 10))

    quantile = 1 - 10 ** (-1 * transformed_quantile_result.x)
    ratio_threshold = estimated_max_beads_only.quantile(quantile)
    total_hits = (smoothed_ratios_df > ratio_threshold).sum().sum()

    if extra_result_info_dict is not None:
        extra_result_info_dict.update({
            'smoothing': smoothing,
            'ratio_threshold': ratio_threshold,
            'quantile': quantile,
            'estimated_false_positives': (
                (1 - quantile) * len(pull_down_samples) * len(smoothed_ratios_df)),
            'clone_hits': (smoothed_ratios_df > ratio_threshold).any(1).sum(),
            'total_hits': total_hits,
            'median_hits_per_sample': (
                smoothed_ratios_df > ratio_threshold).sum(0).median()
        })
    return total_hits


