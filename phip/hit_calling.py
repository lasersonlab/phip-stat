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
    Call hits at a particular FDR.

    Parameters
    ----------
    counts_df : pandas.DataFrame
        clones x samples dataframe of read counts
    beads_only_samples : list of string
        Names of beads-only samples. These names should be columns in counts_df.
    fdr : float
        False discovery rate
    smoothing_bracket : tuple of float
        Initial estimate of smoothing values to consider. Selected smoothing
        value may or may not fall in this interval.
    verbosity : int
        verbosity: no output (0), result summary only (1), or progress (2)

    Returns
    -------
    pandas.DataFrame
        clones x samples dataframe of results. Entries above 1.0 are the hits.

        Higher values indicate more evidence for a hit, but there is no simple
        interpretation of these values besides whether they are above or below
        1.0.
    """
    start = time.time()

    all_samples = counts_df.columns.tolist()
    pull_down_samples = [s for s in all_samples if s not in beads_only_samples]

    data_df = counts_df.copy()
    data_df["max_beads_only"] = data_df[beads_only_samples].max(1)
    data_df["second_max_beads_only"] = [
        sorted(row)[-2]
        for (_, row) in
        data_df[beads_only_samples].iterrows()
    ]

    iteration_box = [0]
    start = time.time()

    def function_to_minimize(current_smoothing):
        current_info = {} if verbosity > 1 else {}
        current_total_hits = hits_at_smoothing(
            smoothing=current_smoothing,
            data_df=data_df,
            beads_only_samples=beads_only_samples,
            pull_down_samples=pull_down_samples,
            fdr=fdr,
            extra_result_info_dict=current_info)
        if verbosity > 1:
            print("[Iteration %5d, %5.0f sec. elapsed]: smoothing=%0.2f\n%s\n" % (
                iteration_box[0],
                time.time() - start,
                current_info['smoothing'],
                "\n".join(
                    "    %s=%s" % (key, value)
                    for (key, value)
                    in sorted(current_info.items())
                    if key != 'smoothing')
            ))
        iteration_box[0] += 1
        return -1 * current_total_hits

    result = minimize_scalar(
        function_to_minimize,
        bracket=smoothing_bracket,
        method="brent",
        tol=0.1)

    smoothing = result.x

    # Call hits_at_smoothing one more time to populate info dict.
    info = {}
    hits_at_smoothing(
        smoothing=smoothing,
        data_df=data_df,
        beads_only_samples=beads_only_samples,
        pull_down_samples=pull_down_samples,
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

    smoothed_ratios_df = pd.DataFrame(index=data_df.index)
    for s in all_samples:
        smoothed_ratios_df[s] = (
            (np.maximum(data_df[s], 0) + smoothing) /
            (np.maximum(data_df["max_beads_only"], 0) + smoothing)
        ).astype("float32")

    return smoothed_ratios_df / ratio_threshold


def hits_at_smoothing(
        smoothing,
        data_df,
        beads_only_samples,
        pull_down_samples,
        fdr,
        extra_result_info_dict=None):
    """
    Compute the total number of hits at a particular smoothing value and FDR.

    Helper function for do_hit_calling().

    Parameters
    ----------
    smoothing : float
        Smoothing value
    data_df : pandas.DataFrame
        clones x samples dataframe of read counts. Should also include two
        additional precomputed columns:
            - max_beads_only
            - second_max_beads_only
    beads_only_samples : list of string
        Names of beads-only samples. These names should be columns in data_df.
    pull_down_samples : list of string
        Names of pull down samples. These names should be columns in data_df.
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
    # The (c, s) entry in this matrix gives the ratio of the read counts for
    # clone c and sample s to the max read count across beads-only samples for
    # clone c. The ratios are smoothed, i.e. computed as:
    # (a + smoothing) / (b + smoothing).
    smoothed_ratios_df = pd.DataFrame(index=data_df.index)
    for s in all_samples:
        smoothed_ratios_df[s] = (
            (np.maximum(data_df[s], 0) + smoothing) /
            (np.maximum(data_df["max_beads_only"], 0) + smoothing))

    # Suppose, for clone c, the beads only sample with the highest read count
    # (call that value m_1) were actually a pull down sample. Then the
    # smoothed ratio for this sample at clone c would be:
    #   (m_1 + smoothing) / (m_2 + smoothing)
    # where m_2 is the second highest beads-only read count for clone c.
    # This quantity is used the estimate the number of false positives.
    estimated_max_beads_only = (
        (np.maximum(data_df["max_beads_only"], 0) + smoothing) /
        (np.maximum(data_df["second_max_beads_only"], 0) + smoothing))

    # Pick a hit threshold. Ratio values above this value are considered hits.
    #
    # For any ratio threshold, we can use the estimated_max_beads_only quantity
    # to estimate the false discovery rate. Cases where estimated_max_beads_only
    # is greater than ratio_threshold would have been falsely called hits if
    # this sample were considered a pull down sample.
    #
    # We solve a simple optimization problem to find the ratio threshold that
    # most closely approximates the desired FDR.
    #
    # Instead of optimizing the ratio threshold directly, we work in a
    # transformed space that makes for a better-behaved problem,
    # log-transformed quantiles of the estimated_max_beads_only distribution.
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


