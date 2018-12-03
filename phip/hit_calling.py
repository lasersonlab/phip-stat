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

    Parameters
    ----------
    counts_df
    beads_only_samples
    fdr

    Returns
    -------

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

    all_samples = beads_only_samples + pull_down_samples

    smoothed_ratios_df = pd.DataFrame(index=data_df.index)
    for s in all_samples:
        smoothed_ratios_df[s] = (
            (np.maximum(data_df[s], 0) + smoothing) /
            (np.maximum(data_df["max_beads_only"], 0) + smoothing))

    smoothed_ratios_df["estimated_max_beads_only"] = (
        (np.maximum(data_df["max_beads_only"], 0) + smoothing) /
        (np.maximum(data_df["second_max_beads_only"], 0) + smoothing))

    def empirical_fdr(negative_log10_1_minus_quantile):
        quantile = 1 - 10 ** (-1 * negative_log10_1_minus_quantile)
        ratio_threshold = smoothed_ratios_df.estimated_max_beads_only.quantile(
            quantile)
        num_hits = (smoothed_ratios_df > ratio_threshold).sum().sum()
        num_false_hits = (1 - quantile) * len(pull_down_samples) * len(
            smoothed_ratios_df)
        return num_false_hits / num_hits

    transformed_quantile_result = minimize_scalar(
        lambda v: (empirical_fdr(v) - fdr) ** 2, method="bounded",
        bounds=(1, 10))
    quantile = 1 - 10 ** (-1 * transformed_quantile_result.x)
    ratio_threshold = smoothed_ratios_df.estimated_max_beads_only.quantile(
        quantile)

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


