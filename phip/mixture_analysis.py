import numpy
import pandas
from collections import OrderedDict
from multiprocessing import Pool
from functools import partial

from sklearn.mixture import GaussianMixture
import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481


def fit_one_mixture(index_and_row, max_iter):
    (index, row) = index_and_row
    values = row.values.reshape((-1, 1))

    m1 = GaussianMixture(n_components=1, max_iter=max_iter)
    m1.fit(values)

    m2 = GaussianMixture(n_components=2, max_iter=max_iter)
    m2.fit(values)

    result_dict = OrderedDict({"index": index})
    result_dict["m1_u"] = m1.means_.flatten()[0]
    result_dict["m1_s"] = numpy.sqrt(m1.covariances_.flatten()[0])

    m2_means = m2.means_.flatten()
    m2_variances = m2.covariances_.flatten()
    if m2_means[0] < m2_means[1]:
        result_dict["m2_u1"] = m2_means[0]
        result_dict["m2_u2"] = m2_means[1]
        result_dict["m2_s1"] = numpy.sqrt(m2_variances[0])
        result_dict["m2_s2"] = numpy.sqrt(m2_variances[1])
        flip_assignments = False
    else:
        result_dict["m2_u1"] = m2_means[1]
        result_dict["m2_u2"] = m2_means[0]
        result_dict["m2_s1"] = numpy.sqrt(m2_variances[1])
        result_dict["m2_s2"] = numpy.sqrt(m2_variances[0])
        flip_assignments = True

    aic1 = m1.aic(values)
    aic2 = m2.aic(values)
    result_dict["aic1"] = aic1
    result_dict["aic2"] = aic2
    result_dict["aic_diff"] = aic1 - aic2

    assignments = m2.predict_proba(values)[:, 0 if flip_assignments else 1]

    for (i, label) in enumerate(row.index):
        result_dict["centered_%s" % label] = row.iloc[i]
        result_dict["assignment_%s" % label] = assignments[i]
    return result_dict


def do_mixture_analysis(
        counts_df,
        beads_only_samples,
        max_iter=1000,
        num_jobs=1):

    centered_df = numpy.log10(1 + counts_df)
    centered_df -= centered_df.median(0)

    work_function = partial(fit_one_mixture, max_iter=max_iter)
    input_iterator = centered_df.iterrows()
    worker_pool = None
    map_function = map
    if num_jobs != 1:
        worker_pool = Pool(processes=num_jobs)
        map_function = worker_pool.imap_unordered

    result_iterator = map_function(work_function, input_iterator)

    # Progress bar
    result_iterator = tqdm.tqdm(result_iterator, total=len(centered_df))

    result_df = pandas.DataFrame(list(result_iterator))

    if worker_pool:
        worker_pool.join()
        worker_pool.close()

    result_df = result_df.set_index('index')
    print(result_df)

    assignment_columns = [c for c in result_df if c.startswith("assignment_")]
    assignment_columns_beads_only = [
        "assignment_" + name for name in beads_only_samples
    ]
    value_columns = [c for c in result_df if c.startswith("centered_")]

    result_df["sample_hits"] = (result_df[assignment_columns] > 0.5).sum(1)
    result_df["sample_hits_beads_only"] = (
        result_df[assignment_columns_beads_only] > 0.5).sum(1)
    result_df["gap"] = (result_df.m2_u2 - result_df.m2_u1) / result_df.m1_s
    assert (result_df.gap >= 0).all(), result_df.gap

    for gap in [0, 1, 2, 3]:
        sub_df = result_df.loc[result_df.gap >= gap].sort_values(
            "aic_diff", ascending=False).copy()
        sub_df["sample_hits_beads_only_cumulative"] = (
            sub_df.sample_hits_beads_only.cumsum())

        sub_df["fdr"] = (
            sub_df.sample_hits_beads_only_cumulative /
            ((1 + numpy.arange(len(sub_df))) * len(beads_only_samples))
        )

        sub_df = sub_df.loc[sub_df.sample_hits_beads_only == 0].copy()
        sub_df["sample_hits_cumulative"] = sub_df.sample_hits.cumsum()
        for col in sub_df.columns[len(result_df.columns):]:
            result_df["gap%d_%s" % (gap, col)] = sub_df[col]

    analysis_columns = [
        c for c in result_df
        if c not in assignment_columns and c not in value_columns
    ]
    result_df = result_df[analysis_columns + assignment_columns + value_columns]
    return result_df