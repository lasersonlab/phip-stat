import time

import numpy as np
import pandas as pd

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow_probability.python.stats.quantiles import percentile


def do_clipped_factorization(
    counts_df,
    rank=3,
    clip_percentile=99.9,
    learning_rate=1.0,
    minibatch_size=1024 * 32,
    patience=5,
    max_epochs=1000,
    normalize_to_reads_per_million=True,
    log_every_seconds=10,
):
    """
    Attempt to detect and correct for clone and sample batch effects by
    subtracting off a learned low-rank reconstruction of the counts matrix.

    The return value is the clones x samples matrix of residuals after
    correcting for batch effects, with a few additional rows and columns giving
    the learned background effects.

    Implements the factorization:

    X = AB
        where X is (clones x samples), A is (clones x rank), and B is
        (rank x samples)

    by minimizing the "clipped" loss:

        ||minimum(X - AB, percentile(X - AB, clip_percentile)||_2 + unclipped

    The minimum is taken elementwise, and ||...||_2 is the Frobenius norm.
    clip_percentile is a parameter giving the percentile to clip at. The
    clipping makes the factorization robust to outliers, some of which are
    likely phip-seq hits.

    If the above is optimized without an `unclipped` term, a few phage clones
    may have all of their residuals above the truncation threshold. Once this
    happens they will likely stay stuck there since they do not contribute to
    the gradient. The unclipped term fixes this by providing a small nudge
    toward smaller errors without truncation.

    Note that "beads-only" samples are not treated in any special way here.

    The optimization is performed using stochastic gradient descent (SGD) on
    tensorflow.

    Parameters
    ----------
    counts_df : pandas.DataFrame
        Matrix of read counts (clones x samples)
    rank : int
        Rank of low-dimensional background effect matrices A and B
    clip_percentile : float
        Elements with reconstruction errors above this percentile do not
        contribute to the gradient. Aim for a lower-bound on the fraction
        of entries you expect NOT to be hits.
    learning_rate : float
        SGD optimizer learning rate
    minibatch_size : int
        Number of rows per SGD minibatch
    patience : int
        Number of epochs without improvement in training loss to tolerate before
        stopping
    max_epochs : int
        Maximum number of epochs
    normalize_to_reads_per_million : boolean
        Before computing factorization, first divide each column by the total
        number of reads for that sample and multiple by 1 million.
    log_every_seconds : float
        Seconds to wait before printing another optimization status update

    Returns
    -------
    pandas.DataFrame : residuals after correcting for batch effects

    In addition to the clones x samples residuals, rows and columns named
    "_background_0", "_background_1", ... giving the learned background vectors
    are also included.
    """

    # Non-tf setup
    if normalize_to_reads_per_million:
        observed = (counts_df * 1e6 / counts_df.sum(0)).astype("float32")
    else:
        observed = counts_df.astype("float32")
    (n, s) = observed.shape
    if len(counts_df) < minibatch_size:
        minibatch_size = len(counts_df)

    # Placeholders
    target = tf.placeholder(name="target", dtype="float32", shape=[None, s])
    minibatch_indices = tf.placeholder(name="minibatch_indices", dtype="int32")

    # Variables
    a = tf.Variable(np.random.rand(n, rank), name="A", dtype="float32")
    b = tf.Variable(np.random.rand(rank, s), name="B", dtype="float32")
    clip_threshold = tf.Variable(observed.max().max())

    # Derived quantities
    reconstruction = tf.matmul(tf.gather(a, minibatch_indices), b)
    differences = target - reconstruction

    # unclipped_term is based only on the minimum unclipped error for each
    # clone. The intuition is that we know for every clone at least one sample
    # must be a non-hit (e.g. a beads only sample), and so should be well modeled
    # by the background process.
    unclipped_term = tf.reduce_min(tf.pow(differences, 2), axis=1)
    loss = (
        tf.reduce_mean(tf.pow(tf.minimum(differences, clip_threshold), 2))
        + tf.reduce_mean(unclipped_term) / s
    )

    update_clip_value = clip_threshold.assign(percentile(differences, clip_percentile))

    # Training
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    best_cost_value = None
    last_log_at = 0
    with tf.Session() as session:
        session.run(init)
        all_indices = np.arange(observed.shape[0], dtype=int)

        for i in range(max_epochs):
            indices = np.array(list(range(observed.shape[0])))
            np.random.shuffle(indices)
            for minibatch_indices_value in np.array_split(
                indices, int(len(indices) / minibatch_size)
            ):
                minibatch_indices_value = minibatch_indices_value[:minibatch_size]
                if len(minibatch_indices_value) == minibatch_size:
                    feed_dict = {
                        target: observed.values[minibatch_indices_value],
                        minibatch_indices: minibatch_indices_value,
                    }
                    session.run(train_step, feed_dict=feed_dict)

            feed_dict = {target: observed, minibatch_indices: all_indices}
            (clip_threshold_value, cost_value) = session.run(
                [update_clip_value, loss], feed_dict=feed_dict
            )

            # Update best epoch
            if best_cost_value is None or cost_value < best_cost_value:
                best_cost_value = cost_value
                best_epoch = i
                (best_a, best_b) = session.run([a, b], feed_dict=feed_dict)

            # Log
            if log_every_seconds and time.time() - last_log_at > log_every_seconds:
                print(
                    "[Epoch %5d] %f, truncating at %f%s"
                    % (
                        i,
                        cost_value,
                        clip_threshold_value,
                        " [new best]" if i == best_epoch else "",
                    )
                )

            # Stop criterion
            if i - best_epoch > patience:
                print("Early stopping at epoch %d." % i)
                break

    background_names = ["_background_%d" % i for i in range(rank)]
    best_a = pd.DataFrame(best_a, index=observed.index, columns=background_names)
    best_b = pd.DataFrame(best_b, index=background_names, columns=observed.columns)

    results = observed - np.dot(best_a, best_b)
    for name in background_names:
        results[name] = best_a[name]
        results.loc[name] = best_b.loc[name]

    return results
