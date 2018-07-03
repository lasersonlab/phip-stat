import time

import pandas
import numpy

import tensorflow as tf
from tensorflow.contrib.distributions import percentile


def do_clipped_factorization_analysis(
        counts_df,
        rank=3,
        truncate_percentile=99.9,
        learning_rate=1.0,
        minibatch_size=1024 * 32,
        patience=5,
        max_epochs=1000,
        log_every_seconds=10):
    """
    Implements the factorization:

    X = AB
        where X is (n x s), A is (n x r), and B is (r x s)

    by minimizing the clipped loss:

        ||minimum(X - AB, percentile(X - AB, q)||_2 + unclipped_term

    The minimum is taken elementwise, and ||M||_2 is the Frobenius norm.
    q is a parameter giving the percentile to clip at, default value 99.9. This
    truncation makes the factorization robust to outliers, which are our hits.

    If the above is optimized without an `unclipped_term`, a few phage clones
    may have all of their residuals above the truncation threshold. Once this
    happens they will likely stay stuck there since they do not contribute to
    the gradient. The `unclipped_term` fixes this by providing a small nudge
    toward smaller errors without truncation. See implementation for details.
    """

    # Non-tf setup
    reads_per_million = (counts_df * 1e6 / counts_df.sum(0)).astype("float32")
    (n, s) = reads_per_million.shape

    # Placeholders
    target = tf.placeholder(
        name="target", dtype="float32", shape=[None, s])
    minibatch_indices = tf.placeholder(
        name="minibatch_indices", dtype="int32")

    # Variables
    a = tf.Variable(
        numpy.random.rand(n, rank), name="A", dtype="float32")
    b = tf.Variable(
        numpy.random.rand(rank, s), name="B", dtype="float32")
    truncate_threshold = tf.Variable(reads_per_million.max().max())

    # Derived quantities
    reconstruction = tf.matmul(
        tf.gather(a, minibatch_indices), b)
    differences = target - reconstruction

    # unclipped_term is based only on the minimum unclipped error for each
    # clone. The intuition is that we know for every clone at least one sample
    # must be a non-hit (e.g. a beads only sample), and so should be well modeled
    # by the background process.
    unclipped_term = tf.reduce_min(
        tf.pow(differences, 2), axis=1)
    loss = (
        tf.reduce_mean(
            tf.pow(tf.minimum(differences, truncate_threshold), 2))
        + tf.reduce_mean(unclipped_term) / s)

    update_truncate_value = truncate_threshold.assign(
        percentile(differences, truncate_percentile))

    # Training
    train_step = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    best_cost_value = None
    last_log_at = 0
    with tf.Session() as session:
        session.run(init)
        all_indices = numpy.array(list(range(reads_per_million.shape[0])))

        for i in range(max_epochs):
            indices = numpy.array(list(range(reads_per_million.shape[0])))
            numpy.random.shuffle(indices)
            for minibatch_indices_value in numpy.array_split(indices, int(
                            len(indices) / minibatch_size)):
                minibatch_indices_value = minibatch_indices_value[:minibatch_size]
                if len(minibatch_indices_value) == minibatch_size:
                    feed_dict = {
                        target: reads_per_million.values[minibatch_indices_value],
                        minibatch_indices: minibatch_indices_value
                    }
                    session.run(train_step, feed_dict=feed_dict)

            feed_dict = {
                target: reads_per_million,
                minibatch_indices: all_indices,
            }
            (truncate_threshold_value, cost_value) = session.run(
                [update_truncate_value, loss], feed_dict=feed_dict)

            # Update best epoch
            if best_cost_value is None or cost_value < best_cost_value:
                best_cost_value = cost_value
                best_epoch = i
                (best_a, best_b) = session.run([a, b], feed_dict=feed_dict)

            # Log
            if log_every_seconds and time.time() - last_log_at > log_every_seconds:
                print("[Epoch %5d] %f, truncating at %f%s" % (
                    i,
                    cost_value,
                    truncate_threshold_value,
                    ' [new best]' if i == best_epoch else ''))

            # Stop criterion
            if i - best_epoch > patience:
                print("Early stopping at epoch %d." % i)
                break

    background_names = ["_background_%d" % i for i in range(rank)]
    best_a = pandas.DataFrame(
        best_a,
        index=reads_per_million.index,
        columns=background_names)
    best_b = pandas.DataFrame(
        best_b,
        index=background_names,
        columns=reads_per_million.columns)

    results = reads_per_million - numpy.matmul(best_a, best_b)
    for name in background_names:
        results[name] = best_a[name]
        results.loc[name] = best_b.loc[name]

    return results


