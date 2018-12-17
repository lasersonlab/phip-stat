from nose.tools import assert_greater, assert_less

import collections
import time
import cProfile
import pstats

import pandas as pd
import numpy as np
from numpy.random import normal


from phip import hit_calling


def test_speed(profile=False):
    starts = collections.OrderedDict()
    timings = collections.OrderedDict()
    profilers = collections.OrderedDict()

    def start(name):
        starts[name] = time.time()
        if profile:
            profilers[name] = cProfile.Profile()
            profilers[name].enable()

    def end(name):
        timings[name] = time.time() - starts[name]
        if profile:
            profilers[name].disable()

    num_clones = 100000
    num_beads_only = 8
    num_pull_down = 200
    num_hits = 1000

    data_df = pd.DataFrame(index=[
        "clone_%d" % (i + 1) for i in range(num_clones)
    ])
    means = np.random.normal(0, 10, num_clones)**2
    for i in range(num_beads_only):
        data_df["beads_only_%d" % (i + 1)] = np.random.poisson(means)
    for i in range(num_pull_down):
        data_df["pull_down_%d" % (i + 1)] = np.random.poisson(means)

    beads_only_samples = [c for c in data_df if c.startswith("beads")]
    pull_down_samples = [c for c in data_df if c.startswith("pull_down")]

    # Add some hits
    hit_pairs = set()  # set of (sample, clone)
    while len(hit_pairs) < num_hits:
        sample = np.random.choice(pull_down_samples)
        clone = np.random.choice(data_df.index)
        data_df.loc[clone, sample] = (
            data_df.loc[clone, sample]**2 + 100)
        hit_pairs.add((sample, clone))

    # Also to test that normalization works, make one beads-only sample way
    # bigger than th eother columns.
    data_df["beads_only_1"] *= 1e6

    start("hit_calling")
    hit_calling.do_hit_calling(data_df, beads_only_samples)
    end("hit_calling")

    print("SPEED BENCHMARK")
    print("Results:\n%s" % str(pd.Series(timings)))

    return dict(
        (key, pstats.Stats(value)) for (key, value) in profilers.items())


if __name__ == '__main__':
    # If run directly from python, do profiling and leave the user in a shell
    # to explore results.

    result = test_speed(profile=True)

    for (name, stats) in result.items():
        print("**** %s ****" % name)
        stats.sort_stats("cumtime").reverse_order().print_stats()
        print("")

    # Leave in ipython
    # locals().update(result)
    # import ipdb  # pylint: disable=import-error
    # ipdb.set_trace()