from nose.tools import assert_greater, assert_less

import pandas as pd
import numpy as np
from numpy.random import normal

from click.testing import CliRunner

from phip import cli


def invoke_and_assert_success(runner, command, args):
    result = runner.invoke(command, args)
    print(result.output)
    if result.exit_code != 0:
        print("Command failed [exit code %d]: %s %s" % (
            result.exit_code, str(command), str(args)))
        raise result.exception
    return result


def test_clipped_factorization_model():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # We generate synthetic data by taking the outer product of two random
        # vectors and then corrupting a single element. This should be well
        # modeled by our model using rank-1 factorization matrices.

        clone_backgrounds = normal(0, 10, (int(1e4),1))**2
        sample_backgrounds = normal(0, 10, (int(1e1),1))**2
        background = np.outer(clone_backgrounds, sample_backgrounds)
        background_df = pd.DataFrame(
            background,
            index=["clone_%d" % (i + 1) for i in range(background.shape[0])],
            columns=["sample_%d" % (i + 1) for i in range(background.shape[1])])

        data_df = background_df.copy()

        # Corrupt one entry.
        data_df.iloc[0, 0] = 1e10

        data_df.to_csv("input.tsv", sep="\t", index=True)

        # In the first test, we disable clipping (clip-percentile > 100) and
        # test that the learned background effects try to account for the
        # corrupted element. In the second test we lower the clip-percentile
        # and verify that the corrupted element is ignored.
        # ***
        # FIRST TEST: No clipping (clip-percentile > 100.0)
        command_result = invoke_and_assert_success(
            runner,
            cli.clipped_factorization_model, [
                "-i", "input.tsv",
                "-o", "output.tsv",
                "--learning-rate", "100.0",
                "--max-epochs", "10000",
                "--patience", "10",
                "--rank", "1",
                "--clip-percentile", "101.0",
                "--discard-sample-reads-fraction", "0.0",
                "--no-normalize-to-reads-per-million",
        ])
        result_df = pd.read_table("output.tsv", index_col=0)
        print(result_df)

        # Outlier entry is accounted for by background effects. The background
        # effects are high and the entry itself (the residual) is low.
        assert_greater(
            (
                result_df.loc["clone_1", "_background_0"] *
                result_df.loc["_background_0", "sample_1"]),
            1e7)
        assert_less(result_df.iloc[0, 0], 1e9)

        # ***
        # SECOND TEST: with clipping.
        command_result = invoke_and_assert_success(
            runner,
            cli.clipped_factorization_model, [
                "-i", "input.tsv",
                "-o", "output.tsv",
                "--learning-rate", "100.0",
                "--max-epochs", "10000",
                "--patience", "10",
                "--rank", "1",
                "--clip-percentile", "99.9",
                "--discard-sample-reads-fraction", "0.0",
                "--no-normalize-to-reads-per-million",
            ])
        result_df = pd.read_table("output.tsv", index_col=0)
        print(result_df)
        result_samples_df = result_df.loc[
            [c for c in background_df.index if not c.startswith("_")],
            [s for s in background_df.columns if not s.startswith("_")],
        ].copy()

        # Outlier entry is NOT accounted for by background effects.
        assert_less(
            (
                result_df.loc["clone_1", "_background_0"] *
                result_df.loc["_background_0", "sample_1"]),
            1e6)

        assert_greater(result_samples_df.iloc[0, 0], 1e9)
        result_samples_df.iloc[0, 0] = 0.0

        # Overall most entries should be accounted for well by background, so
        # mean of the entries should be much less in the processed matrix than in
        # the background matrix.
        assert_less(
            result_samples_df.values.flatten().mean(),
            background_df.values.flatten().mean() / 2)


def test_call_hits():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Check that hit calling accuracy is reasonable, both with and without
        # the clipped factorization preprocessing step, on a simulated dataset.
        #
        # We simulate background noise by first sampling a random mean for each
        # phage clone (square of a normal distribution). We then sample beads-
        # only and pull down (i.e. non-beads-only) count values from a poisson
        # distribution centered at the given per-clone means. Finally, we
        # corrupt a set number of entries, corresponding to our true hits.
        #
        num_clones = 10000
        num_beads_only = 16
        num_pull_down = 100
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

        data_df.to_csv("input.tsv", sep="\t", index=True)

        # Invocation 1: WITHOUT clipped factorization pre-processing step.
        invoke_and_assert_success(
            runner,
            cli.call_hits, [
                "-i", "input.tsv",
                "-o", "hits.no_background_model.tsv",
                "--fdr", "0.15",
        ])

        # Run clipped factorization pre processing step.
        invoke_and_assert_success(
            runner,
            cli.clipped_factorization_model, [
                "-i", "input.tsv",
                "-o", "residuals.tsv",
                "--max-epochs", "100",
                "--discard-sample-reads-fraction", "0.0"
        ])

        # Invocation 2: WITH clipped factorization pre-processing step.
        invoke_and_assert_success(
            runner,
            cli.call_hits, [
                "-i", "residuals.tsv",
                "-o", "hits.with_background_model.tsv",
                "--fdr", "0.15",
        ])
        filenames = [
            "hits.no_background_model.tsv",
            "hits.with_background_model.tsv"
        ]

        # Check accuracy of both hit calling runs.
        for filename in filenames:
            print("Checking hit calling results: %s" % filename)
            result_df = pd.read_table(filename, index_col=0)
            for col in result_df:
                if col.startswith("_background"):
                    del result_df[col]
            result_df = result_df.loc[
                ~result_df.index.str.startswith("_background")
            ]
            print(result_df)

            result_df_with_actual_hits_zeroed_out = result_df.copy()
            hit_values = []
            for (sample, clone) in hit_pairs:
                hit_values.append(result_df.loc[clone, sample])
                result_df_with_actual_hits_zeroed_out.loc[clone, sample] = 0.0
            hit_values = pd.Series(hit_values, index=hit_pairs)

            sensitivity = (hit_values > 1.0).mean()
            total_hits_called = (result_df > 1.0).sum().sum()
            num_hit_calls_that_are_wrong = (
                result_df_with_actual_hits_zeroed_out > 1.0).sum().sum()
            num_actual_negatives = num_pull_down * num_hits - len(hit_pairs)
            specificity = 1 - (num_hit_calls_that_are_wrong / num_actual_negatives)
            empirical_fdr = num_hit_calls_that_are_wrong / total_hits_called

            print(
                "Sensitivity: %0.5f, specificity [%d false hits]: %0.5f, "
                "empirical fdr: %0.5f" % (
                    sensitivity,
                    num_hit_calls_that_are_wrong,
                    specificity,
                    empirical_fdr))

            assert_greater(sensitivity, 0.8)
            assert_greater(specificity, 0.8)
            assert_less(empirical_fdr, 0.3)
