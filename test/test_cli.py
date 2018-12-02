from nose.tools import assert_greater, assert_less

import pandas as pd
import numpy as np
from numpy.random import normal

from click.testing import CliRunner

from phip import cli


def invoke_and_assert_success(runner, command, args):
    result = runner.invoke(command, args)
    if result.exit_code != 0:
        print("Command failed [exit code %d]: %s %s" % (
            result.exit_code, str(command), str(args)))
        print(result.output)
        raise result.exception
    return result


def test_clipped_factorization_model():
    runner = CliRunner()
    with runner.isolated_filesystem():
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

        # No truncation (truncate-percentile > 100.0)
        # Here we expect that the background effects will be used to model the
        # corrupted entry.
        command_result = invoke_and_assert_success(
            runner,
            cli.clipped_factorization_model, [
                "-i", "input.tsv",
                "-o", "output.tsv",
                "--learning-rate", "100.0",
                "--max-epochs", "10000",
                "--patience", "10",
                "--rank", "1",
                "--truncate-percentile", "101",
                "--discard-sample-reads-fraction", "0.0",
                "--no-normalize-to-reads-per-million",
        ])
        print(command_result.output)
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

        # With truncation. Here we expect the corrupted entry to be excluded
        # and have little impact on the background effects.
        command_result = invoke_and_assert_success(
            runner,
            cli.clipped_factorization_model, [
                "-i", "input.tsv",
                "-o", "output.tsv",
                "--learning-rate", "100.0",
                "--max-epochs", "10000",
                "--patience", "10",
                "--rank", "1",
                "--truncate-percentile", "99.9",
                "--discard-sample-reads-fraction", "0.0",
                "--no-normalize-to-reads-per-million",
            ])
        print(command_result.output)
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
            background_df.values.flatten().mean() / 10)



