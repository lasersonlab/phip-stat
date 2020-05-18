# Copyright 2016 Uri Laserson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gzip
import json
import os
import pathlib
import re
import sys
from collections import Counter, OrderedDict
from functools import reduce
from glob import glob
from os import path as osp
from os.path import join as pjoin
from subprocess import PIPE, Popen

import numpy as np
import pandas as pd
from click import Choice, Path, command, group, option
from tqdm import tqdm

from phip.utils import (
    DEFAULT_FDR,
    DEFAULT_REFERENCE_QUANTILE,
    compute_size_factors,
    readfq)


# handle gzipped or uncompressed files
def open_maybe_compressed(*args, **kwargs):
    if args[0].endswith(".gz"):
        # gzip modes are different from default open modes
        if len(args[1]) == 1:
            args = (args[0], args[1] + "t") + args[2:]
        compresslevel = kwargs.pop("compresslevel", 6)
        return gzip.open(*args, **kwargs, compresslevel=compresslevel)
    else:
        return open(*args, **kwargs)


@group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """phip -- PhIP-seq analysis tools"""
    pass


@cli.command(name="truncate-fasta")
@option(
    "-i",
    "--input",
    required=True,
    type=Path(exists=True, dir_okay=False),
    help="input fasta",
)
@option("-o", "--output", required=True, type=Path(exists=False), help="output fasta")
@option(
    "-k",
    "--length",
    required=True,
    type=int,
    help="length of starting subsequence to extract",
)
def truncate_fasta(input, output, length):
    """Truncate each sequence of a fasta file."""
    with open(input, "r") as ip, open(output, "w") as op:
        for (n, s, q) in readfq(ip):
            print(f">{n}\n{s[:length]}", file=op)


@cli.command(name="merge-kallisto-tpm")
@option(
    "-i",
    "--input",
    required=True,
    type=Path(exists=True, file_okay=False),
    help="input dir containing kallisto results",
)
@option("-o", "--output", required=True, type=Path(exists=False), help="output path")
def merge_kallisto_tpm(input, output):
    """Merge kallisto abundance results.

    Input directory should contain sample-named subdirectories, each containing
    an abundance.tsv file.  This command will generate a single tab-delim
    output file with each column containing the tpm values for that sample.
    """
    samples = os.listdir(input)
    iterators = [open(pjoin(input, s, "abundance.tsv"), "r") for s in samples]
    with open(output, "w") as op:
        it = zip(*iterators)
        # burn headers of input files and write header of output file
        _ = next(it)
        print("id\t{}".format("\t".join(samples)), file=op)
        for lines in it:
            fields_array = [line.split("\t") for line in lines]
            # check that join column is the same
            assert all([fields[0] == fields_array[0][0] for fields in fields_array])
            merged_fields = [fields_array[0][0]] + [f[4].strip() for f in fields_array]
            print("\t".join(merged_fields), file=op)


@cli.command(name="gamma-poisson-model")
@option(
    "-i",
    "--input",
    required=True,
    type=Path(exists=True, dir_okay=False),
    help="input counts file (tab-delim)",
)
@option(
    "-o", "--output", required=True, type=Path(exists=False), help="output directory"
)
@option(
    "-t",
    "--trim-percentile",
    default=99.9,
    help="lower percent of data to keep for model fitting",
)
@option(
    "-d", "--index-cols", default=1, help="number of columns to use as index/row-key"
)
def gamma_poisson_model(input, output, trim_percentile, index_cols):
    """Fit a gamma-poisson model.

    Compute -log10(pval) for each (possibly-normalized) count.
    """
    from phip.gampois import gamma_poisson_model as model

    counts = pd.read_csv(input, sep="\t", header=0, index_col=list(range(index_cols)))
    os.makedirs(output, exist_ok=True)
    alpha, beta, rates, mlxp = model(counts, trim_percentile)
    with open(pjoin(output, "parameters.json"), "w") as op:
        json.dump(
            {
                "alpha": alpha,
                "beta": beta,
                "trim_percentile": trim_percentile,
                "background_rates": list(rates),
            },
            op,
        )
    mlxp.to_csv(pjoin(output, "mlxp.tsv"), sep="\t", float_format="%.2f")


@cli.command(name="clipped-factorization-model")
@option(
    "-i",
    "--input",
    required=True,
    type=Path(exists=True, dir_okay=False),
    help="input counts file (tab-delim)",
)
@option(
    "-o",
    "--output",
    required=False,
    type=Path(exists=False),
    help="output file or directory. If ends in .tsv, will be treated as file",
)
@option(
    "-d", "--index-cols", default=1, help="number of columns to use as index/row-key"
)
@option("--rank", default=3, show_default=True, help="matrix rank")
@option(
    "--clip-percentile",
    default=99.9,
    show_default=True,
    help="percentile thershold to clip at",
)
@option(
    "--learning-rate",
    default=1.0,
    show_default=True,
    help="learning rate for Adam optimizer",
)
@option(
    "--minibatch-size", default=1024 * 32, show_default=True, help="rows per minibatch"
)
@option(
    "--patience",
    default=5,
    show_default=True,
    help="number of epochs of no improvement to wait before early stopping",
)
@option("--max-epochs", default=1000, show_default=True, help="maximum epochs")
@option(
    "--discard-sample-reads-fraction",
    default=0.01,
    show_default=True,
    help="Discard samples with fewer than X * m reads, where m is the median "
    "number of reads across samples",
)
@option(
    "--no-normalize-to-reads-per-million",
    is_flag=True,
    help="Work directly on read counts, not counts divided by sample totals",
)
@option(
    "--log-every-seconds",
    default=1,
    show_default=True,
    help="write progress no more often than every N seconds",
)
def clipped_factorization_model(
    input,
    output,
    index_cols,
    rank,
    clip_percentile,
    learning_rate,
    minibatch_size,
    patience,
    max_epochs,
    discard_sample_reads_fraction,
    no_normalize_to_reads_per_million,
    log_every_seconds,
):
    """Fit matrix factorization model.

    Computes residuals from a matrix factorization model. Specifically, attempt
    to detect and correct for clone and sample batch effects by subtracting off
    a learned low-rank reconstruction of the given counts matrix.

    The result is the (clones x samples) matrix of residuals after correcting for
    batch effects. A few additional rows and columns (named _background_0,
    _background_1, ...) giving the learned effects are also included.
    """
    from phip.clipped_factorization import do_clipped_factorization

    counts = pd.read_csv(input, sep="\t", header=0, index_col=list(range(index_cols)))

    total_reads = counts.sum()
    expected_reads = total_reads.median()
    for sample in counts.columns:
        if total_reads[sample] / expected_reads < discard_sample_reads_fraction:
            print(
                "[!!] EXCLUDING SAMPLE %s DUE TO INSUFFICIENT READS "
                "(%d vs. sample median %d)"
                % (sample, total_reads[sample], expected_reads)
            )
            del counts[sample]

    result_df = do_clipped_factorization(
        counts,
        rank=rank,
        clip_percentile=clip_percentile,
        learning_rate=learning_rate,
        minibatch_size=minibatch_size,
        patience=patience,
        max_epochs=max_epochs,
        normalize_to_reads_per_million=not no_normalize_to_reads_per_million,
        log_every_seconds=log_every_seconds,
    )

    if output.endswith(".tsv"):
        output_path = output
    else:
        os.makedirs(output)
        output_path = pjoin(output, "mixture.tsv")
    result_df.to_csv(output_path, sep="\t", float_format="%.2f")
    print("Wrote: %s" % output_path)


@cli.command(name="call-hits")
@option(
    "-i",
    "--input",
    required=True,
    type=Path(exists=True, dir_okay=False),
    help="input counts file (tab-delim)",
)
@option(
    "-o",
    "--output",
    required=False,
    type=Path(exists=False),
    help="output file or directory. If ends in .tsv, will be treated as file",
)
@option(
    "-d", "--index-cols", default=1, help="number of columns to use as index/row-key"
)
@option(
    "--beads-regex",
    default=".*beads.*",
    show_default=True,
    help="samples with names matching this regex are considered beads-only",
)
@option(
    "--ignore-columns-regex",
    default="^_background.*",
    show_default=True,
    help="ignore columns matching the given regex (evaluated in case-insensitive"
    " mode.) Ignored columns are passed through to output without processing.",
)
@option(
    "--ignore-rows-regex",
    default="^_background.*",
    show_default=True,
    help="ignore rows matching the given regex (evaluated in case-insensitive "
    "mode). Ignored rows are passed through to output without processing.",
)
@option(
    "--fdr", default=DEFAULT_FDR, show_default=True, help="target false discovery rate"
)
@option(
    "--reference-quantile",
    default=DEFAULT_REFERENCE_QUANTILE,
    show_default=True,
    help="Percentile to take of each clone's beads-only samples."
)
@option(
    "--discard-sample-reads-fraction",
    default=0.01,
    show_default=True,
    help="Discard samples with fewer than X * m reads, where m is the median "
    "number of reads across samples",
)
@option(
    "--normalize-to-reads-per-million",
    type=Choice(["always", "never", "guess"]),
    default="guess",
    show_default=True,
    help="Divide counts by totals per sample. Recommended "
    "when running directly on raw read counts (as opposed to matrix "
    'factorization residuals). If set to "guess" then the counts matrix '
    "will be left as-is if it contains negative entries, and otherwise "
    "will be normalized.",
)
@option(
    "--verbosity",
    default=2,
    show_default=True,
    help="verbosity: no output (0), result summary only (1), or progress (2)",
)
def call_hits(
    input,
    output,
    index_cols,
    beads_regex,
    ignore_columns_regex,
    ignore_rows_regex,
    fdr,
    reference_quantile,
    discard_sample_reads_fraction,
    normalize_to_reads_per_million,
    verbosity,
):
    """Call hits at specified FDR using a heuristic.

    Either raw read counts or the result of the clipped-factorization-model
    sub-command can be provided.

    The result is a matrix of shape (clones x samples). Entries above 1.0 in
    this matrix indicate hits. Higher values indicate more evidence for a
    hit, but there is no simple interpretation of these values beyond whether
    they are below/above 1.0.

    See the documentation for `hit_calling.do_hit_calling()` for details on
    the implementation.
    """
    from phip.hit_calling import do_hit_calling

    original_counts = pd.read_csv(
        input, sep="\t", header=0, index_col=list(range(index_cols))
    )
    counts = original_counts
    print("Read input matrix: %d clones x %d samples." % counts.shape)
    print("Columns: %s" % " ".join(counts.columns))

    columns_to_ignore = [
        s
        for s in counts.columns
        if ignore_columns_regex
        and re.match(ignore_columns_regex, s, flags=re.IGNORECASE)
    ]
    if columns_to_ignore:
        print(
            "Ignoring %d columns matching regex '%s': %s"
            % (
                len(columns_to_ignore),
                ignore_columns_regex,
                " ".join(columns_to_ignore),
            )
        )
        counts = counts[[c for c in counts.columns if c not in columns_to_ignore]]

    rows_to_ignore = [
        s
        for s in counts.index
        if ignore_rows_regex
        and index_cols == 1
        and re.match(ignore_rows_regex, s, flags=re.IGNORECASE)
    ]
    if rows_to_ignore:
        print(
            "Ignoring %d rows matching regex '%s': %s"
            % (len(rows_to_ignore), ignore_rows_regex, " ".join(rows_to_ignore))
        )
        counts = counts.loc[~counts.index.isin(rows_to_ignore)]

    total_reads = counts.sum()
    expected_reads = total_reads.median()

    if (counts > 0).all().all():
        for sample in counts.columns:
            if total_reads[sample] / expected_reads < discard_sample_reads_fraction:
                print(
                    "[!!] EXCLUDING SAMPLE %s DUE TO INSUFFICIENT READS "
                    "(%d vs. sample median %d)"
                    % (sample, total_reads[sample], expected_reads)
                )
                del counts[sample]

    beads_only_samples = [
        s for s in counts.columns if re.match(beads_regex, s, flags=re.IGNORECASE)
    ]
    print(
        "Beads-only regex '%s' matched %d samples: %s"
        % (beads_regex, len(beads_only_samples), " ".join(beads_only_samples))
    )

    result_df = do_hit_calling(
        counts,
        beads_only_samples=beads_only_samples,
        reference_quantile=reference_quantile,
        fdr=fdr,
        normalize_to_reads_per_million={"always": True, "never": False, "guess": None}[
            normalize_to_reads_per_million
        ],
        verbosity=verbosity,
    )

    full_result_df = original_counts.copy()
    for column in result_df.columns:
        full_result_df.loc[result_df.index, column] = result_df[column]

    if output.endswith(".tsv"):
        output_path = output
    else:
        os.makedirs(output)
        output_path = pjoin(output, "hits.tsv")
    full_result_df.to_csv(output_path, sep="\t", float_format="%.4f")
    print("Wrote: %s" % output_path)


# TOOLS THAT SHOULD BE USED RARELY


@cli.command(name="zip-reads-and-barcodes")
@option(
    "-i",
    "--input",
    type=Path(exists=True, dir_okay=False),
    required=True,
    help="reads fastq file",
)
@option(
    "-b",
    "--barcodes",
    type=Path(exists=True, dir_okay=False),
    required=True,
    help="indexes/barcodes fastq file",
)
@option(
    "-m",
    "--mapping",
    type=Path(exists=True, dir_okay=False),
    required=True,
    help="barcode to sample mapping (tab-delim, no header line)",
)
@option(
    "-o", "--output", type=Path(exists=False), required=True, help="output directory"
)
@option(
    "-z", "--compress-output", is_flag=True, help="gzip-compress output fastq files"
)
@option(
    "-n",
    "--no-wrap",
    is_flag=True,
    help="fastq inputs are not wrapped (i.e., 4 lines per record)",
)
def zip_reads_barcodes(input, barcodes, mapping, output, compress_output, no_wrap):
    """Zip reads with barcodes and split into files.

    Some older versions of the Illumina pipeline would not annotate the reads
    with their corresponding barcodes, but would leave the barcode reads in a
    separate fastq file. This tool will take both fastq files and will modify
    the main fastq record to add the barcode to the header line (in the same
    place Illumina would put it). It will the write one file per sample as
    provided in the mapping.

    This should only be necessary on older data files. Newer pipelines that use
    bcl2fastq2 or the "generate fastq" pipeline in Basespace (starting 9/2016)
    should not require this.

    This tool requires that the reads are presented in the same order in the
    two input files (which should be the case).

    This tool should be used very rarely.
    """
    from .utils import load_mapping, edit1_mapping

    if no_wrap:
        from .utils import read_fastq_nowrap as fastq_parser
    else:
        from .utils import readfq as fastq_parser
    os.makedirs(output, mode=0o755)
    input = osp.abspath(input)
    barcodes = osp.abspath(barcodes)

    # generate all possible edit-1 BCs
    bc2sample = edit1_mapping(load_mapping(mapping))

    with open_maybe_compressed(input, "r") as r_h, open_maybe_compressed(
        barcodes, "r"
    ) as b_h:
        # open file handles for each sample
        ext = "fastq.gz" if compress_output else "fastq"
        output_handles = {
            s: open_maybe_compressed(
                pjoin(output, "{s}.{ext}".format(s=s, ext=ext)), "w"
            )
            for s in set(bc2sample.values())
        }
        try:
            for (r_n, r_s, r_q), (b_n, b_s, b_q) in zip(
                tqdm(fastq_parser(r_h)), fastq_parser(b_h)
            ):
                assert r_n.split(maxsplit=1)[0] == b_n.split(maxsplit=1)[0]
                try:
                    print(
                        "@{r_n}\n{r_s}\n+\n{r_q}".format(r_n=r_n, r_s=r_s, r_q=r_q),
                        file=output_handles[bc2sample[b_s]],
                    )
                except KeyError:
                    continue
        finally:
            for h in output_handles.values():
                h.close()


@cli.command(name="merge-columns")
@option(
    "-i", "--input", required=True, help="input path (directory of tab-delim files)"
)
@option("-o", "--output", required=True, help="output path")
@option(
    "-m",
    "--method",
    type=Choice(["iter", "outer"]),
    default="iter",
    help="merge/join method",
)
@option(
    "-p",
    "--position",
    type=int,
    default=1,
    help="the field position to merge (0-indexed)",
)
@option(
    "-d", "--index-cols", default=1, help="number of columns to use as index/row-key"
)
def merge_columns(input, output, method, position, index_cols):
    """Merge tab-delimited files.

    input must be a directory containing `.tsv` files to merge.

    method: iter -- concurrently iterate over lines of all files; assumes
                    row-keys are identical in each file

    method: outer -- bona fide outer join of data in each file; loads all files
                     into memory and joins using pandas
    
    method: prealloc -- preallocate an array to hold all values; then read each
                        file into the array
    """
    def load(path):
        icols = list(range(index_cols))
        ucols = icols + [position]
        return pd.read_csv(
            path, sep="\t", header=0, dtype=str, index_col=icols, usecols=ucols
        )
    input_dir = os.path.abspath(input)
    output_file = os.path.abspath(output)
    input_files = glob(pjoin(input_dir, "*.tsv"))
    if method == "iter":
        file_iterators = [open(f, "r") for f in input_files]
        file_headers = [osp.splitext(osp.basename(f))[0] for f in input_files]
        with open(output_file, "w") as op:
            # iterate through lines
            for lines in zip(*file_iterators):
                fields_array = [
                    [field.strip() for field in line.split("\t")] for line in lines
                ]
                # check that join column is the same
                for fields in fields_array[1:]:
                    assert fields_array[0][:index_cols] == fields[:index_cols]
                merged_fields = fields_array[0][:index_cols] + [
                    f[position] for f in fields_array
                ]
                print("\t".join(merged_fields), file=op)
    elif method == "outer":
        dfs = [load(path) for path in input_files]
        merge = lambda l, r: pd.merge(
            l, r, how="outer", left_index=True, right_index=True
        )
        df = reduce(merge, dfs).fillna(0)
        df.to_csv(output, sep="\t", float_format="%.2f")
    elif method == "prealloc":
        # iterate through just the first file to generate the row names
        with open(input_files[0], "r") as ip:
            row_names = ["\t".join(line.split("\t")[:index_cols]) for line in ip]
        data = np.zeros((len(row_names), len(input_files)))
        column_names = []
        def load_into(path, i):
            df = load(path)
            column_names.append(df.columns[0])
            data[:, i] = df.iloc[:, 0]
        for i, path in enumerate(input_files):
            load_into(path, i)
        df = pd.DataFrame(data, index=row_names, columns=column_names, copy=False)
        df.to_csv(output, sep="\t", float_format="%.2f")


@cli.command(name="normalize-counts")
@option("-i", "--input", required=True, help="input counts (tab-delim)")
@option("-o", "--output", required=True, help="output path")
@option(
    "-m",
    "--method",
    type=Choice(["col-sum", "size-factors"]),
    default="size-factors",
    help="normalization method",
)
@option(
    "-d", "--index-cols", default=1, help="number of columns to use as index/row-key"
)
def normalize_counts(input, output, method, index_cols):
    """Normalize count matrix.

    Two methods for normalizing are available:
    * Size factors from Anders and Huber 2010 (similar to TMM)
    * Normalize to constant column-sum of 1e6
    """
    df = pd.read_csv(input, sep="\t", header=0, index_col=list(range(index_cols)))
    if method == "col-sum":
        normalized = df / (df.sum() / 1e6)
    elif method == "size-factors":
        factors = compute_size_factors(df.values)
        normalized = df / factors
    normalized.to_csv(output, sep="\t", float_format="%.2f")


@cli.command(name="count-exact-matches")
@option(
    "-i",
    "--input",
    required=True,
    type=Path(exists=True, dir_okay=False),
    help="input fastq (gzipped ok)",
)
@option(
    "-o",
    "--output",
    required=True,
    type=Path(exists=False, dir_okay=False),
    help="output tsv",
)
@option(
    "-r",
    "--reference",
    required=True,
    type=Path(exists=True, dir_okay=False),
    help="path to reference (input) counts file (tab-delim)",
)
@option(
    "-l",
    "--read-length",
    required=True,
    type=int,
    help="read length (or, number of bases to use for matching)",
    metavar="<read-length>",
)
@option("--sample", type=str, help="sample name [default: filename stem]")
def count_exact_matches(input, output, reference, read_length, sample):
    """Match reads to reference exactly.

    Takes the first <read-length> bases of each read and attempt to match it
    exactly to the reference sequences. Computes the number of matches for each
    reference.
    """
    # load reference
    seq_to_ref = OrderedDict()
    with open(reference, "r") as ip:
        for (ref_name, seq, _) in readfq(ip):
            seq_to_ref[seq[:read_length]] = ref_name

    num_reads = 0
    num_matched = 0
    counts = Counter()
    with open_maybe_compressed(input, "r") as ip:
        for (name, seq, _) in tqdm(readfq(ip)):
            num_reads += 1
            refname = seq_to_ref.get(seq[:read_length])
            if refname is not None:
                num_matched += 1
                counts[refname] += 1

    print(
        "num_reads: {}\nnum_matched: {}\nfrac_matched: {}".format(
            num_reads, num_matched, num_matched / num_reads
        ),
        file=sys.stderr,
    )

    if not sample:
        sample = pathlib.Path(input).stem

    with open(output, "w") as op:
        print(f"id\t{sample}", file=op)
        for (_, refname) in seq_to_ref.items():
            print(f"{refname}\t{counts[refname]}", file=op)


# DEPRECATED TOOLS


@cli.command(name="split-fastq", deprecated=True)
@option("-i", "--input", required=True, help="input path (fastq file)")
@option("-o", "--output", required=True, help="output path (directory)")
@option("-n", "--chunk-size", type=int, required=True, help="number of reads per chunk")
def split_fastq(input, output, chunk_size):
    """Split fastq files into smaller chunks."""
    input_file = osp.abspath(input)
    output_dir = osp.abspath(output)
    os.makedirs(output_dir, mode=0o755)

    # convenience functions
    output_file = lambda i: pjoin(output_dir, "part.{0}.fastq".format(i))

    with open_maybe_compressed(input_file, "r") as input_handle:
        num_processed = 0
        file_num = 1
        for (name, seq, qual) in readfq(input_handle):
            if num_processed == 0:
                op = open_maybe_compressed(output_file(file_num), "w")
            print(f"@{name}\n{seq}\n+\n{qual}", file=op)
            num_processed += 1
            if num_processed == chunk_size:
                op.close()
                num_processed = 0
                file_num += 1
        if not op.closed:
            op.close()


@cli.command(name="align-parts", deprecated=True)
@option("-i", "--input", required=True, help="input path (directory of fastq parts)")
@option("-o", "--output", required=True, help="output path (directory)")
@option(
    "-x", "--index", required=True, help="bowtie index (e.g., as specified to bowtie2)"
)
@option(
    "-b",
    "--batch-submit",
    default="",
    help="batch submit command to prefix bowtie command invocation",
)
@option(
    "-p",
    "--threads",
    default=1,
    help="Number of threads to specify for each invocation of bowtie",
)
@option(
    "-3",
    "--trim3",
    default=0,
    help="Number of bases to trim off of 3-end (passed to bowtie)",
)
@option("-d", "--dry-run", is_flag=True, help="Dry run; print out commands to execute")
def align_parts(input, output, index, batch_submit, threads, trim3, dry_run):
    """Align fastq files to peptide reference."""
    input_dir = osp.abspath(input)
    output_dir = osp.abspath(output)
    if not dry_run:
        os.makedirs(output_dir, mode=0o755)
    bowtie_cmd_template = (
        "bowtie -n 3 -l 100 --best --nomaqround --norc -k 1 -p {threads} "
        "-3 {trim3} --quiet {index} {input} {output}"
    )
    for input_file in glob(pjoin(input_dir, "*.fastq")):
        output_file = pjoin(
            output_dir, osp.splitext(osp.basename(input_file))[0] + ".aln"
        )
        bowtie_cmd = bowtie_cmd_template.format(
            index=index,
            input=input_file,
            output=output_file,
            threads=threads,
            trim3=trim3,
        )
        submit_cmd = "{batch_cmd} {app_cmd}".format(
            batch_cmd=batch_submit, app_cmd=bowtie_cmd
        )
        if dry_run:
            print(submit_cmd.strip())
        else:
            p = Popen(
                submit_cmd.strip(), shell=True, stdout=PIPE, universal_newlines=True
            )
            print(p.communicate()[0])


@cli.command(name="compute-counts", deprecated=True)
@option("-i", "--input", required=True, help="input path (directory of aln files)")
@option("-o", "--output", required=True, help="output path (directory)")
@option(
    "-r",
    "--reference",
    required=True,
    help="path to reference (input) counts file (tab-delim)",
)
def compute_counts(input, output, reference):
    """Compute counts from aligned bam file."""
    input_dir = osp.abspath(input)
    output_dir = osp.abspath(output)
    os.makedirs(output_dir, mode=0o755)

    # load reference (i.e., input) counts
    ref_names = []
    ref_counts = []
    with open(reference, "r") as ip:
        # burn header
        _ = next(ip)
        for line in ip:
            fields = line.split("\t")
            ref_names.append(fields[0].strip())
            ref_counts.append(round(float(fields[1])))

    # compute count dicts
    for input_file in glob(pjoin(input_dir, "*.aln")):
        print(input_file)
        sys.stdout.flush()
        counts = {}
        sample = osp.splitext(osp.basename(input_file))[0]
        # accumulate counts
        with open(input_file, "r") as ip:
            for line in ip:
                ref_clone = line.split("\t")[2].strip()
                counts[ref_clone] = counts.get(ref_clone, 0) + 1
        # write counts
        output_file = pjoin(output_dir, sample + ".tsv")
        with open(output_file, "w") as op:
            print("id\tinput\t{0}".format(sample), file=op)
            for (ref_name, ref_count) in zip(ref_names, ref_counts):
                record = "{0}\t{1}\t{2}".format(
                    ref_name, ref_count, counts.get(ref_name, 0)
                )
                print(record, file=op)


@cli.command(name="gen-covariates", deprecated=True)
@option("-i", "--input", required=True, help="input path to merged count file")
@option(
    "-s", "--substring", required=True, help="substring to match against column names"
)
@option("-o", "--output", required=True, help="output file (recommend .tsv extension)")
def gen_covariates(input, substring, output):
    """Compute covariates for input to stat model.

    The input (`-i`) should be the merged counts file.  Each column name is
    matched against the given substring.  The median coverage-normalized value
    of each row from the matching columns will be output into a tab-delim file.
    This file can be used as the "reference" values for computing p-values.
    """
    input_file = osp.abspath(input)
    output_file = osp.abspath(output)
    counts = pd.read_csv(input_file, sep="\t", header=0, index_col=0)
    matched_columns = [col for col in counts.columns if substring in col]
    sums = counts[matched_columns].sum()
    normed = counts[matched_columns] / sums * sums.median()
    medians = normed.median(axis=1)
    medians.name = "input"
    medians.to_csv(output_file, sep="\t", header=True, index_label="id")


@cli.command(name="compute-pvals", deprecated=True)
@option("-i", "--input", required=True, help="input path")
@option("-o", "--output", required=True, help="output path")
@option(
    "-b",
    "--batch-submit",
    help="batch submit command to prefix pval command invocation",
)
@option(
    "-d",
    "--dry-run",
    is_flag=True,
    help="Dry run; print out commands to execute for batch submit",
)
def compute_pvals(input, output, batch_submit, dry_run):
    """Compute p-values from counts."""
    from .genpois import (
        estimate_GP_distributions,
        lambda_theta_regression,
        precompute_pvals,
    )

    if batch_submit is not None:
        # run compute-pvals on each file using batch submit command
        input_dir = osp.abspath(input)
        output_dir = osp.abspath(output)
        if not dry_run:
            os.makedirs(output_dir, mode=0o755)
        pval_cmd_template = "phip compute-pvals -i {input} -o {output}"
        for input_file in glob(pjoin(input_dir, "*.tsv")):
            sample = osp.splitext(osp.basename(input_file))[0]
            output_file = pjoin(output_dir, "{0}.pvals.tsv".format(sample))
            pval_cmd = pval_cmd_template.format(input=input_file, output=output_file)
            submit_cmd = "{batch_cmd} {app_cmd}".format(
                batch_cmd=batch_submit, app_cmd=pval_cmd
            )
            if dry_run:
                print(submit_cmd.strip())
            else:
                p = Popen(
                    submit_cmd.strip(), shell=True, stdout=PIPE, universal_newlines=True
                )
                print(p.communicate()[0])
    else:
        # actually compute p-vals on single file
        input_file = osp.abspath(input)
        output_file = osp.abspath(output)
        clones = []
        samples = None
        input_counts = []
        output_counts = []
        with open(input_file, "r") as ip:
            header_fields = next(ip).split("\t")
            samples = [f.strip() for f in header_fields[2:]]
            for line in tqdm(ip, desc="Loading data"):
                fields = line.split("\t")
                clones.append(fields[0].strip())
                input_counts.append(int(fields[1]))
                output_counts.append(np.int_(fields[2:]))
        input_counts = np.asarray(input_counts)
        # pseudocounts to combat negative regressed theta:
        output_counts = np.asarray(output_counts) + 1
        uniq_input_values = list(set(input_counts))

        # Estimate generalized Poisson distributions for every input count
        (lambdas, thetas, idxs) = estimate_GP_distributions(
            input_counts, output_counts, uniq_input_values
        )

        # Regression on all of the theta and lambda values computed
        (lambda_fits, theta_fits) = lambda_theta_regression(lambdas, thetas, idxs)

        # Precompute CDF for possible input-output combinations
        uniq_combos = []
        for i in range(output_counts.shape[1]):
            uniq_combos.append(set(zip(input_counts, output_counts[:, i])))
        log10pval_hash = precompute_pvals(lambda_fits, theta_fits, uniq_combos)

        # Compute p-values for each clone using regressed GP parameters
        with open(output_file, "w") as op:
            header = "\t".join(["id"] + samples)
            print(header, file=op)
            for (clone, ic, ocs) in zip(
                tqdm(clones, desc="Writing scores"), input_counts, output_counts
            ):
                fields = [clone]
                for (i, oc) in enumerate(ocs):
                    fields.append("{:.2f}".format(log10pval_hash[(i, ic, oc)]))
                print("\t".join(fields), file=op)
