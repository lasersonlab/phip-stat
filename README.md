# `phip-stat`: tools for analyzing PhIP-seq data

The PhIP-seq assay was first described in
[Larman et. al.](https://dx.doi.org/10.1038/nbt.1856).

The overall flow of the pipeline (along with the associated command name) is as
follows:

1.  `split-fastq`: Split `.fastq` file into smaller parts if desired/necessary.
    Sample-demultiplexed files can skip this step if the parts are small
    enough.

2.  `align-parts`: Align the raw reads to the reference PhIP-seq library using
    `bowtie`.

3.  `groupby-sample`: Reorganize alignments by sample, if reads were not
    demultiplexed to begin with.

4.  `compute-counts`: Compute counts for each possible peptide clone in the
    PhIP-seq library.

5.  `compute-pvals`: Compute p-values for enrichments for each peptide clone in
    each sample.

6.  `merge-columns`: Optionally merge p-value vectors into a single
    matrix/tab-delim file for further analysis.

The tools are implemented in Python and can dispatch jobs to an HPC job
scheduler such as LSF or Grid Engine.

Please submit [Issues](https://github.com/lasersonlab/phip-stat/issues) to
report any problems.  This code implements the statistical model as described
in the original Larman et al paper.  There are multiple alternative models
currently in development.


## Getting started/installation

The tools are (should be) compatible with Python 2.6+ and Python 3.4+.  Install
with `pip`

```bash
pip install phip-stat
```

or from GitHub

```bash
git clone https://github.com/lasersonlab/phip-stat.git
cd phip-stat
python setup.py install
```

or download a release
[directly from GitHub](https://github.com/lasersonlab/phip-stat/releases).

Requirements for `phip` (which should get installed by `pip`):

*   `click`
*   `biopython`
*   `numpy`
*   `scipy`

Additionally, the pipeline makes use of `bowtie` for short read alignment, and
expects it to be available on the `PATH`.

When running on a cluster, we assume that you can access a common filesystem
from all nodes (e.g. NFS), which is common for HPC computing environments.  It
is also important that the `phip-stat` package is installed into a Python
installation that will be invoked across the cluster.  We recommend Anaconda
for easily installing Python into a local user directory.


## Running the pipeline

All the pipeline tools are accessed through the executable `phip`.  (Your
`PATH` may need to be modified to include the `bin/` directory of your Python
installation.)  Their options/usage can be invoked by passing `-h`.

```
$ phip -h
Usage: phip [OPTIONS] COMMAND [ARGS]...

  phip -- PhIP-seq analysis tools

Options:
  -h, --help  Show this message and exit.

Commands:
  align-parts     align fastq files to peptide reference
  compute-counts  compute counts from aligned bam file
  compute-pvals   compute p-values from counts
  groupby-sample  group alignments by sample
  merge-columns   merge tab-delim files
  split-fastq     split fastq files into smaller chunks
```


### Pipeline inputs

The PhIP-seq pipeline as implemented requires:

1.  Read data from a PhIP-seq experiment (the raw data)

2.  Reference file for PhIP-seq library or bowtie index for alignment

3.  Input counts from sequencing the PhIP-seq library without any
    immunoprecipitation


### Generate per-sample alignment files

If you start with a single `.fastq` file that contains all the reads together,
we'll first split them into smaller chunks for alignment.

```bash
phip split-fastq -n 2000000 -i path/to/input.fastq -o path/to/workdir/parts
```

Note that for each pipeline tool, you can invoke a help message for more
information about options.  For example

```
$ phip split-fastq -h
Usage: phip split-fastq [OPTIONS]

  split fastq files into smaller chunks

Options:
  -i, --input TEXT          input path (fastq file)
  -o, --output TEXT         output path (directory)
  -n, --chunk-size INTEGER  number of reads per chunk
  -h, --help                Show this message and exit.
```

We then align each read to the reference PhIP-seq library using `bowtie`
(making sure to set the right queue):

```bash
phip align-parts \
    -i workdir/parts -o workdir/alns \
    -x path/to/index_name.ebwt \
    -b "bsub -q short"
```

Note: `align-parts` works by constructing a `bowtie` command and executing it
by prefixing it with the command given in to the `-b` option.  Each invocation
is executed and blocks to completion, which is instantaneous if submitting to a
batch scheduler such as LSF (as shown).  If omitted or given whitespace as a
string, each command will be executed in serial.

We reorganize the resulting alignments by sample, assuming that the sample
barcode is contained in the "query id" of each read.

```bash
phip groupby-sample -i workdir/alns -o workdir/alns_by_sample -m mapping.tsv
```

The `mapping.tsv` file is a tab-delimited file whose first column is a list of
barcode sequences and second column is a corresponding sample identifier (which
should work nicely as a file name).

If you instead started this pipeline with pre-demultiplexed `.fastq` files, you
can start with `align-parts` immediately and skip the `groupby-sample`.  We
assume the sample identifier is the base part of the filename (e.g.,
`sample1.fastq`).


### Generating p-values for enrichments

Once we have per-sample alignment files, we will generate count vectors for
each sample followed by enrichment p-values (using the "input" vector for the
particular phage library).

```bash
phip compute-counts \
    -i workdir/alns_by_sample -o workdir/counts -r path/to/reference/counts.tsv
```

The count-generation is performed single-threaded/locally, but the p-value
computation is more CPU-intensive so it can be parallelized with a job
scheduler.

```bash
phip compute-pvals -i workdir/counts -o workdir/pvals -b "bsub -q short"
```

It schedules single-file p-value computations using the same command without
the `-b`.  To manually compute p-values on just a single count file, execute

```bash
phip compute-pvals -i workdir/counts/sample1.counts.tsv -o sample1.pvals.tsv
```

Finally, the p-values from all samples can be merged into a single tab-
delimited file:

```bash
phip merge-columns -i workdir/pvals -o pvals.tsv -p 2
```

The `-p 2` tells the command to take the 3rd column from each file (0-indexed).
