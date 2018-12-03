# `phip-stat`: tools for analyzing PhIP-seq data

The PhIP-seq assay was first described in [Larman et.
al.](https://dx.doi.org/10.1038/nbt.1856). This repo contains code for
processing raw PhIP-seq data into analysis-ready enrichment scores.

The overall flow of the pipeline is for each sample

1. quantify the number of reads observed for each possible peptide in the phage library (using kallisto), and
2. convert those values to p-values by fitting to a Gamma-Poisson distribution

An entire NextSeq run with 500M reads can be processed in <30 min on a 4-core laptop.

Please submit [issues](https://github.com/lasersonlab/phip-stat/issues) to
report any problems.  This code implements the statistical model as described
in the original Larman et al paper.  There are multiple alternative models
currently in development.


## Installation

`phip-stat` was developed with Python 3 but should also be compatible with
Python 2. Please submit as issue if there are any problems.

```bash
pip install phip-stat
```

or to install the latest development version from GitHub

```bash
pip install git+https://github.com/lasersonlab/phip-stat.git
```

or download a release
[directly from GitHub](https://github.com/lasersonlab/phip-stat/releases).

Dependencies for `phip` (which should get installed by `pip`):

*   `click`
*   `biopython`
*   `numpy`
*   `scipy`

Additionally, the pipeline makes use of `bowtie` for short read alignment, and
expects it to be available on the `PATH`.

When running on a cluster, we assume that you can access a common filesystem
from all nodes (e.g. NFS), which is common for academic HPC computing
environments.  It is also important that the `phip-stat` package is installed
into the Python distribution that will be invoked across the cluster.  We
recommend `conda` for easily installing Python into a local user directory (see
appendix).


## Running the pipeline

All the pipeline tools are accessed through the executable `phip`.  (Your
`PATH` may need to be modified to include the `bin/` directory of your Python
installation.)  A list of commands can be obtained by passing `-h`.

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
  join-barcodes   annotate Illumina reads with barcodes Some...
  merge-columns   merge tab-delim files
  split-fastq     split fastq files into smaller chunks
```

Options/usage for a specific command can also be obtained with `-f`, for
example:

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

We then align each read to the reference PhIP-seq library using `bowtie`
(making sure to set the right queue):

```bash
phip align-parts \
    -i workdir/parts -o workdir/alns \
    -x path/to/index \ # implying path/to/index.1.ebwt exists etc.
    -b "bsub -q short"
```

Note: `align-parts` works by constructing a `bowtie` command and executing it
by prefixing it with the command given in to the `-b` option.  Each invocation
is executed and blocks to completion, which is instantaneous if submitting to a
batch scheduler such as LSF (as shown).  If omitted or given whitespace as a
string, each command will be executed serially.

Next, we reorganize the resulting alignments by sample, assuming that the
sample barcode is contained in the "query id" of each read.

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
phip merge-columns -i workdir/pvals -o pvals.tsv -p 1
```

The `-p 1` tells the command to take the 2nd column from each file (0-indexed).

## Running unit tests
Unit tests use the `nose` package and can be run with:

```
$ pip install nose  # if not already installed
$ nosetests -sv test/
```

## Appendix

### Using `conda` for easily managing packages

Conda is a package manager that works on multiple operating systems and is
closely connected to the Anaconda and Miniconda distributions of Python. Using
Conda makes it incredibly easy to install a full Python distribution into your
local directory, along with all the heavy-weight packages. It also manages many
non-Python packages, including bowtie, for exmaple.

```bash
curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh > miniconda3.sh
bash miniconda3.sh -b -p $HOME/miniconda3
# ADD $HOME/miniconda3/bin to your PATH in $HOME/.bash_profile
conda install -y numpy scipy biopython click
conda install -y bowtie
```

This will install Python 3 and bowtie into your home directory, completely
isolated from the system Python installation.


### Joining reads to barcodes/indexes on older Illumina data

If you use the Basespace "generate fastq" pipeline or the MiSeq local pipeline
for creating `.fastq` files on indexed runs, then reads that do not match an
index (i.e., all of the reads if you will manually demultiplex) will go into an
"Unidentified" file that does NOT include the index sequence for that read.
Instead, the indexes are available in a separate `.fastq` file.  To use these
data sets, the reads `.fastq` file must be rewritten to include the index
sequence in the read header.  This can be accomplished with the `phip join-
barcodes` command.
