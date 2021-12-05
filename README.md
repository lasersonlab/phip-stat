# phip-stat: tools for analyzing PhIP-seq data

**NOTE**: This project is no longer being maintained. Please see the phippery, phip-flow, and related projects maintained by Erick Matsen's group:
https://github.com/matsengrp/phippery
https://github.com/matsengrp/phip-flow


The PhIP-seq assay was first described in [Larman et.
al.](https://dx.doi.org/10.1038/nbt.1856). This repo contains code for
processing raw PhIP-seq data into analysis-ready enrichment scores.

This code also implements multiple statistical models for processing PhIP-seq
data, including the model described in the original Larman et al paper
(`generalized-poisson-model`).  We currently recommend using one of the newer
models implemented here (e.g., `gamma-poisson-model`).

Please submit [issues](https://github.com/lasersonlab/phip-stat/issues) to
report any problems.


## Installation

phip-stat runs on Python 3.6+ and minimally depends on click, tqdm, numpy, scipy,
and pandas. The matrix factorization model also requires tensorflow.

```bash
pip install phip-stat
```

or to install the latest development version from GitHub

```bash
pip install git+https://github.com/lasersonlab/phip-stat.git
```


## Usage

The overall flow of the pipeline is

1. align — for each sample count the number of reads derived from each
   possible library member

2. merge — combine the count values from all samples into a single count
   matrix

3. model — normalize counts and train a model to compute enrichment
   scores/hits

An entire NextSeq run with 500M reads can be processed in <30 min on a 4-core
laptop (if aligning with a tool like kallisto).

### Command-line interface

All the pipeline tools are accessed through the `phip` executable. All
(sub)command usage/options can be obtained by passing `-h`.

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

### Example pipeline 1: kallisto alignment followed by Gamma-Poisson model

This pipeline will use kallisto to pseudoalign the reads to the reference.
Because the output of each alignment step is a directory, the merge step uses a
CLI tool designed for this directory structure. The counts are also
pre-normalized.

```bash
# 1. align
kallisto quant --single --plaintext --fr-stranded -l 75 -s 0.1 -t 4 \
    -i reference.idx -o sample_counts/sample1 sample1.fastq.gz
# ...
kallisto quant --single --plaintext --fr-stranded -l 75 -s 0.1 -t 4 \
    -i reference.idx -o sample_counts/sampleN sampleN.fastq.gz

# 2. merge
phip merge-kallisto-tpm -i sample_counts -o cpm.tsv

# 3. model
phip gamma-poisson-model -t 99.9 -i cpm.tsv -o gamma-poisson
```

### Example pipeline 2: exact-matching reads followed by matrix factorization

This pipeline will match each read to the reference exactly (or a chosen subset
of the read) followed by merging into a single matrix. The matrix is then
factored with a low-rank approximation (allowing for clipping) and "hits" are
called with a heuristic.

```bash
# 1. align
phip count-exact-matches -r reference.fasta -l 75 -o sample_counts/sample1.counts.tsv sample1.fastq.gz
# ...
phip count-exact-matches -r reference.fasta -l 75 -o sample_counts/sampleN.counts.tsv sampleN.fastq.gz

# 2. merge
phip merge-columns -m iter -i sample_counts -o counts.tsv

# 3. model
phip clipped-factorization-model --rank 2 -i counts.tsv -o residuals.tsv
phip call-hits -i residuals.tsv -o hits.tsv --beads-regex ".*BEADS_ONLY.*"
```

### Example pipeline 3: bowtie2 alignment followed by normalization and Gamma-Poisson

This example uses bowtie2, which should give the maximum sensitivity at the
expense of speed. The main bowtie2 command accomplishes the following: align
reads to reference, sort and convert to BAM, compute coverage depth at each
position of each clone, for each clone take only the largest number observed,
finally sort by clone identifier.

```bash
# 1. align
echo "id\tsample1" > sample_counts/sample1.tsv
bowtie2 -p 4 -x reference_index -U sample1.fastq.gz \
    | samtools sort -O BAM \
    | samtools depth -aa -m 100000000 - \
    | awk 'BEGIN {OFS="\t"} {counts[$1] = ($3 < counts[$1]) ? counts[$1] : $3} END {for (c in counts) {print c, counts[c]}}' \
    | sort -k 1 \
    >> sample_counts/sample1.tsv
# ...
echo "id\tsampleN" > sample_counts/sampleN.tsv
bowtie2 -p 4 -x reference_index -U sampleN.fastq.gz \
    | samtools sort -O BAM \
    | samtools depth -aa -m 100000000 - \
    | awk 'BEGIN {OFS="\t"} {counts[$1] = ($3 < counts[$1]) ? counts[$1] : $3} END {for (c in counts) {print c, counts[c]}}' \
    | sort -k 1 \
    >> sample_counts/sampleN.tsv

# 2. merge -- NOTE: this performs a pandas outer join and loads all counts into memory
phip merge-columns -m outer -i sample_counts -o counts.tsv

# 3. model
phip normalize-counts -m size-factors -i counts.tsv -o normalized_counts.tsv
phip gamma-poisson-model -t 99.9 -i normalized_counts.tsv -o gamma-poisson
```

### Snakemake recipes

We include several example Snakemake recipes for easily processing large sets of
samples at once, e.g.,
`workflows/example-kallisto-GamPois-factorization.snakefile`. In general the
configuration section must be edited to specify the location of the raw
sequencing data.


## Running unit tests
Unit tests use the `nose` package and can be run with:

```
$ pip install nose  # if not already installed
$ nosetests -sv test/
```
