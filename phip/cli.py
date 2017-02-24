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

from __future__ import print_function

import os
import sys
from os import path as osp
from os.path import join as pjoin
from glob import glob
from subprocess import Popen, PIPE
if sys.version_info[0] == 2:
    from itertools import izip as zip

from click import group, command, option
import numpy as np
from Bio import SeqIO
from Bio.SeqIO.QualityIO import FastqGeneralIterator

from phip.gp import (
    estimate_GP_distributions, lambda_theta_regression, precompute_pvals)


@group(context_settings={'help_option_names': ['-h', '--help']})
def cli():
    """phip -- PhIP-seq analysis tools"""
    pass


@cli.command(name='join-barcodes')
@option('-i', '--input', required=True, help='reads fastq file')
@option('-b', '--barcodes', required=True, help='barcodes fastq file')
@option('-o', '--output', required=True, help='output fastq file')
def join_barcodes(input, barcodes, output):
    """annotate Illumina reads with barcodes

    Some older versions of the Illumina pipeline would not annotate the reads
    their corresponding barcodes, but would leave the barcode reads in a
    separate fastq file. This tool will take both fastq files and will modify
    the main reads to add the barcode to the header line (in the same place
    Illumina would put it).

    This should only be necessary on older data files. Newer pipelines that use
    bcl2fastq2 or the "generate fastq" pipeline in Basespace (starting 9/2016)
    should not require this.

    This tool assumes that the reads are presented in the same order in the two
    input files.
    """
    r_f = osp.abspath(input)
    b_f = osp.abspath(barcodes)
    o_f = osp.abspath(output)
    fastq_template = '@{0}\n{1}\n+\n{2}\n'.format
    with open(r_f, 'r') as r_h, open(b_f, 'r') as b_h, open(o_f, 'w') as o_h:
        r_it = FastqGeneralIterator(r_h)
        b_it = FastqGeneralIterator(b_h)
        o_write = o_h.write
        for read, barcode in zip(r_it, b_it):
            assert read[0] == barcode[0]
            parts = read[0].rsplit(':', maxsplit=1)
            parts[1] = str(barcode[1])
            o_write(fastq_template(':'.join(parts), read[1], read[2]))


@cli.command(name='split-fastq')
@option('-i', '--input', required=True, help='input path (fastq file)')
@option('-o', '--output', required=True, help='output path (directory)')
@option('-n', '--chunk-size', type=int, required=True,
        help='number of reads per chunk')
def split_fastq(input, output, chunk_size):
    """split fastq files into smaller chunks"""
    input_file = osp.abspath(input)
    output_dir = osp.abspath(output)
    os.makedirs(output_dir, mode=0o755)

    # convenience functions
    output_file = lambda i: pjoin(output_dir, 'part.{0}.fastq'.format(i))
    fastq_template = '@{0}\n{1}\n+\n{2}\n'.format

    with open(input_file, 'r') as input_handle:
        num_processed = 0
        file_num = 1
        for record in FastqGeneralIterator(input_handle):
            if num_processed == 0:
                op = open(output_file(file_num), 'w')
                write = op.write
            write(fastq_template(*record))
            num_processed += 1
            if num_processed == chunk_size:
                op.close()
                num_processed = 0
                file_num += 1
        if not op.closed:
            op.close()


@cli.command(name='align-parts')
@option('-i', '--input', required=True,
        help='input path (directory of fastq parts)')
@option('-o', '--output', required=True,
        help='output path (directory)')
@option('-x', '--index', required=True,
        help='bowtie index (e.g., as specified to bowtie2)')
@option('-b', '--batch-submit', default='',
        help='batch submit command to prefix bowtie command invocation')
@option('-p', '--threads', default=1,
        help='Number of threads to specify for each invocation of bowtie')
def align_parts(input, output, index, batch_submit, threads):
    """align fastq files to peptide reference"""
    input_dir = osp.abspath(input)
    output_dir = osp.abspath(output)
    os.makedirs(output_dir, mode=0o755)
    bowtie_cmd_template = (
        'bowtie -n 3 -l 100 --best --nomaqround --norc -k 1 -p {threads} '
        '--quiet {index} {input} {output}')
    for input_file in glob(pjoin(input_dir, '*.fastq')):
        output_file = pjoin(output_dir,
                            osp.splitext(osp.basename(input_file))[0] + '.aln')
        bowtie_cmd = bowtie_cmd_template.format(index=index,
                                                input=input_file,
                                                output=output_file,
                                                threads=threads)
        submit_cmd = '{batch_cmd} {app_cmd}'.format(batch_cmd=batch_submit,
                                                    app_cmd=bowtie_cmd)
        p = Popen(submit_cmd.strip(), shell=True, stdout=PIPE,
                  universal_newlines=True)
        print(p.communicate()[0])


@cli.command(name='groupby-sample')
@option('-i', '--input', required=True,
        help='input path (directory of aln parts)')
@option('-o', '--output', required=True, help='output path (directory)')
@option('-m', '--mapping', required=True,
        help='barcode to sample mapping (tab-delim)')
def groupby_sample(input, output, mapping):
    """group alignments by sample"""
    input_dir = osp.abspath(input)
    output_dir = osp.abspath(output)
    os.makedirs(output_dir, mode=0o755)

    def one_base_mutants(seq):
        alphabet = set(['A', 'C', 'G', 'T', 'N'])
        for i in range(len(seq)):
            for alt in alphabet - set([seq[i].upper()]):
                yield seq[:i] + alt + seq[i + 1:]

    # load sample mapping and open output handles
    bc2sample = {}
    output_handles = {}
    with open(mapping, 'r') as ip:
        for line in ip:
            (bc, sample) = line.split()
            bc2sample[bc] = sample
            for mut in one_base_mutants(bc):
                bc2sample[mut] = sample
            output_handles[sample] = open(
                pjoin(output_dir, sample + '.aln'), 'w')

    for input_file in glob(pjoin(input_dir, '*.aln')):
        with open(input_file, 'r') as ip:
            for line in ip:
                bc = line.split()[1].split(':')[-1]
                try:
                    sample = bc2sample[bc]
                except KeyError:
                    continue
                output_handles[sample].write(line)


@cli.command(name='compute-counts')
@option('-i', '--input', required=True,
        help='input path (directory of aln files)')
@option('-o', '--output', required=True, help='output path (directory)')
@option('-r', '--reference', required=True,
        help='path to reference (input) counts file (tab-delim)')
def compute_counts(input, output, reference):
    """compute counts from aligned bam file"""
    input_dir = osp.abspath(input)
    output_dir = osp.abspath(output)
    os.makedirs(output_dir, mode=0o755)

    # load reference (i.e., input) counts
    ref_names = []
    ref_counts = []
    with open(reference, 'r') as ip:
        # burn header
        _ = next(ip)
        for line in ip:
            fields = line.split('\t')
            ref_names.append(fields[0].strip())
            ref_counts.append(int(fields[1]))

    # compute count dicts
    for input_file in glob(pjoin(input_dir, '*.aln')):
        print(input_file)
        sys.stdout.flush()
        counts = {}
        sample = osp.splitext(osp.basename(input_file))[0]
        # accumulate counts
        with open(input_file, 'r') as ip:
            for line in ip:
                ref_clone = line.split('\t')[2].strip()
                counts[ref_clone] = counts.get(ref_clone, 0) + 1
        # write counts
        output_file = pjoin(output_dir, sample + '.tsv')
        with open(output_file, 'w') as op:
            print('id\tinput\t{0}'.format(sample), file=op)
            for (ref_name, ref_count) in zip(ref_names, ref_counts):
                record = '{0}\t{1}\t{2}'.format(
                    ref_name, ref_count, counts.get(ref_name, 0))
                print(record, file=op)


@cli.command(name='compute-pvals')
@option('-i', '--input', required=True, help='input path')
@option('-o', '--output', required=True, help='output path')
@option('-b', '--batch-submit',
        help='batch submit command to prefix pval command invocation')
def compute_pvals(input, output, batch_submit):
    """compute p-values from counts"""
    if batch_submit is not None:
        # run compute-pvals on each file using batch submit command
        input_dir = osp.abspath(input)
        output_dir = osp.abspath(output)
        os.makedirs(output_dir, mode=0o755)
        pval_cmd_template = 'phip compute-pvals -i {input} -o {output}'
        for input_file in glob(pjoin(input_dir, '*.tsv')):
            sample = osp.splitext(osp.basename(input_file))[0]
            output_file = pjoin(output_dir, '{0}.pvals.tsv'.format(sample))
            pval_cmd = pval_cmd_template.format(
                input=input_file, output=output_file)
            submit_cmd = '{batch_cmd} {app_cmd}'.format(
                batch_cmd=batch_submit, app_cmd=pval_cmd)
            p = Popen(submit_cmd.strip(), shell=True, stdout=PIPE,
                      universal_newlines=True)
            print(p.communicate()[0])
    else:
        # actually compute p-vals on single file
        # Load data
        input_file = osp.abspath(input)
        output_file = osp.abspath(output)
        clones = []
        samples = None
        input_counts = []
        output_counts = []
        with open(input_file, 'r') as ip:
            header_fields = next(ip).split('\t')
            samples = [f.strip() for f in header_fields[2:]]
            for line in ip:
                fields = line.split('\t')
                clones.append(fields[0].strip())
                input_counts.append(int(fields[1]))
                output_counts.append(np.int_(fields[2:]))
        input_counts = np.asarray(input_counts)
        # pseudocounts to combat negative regressed theta:
        output_counts = np.asarray(output_counts) + 1
        uniq_input_values = list(set(input_counts))

        # Estimate generalized Poisson distributions for every input count
        (lambdas, thetas, idxs) = estimate_GP_distributions(input_counts,
                                                            output_counts,
                                                            uniq_input_values)

        # Regression on all of the theta and lambda values computed
        (lambda_fits, theta_fits) = lambda_theta_regression(lambdas,
                                                            thetas,
                                                            idxs)

        # Precompute CDF for possible input-output combinations
        uniq_combos = []
        for i in range(output_counts.shape[1]):
            uniq_combos.append(set(zip(input_counts, output_counts[:, i])))
        log10pval_hash = precompute_pvals(lambda_fits, theta_fits, uniq_combos)

        # Compute p-values for each clone using regressed GP parameters
        with open(output_file, 'w') as op:
            header = '\t'.join(['id'] + samples)
            print(header, file=op)
            for (clone, ic, ocs) in zip(clones, input_counts, output_counts):
                fields = [clone]
                for (i, oc) in enumerate(ocs):
                    fields.append('{0:f}'.format(log10pval_hash[(i, ic, oc)]))
                print('\t'.join(fields), file=op)


@cli.command(name='merge-columns')
@option('-i', '--input', required=True,
        help='input path (directory of tab-delim files)')
@option('-o', '--output', required=True, help='output path')
@option('-p', '--position', type=int, default=1,
        help='the field position to merge (0-indexed)')
def merge_columns(input, output, position):
    """merge tab-delim files"""
    input_dir = os.path.abspath(input)
    output_file = os.path.abspath(output)

    input_files = glob(pjoin(input_dir, '*.tsv'))
    file_iterators = [open(f, 'r') for f in input_files]
    file_headers = [osp.splitext(osp.basename(f))[0] for f in input_files]

    with open(output_file, 'w') as op:
        # iterate through lines
        for lines in zip(*file_iterators):
            fields_array = [[field.strip() for field in line.split('\t')]
                            for line in lines]
            # check that join column is the same
            for fields in fields_array[1:]:
                assert fields_array[0][0] == fields[0]
            merged_fields = ([fields_array[0][0]] +
                             [f[position] for f in fields_array])
            print('\t'.join(merged_fields), file=op)
