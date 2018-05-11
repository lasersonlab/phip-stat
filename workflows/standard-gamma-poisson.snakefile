import re
import os.path as osp
from glob import glob
from collections import namedtuple


def fastx_stem(path):
    m = re.match('(.+)(?:\.fast[aq]|\.fna|\.f[aq])(?:\.gz)?$', osp.basename(path))
    if m is None:
        raise ValueError(
            'Path {} does not look like a fast[aq] file'.format(path))
    return m.group(1)


def parse_illumina_fastq_name(path):
    """Parse Illumina fastq file name"""
    stem = fastx_stem(path)
    m = re.match('(.*)_S(\d+)_L(\d+)_([RI][12])_001', stem)
    IlluminaFastq = namedtuple(
        'IlluminaFastq', ['sample', 'sample_num', 'lane', 'read', 'path'])
    return IlluminaFastq(sample=m.group(1),
                         sample_num=int(m.group(2)),
                         lane=int(m.group(3)),
                         read=m.group(4),
                         path=path)


###############################################################################
# <CONFIGURATION>
# define config['samples'] as dict[sample_name, list(sample_abs_path)]
# e.g.,
# config['samples'] = {'sample1': ['sample1.1.fastq.gz', 'sample1.2.fastq.gz']}
config['samples'] = {}
for f in glob('/Users/laserson/Downloads/BaseSpace/phip-14-48923876/*/*.fastq.gz'):
    p = parse_illumina_fastq_name(f)
    config['samples'].setdefault(p.sample, []).append(p.path)

config['ref_fasta'] =
config['read_length'] =
# </CONFIGURATION>
###############################################################################


rule all:
    input:
        'cpm.tsv',
        'mlxp.tsv'


rule create_kmer_reference:
    input:
        config['ref_fasta']
    output:
        temp('reference.fasta')
    params:
        # kallisto needs a bit of extra sequence in the reference
        k = int(config['read_length']) + round(0.2 * int(config['read_length']))
    shell:
        'phip truncate-fasta -k {params.k} -i {input} -o {output}'


rule kallisto_index:
    input:
        'reference.fasta'
    output:
        temp('kallisto.idx')
    shell:
        'kallisto index -i {output} {input}'


rule quantify_phip:
    input:
        index = 'kallisto.idx',
        samples = lambda wildcards: config['samples'][wildcards.sample]
    output:
        'kallisto/{sample}/abundance.tsv'
    params:
        output_dir = 'kallisto/{sample}',
        rl = config['read_length'],
        sd = 0.1,
    shell:
        'kallisto quant --single --plaintext --fr-stranded -l {params.rl} -s {params.sd} -t {threads} -i {input.index} -o {params.output_dir} {input.samples}'


rule merge_counts:
    input:
        expand('kallisto/{sample}/abundance.tsv', sample=config['samples']),
    output:
        'cpm.tsv'
    params:
        input_dir = 'kallisto'
    shell:
        'phip merge-kallisto-tpm -i {params.input_dir} -o {output}'


rule compute_pvals:
    input:
        'cpm.tsv'
    output:
        'gamma-poisson',
        'gamma-poisson/mlxp.tsv'
    params:
        trim_percentile = 99.9
    shell:
        'phip gamma-poisson-model -t {params.trim_percentile} -i {input} -o {output[0]}'
