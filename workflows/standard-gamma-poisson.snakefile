import os.path as osp
from glob import glob

# list of all input fastq files
input_files = glob(osp.expanduser(config['input_glob']))


#############################################################
# EDIT THIS: define config['samples']
#
# config['samples'] = {'sample1': ['sample1.1.fastq.gz', 'sample1.2.fastq.gz']}
#
# config['samples'] is a dictionary where each key is the name of a sample and
# each value is a list of paths to fastq files for that sample
from llutil.utils import parse_illumina_fastq_name
config['samples'] = {}
for f in input_files:
    p = parse_illumina_fastq_name(f)
    config['samples'].setdefault(p.sample, []).append(p.path)
#############################################################


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
        'mlxp.tsv'
    params:
        trim_percentile = 99.9
    shell:
        'phip gamma-poisson-model -t {trim_percentile} -i {input} -o {output}'
