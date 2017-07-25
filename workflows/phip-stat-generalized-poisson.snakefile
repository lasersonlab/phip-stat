import os.path as osp
from glob import glob

from llutil.utils import parse_illumina_fastq_name


input_files = glob(osp.expanduser(config['input_glob']))
config['samples'] = {}
for f in input_files:
    p = parse_illumina_fastq_name(f)
    config['samples'].setdefault(p.sample, []).append(p.path)


rule all:
    input:
        'cpm.tsv',
        'pvals.tsv'


rule create_kmer_reference:
    input:
        config['ref_fasta']
    output:
        temp('reference.fasta')
    params:
        k = int(config['read_length']) + 1
    run:
        from Bio import SeqIO
        with open(output[0], 'w') as op:
            for sr in SeqIO.parse(input[0], 'fasta'):
                print(sr[:params.k].format('fasta'), end='', file=op)


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
        'kallisto/{sample}'
    params:
        rl = config['read_length'],
        sd = 0.1,
    shell:
        'kallisto quant --single --plaintext --fr-stranded -l {params.rl} -s {params.sd} -t {threads} -i {input.index} -o {output} {input.samples}'


rule join_input_output:
    input:
        'kallisto/{sample}'
    output:
        'iopairs/{sample}.cpm.tsv'
    params:
        input_path = config['ref_counts']
    run:
        from os.path import join as pjoin
        import pandas as pd
        ref_counts = pd.read_csv(params.input_path, sep='\t', header=0)
        output_counts = pd.read_csv(
            pjoin(input[0], 'abundance.tsv'), sep='\t', header=0)
        joined = pd.merge(
            ref_counts, output_counts, how='left', left_on='id',
            right_on='target_id')
        # add a "count" column of integers
        joined[wildcards.sample] = joined['tpm'].apply(lambda x: round(x))
        joined.to_csv(
            output[0], sep='\t', columns=['id', 'input', wildcards.sample],
            index=False)


rule compute_pvals:
    input:
        'iopairs/{sample}.cpm.tsv'
    output:
        'pvals/{sample}.pvals.tsv'
    shell:
        'phip compute-pvals -i {input} -o {output}'


rule merge_counts:
    input:
        expand('iopairs/{sample}.cpm.tsv', sample=config['samples']),
    output:
        'cpm.tsv'
    params:
        input_dir = 'iopairs'
    shell:
        'phip merge-columns -i {params.input_dir} -o {output} -p 2'


rule merge_pvals:
    input:
        expand('pvals/{sample}.pvals.tsv', sample=config['samples']),
    output:
        'pvals.tsv'
    shell:
        'phip merge-columns -i pvals -o {output} -p 1'
