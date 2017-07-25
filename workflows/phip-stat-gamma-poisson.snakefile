import os.path as osp
from glob import glob

from llutil.utils import parse_illumina_fastq_name

# list of all input fastq files
input_files = glob(osp.expanduser(config['input_glob']))

# organize the input files into samples
# config['samples'] is a dictionary where each key is the name of a sample and
# each value is a list of paths to fastq files for that sample
config['samples'] = {}
for f in input_files:
    p = parse_illumina_fastq_name(f)
    config['samples'].setdefault(p.sample, []).append(p.path)


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
    run:
        samples = [p.split('/')[1] for p in input]
        iterators = [open(p, 'r') for p in input]
        with open(output[0], 'w') as op:
            it = zip(*iterators)
            # burn headers of input files and write header of output file
            _ = next(it)
            print('id\t{}'.format('\t'.join(samples)), file=op)
            for lines in it:
                fields_array = [line.split('\t') for line in lines]
                # check that join column is the same
                assert all([fields[0] == fields_array[0][0] for fields in fields_array])
                merged_fields = [fields_array[0][0]] + [f[4].strip() for f in fields_array]
                print('\t'.join(merged_fields), file=op)


rule compute_pvals:
    input:
        'cpm.tsv'
    output:
        'mlxp.tsv'
    run:
        import pandas as pd
        import numpy as np
        import scipy as sp
        import scipy.stats
        cpm = pd.read_csv(input[0], sep='\t', header=0, index_col=0)
        upper_bound = sp.stats.scoreatpercentile(cpm.values, 99.9)
        trimmed_means = cpm.apply(lambda x: x[x <= upper_bound].mean(), axis=1, raw=True).values

        # definte log likelihood
        m = len(trimmed_means)
        s = trimmed_means[trimmed_means > 0].sum()
        sl = np.log(trimmed_means[trimmed_means > 0]).sum()
        def ll(x):
            return -1 * (m * x[0] * np.log(x[1]) - m * sp.special.gammaln(x[0]) + (x[0] - 1) * sl - x[1] * s)

        param = sp.optimize.minimize(ll, np.asarray([2, 1]), bounds=[(np.nextafter(0, 1), None), (np.nextafter(0, 1), None)])
        alpha, beta = param.x

        # assumes the counts for each clone are Poisson distributed with the learned Gamma prior
        # Therefore, the posterior is Gamma distributed, and we use the expression for its expected value
        trimmed_sums = cpm.apply(lambda x: x[x <= upper_bound].sum(), axis=1, raw=True).values
        trimmed_sizes = cpm.apply(lambda x: (x <= upper_bound).sum(), axis=1, raw=True).values
        background_rates = (alpha + trimmed_sums) / (beta + trimmed_sizes)

        # mlxp is "minus log 10 pval"
        mlxp = []
        for i in range(cpm.shape[0]):
            mlxp.append(-sp.stats.poisson.logsf(cpm.values[i], background_rates[i]) / np.log(10))
        mlxp = np.asarray(mlxp)
        mlxp = pd.DataFrame(data=mlxp, index=cpm.index, columns=cpm.columns).replace(np.inf, -np.log10(np.finfo(np.float64).tiny))
        mlxp.to_csv(output[0], sep='\t', index_label='id', float_format='%.2f')
