from glob import glob
from collections import defaultdict

from phip.utils import parse_illumina_fastq_name


###############################################################################
# <CONFIGURATION>
# define config['samples'] as dict[sample_name, list(sample_abs_paths)]
# e.g.,
# config['samples'] = {'sample1': ['sample1.1.fastq.gz', 'sample1.2.fastq.gz']}
config['samples'] = defaultdict(list)
for f in glob('/Users/laserson/Downloads/BaseSpace/phip-14/*/*.fastq.gz'):
    p = parse_illumina_fastq_name(f)
    config['samples'][p.sample].append(p.path)

config['read_length'] =
config['kallisto_index'] =  # /path/to/index
config['factorization_rank'] = 1
# </CONFIGURATION>
###############################################################################


rule all:
    input:
        'cpm.tsv',
        'gamma-poisson',
        'matrix-residuals.tsv',
        'called-hits.tsv'


rule kallisto_pseudoalign:
    input:
        lambda wildcards: config['samples'][wildcards.sample]
    output:
        directory('kallisto/{sample}')
    params:
        idx = config['kallisto_index'],
        rl = config['read_length'],
        sd = 0.01,
    threads: 4
    shell:
        """
        kallisto quant --single --plaintext --fr-stranded \
            -l {params.rl} -s {params.sd} -t {threads} -i {params.idx} -o {output} {input}
        """


rule merge_kallisto_tpm:
    input:
        expand('kallisto/{sample}', sample=config['samples'])
    output:
        'cpm.tsv'
    params:
        parent_dir = 'kallisto'
    shell:
        'phip merge-kallisto-tpm -i {params.parent_dir} -o {output}'


rule compute_gamma_poisson:
    input:
        'cpm.tsv'
    output:
        directory('gamma-poisson')
    params:
        trim_percentile = 99.9
    shell:
        'phip gamma-poisson-model -t {params.trim_percentile} -i {input} -o {output}'


rule clipped_factorization_model:
    input:
        'cpm.tsv'
    output:
        'matrix-residuals.tsv'
    params:
        rank = config['factorization_rank']
    shell:
        """
        phip clipped-factorization-model -i {input} -o {output} --rank {params.rank} --no-normalize-to-reads-per-million
        """


rule call_hits:
    input:
        'matrix-residuals.tsv'
    output:
        'called-hits.tsv'
    shell:
        'phip call-hits -i {input} -o {output}'
