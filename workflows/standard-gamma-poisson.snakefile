import re
import os.path as osp
from glob import glob
from collections import namedtuple

from tqdm import tqdm


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

config['reference_index'] =
# </CONFIGURATION>
###############################################################################


rule all:
    input:
        'normalized_match_counts.tsv',
        'match_counts.tsv',
        'alignment_counts.tsv',
        'gamma-poisson/mlxp.tsv'


rule align_ref:
    input:
        lambda wildcards: config['samples'][wildcards.sample]
    output:
        'alignments/{sample}.bam',
    params:
        reference_index = config['reference_index']
    threads: 1
    shell:
        """
        cat {input} \
            | gunzip \
            | bowtie2 -p {threads} --norc -x {params.reference_index} -U - \
            | tqdm \
            | samtools sort -O BAM -o {output}
        """


rule compute_alignment_counts:
    input:
        'alignments/{sample}.bam'
    output:
        'alignment_counts/{sample}.alignment_counts.tsv'
    shell:
        """
        echo "id\t{wildcards.sample}" > {output}
        samtools depth -aa -m 10000000 {input} \
            | tqdm \
            | awk 'BEGIN {{OFS="\t"}} {{counts[$1] = ($3 < counts[$1]) ? counts[$1] : $3}} END {{for (c in counts) {{print c, counts[c]}}}}' \
            | sort -k 1 \
            >> {output}
        """


rule compute_match_counts:
    input:
        'alignments/{sample}.bam'
    output:
        'match_counts/{sample}.match_counts.tsv'
    run:
        from collections import Counter
        from pysam import AlignmentFile
        counts = Counter()
        bamfile = AlignmentFile(input[0], 'rb')
        for aln in tqdm(bamfile, position=1):
            if aln.flag == 0 and len(aln.cigar) == 1 and aln.cigar[0] == (0, aln.query_length):
                counts[aln.reference_name] += 1
        with open(output[0], 'w') as op:
            print('id\t{}'.format(wildcards.sample), file=op)
            for n in bamfile.header.references:
                print('{}\t{}'.format(n, counts[n]), file=op)


rule merge_alignment_counts:
    input:
        expand('alignment_counts/{sample}.alignment_counts.tsv', sample=config['samples'])
    output:
        'alignment_counts.tsv'
    params:
        input_dir = 'alignment_counts'
    shell:
        'phip merge-columns -i {params.input_dir} -o {output} -m iter'


rule merge_match_counts:
    input:
        expand('match_counts/{sample}.match_counts.tsv', sample=config['samples'])
    output:
        'match_counts.tsv'
    params:
        input_dir = 'match_counts'
    shell:
        'phip merge-columns -i {params.input_dir} -o {output} -m iter'


rule normalize_match_counts:
    input:
        'match_counts.tsv'
    output:
        'normalized_match_counts.tsv'
    shell:
        'phip normalize-counts -i {input} -o {output} -m size-factors'


rule compute_pvals:
    input:
        'normalized_match_counts.tsv'
    output:
        'gamma-poisson',
        'gamma-poisson/mlxp.tsv'
    params:
        trim_percentile = 99.9
    shell:
        'phip gamma-poisson-model -t {params.trim_percentile} -i {input} -o {output[0]}'
