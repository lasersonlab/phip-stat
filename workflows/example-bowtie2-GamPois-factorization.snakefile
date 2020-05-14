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
config['bowtie2_index'] =  # /path/to/index
config['factorization_rank'] = 1
# </CONFIGURATION>
###############################################################################


rule all:
    input:
        "alignment_totals.tsv",
        'counts.tsv',
        'gamma-poisson',
        'matrix-residuals.tsv',
        'called-hits.tsv'

rule bowtie2_align:
    input:
        lambda wildcards: config['samples'][wildcards.sample]
    output:
        "bam/{sample}.sorted.bam"
    params:
        idx = config['bowtie2_index']
    threads: 4
    shell:
        """
        bowtie2 -p {threads} -x {params.idx} -U {input} \
            | samtools sort -O BAM \
            > {output}
        """


rule count_alignments:
    input:
        "bam/{sample}.sorted.bam"
    output:
        "counts/{sample}.tsv"
    shell:
        """
        echo "id\t{wildcards.sample}" > {output}
        samtools depth -aa -m 0 {input} \
            | awk 'BEGIN {{OFS="\t"}} {{c[$1] = ($3 < c[$1]) ? c[$1] : $3}} END {{for (k in c) {{print k, c[k]}}}}' \
            | sort -k 1 \
            >> {output}
        """


rule flagstat:
    input:
        "bam/{sample}.sorted.bam"
    output:
        "flagstat/{sample}.txt"
    shell:
        "samtools flagstat {input} > {output}"



rule alignment_totals:
    input:
        expand('flagstat/{sample}.txt', sample=config['samples'])
    output:
        "alignment_totals.tsv"
    run:
        import re
        with open(output[0], "w") as op:
            print("sample\ttotal_reads\taligned_reads\tfraction_aligned", file=op)
            for f in input:
                sample = re.fullmatch("flagstat/(.*)\.txt", f)[1]
                with open(f, "r") as ip:
                    lines = ip.readlines()
                assert "total" in lines[0]
                assert "mapped" in lines[4]
                total_reads = int(lines[0].split()[0])
                aligned_reads = int(lines[4].split()[0])
                fraction_aligned = aligned_reads / total_reads
                print(f"{sample}\t{total_reads}\t{aligned_reads}\t{fraction_aligned:.3f}", file=op)


rule merge_counts:
    input:
        expand('counts/{sample}.tsv', sample=config['samples'])
    output:
        'counts.tsv'
    params:
        parent_dir = 'counts'
    shell:
        'phip merge-columns -m iter -d 1 -p 1 -i {params.parent_dir} -o {output}'


rule normalize_counts:
    input:
        'counts.tsv'
    output:
        'counts.normalized.tsv'
    shell:
        "phip normalize-counts -m size-factors -d 1 -i {input} -o {output}"


rule compute_gamma_poisson:
    input:
        'counts.normalized.tsv'
    output:
        directory('gamma-poisson')
    params:
        trim_percentile = 99.9
    shell:
        'phip gamma-poisson-model -t {params.trim_percentile} -i {input} -o {output}'


rule clipped_factorization_model:
    input:
        'counts.normalized.tsv'
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
