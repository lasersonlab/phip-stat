# phip-stat workflow for single-end sequencing of single-genome shotgun library

from glob import glob
from llutil.utils import parse_illumina_fastq_name

###############################################################################
# <CONFIGURATION>

# define config['samples'] as dict[sample_name, list(sample_path)]
# e.g.,
# config['samples'] = {'sample1': ['sample1.1.fastq.gz', 'sample1.2.fastq.gz']}
config['samples'] = {}
for f in glob('/Users/laserson/Downloads/BaseSpace/phip-14-48923876/*/*.fastq.gz'):
    p = parse_illumina_fastq_name(f)
    config['samples'].setdefault(p.sample, []).append(p.path)

config['species_index'] = '/Users/laserson/refdata/genomes/boa/bowtie2/boa'
config['species_gff'] = '/Users/laserson/refdata/genomes/boa/GCF_001314995.1_ASM131499v1_genomic.gff'

# </CONFIGURATION>
###############################################################################


rule all:
    input:
        'model/mlxp.tsv',
        'model/mlxp.inframe.tsv'


rule align_to_genome:
    input:
        lambda wildcards: config['samples'][wildcards.sample]
    output:
        'aligned/{sample}.bam'
    params:
        species_index = config['species_index']
    threads: 4
    shell:
        """
        bowtie2 -p {threads} -x {params.species_index} -U {input} --no-unal \
            | samtools sort -@ {threads} -O BAM -o {output}
        """


rule filter_gene_annot:
    input:
        config['species_gff']
    output:
        temp('genes.gff')
    shell:
        r"""cat {input} | awk -F \t '/^[^#]/ && $3 == "gene"' > {output}"""


rule annotate_reads:
    input:
        aligned_reads = 'aligned/{sample}.bam',
        gene_annot = 'genes.gff'
    output:
        # chrom, start, end, read_id, inframe, strand, annot
        'annot/{sample}.bed'
    shell:
        r"""
        bedtools intersect -s -wao -bed -a {input.aligned_reads} -b {input.gene_annot} \
            | awk -F '\t' 'BEGIN {{OFS="\t"}}
                {{
                    inframe = ($16 >= 0) && (($6 == "+" && ($2 - $16 + 1) % 3 == 0) || ($6 == "-" && ($3 - $17) % 3 == 0))
                    print $1, $2, $3, $4, inframe, $6, $21
                }}' \
            > {output}
        """


rule compute_counts:
    input:
        'annot/{sample}.bed'
    output:
        # chrom, start/end, strand, inframe, annot, count
        'counts/{sample}.tsv'
    shell:
        r"""
        echo -e 'chr\tpos\tstrand\tinframe\tannot\t{wildcards.sample}' > {output}
        cat {input} \
            | awk -F '\t' 'BEGIN {{OFS="\t"}} {{print $1, ($6 == "+") ? $2 : $3, $6, $5, $7}}' \
            | sort \
            | uniq -c \
            | awk 'BEGIN {{OFS="\t"}} {{print $2, $3, $4, $5, $6, $1}}' \
            >> {output}
        """


rule merge_counts:
    input:
        expand('counts/{sample}.tsv', sample=config['samples'])
    output:
        'counts.tsv'
    params:
        input_dir = 'counts'
    shell:
        'phip merge-columns -i {params.input_dir} -o {output} -p 5 -d 5 -m outer'


rule filter_low_counts:
    input:
        'counts.tsv'
    output:
        'counts.filtered.tsv'
    params:
        index_cols = 5
    shell:
        r"""
        cat {input} \
            | awk -F '\t' '{{
                    pos = 0
                    sum = 0
                    for (i = {params.index_cols} + 1; i <= NF; i++) {{
                        sum += $i
                        pos += ($i > 0) ? 1 : 0
                    }}
                    if (sum > 2 || pos > 1) {{
                        print $0
                    }}
                }}' \
            > {output}
        """


rule normalize_counts:
    input:
        'counts.filtered.tsv'
    output:
        'norm_counts.tsv'
    shell:
        'phip normalize-counts -i {input} -o {output} -m size-factors -d 5'


rule compute_mlxp:
    input:
        'norm_counts.tsv'
    output:
        'model/mlxp.tsv',
        'model/parameters.json'
    params:
        output_dir = 'model',
        trim_percentile = 99.9
    shell:
        'phip gamma-poisson-model -t {params.trim_percentile} -i {input} -o {params.output_dir} -d 5'


rule filter_inframe_mlxp:
    input:
        'model/mlxp.tsv',
    output:
        'model/mlxp.inframe.tsv'
    shell:
        r"""
        head -n 1 {input} > {output}
        awk -F '\t' '$4 == 1' {input} >> {output}
        """
