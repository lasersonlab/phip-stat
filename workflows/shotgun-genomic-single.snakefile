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
config['species_fasta'] = '/Users/laserson/refdata/genomes/boa/GCF_001314995.1_ASM131499v1_genomic.fna'
config['fragment_size_nt'] = 600

# </CONFIGURATION>
###############################################################################


rule all:
    input:
        'model/mlxp.tsv',
        'annot.tsv'


rule align_reads_to_genome:
    input:
        lambda wildcards: config['samples'][wildcards.sample]
    output:
        'aligned_reads/{sample}.bam'
    params:
        species_index = config['species_index']
    threads: 4
    shell:
        """
        bowtie2 -p {threads} -x {params.species_index} -U {input} --no-unal \
            | samtools sort -@ {threads} -O BAM -o {output}
        """


rule count_fragments:
    input:
        'aligned_reads/{sample}.bam'
    output:
        'fragment_counts/{sample}.tsv'
    shell:
        r"""
        echo -e 'chr\tpos\tstrand\t{wildcards.sample}' > {output}
        cat {input} \
            | bedtools bamtobed -i stdin \
            | awk -F '\t' 'BEGIN {{OFS="\t"}} {{print $1, ($6 == "+") ? $2 : $3, $6}}' \
            | sort \
            | uniq -c \
            | awk 'BEGIN {{OFS="\t"}} {{print $2, $3, $4, $1}}' \
            >> {output}
        """


rule merge_counts:
    input:
        expand('fragment_counts/{sample}.tsv', sample=config['samples'])
    output:
        'counts.tsv'
    params:
        input_dir = 'fragment_counts'
    shell:
        'phip merge-columns -i {params.input_dir} -o {output} -p 3 -d 3 -m outer'


rule filter_low_counts:
    input:
        'counts.tsv'
    output:
        'counts.filtered.tsv'
    params:
        index_cols = 3
    shell:
        r"""
        # if fragment seen in two different pulldowns,
        # or if not, it's seen at least 3 times, write it
        head -n 1 {input} > {output}
        tail -n +2 {input} \
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
            >> {output}
        """


rule normalize_counts:
    input:
        'counts.filtered.tsv'
    output:
        'norm_counts.tsv'
    shell:
        'phip normalize-counts -i {input} -o {output} -m size-factors -d 3'


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
        'phip gamma-poisson-model -t {params.trim_percentile} -i {input} -o {params.output_dir} -d 3'


rule create_cds_reference:
    input:
        config['species_gff']
    output:
        temp('cdss.gff')
    shell:
        r"""cat {input} | awk -F \t '/^[^#]/ && $3 == "CDS"' > {output}"""


rule contig_lengths:
    input:
        config['species_fasta']
    output:
        temp('chrom_sizes.tsv')
    shell:
        """
        samtools dict -H {input} \
            | awk 'BEGIN {{FS="\t"; OFS="\t"}} {{print substr($2, 4), substr($3, 4)}}' \
            > {output}
        """


rule join_fragments_to_cdss:
    input:
        fragment_counts = 'counts.tsv',
        cds_annot = 'cdss.gff',
        chrom_sizes = 'chrom_sizes.tsv'
    output:
        'fragments.tsv'
    params:
        size = config['fragment_size_nt']
    shell:
        r"""
        echo -e 'chr\tfrag_start\tfrag_end\tstrand\tcds_start\tcds_end\tannot' > {output}
        tail -n +2 {input.fragment_counts} \
            | awk -F '\t' 'BEGIN {{OFS="\t"}} {{print $1, $2, $2, ".", ".", $3}}' \
            | bedtools slop -s -l 0 -r {params.size} -i stdin -g {input.chrom_sizes} \
            | bedtools intersect -s -wao -a stdin -b {input.cds_annot} \
            | awk -F '\t' 'BEGIN {{OFS="\t"}} {{print $1, $2, $3, $6, $10, $11, $15}}' \
            >> {output}
        """


rule annotate_fragment_list:
    input:
        'fragments.tsv',
        config['species_fasta']
    output:
        'annot.tsv'
    run:
        from csv import DictReader
        from collections import Counter

        from tqdm import tqdm
        from Bio import SeqIO

        chrom_dict = SeqIO.to_dict(SeqIO.parse(input[1], 'fasta'))

        fieldnames = ['chr', 'frag_start', 'frag_end', 'strand', 'cds_start', 'cds_end', 'annot']
        annots = dict()
        with open(input[0], 'r') as ip, open(output[0], 'w') as op:
            _ = next(ip)  # skip input header
            # write output header
            print(f'chr\tposition\tstrand\thas_overlap\thas_inframe_overlap\tbases_till_overlap\tfragment_nt\tfragment_aa\tannot', file=op)
            for r in tqdm(DictReader(ip, fieldnames=fieldnames, delimiter='\t', dialect='unix')):
                if r['strand'] not in ['+', '-']:
                    raise ValueError('strand must be + or -')

                cds_start = int(r['cds_start'])
                cds_end = int(r['cds_end'])
                frag_start = int(r['frag_start'])
                frag_end = int(r['frag_end'])

                # is the fragment on the positive strand of the chr?
                is_positive_strand = int(r['strand'] == '+')
                # does the fragment overlap any features in the bed file?
                # note: this value comes from bedtools, which uses -1 for no
                # overlap
                has_overlap = int(cds_start >= 0)
                # the python-style index where the fragment fragment starts
                position = frag_start if is_positive_strand else frag_end

                # compute base differential between fragment and CDS
                if is_positive_strand:
                    diff = position - cds_start + 1
                else:
                    diff = cds_end - position

                # extract nt and aa sequences
                fragment_nt = chrom_dict[r['chr']][frag_start:frag_end].seq
                if not is_positive_strand:
                    fragment_nt = fragment_nt.reverse_complement()
                frame = len(fragment_nt) % 3
                fragment_aa = fragment_nt[:len(fragment_nt) - frame].translate().split('*')[0]

                # how many bases from the 5' end of the fragment till we
                # get to the overlap
                bases_till_overlap = max(-diff, 0) if has_overlap else ''
                correct_frame = diff % 3 == 0
                translation_reaches_overlap = (len(fragment_aa) * 3 > bases_till_overlap) if has_overlap else False
                has_inframe_overlap = int(has_overlap and correct_frame and translation_reaches_overlap)

                record = (
                    r['chr'], position, r['strand'], has_overlap,
                    has_inframe_overlap, bases_till_overlap, fragment_nt,
                    fragment_aa, r['annot'])
                record_string = '\t'.join(map(str, record))
                print(record_string, file=op)
