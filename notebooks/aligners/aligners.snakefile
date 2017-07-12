

rule all:
    input:
        expand('counts/bowtie-{k}mer.tsv', k=[50, 51, 60]),
        expand('counts/bowtie2-{k}mer.tsv', k=[50, 51, 60]),
        expand('counts/kallisto-{k}mer.tsv', k=[50, 51, 60]),



rule sample_reads:
    input:
        '/Users/laserson/Downloads/sjogrens/Undetermined_S0_L005_R1_001.fastq.gz'
    output:
        'sample.fastq.gz'
    run:
        import gzip
        import random
        from Bio.SeqIO.QualityIO import FastqGeneralIterator
        total_reads = int(shell('cat {input} | gunzip | wc -l', read=True)) // 4
        idxs = set(random.sample(range(total_reads), 1000000))
        with gzip.open(input[0], 'rt') as ip:
            with gzip.open(output[0], 'wt') as op:
                for i, (title, seq, qual) in enumerate(FastqGeneralIterator(ip)):
                    if i in idxs:
                        print(f'@{title}\n{seq}\n+\n{qual}', file=op)


rule create_kmer_ref:
    input:
        '/Users/laserson/repos/phage_libraries_private/human90/human90-ref.fasta'
    output:
        'refs/human90-{k}mer-ref.fasta'
    run:
        from Bio import SeqIO
        with open(output[0], 'w') as op:
            for sr in SeqIO.parse(input[0], 'fasta'):
                print(sr[:int(wildcards.k)].format('fasta'), end='', file=op)


rule build_bowtie_index:
    input:
        'refs/human90-{k}mer-ref.fasta'
    output:
        'indexes/bowtie/human90-{k}mer.1.ebwt',
        'indexes/bowtie/human90-{k}mer.2.ebwt',
        'indexes/bowtie/human90-{k}mer.3.ebwt',
        'indexes/bowtie/human90-{k}mer.4.ebwt',
        'indexes/bowtie/human90-{k}mer.rev.1.ebwt',
        'indexes/bowtie/human90-{k}mer.rev.2.ebwt',
    params:
        index = 'indexes/bowtie/human90-{k}mer',
    shell:
        'bowtie-build -q {input} {params.index}'


rule build_bowtie2_index:
    input:
        'refs/human90-{k}mer-ref.fasta'
    output:
        'indexes/bowtie2/human90-{k}mer.1.bt2',
        'indexes/bowtie2/human90-{k}mer.2.bt2',
        'indexes/bowtie2/human90-{k}mer.3.bt2',
        'indexes/bowtie2/human90-{k}mer.4.bt2',
        'indexes/bowtie2/human90-{k}mer.rev.1.bt2',
        'indexes/bowtie2/human90-{k}mer.rev.2.bt2',
    params:
        index = 'indexes/bowtie2/human90-{k}mer',
    shell:
        'bowtie2-build -q {input} {params.index}'


rule build_kallisto_index:
    input:
        'refs/human90-{k}mer-ref.fasta'
    output:
        'indexes/kallisto/human90-{k}mer.idx'
    shell:
        'kallisto index -i {output} {input}'



rule bowtie_align:
    input:
        'sample.fastq.gz',
        'indexes/bowtie/human90-{k}mer.1.ebwt',
    output:
        'alns/bowtie-{k}mer.aln'
    params:
        index = 'indexes/bowtie/human90-{k}mer'
    shell:
        'bowtie -n 3 -l 100 --best --nomaqround --norc -k 1 -p 3 -S --sam-nohead --quiet {params.index} {input[0]} {output}'


rule bowtie2_align:
    input:
        'sample.fastq.gz',
        'indexes/bowtie2/human90-{k}mer.1.bt2',
    output:
        'alns/bowtie2-{k}mer.aln'
    params:
        index = 'indexes/bowtie2/human90-{k}mer'
    shell:
        'bowtie2 --norc -p 3 -k 1 --no-head -x {params.index} -U {input[0]} -S {output}'


rule compute_counts:
    input:
        'alns/{sample}.aln',
        '/Users/laserson/repos/phage_libraries_private/human90/inputs/human90-larman1-input.tsv'
    output:
        'counts/{sample}.tsv'
    run:
        ref_names = []
        ref_counts = []
        with open(input[1], 'r') as ip:
            # burn header
            _ = next(ip)
            for line in ip:
                fields = line.split('\t')
                ref_names.append(fields[0].strip())
                ref_counts.append(round(float(fields[1])))

        counts = {}
        # accumulate counts
        with open(input[0], 'r') as ip:
            for line in ip:
                ref_clone = line.split('\t')[2].strip()
                counts[ref_clone] = counts.get(ref_clone, 0) + 1
        # write counts
        with open(output[0], 'w') as op:
            print('id\tinput\t{0}'.format(wildcards.sample), file=op)
            for (ref_name, ref_count) in zip(ref_names, ref_counts):
                record = '{0}\t{1}\t{2}'.format(
                    ref_name, ref_count, counts.get(ref_name, 0))
                print(record, file=op)


rule kallisto_quant:
    input:
        'sample.fastq.gz',
        'indexes/kallisto/human90-{k}mer.idx',
    output:
        temp('tmp_kallisto_counts_{k}mer'),
        'alns/kallisto-{k}mer.sam',
        'counts/kallisto-{k}mer.tsv'
    params:
        read_len = 50
    shell:
        """
        kallisto quant --pseudobam --single --plaintext --fr-stranded -l {params.read_len} -s 0.1 -t 1 -i {input[1]} -o {output[0]} {input[0]} > {output[1]}
        cp {output[0]}/abundance.tsv {output[2]}
        """


