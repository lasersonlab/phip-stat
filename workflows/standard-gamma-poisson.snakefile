import re
import gzip
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


def readfq(fp): # this is a generator function
    last = None # this is a buffer keeping the last unprocessed line
    while True: # mimic closure; is it a bad idea?
        if not last: # the first record or a record following a fastq
            for l in fp: # search for the start of the next record
                if l[0] in '>@': # fasta/q header line
                    last = l[:-1] # save this line
                    break
        if not last: break
        name, seqs, last = last[1:].partition(" ")[0], [], None
        for l in fp: # read the sequence
            if l[0] in '@+>':
                last = l[:-1]
                break
            seqs.append(l[:-1])
        if not last or last[0] != '+': # this is a fasta record
            yield name, ''.join(seqs), None # yield a fasta record
            if not last: break
        else: # this is a fastq record
            seq, leng, seqs = ''.join(seqs), 0, []
            for l in fp: # read the quality
                seqs.append(l[:-1])
                leng += len(l) - 1
                if leng >= len(seq): # have read enough quality
                    last = None
                    yield name, seq, ''.join(seqs); # yield a fastq record
                    break
            if last: # reach EOF before reading enough quality
                yield name, seq, None # yield a fasta record instead
                break


###############################################################################
# <CONFIGURATION>
# define config['samples'] as dict[sample_name, list(sample_abs_path)]
# e.g.,
# config['samples'] = {'sample1': ['sample1.1.fastq.gz', 'sample1.2.fastq.gz']}
config['samples'] = {}
for f in glob('/Users/laserson/Downloads/BaseSpace/phip-14-48923876/*/*.fastq.gz'):
    p = parse_illumina_fastq_name(f)
    config['samples'].setdefault(p.sample, []).append(p.path)

# NOTE: this should be an "effective read length": the number of bases to use
# from the reference and from the reads. e.g., if seqencing 150 bp on the
# human36 lib, some of the read will be of the adaptor
config['read_length'] =
config['reference_fasta'] =
# </CONFIGURATION>
###############################################################################


rule all:
    input:
        'normalized_counts.tsv',
        'counts.tsv',
        'gamma-poisson'


rule compute_counts:
    input:
        lambda wildcards: config['samples'][wildcards.sample]
    output:
        'counts/{sample}.counts.tsv',
    params:
        read_length = config['read_length'],
        reference_fasta = config['reference_fasta']
    run:
        from collections import Counter, OrderedDict
        # load reference
        seq_to_ref = OrderedDict()
        with open(params.reference_fasta, 'r') as ip:
            for (ref_name, seq, _) in readfq(ip):
                seq_to_ref[seq[:params.read_length]] = ref_name

        num_reads = 0
        num_matched = 0
        counts = Counter()
        for input_file in input:
            with gzip.open(input_file, 'rt') as ip:
                for (name, seq, _) in tqdm(readfq(ip)):
                    num_reads += 1
                    refname = seq_to_ref.get(seq)
                    if refname is not None:
                        num_matched += 1
                        counts[refname] += 1

        print(
            'num_reads: {}\nnum_matched: {}\nfrac_matched: {}'.format(
                num_reads, num_matched, num_matched / num_reads),
            file=sys.stderr)

        with open(output[0], 'w') as op:
            print('id\t{}'.format(wildcards.sample), file=op)
            for (_, refname) in seq_to_ref.items():
                print('{}\t{}'.format(refname, counts[refname]), file=op)


rule merge_counts:
    input:
        expand('counts/{sample}.counts.tsv', sample=config['samples'])
    output:
        'counts.tsv'
    params:
        input_dir = 'counts'
    shell:
        'phip merge-columns -i {params.input_dir} -o {output} -m iter'


rule normalize_counts:
    input:
        'counts.tsv'
    output:
        'normalized_counts.tsv'
    shell:
        'phip normalize-counts -i {input} -o {output} -m size-factors'


rule compute_gamma_poisson:
    input:
        'normalized_counts.tsv'
    output:
        directory('gamma-poisson')
    params:
        trim_percentile = 99.9
    shell:
        'phip gamma-poisson-model -t {params.trim_percentile} -i {input} -o {output}'
