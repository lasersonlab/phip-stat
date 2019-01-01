# Copyright 2017 Uri Laserson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def one_base_mutants(seq):
        alphabet = set(['A', 'C', 'G', 'T', 'N'])
        for i in range(len(seq)):
            for alt in alphabet - set([seq[i].upper()]):
                yield seq[:i] + alt + seq[i + 1:]


def load_mapping(path):
    """Load barcode-to-sample mapping

    Expects a tab-delimited file: the first column is the barcode sequence, the
    second column is the sample name.  No header line.
    """
    bc2sample = {}
    with open(path, 'r') as ip:
        for line in ip:
            (bc, sample) = [field.strip() for field in line.split('\t')]
            bc2sample[bc] = sample
    return bc2sample


def edit1_mapping(mapping):
    """Insert edit-distance=1 seqs into barcode-to-sample mapping

    Input should be a dictionary of barcode sequence to sample string.  The
    keys should be in the alphabet ACGTN.
    """
    extended_mapping = mapping.copy()
    for bc in mapping.keys():
        for mut in one_base_mutants(bc):
            if mut in extended_mapping:
                raise ValueError(
                    '{} already in dict: BCs are within 1 edit'.format(mut))
            extended_mapping[mut] = mapping[bc]
    return extended_mapping


def compute_size_factors(counts):
    """Compute size factors from Anders and Huber 2010

    counts is a numpy array
    """
    import numpy as np
    masked = np.ma.masked_equal(counts, 0)
    geom_means = np.ma.exp(np.ma.log(masked).sum(axis=1) / (~masked.mask).sum(axis=1)).data[np.newaxis].T
    return np.ma.median(masked / geom_means, axis=0).data


# fmt: off
# https://github.com/lh3/readfq
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
# fmt: on


def read_fastq_nowrap(fp):
    raw = [None, None, None, None]
    for (i, line) in enumerate(fp):
        mod = i % 4
        if mod == 0 and line[0] != '@':
            raise ValueError(f'{line} expected to start with "@"')
        if mod == 2 and line != '+\n':
            raise ValueError(f'{line} expected to contain just "+"')
        raw[mod] = line
        if mod == 3:
            rec = (raw[0][1:].rstrip(), raw[1].rstrip(), raw[3].rstrip())
            if len(rec[1]) != len(rec[2]):
                raise ValueError(f'{rec} seq and qual are diff length')
            yield rec
    else:
        if mod != 3:
            raise ValueError('wrong number of lines in file')
