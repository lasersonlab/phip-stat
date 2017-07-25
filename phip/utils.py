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
            (bc, sample) = [field.strip() for field in line.split()]
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
                raise ValueError(f'{mut} already in dict: BCs are within 1 edit')
            extended_mapping[mut] = mapping[bc]
    return extended_mapping
