#! /usr/bin/env python

import os
import argparse
import glob
import re

bcre = re.compile(r'#(.*)/')

def hamming1(s):
    s = s.upper()
    alts = {'A':'CGTN','C':'AGTN','G':'ACTN','T':'ACGN'}
    mutants = []
    for i in range(len(s)):
        for alt in alts[s[i]]:
            mutant = s[:i] + alt + s[i+1:]
            mutants.append(mutant)
    return mutants

argparser = argparse.ArgumentParser(description=None)
argparser.add_argument('-i','--input',required=True)
argparser.add_argument('-o','--output',required=True)
argparser.add_argument('-m','--mapping',required=True)
args = argparser.parse_args()

input_dir = os.path.abspath(args.input)
output_dir = os.path.abspath(args.output)
os.makedirs(output_dir,mode=0755)
mapping_file = args.mapping

# load barcode mapping and open outhandles
barcode2sample = {}
outhandles = {}
with open(mapping_file,'r') as ip:
    for line in ip:
        data = line.split()
        bc = data[0]
        sample = data[1]
        barcode2sample[bc] = sample
        for mut in hamming1(bc):
            barcode2sample[mut] = sample
        outhandles[sample] = open(os.path.join(output_dir,data[1]+'.aln'),'w')

# iterate through alignments
for infilename in glob.glob(os.path.join(input_dir,'*.aln')):
    with open(infilename,'r') as ip:
        for line in ip:
            # read_name = line.split()[0]
            # bc = bcre.search(read_name).group(1)
            bc = line.split()[1].split(':')[-1]
            try:
                sample = barcode2sample[bc]
            except KeyError:
                continue
            outhandles[sample].write(line)
