#! /usr/bin/env python

import os
import argparse
import glob
import re

bcre = re.compile(r'#(.*)/')

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
        barcode2sample[data[0]] = data[1]
        outhandles[data[1]] = open(os.path.join(output_dir,data[1]+'.aln'),'w')

# iterate through alignments
for infilename in glob.glob(os.path.join(input_dir,'*.aln')):
    with open(infilename,'r') as ip:
        for line in ip:
            read_name = line.split()[0]
            bc = bcre.search(read_name).group(1)
            try:
                sample = barcode2sample[bc]
            except KeyError:
                continue
            outhandles[sample].write(line)
