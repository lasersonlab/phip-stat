#! /usr/bin/env python

import os
import sys
import glob
import argparse


argparser = argparse.ArgumentParser(description=None)
argparser.add_argument('-i','--input',required=True)
argparser.add_argument('-o','--output',required=True)
argparser.add_argument('-r','--refcounts',required=True)
args = argparser.parse_args()

input_dir = os.path.abspath(args.input)
output_dir = os.path.abspath(args.output)
os.makedirs(output_dir,mode=0755)
reference_count_file = args.refcounts

# load reference counts
reference_names = []
reference_counts = []
with open(reference_count_file,'r') as ip:
    for line in ip:
        data = line.split(',')
        reference_names.append(data[0].strip())
        reference_counts.append(int(data[1]))

# generate count dict
for infilename in glob.glob(os.path.join(input_dir,'*.aln')):
    sys.stdout.write("%s\n" % infilename)
    sys.stdout.flush()
    counts = {}
    sample = '.'.join(os.path.basename(infilename).split('.')[:-1])
    with open(infilename,'r') as ip:
        for line in ip:
            ref_clone = line.split('\t')[2].strip()
            counts[ref_clone] = counts.get(ref_clone,0) + 1
    
    # output counts
    output_file = os.path.join(output_dir,"%s.csv" % sample)
    with open(output_file,'w') as op:
        print >>op, '# ' + ','.join(["ref_clone","ref_input",sample])  # header line
        for (ref_clone,ref_count) in zip(reference_names,reference_counts):
            record = [ref_clone,str(ref_count),str(counts.get(ref_clone,0))]
            print >>op, ','.join(record)
