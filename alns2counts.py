import os
import argparse
import glob

argparser = argparse.ArgumentParser(description=None)
argparser.add_argument('-i','--input',required=True)
argparser.add_argument('-o','--output',required=True)
argparser.add_argument('-r','--refcounts',required=True)
args = argparser.parse_args()

input_dir = os.path.abspath(args.input)
output_file = os.path.abspath(args.output)
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
samples = []
counts = {}
for infilename in glob.glob(os.path.join(input_dir,'*.aln')):
    basename = '.'.join(os.path.basename(infilename).split('.')[:-1])
    samples.append(basename)
    counts[basename] = {}
    with open(infilename,'r') as ip:
        for line in ip:
            ref_clone = line.split('\t')[2].strip()
            counts[basename][ref_clone] = counts.get(ref_clone,0) + 1

# output counts
with open(output_file,'w') as op:
    print >>op, '# ' + ','.join(["ref_clone","ref_input"]+samples)  # header line
    for (ref_clone,ref_count) in zip(reference_names,reference_counts):
        record = ['ref_clone',str(ref_count)]
        for sample in samples:
            record.append(str(counts[sample].get(ref_clone,0)))
        print >>op, ','.join(record)
