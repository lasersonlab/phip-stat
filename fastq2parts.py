import os
import argparse

from Bio import SeqIO

argparser = argparse.ArgumentParser(description=None)
argparser.add_argument('-i','--input',required=True)
argparser.add_argument('-o','--output',required=True)
argparser.add_argument('-p','--packetsize',type=int,required=True)
args = argparser.parse_args()

input_filename = args.input
output_dir = os.path.abspath(args.output)
os.makedirs(output_dir,mode=0755)
packetsize = args.packetsize

num_processed = 0
file_num = 1
outfilename = os.path.join(work_dir,parts_dir,'part.%s.fastq' % file_num)
for record in SeqIO.parse(input_filename,'fastq'):
    if num_processed == 0:
        op = open(outfilename,'w')
    
    print >>op, record
    num_processed += 1
    
    if num_processed == packetsize:
        op.close()
        num_processed = 0
        file_num += 1
        outfilename = os.path.join(work_dir,parts_dir,'part.%s.fastq' % file_num)

if not op.closed:
    op.close()
