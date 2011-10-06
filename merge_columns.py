#! /usr/bin/env python

import os
import sys
import glob
import argparse
import subprocess

header = lambda f: os.path.splitext(os.path.basename(f))[0]

argparser = argparse.ArgumentParser(description=None)
argparser.add_argument('-i','--input',required=True)
argparser.add_argument('-o','--output',required=True)
argparser.add_argument('-f','--field',type=int,default=1)
args = argparser.parse_args()

input_dir = os.path.abspath(args.input)
output_file = os.path.abspath(args.output)

input_files = glob.glob(os.path.join(input_dir,'*.csv'))

join_column = None
merged_data = []
for infilename in input_files:
    sys.stdout.write("Reading %s..." % infilename)
    sys.stdout.flush()
    
    file_data = []
    with open(infilename,'r') as ip:
        for line in ip:
            if line.startswith('#'): continue
            row_data = [d.strip() for d in line.split(',')]
            file_data.append(row_data)
    col_data = map(list,zip(*file_data))
    
    if join_column == None:
        join_column = col_data[0]
    else:
        assert join_column == col_data[0]
    
    merged_data.append([header(infilename)] + col_data[args.field])
    
    sys.stdout.write("finished.\n")
    sys.stdout.flush()

merged_data = [[''] + join_column] + merged_data

output_data = zip(*merged_data)
with open(output_file,'w') as op:
    for row in output_data:
        print >>op, ','.join(row)




# ===============
# = bash method =
# ===============

# import os
# import glob
# import argparse
# import subprocess
# 
# argparser = argparse.ArgumentParser(description=None)
# argparser.add_argument('-i','--input',required=True)
# argparser.add_argument('-o','--output',required=True)
# args = argparser.parse_args()
# 
# input_dir = os.path.abspath(args.input)
# output_file = os.path.abspath(args.output)
# 
# input_files = glob.glob(os.path.join(input_dir,'*.csv'))
# 
# join_cmd = 'cat %s' % input_files[0]
# for infilename in input_files[1:]:
#     join_cmd += " | join -t, - %s" % infilename
# 
# join_cmd += ' > %s' % output_file
# 
# p = subprocess.Popen(join_cmd,shell=True)
# p.wait()
