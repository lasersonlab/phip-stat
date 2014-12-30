#! /usr/bin/env python
# Copyright 2014 Uri Laserson
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

# =====================
# = lazy zip() method =
# =====================

import os
import sys
import glob
import argparse
import itertools
import string

header = lambda f: os.path.splitext(os.path.basename(f))[0]

argparser = argparse.ArgumentParser(description=None)
argparser.add_argument('-i','--input',required=True)
argparser.add_argument('-o','--output',required=True)
argparser.add_argument('-f','--field',type=int,default=1)
args = argparser.parse_args()

input_dir = os.path.abspath(args.input)
output_file = os.path.abspath(args.output)

input_files = glob.glob(os.path.join(input_dir,'*.csv'))
file_iterators = [open(f,'r') for f in input_files]
file_headers = map(header,input_files)

with open(output_file,'w') as op:
    # write header
    print >>op, ','.join(['']+file_headers)
    
    # iterate through lines
    for lines in itertools.izip(*file_iterators):
        # ignore comment header lines; only checks first file
        if lines[0].startswith('#'): continue
        data = [map(string.strip,line.split(',')) for line in lines]
        # check that join column is the same
        for datum in data[1:]: assert data[0][0] == datum[0]
        print >>op, ','.join([data[0][0]]+[datum[args.field] for datum in data])


# ================
# = zip() method =
# ================

# import os
# import sys
# import glob
# import argparse
# 
# header = lambda f: os.path.splitext(os.path.basename(f))[0]
# 
# argparser = argparse.ArgumentParser(description=None)
# argparser.add_argument('-i','--input',required=True)
# argparser.add_argument('-o','--output',required=True)
# argparser.add_argument('-f','--field',type=int,default=1)
# args = argparser.parse_args()
# 
# input_dir = os.path.abspath(args.input)
# output_file = os.path.abspath(args.output)
# 
# input_files = glob.glob(os.path.join(input_dir,'*.csv'))
# 
# join_column = None
# merged_data = []
# for infilename in input_files:
#     sys.stdout.write("Reading %s..." % infilename)
#     sys.stdout.flush()
#     
#     file_data = []
#     with open(infilename,'r') as ip:
#         for line in ip:
#             if line.startswith('#'): continue
#             row_data = [d.strip() for d in line.split(',')]
#             file_data.append(row_data)
#     col_data = map(list,zip(*file_data))
#     
#     if join_column == None:
#         join_column = col_data[0]
#     else:
#         assert join_column == col_data[0]
#     
#     merged_data.append([header(infilename)] + col_data[args.field])
#     
#     sys.stdout.write("finished.\n")
#     sys.stdout.flush()
# 
# merged_data = [[''] + join_column] + merged_data
# 
# output_data = zip(*merged_data)
# with open(output_file,'w') as op:
#     for row in output_data:
#         print >>op, ','.join(row)




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
