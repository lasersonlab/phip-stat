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

import os
import argparse
import glob
import subprocess

def submit_to_LSF(queue, duration, LSFopfile, cmd_to_submit, mem_usage=None):
    # wrap command to submit in quotations
    cmd_to_submit = r'"%s"' % cmd_to_submit.strip(r'"')
    LSF_params = {'LSFoutput': LSFopfile,
                  'queue': queue,
                  'duration': duration}
    LSF_cmd = 'bsub -q %(queue)s -W %(duration)s -o %(LSFoutput)s' % LSF_params
    if mem_usage != None:
        LSF_cmd += r' -R "rusage[mem=%d]"' % mem_usage
    cmd = ' '.join([LSF_cmd,cmd_to_submit])
    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    #p.wait()
    return p.stdout.read().split('<')[1].split('>')[0]


argparser = argparse.ArgumentParser(description=None)
argparser.add_argument('-i','--input',required=True)
argparser.add_argument('-o','--output',required=True)
argparser.add_argument('-x','--index',required=True)
argparser.add_argument('-l','--logs',required=True)
argparser.add_argument('-q','--queue',required=True)
argparser.add_argument('-W','--duration',required=True)
args = argparser.parse_args()

input_dir = os.path.abspath(args.input)
output_dir = os.path.abspath(args.output)
os.makedirs(output_dir,mode=0755)
log_dir = os.path.abspath(args.logs)
os.makedirs(log_dir,mode=0755)

params = {
    'index_dir'  : os.path.dirname(args.index),
    'index_name' : os.path.basename(args.index)
}

bowtie_cmd = 'BOWTIE_INDEXES=%(index_dir)s bowtie -n 3 -l 100 --best --nomaqround --norc -k 1 --quiet %(index_name)s %(reads)s %(alignments)s'

for infilename in glob.glob(os.path.join(input_dir,'*.fastq')):
    basename = '.'.join(os.path.basename(infilename).split('.')[:-1])
    outfilename = os.path.join(output_dir,basename+'.aln')
    logfilename = os.path.join(log_dir,basename+'.log')
    params['reads'] = infilename
    params['alignments'] = outfilename
    
    print submit_to_LSF(args.queue, args.duration, logfilename, bowtie_cmd % params)
