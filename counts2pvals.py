#! /usr/bin/env python

import sys
import argparse

import numpy as np
import scipy as sp
import scipy.optimize

argparser = argparse.ArgumentParser(description=None)
argparser.add_argument('-i','--input',required=True)
argparser.add_argument('-o','--output',required=True)
args = argparser.parse_args()

inhandle = open(args.input,'r')
outhandle = open(args.output,'w')

def GP_lambda_likelihood(counts):
    # compute inputs to likelihood function
    (nx,x) = np.histogram(counts,bins=range(max(counts)+2))
    x = x[:-1]
    n = len(counts)
    x_bar = sum(counts) / float(n)
    
    # check condition for unique root
    if sum(nx[2:]*x[2:]*(x[2:]-1)) - n*(x_bar**2) <= 0:
        sys.stderr.write("Condition for uniqueness of lambda is not met.\n    x: %s\n    n: %s\n    x_bar: %s\n" % (x,n,x_bar)); sys.stderr.flush()
    
    return lambda lam: sum(nx*(x*(x-1)/(x_bar+(x-x_bar)*lam))) - n*x_bar

#################################################

# DEPRECATED
def GP_pmf(x,theta,lambd):
    log = np.log
    logP = log(theta) + (x-1)*log(theta+x*lambd) - (theta+x*lambd) - np.sum(log(np.arange(1,x+1)))
    return np.exp(logP)

# DEPRECATED
def GP_cdf(x,theta,lambd):
    return np.sum([GP_pmf(y,theta,lambd) for y in np.arange(x+1)])

# DEPRECATED
def GP_sf(x,theta,lambd):
    return reduce(lambda x,y: x-y, [1]+[GP_pmf(y,theta,lambd) for y in np.arange(x+1)])

# DEPRECATED
def GP_cdf_parallel(x,theta,lambd):
    log = np.log
    y = np.arange(x+1)
    logPxs = log(theta) + (y-1)*log(theta+y*lambd) - (theta+y*lambd) - np.sum(np.log((np.tri(x+1) * np.arange(x+1) + np.tri(x+1,k=-1).transpose())[:,1:]),axis=1)
    return np.sum(np.exp(logPxs))

#################################################

def log_GP_pmf(x,theta,lambd):
    log = np.log
    logP = log(theta) + (x-1)*log(theta+x*lambd) - (theta+x*lambd) - np.sum(log(np.arange(1,x+1)))
    return logP

def log_GP_sf(x,theta,lambd):
    extensions = 20
    start = x + 1
    end = x + 100
    pmf = [log_GP_pmf(y,theta,lambd) for y in xrange(start,end)]
    while extensions > 0:
        accum = np.logaddexp.accumulate( pmf )
        if accum[-1] == accum[-2]: return accum[-1]
        start = end
        end += 100
        pmf += [log_GP_pmf(y,theta,lambd) for y in xrange(start,end)]
        extensions -= 1
    # raise ValueError
    return np.nan


# Load data
sys.stderr.write("Loading data...\n"); sys.stderr.flush()
clones = []
input_counts = []
output_counts = []
for line in inhandle:
    if line.startswith('#'): continue
    data = line.split(',')
    clones.append( data[0].strip() )
    input_counts.append( int(data[1]) )
    output_counts.append( np.int_(data[2:]) )

input_counts = np.asarray(input_counts)
output_counts = np.asarray(output_counts)
uniq_input_values = list(set(input_counts))
sys.stderr.write("Num clones: %s\nInput vec shape: %s\nOutput array shape: %s\n" % (len(clones),input_counts.shape,output_counts.shape)); sys.stderr.flush()

# Estimate generalized Poisson distributions for every input count
sys.stderr.write("Computing lambdas and thetas for %i different input values...\n" % len(uniq_input_values)); sys.stderr.flush()
lambdas = []
thetas = []
idxs = []
for i in xrange(output_counts.shape[1]):    # for each output column...
    sys.stderr.write("    working on output column %i\n" % i); sys.stderr.flush()
    lambdas.append([])
    thetas.append([])
    idxs.append([])
    for input_value in uniq_input_values:   # ...compute lambdas/thetas
        # compute lambda
        curr_counts = output_counts[np.logical_and(input_counts == input_value,output_counts[:,i] > 1),i]
        if len(curr_counts) < 50:
            continue
        idxs[-1].append(input_value)
        H = GP_lambda_likelihood(curr_counts)
        lambd = sp.optimize.fsolve(H,0.5)[0]
        lambdas[-1].append( lambd )
    
        # compute theta
        n = len(curr_counts)
        x_bar = sum(curr_counts) / float(n)
        theta =  x_bar * (1 - lambd)
        thetas[-1].append( theta )

# Regression on all of the theta and lambda values computed
sys.stderr.write("Regression on lambdas and thetas...\n"); sys.stderr.flush()
lambda_fits = []
theta_fits = []
for i in xrange(output_counts.shape[1]):
    sys.stderr.write("    working on output column %i\n" % i); sys.stderr.flush()
    lambda_fit = lambda x: np.mean(lambdas[i])
    coeffs = np.polyfit(idxs[i],thetas[i],1)
    theta_fit = lambda x: coeffs[0]*x + coeffs[1]
    lambda_fits.append(lambda_fit)
    theta_fits.append(theta_fit)

# Precompute CDF for possible input-output combinations
sys.stderr.write("Precomputing pval combos..."); sys.stderr.flush()
uniq_combos = []
for i in xrange(output_counts.shape[1]):
    uniq_combos.append( set(zip(input_counts,output_counts[:,i])) )
sys.stderr.write("computing %i combos\n" % sum([len(u) for u in uniq_combos])); sys.stderr.flush()
log10pval_hash = {}
j = 0
for (i,u) in enumerate(uniq_combos):
    for (ic,oc) in u:
        if j % 1000 == 0: sys.stderr.write("...computed %i p-vals\n" % j); sys.stderr.flush()
        log_pval = log_GP_sf(oc,theta_fits[i](ic),lambda_fits[i](ic))
        log10pval_hash[(i,ic,oc)] = log_pval * np.log10( np.e ) * -1.
        # pval = GP_sf(oc,theta_fits[i](ic),lambda_fits[i](ic))
        # log10pval_hash[(i,ic,oc)] = -np.log10(pval)
        j += 1

# Compute p-values for each clone using regressed GP parameters
sys.stderr.write("Computing actual pvals...\n"); sys.stderr.flush()
for (clone,ic,ocs) in zip(clones,input_counts,output_counts):
    output_string = clone
    for (i,oc) in enumerate(ocs): output_string += ",%f" % log10pval_hash[(i,ic,oc)]
    print >>outhandle, output_string

# DEBUG
# logpvals = np.asarray(logpvals)
# clones = np.asarray(clones)
# for (c,p) in zip(clones[logpvals>6],logpvals[logpvals>8]):
#     print "%s\t%s" % (c,p)
