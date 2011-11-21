import sys

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd

# set parameters
t = 4
alpha = 10
# x_m = 1, always

# load data
df = pd.read_csv('BC.serum.HC.3257.1.csv',index_col=0)
df.columns = pd.Index(['input','output'])
X = df['input']
Z = df['output']
Znorm = Z / float(Z.sum())
N = len(X)

def sample_w_conditional(w,theta):
    proposals = w + np.random.randn(N) # proposed moves for w_i
    proposals = np.choose(proposals < 1,(proposals,2-proposals)) # reflect about w == 1
    accept = np.log(np.random.rand(N))  # log of uniform variates for acceptance
    log_ratio = (t+1) * (np.log(w) - np.log(proposals)) + (proposals - w) * alpha * Znorm * np.log(theta)
    new_w = np.choose(accept < log_ratio,(w,proposals))
    return new_w

def loglikelihood(w,theta):
    return np.sum(np.log(sp.stats.pareto.pdf(w,t))) + np.sum((alpha*Znorm*w - 1) * np.log(theta) - sp.special.gammaln(alpha*Znorm*w)) + sp.special.gammaln(np.sum(alpha*Znorm*w))


# optional - aggregate data into proteins
# TODO

# generate initial configuration on fitness values w
w = np.random.pareto(t,N) + 1

# start Gibbs sampling loop
iterations = 500

likelihoods = []
for i in xrange(iterations):
    if i % 10 == 0:
        sys.stdout.write("%i " % i)
        sys.stdout.flush()
    
    # sample from conditional over theta
    theta = np.random.dirichlet(alpha*Znorm*w+X)
    
    # sample from conditional on fitness w
    w = sample_w_conditional(w,theta)
    
    # compute log likelihood
    likelihoods.append(loglikelihood(w,theta))
    
    # write w values to disk





