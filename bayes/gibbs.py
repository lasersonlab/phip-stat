import sys

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd

from numpy import log,sum
from numpy.random import permutation
from scipy.special import gammaln

# set parameters
t = 1
alpha = 1000
# x_m = 1, always

# load data
df = pd.read_csv('BC.serum.HC.3257.1.csv',index_col=None)
df.columns = pd.Index(['peptide','input','output'])

# optional - aggregate data into proteins
df['protein'] = df['peptide'].apply(lambda s: '_'.join(s.split('_')[:-1]))
df['ratio'] = np.float_(df['output'])/df['input']
grouped_by_protein = df.groupby('protein')
indices = [max(group,key=lambda i: df.ix[i]['ratio']) for group in grouped_by_protein.groups.itervalues()]
agg_by_protein = df.reindex(indices)

Z = agg_by_protein['input']
X = agg_by_protein['output']

# Z = df['input']
# X = df['output']

Z = Z + 1   # add pseudocount
Znorm = Z / float(Z.sum())
N = len(X)
n = sum(X)

def sample_w_conditional(w,theta):
    # precompute random variates
    w_star = w + np.random.randn(N) # proposed moves for w_i
    w_star = np.choose(w_star < 1,(w_star,2-w_star)) # reflect about w == 1
    accept = log(np.random.rand(N))  # log of uniform variates for acceptance
    
    # metropolis-hastings
    for i in permutation(N):
        sum_aZw_not_i = alpha * (sum(Znorm*w) - Znorm[i]*w[i])
        log_ratio = (t+1) * (log(w[i]) - log(w_star[i])) + \
                    (w_star[i] - w[i]) * alpha * Znorm[i] * log(theta[i]) + \
                    gammaln(alpha*Znorm[i]*w[i]) - gammaln(alpha*Znorm[i]*w_star[i]) + \
                    gammaln(alpha*Znorm[i]*w_star[i] + sum_aZw_not_i) - gammaln(alpha*Znorm[i]*w[i] + sum_aZw_not_i)
        
        if accept[i] < log_ratio:
            w[i] = w_star[i]

# log likelihood functions
logfactorial = lambda n: sum(log(range(1,n+1)))
a = N*log(t)
b = logfactorial(n)
c = sum([logfactorial(x) for x in X])

def loglikelihood_w(w):
    return a - (t+1)*sum(log(w))

def loglikelihood_theta(w,theta):
    return sum((alpha*Znorm*w-1)*log(theta)) + gammaln(sum(alpha*Znorm*w)) - sum(gammaln(alpha*Znorm*w))

def loglikelihood_X(theta):
    return b - c + sum(X*log(theta))

def loglikelihood(w,theta): # also Z and X, but they are constant
    # computes joint probability of all variables
    return loglikelihood_w(w) + loglikelihood_theta(w,theta) + loglikelihood_X(theta)


# generate initial configuration on fitness values w
w = np.random.pareto(t,N) + 1

# start Gibbs sampling loop
iterations = 25

loglikelihoods = []
loglikelihoods_w = []
loglikelihoods_theta = []
loglikelihoods_X = []
for i in xrange(iterations):
    if i % 1 == 0:
        sys.stdout.write("%i " % i)
        sys.stdout.flush()
    
    # sample from conditional over theta
    theta = np.random.dirichlet(alpha*Znorm*w+X)
    
    # sample from conditional on fitness w
    # this function modifies w in place
    sample_w_conditional(w,theta)
    
    # compute log likelihood
    # likelihoods.append(loglikelihood(w,theta))
    loglikelihoods_w.append(loglikelihood_w(w))
    loglikelihoods_theta.append(loglikelihood_theta(w,theta))
    loglikelihoods_X.append(loglikelihood_X(theta))
    loglikelihoods.append(loglikelihoods_w[-1] + loglikelihoods_theta[-1] + loglikelihoods_X[-1])
    
    # write w values to disk

plt.close('all')
plt.plot(loglikelihoods_w,label='w')
plt.plot(loglikelihoods_theta,label='theta')
plt.plot(loglikelihoods_X,label='X')
plt.plot(loglikelihoods,label='combined')
plt.legend()
plt.gcf().show()

plt.figure()
plt.hist(theta,bins=100,log=True)
plt.gcf().show()

plt.figure()
plt.hist(w,bins=100,log=True)
plt.gcf().show()
