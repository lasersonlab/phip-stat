import sys

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd

from numpy import log,sum
from numpy.random import permutation
from scipy.special import gammaln

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

# take a fraction of data
# X = X[0:200]
# Z = Z[0:200]

Z = Z + 1   # add pseudocount
Znorm = Z / float(Z.sum())
N = len(X)
n = sum(X)

# set parameters
t = 4
alpha = sum(X)
# x_m = 1, always


def sample_w_conditional(w,theta):
    # precompute random variates
    w_star = w + np.random.randn(N) # proposed moves for w_i
    w_star = np.choose(w_star < 1,(w_star,2-w_star)) # reflect about w == 1
    accept = log(np.random.rand(N))  # log of uniform variates for acceptance
    
    # metropolis-hastings
    num_accepted = 0
    for i in permutation(N):
        sum_aZw_not_i = alpha * (sum(Znorm*w) - Znorm[i]*w[i])
        log_ratio = (t+1) * (log(w[i]) - log(w_star[i])) + \
                    (w_star[i] - w[i]) * alpha * Znorm[i] * log(theta[i]) + \
                    gammaln(alpha*Znorm[i]*w[i]) - gammaln(alpha*Znorm[i]*w_star[i]) + \
                    gammaln(alpha*Znorm[i]*w_star[i] + sum_aZw_not_i) - gammaln(alpha*Znorm[i]*w[i] + sum_aZw_not_i)
        
        if accept[i] < log_ratio:
            w[i] = w_star[i]
            num_accepted += 1
    
    return float(num_accepted) / N

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
iterations = 3000

ws = []
loglikelihoods = []
loglikelihoods_w = []
loglikelihoods_theta = []
loglikelihoods_X = []
frac_accepted = []
for i in xrange(iterations):
    if i % 1 == 0:
        sys.stdout.write("%i " % i)
        sys.stdout.flush()
    
    # sample from conditional over theta
    theta = np.random.dirichlet(alpha*Znorm*w+X)
    
    # sample from conditional on fitness w
    # this function modifies w in place
    frac_accepted.append( sample_w_conditional(w,theta) )
    
    # compute log likelihood
    # likelihoods.append(loglikelihood(w,theta))
    loglikelihoods_w.append(loglikelihood_w(w))
    loglikelihoods_theta.append(loglikelihood_theta(w,theta))
    loglikelihoods_X.append(loglikelihood_X(theta))
    loglikelihoods.append(loglikelihoods_w[-1] + loglikelihoods_theta[-1] + loglikelihoods_X[-1])
    
    # write w values to disk
    ws.append(w.copy())


import matplotlib.pyplot as plt
plt.close('all')
plt.figure()
plt.plot(loglikelihoods_w,label='w')
plt.plot(loglikelihoods_theta,label='theta')
plt.plot(loglikelihoods_X,label='X')
plt.plot(loglikelihoods,label='combined')
plt.legend(loc=4)
plt.gcf().show()
plt.gcf().savefig('loglikelihood_full.png')

plt.figure()
plt.plot(frac_accepted)
plt.gcf().show()
plt.gcf().savefig('frac_accepted_full.png')

plt.figure()
plt.hist(theta,bins=100,log=True)
plt.gcf().show()
plt.gcf().savefig('theta_final_hist.png')

plt.figure()
plt.hist(w,bins=100,log=True)
plt.gcf().show()
plt.gcf().savefig('w_final_hist.png')

plt.figure()
plt.plot(range(1,len(w)+1),sorted(w,reverse=True),'o-b',clip_on=False)
plt.xscale('log')
plt.gcf().show()
plt.gcf().savefig('w_final_values.png')

import matplotlib as mpl
import matplotlib.colors
import matplotlib.cm
norm = mpl.colors.normalize(0,len(ws)-1)
plt.figure()
for (i,w_current) in enumerate(ws):
    if i % 200 == 0:
        temp = plt.hist(np.log10(w_current),bins=100,log=True,histtype='step',color=mpl.cm.jet(norm(i)),linewidth=1,alpha=0.5)
        # temp = plt.hist(w_current,bins=100,log=True,histtype='step',color=mpl.cm.jet(norm(i)),linewidth=1,alpha=0.5)

# plt.axis([0,2500,1,1e5])
plt.gcf().show()
plt.gcf().savefig('ws_hist_evolution_log_t4.png')
# plt.gcf().savefig('ws_hist_evolution_log.png')

plt.figure()
w_sums = [sum(w_current) for w_current in ws]
plt.plot(w_sums)
plt.gcf().show()

# NP_002745.1_13 is the positive control