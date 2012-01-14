import os
import sys
import random

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd

from numpy import log,sum,pi
from numpy.random import permutation
from scipy.special import gammaln

output_dir = os.path.abspath(sys.argv[3])
os.makedirs(output_dir,mode=0755)

# load data
full_df = pd.read_csv(sys.argv[2],index_col=None)
full_df.columns = pd.Index(['peptide','input','output'])

# optional - subsample rows to make problem smaller
df = full_df.ix[random.sample(xrange(full_df.shape[0]),2000)]

Z = np.array(df['input'])
X = np.array(df['output'])

Z = Z + 1   # add pseudocount
N = len(X)
n = sum(X)

# parameters for w distributions
sigma = float(sys.argv[1])
mu = sigma ** 2


# conditional distribution of w; more MCMC
def sample_w_conditional(w,theta):
    # precompute random variates
    w_star = w * np.random.lognormal(0.1**2,0.1,N) # proposed moves for w_i
    accept = log(np.random.rand(N))  # log of uniform variates for acceptance
    
    # metropolis-hastings
    num_accepted = 0
    for i in permutation(N):
        sum_Zw_not_i = sum(Z*w) - Z[i]*w[i]
        log_ratio = log(w[i]) - log(w_star[i]) - \
                    ((log(w_star[i]) - sigma**2)**2 + (log(w[i]) - sigma**2)**2) / (2*sigma**2) + \
                    (w_star[i] - w[i]) * Z[i] * log(theta[i]) + \
                     gammaln(sum_Zw_not_i + Z[i]*w_star[i]) - gammaln(Z[i]*w_star[i]) - \
                    (gammaln(sum_Zw_not_i + Z[i]*w[i]     ) - gammaln(Z[i]*w[i]    ))
        
        if accept[i] < log_ratio:
            w[i] = w_star[i]
            num_accepted += 1
    
    return float(num_accepted) / N


# log likelihood functions
logfactorial = lambda n: sum(log(range(1,n+1)))
a = -N * log(2*pi*sigma**2) / 2
b = logfactorial(n)
c = sum([logfactorial(x) for x in X])

def loglikelihood_w(w):
    return a - sum(log(w)) - sum((log(w) - mu)**2) / (2*sigma**2)

def loglikelihood_theta(w,theta):
    return sum((Z*w-1)*log(theta)) + gammaln(sum(Z*w)) - sum(gammaln(Z*w))

def loglikelihood_X(theta):
    return b - c + sum(X*log(theta))

def loglikelihood(w,theta): # also Z and X, but they are constant
    # computes joint probability of all variables
    return loglikelihood_w(w) + loglikelihood_theta(w,theta) + loglikelihood_X(theta)


# start Gibbs sampling loop
iterations = 5000

# generate initial configuration on fitness values w
w = np.random.lognormal(mu,sigma,N)

ws = []
loglikelihoods = []
loglikelihoods_w = []
loglikelihoods_theta = []
loglikelihoods_X = []
frac_accepted = []
for i in xrange(iterations):
    if i % 10 == 0:
        sys.stdout.write("%i " % i)
        sys.stdout.flush()
    
    # sample from conditional over theta
    theta = np.random.dirichlet(Z*w+X)
    
    # sample from conditional on fitness w
    # this function modifies w in place
    frac_accepted.append( sample_w_conditional(w,theta) )
    
    # compute log likelihood
    loglikelihoods_w.append(loglikelihood_w(w))
    loglikelihoods_theta.append(loglikelihood_theta(w,theta))
    loglikelihoods_X.append(loglikelihood_X(theta))
    loglikelihoods.append(loglikelihoods_w[-1] + loglikelihoods_theta[-1] + loglikelihoods_X[-1])
    
    # save intermediate w values for later
    ws.append(w.copy())


# figures

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm

# loglikelihood plots
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(loglikelihoods_w,label='w')
ax.plot(loglikelihoods_theta,label='theta')
ax.plot(loglikelihoods_X,label='X')
ax.plot(loglikelihoods,label='combined')
ax.set_xlabel('iteration')
ax.set_title('log likelihoods')
ax.legend(loc=4)
# fig.show()
fig.savefig(os.path.join(output_dir,'loglikelihoods.png'))

# fraction of accepted moves in metropolis-hastings
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(frac_accepted)
ax.set_xlabel('iteration')
ax.set_ylabel('frac moves accepted')
# fig.show()
fig.savefig(os.path.join(output_dir,'frac_accepted.png'))

# ranked w-values
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1,len(w)+1),sorted(np.log10(w),reverse=True),'o-b',clip_on=False)
ax.set_xlabel('rank')
ax.set_ylabel('log10(w)')
# fig.show()
fig.savefig(os.path.join(output_dir,'ranked_w_final.png'))

# histogram evolution of w-values
norm = mpl.colors.normalize(0,len(ws)-1)
fig = plt.figure()
ax = fig.add_subplot(111)
for (i,w_current) in enumerate(ws):
    if i % 200 == 0:
        ax.hist(np.log10(w_current),bins=100,log=True,histtype='step',color=mpl.cm.jet(norm(i)),linewidth=1,alpha=0.5)

# fig.show()
fig.savefig(os.path.join(output_dir,'hist_evolution.png'))

# total weight in w
fig = plt.figure()
ax = fig.add_subplot(111)
w_sums = [sum(np.log10(w_current)) for w_current in ws]
ax.plot(w_sums)
ax.set_xlabel('iteration')
ax.set_ylabel('sum(log10(w))')
# fig.show()
fig.savefig(os.path.join(output_dir,'total_weight.png'))

# plot stds of the w values
ws = np.asarray(ws)
stds = np.std(ws[-1000:,:],axis=0)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sorted(stds))
ax.set_xlabel('rank')
ax.set_ylabel('std of each component in last 1000 w vectors')
# fig.show()
fig.savefig(os.path.join(output_dir,'ranked_stds.png'))

# plot ip/op data colored by estimated w value or variance in estimate
medians = np.median(ws[-1000:,:],axis=0)
clim = max(np.abs(np.min(np.log10(ws))),np.abs(np.max(np.log10(ws))))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Z,X,c=np.log10(medians),cmap=mpl.cm.RdBu,vmin=-clim,vmax=clim,s=25,lw=0.5,clip_on=False,zorder=10)
ax.set_yscale('log')
ax.set_xlabel('input count')
ax.set_ylabel('output count')
ax.axis([0,1200,1,1e3])
bar = fig.colorbar(ax.collections[0])
bar.set_label('log10(w)')
# fig.show()
fig.savefig(os.path.join(output_dir,'raw_data_median_late_ws.png'))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Z,X,c=np.abs(np.log10(ws[5,:])),cmap=mpl.cm.RdBu,vmin=-clim,vmax=clim,s=25,lw=0.5,clip_on=False,zorder=10)
ax.set_yscale('log')
ax.set_xlabel('input count')
ax.set_ylabel('output count')
ax.axis([0,1200,1,1e3])
bar = fig.colorbar(ax.collections[0])
bar.set_label('log10(w)')
# fig.show()
fig.savefig(os.path.join(output_dir,'raw_data_early_ws.png'))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Z,X,c=-stds,cmap=plt.jet(),s=25,lw=0.5,clip_on=False,zorder=10)
ax.set_yscale('log')
ax.set_xlabel('input count')
ax.set_ylabel('output count')
ax.axis([0,1200,1,1e3])
bar = fig.colorbar(ax.collections[0])
bar.set_label('std(w)')
# fig.show()
fig.savefig(os.path.join(output_dir,'raw_data_variance_ws.png'))

# correlate stds with medians:
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.log10(medians),stds)
ax.set_xlabel('log10(median w)')
ax.set_ylabel('-1*std(w)')
# fig.show()
fig.savefig(os.path.join(output_dir,'median_w_vs_std_w.png'))

# plot real and simulated data from the final model
theta_start = np.random.dirichlet(Z*ws[5])
Xsim_start = np.random.multinomial(n,theta)

theta_end = np.random.dirichlet(Z*medians)
Xsim_end = np.random.multinomial(n,theta)

fig = plt.figure()

ax = fig.add_subplot(311)
ax.scatter(Z,X,c='k',s=3,lw=0)
ax.set_yscale('log')
ax.axis([0,2000,1,1e3])
ax.set_ylabel('real data')

ax = fig.add_subplot(312)
plt.scatter(Z,Xsim_start,c='k',s=3,lw=0)
ax.set_yscale('log')
ax.axis([0,2000,1,1e3])
ax.set_ylabel('initial sim')

ax = fig.add_subplot(313)
ax.scatter(Z,Xsim_end,c='k',s=3,lw=0)
ax.set_yscale('log')
ax.axis([0,2000,1,1e3])
ax.set_ylabel('final sim')

# fig.show()
fig.savefig(os.path.join(output_dir,'generated_data.png'))


# NP_002745.1_13 is the positive control

