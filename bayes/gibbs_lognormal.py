import os
import sys
import random

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd

from numpy import log,log10,sum,pi
from numpy.random import permutation
from scipy.special import gammaln

import statstools

interactive = False

output_dir = os.path.abspath(sys.argv[3])
os.makedirs(output_dir,mode=0755)

# load data
full_df = pd.read_csv(sys.argv[2],index_col=None)
full_df.columns = pd.Index(['peptide','input','output'])

# optional - subsample rows to make problem smaller
random_idxs = random.sample(xrange(full_df.shape[0]),500)
df = full_df.ix[random_idxs]
# df = full_df

Z = np.array(df['input'])
X = np.array(df['output'])

Z = Z + 1   # add pseudocount
N = len(X)
n = sum(X)

# parameters for prior w distributions
alpha = 0.05
sigma = float(sys.argv[1])
# mu = sigma ** 2
mu = 0

# generate some synthetic w and output data
wtruth = np.random.lognormal(0,0.5,N)
thetatruth = np.random.dirichlet(Z*wtruth)
X = np.random.multinomial(n,thetatruth)

# conditional distribution of w; more MCMC
def sample_w_conditional(w,theta):
    # precompute random variates
    r = np.random.normal(0,0.1,N)
    w_star = w * np.exp(r)
    accept = log(np.random.rand(N))  # log of uniform variates for acceptance
    
    # metropolis-hastings
    num_accepted = 0
    for i in permutation(N):
        sum_Zw_not_i = sum(alpha*Z*w) - alpha*Z[i]*w[i]
        log_ratio = log(w[i]) - log(w_star[i]) - \
                    ((log(w_star[i]) - mu)**2 + (log(w[i]) - mu)**2) / (2*sigma**2) + \
                    (w_star[i] - w[i]) * alpha * Z[i] * log(theta[i]) + \
                     gammaln(sum_Zw_not_i + alpha*Z[i]*w_star[i]) - gammaln(alpha*Z[i]*w_star[i]) - \
                    (gammaln(sum_Zw_not_i + alpha*Z[i]*w[i]     ) - gammaln(alpha*Z[i]*w[i]    ))
        
        if accept[i] < log_ratio + r[i]: # note: the 2nd term is a Jacobian
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
    return sum((alpha*Z*w-1)*log(theta)) + gammaln(sum(alpha*Z*w)) - sum(gammaln(alpha*Z*w))

def loglikelihood_X(theta):
    return b - c + sum(X*log(theta))

def loglikelihood(w,theta): # also Z and X, but they are constant
    # computes joint probability of all variables
    return loglikelihood_w(w) + loglikelihood_theta(w,theta) + loglikelihood_X(theta)


# start Gibbs sampling loop
iterations = 10000

# generate initial configuration on fitness values w
w = np.random.lognormal(mu,sigma,N)

ws = []
thetas = []
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
    theta = np.random.dirichlet(alpha*Z*w+X)
    
    # sample from conditional on fitness w
    # this function modifies w in place
    frac_accepted.append( sample_w_conditional(w,theta) )
    
    # compute log likelihood
    loglikelihoods_w.append(loglikelihood_w(w))
    loglikelihoods_theta.append(loglikelihood_theta(w,theta))
    loglikelihoods_X.append(loglikelihood_X(theta))
    loglikelihoods.append(loglikelihoods_w[-1] + loglikelihoods_theta[-1] + loglikelihoods_X[-1])
    
    # save intermediate w and theta values for later
    ws.append(w.copy())
    thetas.append(theta.copy())


# write ws to disk:
if not interactive:
    # dfo = pd.DataFrame(ws)
    # dfo.index.name = 'iter'
    # dfo.to_csv(os.path.join(output_dir,'ws.csv'))
    
    df['w'] = w
    df.to_csv(os.path.join(output_dir,'output.csv'),index=False)


# figures

import matplotlib as mpl
if not interactive: mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm

# compute some data for figures
w_sums = [sum(log10(w_current)) for w_current in ws]
ws = np.asarray(ws)
thetas = np.asarray(thetas)
stds = np.std(ws[-1000:,:],axis=0)
medians = np.median(ws[-1000:,:],axis=0)
clim = max(np.abs(np.min(log10(ws))),np.abs(np.max(log10(ws))))
modeslog10 = [h[1][np.argmax(h[0])] for h in (np.histogram(log10(w_component),bins=100,range=(-clim,clim)) for w_component in ws[-1000:,:].T)]
order = np.argsort(ws[-1,:])[::-1]
diffs = np.diff(log10(ws.T))
dlim = max(np.abs(np.min(diffs)),np.abs(np.max(diffs)))
updates = np.sum(diffs != 0,axis=1)
dirichlet_weights = np.sum(ws*Z*alpha,axis=1)
theta_err = np.sqrt(np.sum((thetas - thetatruth)**2,axis=1))
theta_err_L1 = np.sum(np.abs(thetas - thetatruth),axis=1)

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
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'loglikelihoods.png'))

# fraction of accepted moves in metropolis-hastings
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(frac_accepted)
ax.set_xlabel('iteration')
ax.set_ylabel('frac moves accepted')
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'frac_accepted.png'))

# ranked w-values
norm = mpl.colors.normalize(0,len(ws)-1)
fig = plt.figure()
ax = fig.add_subplot(111)
for (i,w_current) in enumerate(ws):
    if i % 200 == 0:
        ax.plot(range(1,len(w_current)+1),sorted(log10(w_current),reverse=True),color=mpl.cm.jet(norm(i)),clip_on=False)

ax.plot(range(1,len(wtruth)+1),sorted(log10(wtruth),reverse=True),'-k',clip_on=False,linewidth=2)
# ax.plot(range(1,len(w)+1),sorted(log10(w),reverse=True),'o-b',clip_on=False)
ax.set_xlabel('rank')
ax.set_ylabel('log10(w)')
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'ranked_w_final.png'))

# theta error
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.plot(range(iterations),theta_err)
ax.plot(range(iterations),theta_err_L1)
ax.set_xlabel('iteration')
ax.set_ylabel('theta error')
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'theta_error.png'))


# weight of parameters into the Dirichlet
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(iterations),dirichlet_weights)
ax.set_xlabel('iteration')
ax.set_ylabel('Z * w')
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'weight_dirichlet_param.png'))


# correlation btwn wtruth and w
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.scatter(log10(w),log10(wtruth),c=statstools.density2d(log10(w),log10(wtruth)),cmap=plt.jet(),s=25,clip_on=False,lw=0.5)
# ax.scatter(log10(ws[100,:]),log10(wtruth),c=statstools.density2d(log10(ws[100,:]),log10(wtruth)),cmap=plt.jet(),s=25,clip_on=False,lw=0.5)
ax.scatter(log10(medians),log10(wtruth),c=statstools.density2d(log10(medians),log10(wtruth)),cmap=plt.jet(),s=25,clip_on=False,lw=0.5)
# ax.scatter(modeslog10,log10(wtruth),c=statstools.density2d(modeslog10,log10(wtruth)),cmap=plt.jet(),s=25,clip_on=False,lw=0.5)
ax.set_xlabel('w')
ax.set_ylabel('w_truth')
if interactive: fig.show()
# else: fig.savefig(os.path.join(output_dir,'ranked_w_final.png'))




# histogram evolution of w-values
norm = mpl.colors.normalize(0,len(ws)-1)
fig = plt.figure()
ax = fig.add_subplot(111)
for (i,w_current) in enumerate(ws):
    if i % 200 == 0:
        ax.hist(log10(w_current),bins=100,log=True,histtype='step',color=mpl.cm.jet(norm(i)),linewidth=1,alpha=0.5)

ax.set_xlabel('log10(w)')
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'hist_evolution.png'))

# total weight in w
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(w_sums)
ax.set_xlabel('iteration')
ax.set_ylabel('sum(log10(w))')
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'total_weight.png'))

# plot stds of the w values
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sorted(stds))
ax.set_xlabel('rank')
ax.set_ylabel('std of each component in last 1000 w vectors')
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'ranked_stds.png'))

# plot ip/op data colored by estimated w value or variance in estimate
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Z,X,c=log10(medians),cmap=mpl.cm.RdBu,vmin=-clim,vmax=clim,s=25,lw=0.5,clip_on=False,zorder=10)
ax.set_yscale('log')
ax.set_xlabel('input count')
ax.set_ylabel('output count')
ax.axis([0,1200,1,1e3])
bar = fig.colorbar(ax.collections[0])
bar.set_label('log10(w)')
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'raw_data_median_late_ws.png'))

# truth
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Z,X,c=log10(wtruth),cmap=mpl.cm.RdBu,vmin=-clim,vmax=clim,s=25,lw=0.5,clip_on=False,zorder=10)
ax.set_yscale('log')
ax.set_xlabel('input count')
ax.set_ylabel('output count')
ax.axis([0,1200,1,1e3])
bar = fig.colorbar(ax.collections[0])
bar.set_label('log10(w)')
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'raw_data_median_late_ws.png'))


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(Z,X,c=np.abs(log10(ws[5,:])),cmap=mpl.cm.RdBu,vmin=-clim,vmax=clim,s=25,lw=0.5,clip_on=False,zorder=10)
# ax.set_yscale('log')
# ax.set_xlabel('input count')
# ax.set_ylabel('output count')
# ax.axis([0,1200,1,1e3])
# bar = fig.colorbar(ax.collections[0])
# bar.set_label('log10(w)')
# if interactive: fig.show()
# else: fig.savefig(os.path.join(output_dir,'raw_data_early_ws.png'))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Z,X,c=-stds,cmap=plt.jet(),s=25,lw=0.5,clip_on=False,zorder=10)
ax.set_yscale('log')
ax.set_xlabel('input count')
ax.set_ylabel('output count')
ax.axis([0,1200,1,1e3])
bar = fig.colorbar(ax.collections[0])
bar.set_label('-1*std(w)')
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'raw_data_variance_ws.png'))

# plot trajectory of each component
segments = tuple([np.c_[np.arange(iterations),log10(trajectory)] for trajectory in ws.T])
coll = mpl.collections.LineCollection(segments,colors=(0,0,0,0.1))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0,iterations])
ax.set_ylim([np.min(log10(ws))*0.9,np.max(log10(ws))*1.1])
ax.add_collection(coll)
ax.set_xlabel('iteration')
ax.set_ylabel('w component')
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'trajectories.png'))

# trajectories of each component as heat map
fig = plt.figure(figsize=(iterations/250.,N/250.))
ax = fig.add_axes([0.1,0.1,0.87,0.87])
ax.imshow(log10(ws.T)[order,:],aspect='auto',interpolation='nearest',cmap=mpl.cm.RdBu,vmin=-clim,vmax=clim)
ax.set_xlabel('iteration')
ax.set_ylabel('w component')
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'trajectories_heat.png'))

# updates (log derivatives) of each component as heat map
# fig = plt.figure(figsize=(iterations/250.,N/250.))
# ax = fig.add_axes([0.1,0.1,0.87,0.87])
# ax.imshow(diffs[order,:],aspect='auto',interpolation='nearest',cmap=mpl.cm.RdBu,vmin=-dlim,vmax=dlim)
# ax.set_xlabel('iteration')
# ax.set_ylabel('w component')
# if interactive: fig.show()
# else: fig.savefig(os.path.join(output_dir,'trajectories_derivatives_heat.png'))

# updates (log derivatives) of each component as "spy"/binary
fig = plt.figure(figsize=(iterations/250.,N/250.))
ax = fig.add_axes([0.1,0.1,0.87,0.87])
ax.spy(diffs[order,:],aspect='auto')
ax.set_xlabel('iteration')
ax.set_ylabel('w component')
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'trajectories_derivatives_binary.png'))

# how many updates does a typical w component get?
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(updates,bins=50)
ax.set_xlabel('num updates for a given w in %i iterations' % iterations)
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'hist_num_updates.png'))

# is there any correlation between the number of updates for a row and it's
# final value?
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.abs(log10(ws.T[order,0])),updates)
ax.set_xlabel('final w value')
ax.set_ylabel('num updates for that values in %i iterations' % iterations)
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'num_updates_vs_final_w.png'))

# is there any correlation between the number of updates for a row and it's
# std deviation over the final rounds
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(stds[order],updates)
ax.set_xlabel('stddev(w) for last 1000 iter')
ax.set_ylabel('num updates for that values in %i iterations' % iterations)
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'num_updates_vs_stds.png'))

# correlate stds with medians:
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(log10(medians),stds)
ax.set_xlabel('log10(median w)')
ax.set_ylabel('std(w)')
if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'median_w_vs_std_w.png'))

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

if interactive: fig.show()
else: fig.savefig(os.path.join(output_dir,'generated_data.png'))


# NP_002745.1_13 is the positive control

