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
import random

import matplotlib as mpl
if __name__ == '__main__':  # if running as script, disable any windowing
    mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import pymc

def autocorrelation(x, normed=True):
    x = np.asarray(x)
    x -= x.mean()   # detrend
    c = np.correlate(x, x, mode='full')
    N = len(x)
    maxlags = N - 1
    
    if normed:
        c /= np.dot(x,x)
    
    lags = np.arange(-maxlags, maxlags+1)
    c = c[N-1-maxlags:N+maxlags]
    
    return lags, c


show = lambda fig, output_dir, output_file: fig.show() if output_dir == None else fig.savefig(os.path.join(output_dir, output_file))

class MCMCAnalysis(object):
    """Generate plots from MCMC Model object"""
    
    def __init__(self, df, M):
        self.r = len(df.columns) - 1    # number of observed vectors
        self.M = M
        self.N = len(M.w.value)
        self.ws = M.w.trace.gettrace()
        self.logratios1 = np.asarray(np.log10(df['X_1'] + 1) - np.log10(np.sum(df['X_1'] + 1)) - np.log10(df['X_0'] + 1) + np.log10(np.sum(df['X_0'] + 1)), dtype=float)
        self.medians = np.median(self.ws[-1000:, :], axis=0)
        self.weights = np.asarray(np.sum(df.values[:, 1:], axis=1), dtype=float)
        self.order_by_median_ws = np.argsort(self.medians)[::-1]
        self.order_by_weight = np.argsort(self.weights)
        self.p5  = sp.stats.scoreatpercentile(self.ws[-1000:, :], 5)
        self.p25 = sp.stats.scoreatpercentile(self.ws[-1000:, :], 25)
        self.p50 = self.medians
        self.p75 = sp.stats.scoreatpercentile(self.ws[-1000:, :], 75)
        self.p95 = sp.stats.scoreatpercentile(self.ws[-1000:, :], 95)
    
    def deviance(self, output_dir=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.M.trace('deviance').gettrace())
        ax.set_xlabel('sample')
        ax.set_ylabel('deviance (-2*loglikelihood)')
        show(fig, output_dir, 'deviance.png')
    
    def w_distributions_ordered_by(self, order, output_dir=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(self.N), self.p5[order],  s=5, c='k', lw=0, zorder=1)
        ax.scatter(range(self.N), self.p95[order], s=5, c='k', lw=0, zorder=1)
        for (pos, low, high) in zip(range(self.N), self.p25[order], self.p75[order]):
            ax.plot([pos, pos], [low, high], color='#bdbdbd', lw=2, zorder=2)
        ax.scatter(range(self.N), self.p50[order], s=10, c='r', linewidths=0, zorder=3)
        ax.axhline(0, zorder=0)
        ax.set_xlabel('w component')
        ax.set_ylabel('w value')
        return fig
    
    def w_distributions_ordered_by_medians(self, output_dir=None):
        fig = self.w_distributions_ordered_by(order=self.order_by_median_ws, output_dir=output_dir)
        show(fig, output_dir, 'w_distributions_ordered_by_medians.png')
    
    def w_distributions_ordered_by_weight(self, output_dir=None):
        fig = self.w_distributions_ordered_by(order=self.order_by_weight, output_dir=output_dir)
        show(fig, output_dir, 'w_distributions_ordered_by_weight.png')
    
    def logratios_vs_ws_by_weights(self, output_dir=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.logratios1, self.medians, c=np.log10(self.weights + 1), cmap=mpl.cm.Blues, s=25, clip_on=False, lw=0.5)
        ax.set_xlabel('log10(ratio)')
        ax.set_ylabel('w')
        bar = fig.colorbar(ax.collections[0])
        bar.set_label('log10(weights)')
        show(fig, output_dir, 'logratios_vs_ws_by_weights.png')
    
    def autocorr_vs_w(self, output_dir=None):
        acorr = np.asarray([np.sum(autocorrelation(self.M.w.trace()[-1000:,i])[1]) for i in xrange(self.N)])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.medians, acorr, s=25, clip_on=False, lw=0.5)
        ax.set_xlabel('w')
        ax.set_ylabel('sum(autocorrelation)')
        show(fig, output_dir, 'autocorr_vs_w.png')
    
    def w_vs_weight(self, output_dir=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.medians, self.weights, s=25, clip_on=False, lw=0.5)
        ax.set_xlabel('w')
        ax.set_ylabel('weight')
        show(fig, output_dir, 'w_vs_weight.png')
        


if __name__ == '__main__':
    
    import argparse

    argparser = argparse.ArgumentParser(description=None)
    argparser.add_argument('--input')
    argparser.add_argument('--output', default='output.hdf5')
    argparser.add_argument('--iterations', type=int, default=3000)
    argparser.add_argument('--subsample', type=int, default=0)
    argparser.add_argument('--truth', action='store_true')
    argparser.add_argument('--verbose', action='store_true')
    args = argparser.parse_args()
    # args = argparser.parse_args('--input /Users/laserson/Dropbox/ElledgeLab/yifan/E7screenRawCount_input_end.csv --verbose --iterations 100000'.split())
    
    def msg(txt):
        sys.stderr.write(txt)
        sys.stderr.flush()
        
    # check if I will dump out tons of figures about the process
    if args.verbose:
        output_dir = os.path.splitext(args.output)[0]
        os.makedirs(output_dir, mode=0755)
        output_file = os.path.basename(args.output)
    else:
        output_dir = os.getcwd()
        output_file = args.output
    
    # load data
    msg("Loading data...")
    full_df = pd.read_csv(args.input, index_col=None)
    columns = ['individual'] + ['X_%i' % i for i in xrange(len(full_df.columns) - 1)]
    full_df.columns = pd.Index(columns)
    msg("finished\n")
    
    # subsample rows to make problem smaller
    if args.subsample > 0:
        random_idxs = random.sample(xrange(full_df.shape[0]), args.subsample)
        df = full_df.ix[random_idxs]
        df.index = xrange(args.subsample)
    else:
        df = full_df
    
    # define the model
    N = df.shape[0]
    pseudocount = 1
    nodes = {}
    nodes['X_0'] = df['X_0'] + pseudocount
    
    # nodes['w'] = pymc.Lognormal('w', mu=0, tau=1, size=N)
    # for i in xrange(1,len(df.columns) - 1):
    #     nodes['alpha_%i' % i] = pymc.Lambda('alpha_%i' % i, lambda Z=nodes['X_%i' % (i-1)], w=nodes['w']: Z * w)
    #     nodes['theta_%i' % i] = pymc.Dirichlet('theta_%i' % i, theta=nodes['alpha_%i' % i])
    #     nodes['X_%i' % i] = pymc.Multinomial('X_%i' % i, n=np.sum(df['X_%i' % i]), p=nodes['theta_%i' % i], value=df['X_%i' % i], observed=True)
    
    nodes['w'] = pymc.Normal('w', mu=0, tau=1, size=N)
    for i in xrange(1,len(df.columns) - 1):
        nodes['alpha_%i' % i] = pymc.Lambda('alpha_%i' % i, lambda Z=nodes['X_%i' % (i-1)], w=nodes['w']: Z * np.exp(w))
        nodes['theta_%i' % i] = pymc.Dirichlet('theta_%i' % i, theta=nodes['alpha_%i' % i])
        nodes['X_%i' % i] = pymc.Multinomial('X_%i' % i, n=np.sum(df['X_%i' % i]), p=nodes['theta_%i' % i], value=df['X_%i' % i], observed=True)
    
    M = pymc.MCMC(nodes, calc_deviance=True, db='hdf5', dbname=os.path.join(output_dir, output_file))
    
    # run the sampling
    M.sample(iter=args.iterations, burn=0, thin=5000)
    M.save_state()
    
    # write output csv
    msg("Writing w values to disk...")
    df['w'] = median_w
    df['std_w'] = std_w
    df['p5_w'] = p5_w
    df['p95_w'] = p95_w
    df.to_csv(os.path.join(output_dir, output_file), index=False, cols=['clone', 'w', 'p5_w', 'p95_w', 'std_w'])
    msg("finished\n")
    
    # figures (verbose output only)
    if args.verbose:
        msg("Computing values for figures...")
        plots = MCMCAnalysis(M)





###############################################################################

# def logp_trace(model):
#     """
#     return a trace of logp for model
#     """
    
#     #init
#     db = model.db
#     n_samples = db.trace('deviance').length()
#     logp = np.empty(n_samples, np.double)
    
#     #loop over all samples
#     for i_sample in xrange(n_samples):
#         #set the value of all stochastic to their 'i_sample' value
#         for stochastic in model.stochastics:
#             try:
#                 value = db.trace(stochastic.__name__)[i_sample]
#                 stochastic.value = value
            
#             except KeyError:
#                 print "No trace available for %s. " % stochastic.__name__
        
#         #get logp
#         logp[i_sample] = model.logp
    
#     return logp
