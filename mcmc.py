import os
import random
import argparse

import pandas as pd
import pymc

argparser = argparse.ArgumentParser(description=None)
argparser.add_argument('--input')
argparser.add_argument('--output', default='output.hdf5')
argparser.add_argument('--prior', default='lognormal')
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
full_df.columns = pd.Index(['individual', 'input', 'output'])
msg("finished\n")


# subsample rows to make problem smaller
if args.subsample > 0:
    random_idxs = random.sample(xrange(full_df.shape[0]), args.subsample)
    df = full_df.ix[random_idxs]
else:
    df = full_df

pseudocount = min(1, np.sum(df['input']) / (100. * len(df['input'])))
Z = np.array(data['input'] + pseudocount)
N = len(Z)
no = np.sum(data['output'])


# define model
w = pymc.Lognormal('w', mu=0, tau=1, size=N)

@pymc.deterministic
def dirichlet_alpha(Z=Z, w=w):
    return (Z * w)

theta = pymc.Dirichlet('theta', theta=dirichlet_alpha)

X = pymc.Multinomial('X', n=no, p=theta, value=data['output'], observed=True)

M = pymc.MCMC([w, dirichlet_alpha, theta, X], db='hdf5', dbname=os.path.join(output_dir, output_file))


# run the sampling
M.sample(iter=args.iterations, burn=3000, thin=200)


def logp_trace(model):
    """
    return a trace of logp for model
    """

    #init
    db = model.db
    n_samples = db.trace('deviance').length()
    logp = np.empty(n_samples, np.double)

    #loop over all samples
    for i_sample in xrange(n_samples):
        #set the value of all stochastic to their 'i_sample' value
        for stochastic in model.stochastics:
            try:
                value = db.trace(stochastic.__name__)[i_sample]
                stochastic.value = value

            except KeyError:
                print "No trace available for %s. " % stochastic.__name__

        #get logp
        logp[i_sample] = model.logp

    return logp
