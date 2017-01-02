# Copyright 2016 Uri Laserson
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

"""Generalized Poisson (GP) model for PhIP-seq"""

from __future__ import print_function

import numpy as np
import scipy as sp
import scipy.optimize


lt1 = 1. - np.finfo(np.float64).epsneg


def GP_lambda_likelihood(counts):
    # compute inputs to likelihood function
    (nx, x) = np.histogram(counts, bins=range(max(counts) + 2))
    x = x[:-1]
    n = len(counts)
    x_bar = sum(counts) / float(n)
    # check condition for unique root
    if sum(nx[2:] * x[2:] * (x[2:]-1)) - n * (x_bar ** 2) <= 0:
        raise ValueError(
            'Condition for uniqueness of lambda is not met.\n'
            '    x: {0}\n    n: {1}\n    x_bar: {2}\n'.format(x, n, x_bar))
    return lambda lam: (sum(nx * (x * (x - 1) / (x_bar + (x - x_bar) * lam))) -
                        n * x_bar)


def log_GP_pmf(x, theta, lambd):
    log = np.log
    logP = (log(theta) + (x - 1) * log(theta + x * lambd) -
            (theta + x * lambd) - np.sum(log(np.arange(1, x + 1))))
    return logP


def log_GP_sf(x, theta, lambd):
    extensions = 20
    start = x + 1
    end = x + 100
    pmf = [log_GP_pmf(y, theta, lambd) for y in xrange(start, end)]
    while extensions > 0:
        accum = np.logaddexp.accumulate(pmf)
        if accum[-1] == accum[-2]:
            return accum[-1]
        start = end
        end += 100
        pmf += [log_GP_pmf(y, theta, lambd) for y in xrange(start, end)]
        extensions -= 1
    return np.nan


def estimate_GP_distributions(input_counts, output_counts, uniq_input_values):
    lambdas = []
    thetas = []
    idxs = []
    for i in range(output_counts.shape[1]):  # for each output column...
        lambdas.append([])
        thetas.append([])
        idxs.append([])
        for input_value in uniq_input_values:  # ...compute lambdas/thetas
            # compute lambda
            curr_counts = output_counts[input_counts == input_value, i]
            if len(curr_counts) < 50:
                continue

            try:    # may fail if MLE doesn't meet uniqueness condition
                H = GP_lambda_likelihood(curr_counts)
            except ValueError:
                continue

            idxs[-1].append(input_value)
            lambd = sp.optimize.brentq(H, 0., lt1)
            lambdas[-1].append( lambd )

            # compute theta
            n = len(curr_counts)
            x_bar = sum(curr_counts) / float(n)
            theta =  x_bar * (1 - lambd)
            thetas[-1].append( theta )
    return (lambdas, thetas, idxs)


def lambda_theta_regression(lambdas, thetas, idxs):
    lambda_fits = []
    theta_fits = []
    for i in range(len(lambdas)):
        lambda_fit = lambda x: np.mean(lambdas[i])
        coeffs = np.polyfit(idxs[i], thetas[i], 1)
        theta_fit = lambda x: coeffs[0] * x + coeffs[1]
        lambda_fits.append(lambda_fit)
        theta_fits.append(theta_fit)
    return (lambda_fits, theta_fits)


def precompute_pvals(lambda_fits, theta_fits, uniq_combos):
    log10pval_hash = {}
    j = 0
    for (i, u) in enumerate(uniq_combos):
        for (ic, oc) in u:
            if j % 1000 == 0:
                print('computed {0} p-vals'.format(j), file=sys.stderr)
                sys.stderr.flush()
            log_pval = log_GP_sf(oc, theta_fits[i](ic), lambda_fits[i](ic))
            log10pval_hash[(i, ic, oc)] = log_pval * np.log10( np.e ) * -1.
            j += 1
