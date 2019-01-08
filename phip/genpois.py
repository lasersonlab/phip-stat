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

import sys

import numpy as np
import scipy as sp
import scipy.optimize
from tqdm import tqdm, trange


lt1 = 1.0 - np.finfo(np.float64).epsneg


def GP_lambda_likelihood(counts):
    # compute inputs to likelihood function
    (nx, x) = np.histogram(counts, bins=range(max(counts) + 2))
    x = x[:-1]
    n = len(counts)
    x_bar = sum(counts) / float(n)
    # check condition for unique root
    if sum(nx[2:] * x[2:] * (x[2:] - 1)) - n * (x_bar ** 2) <= 0:
        raise ValueError(
            "Condition for uniqueness of lambda is not met.\n"
            "    x: {0}\n    n: {1}\n    x_bar: {2}\n".format(x, n, x_bar)
        )
    return lambda lam: (
        sum(nx * (x * (x - 1) / (x_bar + (x - x_bar) * lam))) - n * x_bar
    )


def log_GP_pmf(x, theta, lambd):
    return (
        np.log(theta)
        + (x - 1) * np.log(theta + x * lambd)
        - (theta + x * lambd)
        - sp.special.gammaln(x + 1)
    )


def log_GP_sf(x, theta, lambd):
    count = x + 1
    accum = log_GP_pmf(count, theta, lambd)
    while True:
        count += 1
        new = np.logaddexp(accum, log_GP_pmf(count, theta, lambd))
        if new - accum < 1e-6:
            break
        accum = new
    return accum


def estimate_GP_distributions(input_counts, output_counts, uniq_input_values):
    lambdas = []
    thetas = []
    idxs = []
    for i in trange(
        output_counts.shape[1], desc="GenPois sample estimates"
    ):  # for each output column...
        lambdas.append([])
        thetas.append([])
        idxs.append([])
        for input_value in tqdm(
            uniq_input_values, desc="Unique input values"
        ):  # ...compute lambdas/thetas
            # compute lambda
            curr_counts = output_counts[input_counts == input_value, i]
            if len(curr_counts) < 50:
                continue

            try:  # may fail if MLE doesn't meet uniqueness condition
                H = GP_lambda_likelihood(curr_counts)
            except ValueError:
                continue

            idxs[-1].append(input_value)
            lambd = sp.optimize.brentq(H, 0.0, lt1)
            lambdas[-1].append(lambd)

            # compute theta
            n = len(curr_counts)
            x_bar = sum(curr_counts) / float(n)
            theta = x_bar * (1 - lambd)
            thetas[-1].append(theta)
    return (lambdas, thetas, idxs)


def lambda_theta_regression(lambdas, thetas, idxs):
    lambda_fits = []
    theta_fits = []
    for i in trange(len(lambdas), desc="Parameter regression"):
        try:
            lambda_fit = lambda x: np.mean(lambdas[i])
            coeffs = np.polyfit(idxs[i], thetas[i], 1)
            theta_fit = lambda x: coeffs[0] * x + coeffs[1]
            lambda_fits.append(lambda_fit)
            theta_fits.append(theta_fit)
        except TypeError:
            # occurs when failure to get enough values or non-uniqueness in fit
            lambda_fits.append(None)
            theta_fits.append(None)
    return (lambda_fits, theta_fits)


def precompute_pvals(lambda_fits, theta_fits, uniq_combos):
    total_combos = sum([len(s) for s in uniq_combos])
    log10pval_hash = {}
    j = 0
    with tqdm(desc="Precomputing p-vals", unit="pval", total=total_combos) as pbar:
        for (i, u) in enumerate(uniq_combos):
            for (ic, oc) in u:
                try:
                    log_pval = log_GP_sf(oc, theta_fits[i](ic), lambda_fits[i](ic))
                    log10pval_hash[(i, ic, oc)] = log_pval * np.log10(np.e) * -1.0
                except TypeError:
                    # occurs when the parameter function is invalid, which
                    # occurs when lambda_theta_regression can't compute a fit
                    log10pval_hash[(i, ic, oc)] = -1.0
                j += 1
                if j % 10 == 0:
                    pbar.update(10)
    return log10pval_hash
