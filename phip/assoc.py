# Copyright 2018 Uri Laserson
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

from functools import lru_cache

from numpy import exp, inf, log
from scipy.integrate import dblquad
from scipy.optimize import root
from scipy.special import betaln
from scipy.special import binom as binom_coef
from scipy.special import gammaln
from scipy.stats import binom, lognorm

#        A      B
#      -----   -----
# 0s |  n0A  |  n0B  | n0
#    | -----   ----- |
# 1s |  n1A  |  n1B  | n1
#      -----   -----
#       nA      nB     N
#
# nA and nB are fixed up front by experimental design
#
#
# unassoc model
#
# A and B population are each Binom(nA, p) or Binom(nB, p) with same p
#
# prior on p is Uniform(0, 1)
#
#
# assoc model
#
# Populations are Binom(nA, pA) and Binom(nB, pB)
#
# priors on pA, pB are Uniform(0, 1)


@lru_cache(maxsize=16384)
def log_evidence_unassoc(N, n1, nA, n1A):
    """log P(D | unassoc): two binomial distributions with same p"""
    n0 = N - n1
    nB = N - nA
    n0A = nA - n1A
    n1B = n1 - n1A
    n0B = nB - n1B

    a = gammaln(nA + 1)
    b = gammaln(nB + 1)
    c = gammaln(n1 + 1)
    d = gammaln(n0 + 1)
    e = gammaln(n1A + 1)
    f = gammaln(n0A + 1)
    g = gammaln(n1B + 1)
    h = gammaln(n0B + 1)
    i = gammaln(N + 2)

    return (a + b + c + d) - (e + f + g + h + i)


@lru_cache(maxsize=16384)
def posterior_unassoc(N, n1, nA, n1A):
    """returns Beta parameters for posterior distribution on p in unassoc case"""
    alpha = n1 + 1
    beta = N - n1 + 1
    return (alpha, beta)


@lru_cache(maxsize=16384)
def log_evidence_assoc(N, n1, nA, n1A):
    """log P(D | assoc): two binomial distributions with different p for A and B"""
    n0 = N - n1
    nB = N - nA
    n0A = nA - n1A
    n1B = n1 - n1A
    n0B = n0 - n0A

    a = log(binom_coef(nA, n1A))
    b = log(binom_coef(nB, n1B))
    c = betaln(n1A + 1, n0A + 1)
    d = betaln(n1B + 1, n0B + 1)

    return a + b + c + d


@lru_cache(maxsize=16384)
def posterior_assoc(N, n1, nA, n1A):
    """returns Beta parameters for posterior distributions on pA and pB in assoc case"""
    n0 = N - n1
    nB = N - nA
    n0A = nA - n1A
    n1B = n1 - n1A
    n0B = nB - n1B

    alpha_A = n1A + 1
    beta_A = n0A + 1
    alpha_B = n1B + 1
    beta_B = n0B + 1

    return (alpha_A, beta_A, alpha_B, beta_B)


# @lru_cache(maxsize=16384)
# def log_evidence_assoc(N, n1, nA, n1A):
#     """log P(D | assoc): two binomial distributions with different p for A and B"""
#     nB = N - nA
#     n1B = n1 - n1A

#     def p_and_theta_to_pA_and_pB(p, theta):
#         # nA or nB cannot be 0
#         if theta == 1:
#             return (p, p)
#         # using computer algebra system to compute roots
#         tm1 = theta - 1
#         a = 2 * nA * tm1
#         b = 2 * nB * tm1
#         c = N * p * tm1
#         d = nA + nB * theta
#         radical = np.sqrt(((c - d)**2) + 4 * nA * c)

#         # there are two branches of sol'n
#         # we need to check which is the valid one
#         # this can probably be expressed as some algebraic constraint but meh
#         pA_1 = (c - d + radical) / a
#         pB_1 = (c + d - radical) / b
#         # the other branch of soln is
#         pA_2 = (c - d - radical) / a
#         pB_2 = (c + d + radical) / b

#         soln_1 = pA_1 >= 0 and pA_1 <= 1 and pB_1 >= 0 and pB_1 <= 1
#         soln_2 = pA_2 >= 0 and pA_2 <= 1 and pB_2 >= 0 and pB_2 <= 1
#         sane = (soln_1 and not soln_2) or (soln_2 and not soln_1)

#         if not sane:
#             raise ValueError("Failed to find a valid solution")

#         if soln_1:
#             return (pA_1, pB_1)
#         else:
#             return (pA_2, pB_2)


#     def integrand(p, theta):
#         pA, pB = p_and_theta_to_pA_and_pB(p, theta)
#         return binom.pmf(n1A, nA, pA) * binom.pmf(n1B, nB, pB) * lognorm.pdf(theta, 1)

#     # return log(dblquad(integrand, 0, inf, gfun=lambda theta: 0, hfun=lambda theta: 1)[0])
#     return log(dblquad(integrand, 0, inf, 0, 1)[0])


def expectation_binomial_assoc(N, n1, nA, n1A):

    pass


def compute_2x2(thresholded, cols_A, cols_B):
    """Takes thresholded DataFrame (bool ok) and computes 2x2 contingency table counts

    returns N, n1, nA, n1A
    """
    N = len(cols_A) + len(cols_B)
    nA = len(cols_A)
    n1 = thresholded[cols_A + cols_B].sum(axis=1)
    n1A = thresholded[cols_A].sum(axis=1)
    return pd.DataFrame(
        {"N": N, "n1": n1, "nA": nA, "n1A": n1A}, columns=["N", "n1", "nA", "n1A"]
    )


def compute_assoc(thresholded, cols_A, cols_B):
    """Takes thresholded DataFrame (bool ok) and returns association stats for each row
    """
    contingency_tables = compute_2x2(thresholded, cols_A, cols_B)

    # compute log evidences
    contingency_tables["lev_unassoc"] = [
        log_evidence_unassoc(*tup)
        for tup in contingency_tables[["N", "n1", "nA", "n1A"]].itertuples(index=False)
    ]
    contingency_tables["lev_assoc"] = [
        log_evidence_assoc(*tup)
        for tup in contingency_tables[["N", "n1", "nA", "n1A"]].itertuples(index=False)
    ]

    # bayes factor
    contingency_tables["bf"] = exp(
        contingency_tables["lev_assoc"] - contingency_tables["lev_unassoc"]
    )

    # compute the posterior distribution parameters
    x, y = zip(
        *[
            posterior_unassoc(*tup)
            for tup in contingency_tables[["N", "n1", "nA", "n1A"]].itertuples(
                index=False
            )
        ]
    )
    contingency_tables["posterior_unassoc_alpha"] = x
    contingency_tables["posterior_unassoc_beta"] = y
    x, y, z, w = zip(
        *[
            posterior_assoc(*tup)
            for tup in contingency_tables[["N", "n1", "nA", "n1A"]].itertuples(
                index=False
            )
        ]
    )
    contingency_tables["posterior_assoc_A_alpha"] = x
    contingency_tables["posterior_assoc_A_beta"] = y
    contingency_tables["posterior_assoc_B_alpha"] = z
    contingency_tables["posterior_assoc_B_beta"] = w

    return contingency_tables
