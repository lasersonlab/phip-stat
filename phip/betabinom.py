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

from numpy import asarray, exp
from scipy.optimize import minimize
from scipy.special import betaln, gammaln


def fit_betabinom(counts):
    """Assumes each column is a count vector; counts should be ndarray"""
    Ns = counts.sum(axis=0)

    def nll(x):
        alpha = exp(x[0])
        beta = exp(x[1])
        return (
            counts.size * betaln(alpha, beta)
            - betaln(counts + alpha, Ns - counts + beta).sum()
        )

    result = minimize(nll, asarray([0, 1]), method="Nelder-Mead")

    if result.success:
        alpha, beta = exp(result.x)
        return alpha, beta
    else:
        print(result)
        raise ValueError("Failed to fit")
