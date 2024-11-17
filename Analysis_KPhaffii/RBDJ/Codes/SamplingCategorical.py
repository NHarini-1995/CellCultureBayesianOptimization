
#==========================================
#CoCaBO Algorithm - Function to faciitate sampling of categorical variable
#==========================================
import math
import collections
import pickle
import random
from random import random, uniform

import GPy
import numpy as np
import pandas as pd
from tqdm import tqdm

from Kernel import MixtureViaSumAndProduct, CategoryOverlapKernel
from InitialData_Gen import initialize
from AcquisitionFunctions import EI, PI, UCB, AcquisitionOnSubspace


from scipy.optimize import minimize

from typing import Union, Tuple
from paramz.transformations import Logexp

# Compute the probabilty distribution and draw the categorical variable
def compute_prob_dist_and_draw_hts(Wc_list, gamma_list, C_list, batch_size):
    if batch_size > 1:
        ht_batch_list = np.zeros((batch_size, len(C_list)))
        probabilityDistribution_list = []
        for j in range(len(C_list)): # Loop through the categorical variable
            Wc = Wc_list[j]
            gamma = gamma_list[j]
            C = C_list[j]
            # perform some truncation here - HN Comments, get rid of this in the next version. once other issues are fixed.
            maxW = np.max(Wc)
            temp = np.sum(Wc) * (1.0 / batch_size - gamma / C) / (1 - gamma)
            S0 = []
            # S0 = [idx for idx, val in enumerate(Wc)]
            # if gamma < 1 and maxW >= temp:
            #     # find a threshold alpha
            #     print("I am here")
            #     alpha = estimate_alpha(batch_size, gamma, Wc, C)
            #     S0 = [idx for idx, val in enumerate(Wc) if val > alpha]
            # else:
            #     S0 = []

            # Compute the probability for each category
            probabilityDistribution = distr(Wc, gamma)

            # draw a batch here
            # if batch_size < C:
            #     mybatch_ht = DepRound(probabilityDistribution, k=batch_size)
            # else:
            mybatch_ht = np.random.choice(len(probabilityDistribution), batch_size, p=probabilityDistribution)

            # ht_batch_list size: len(self.C_list) x B
            ht_batch_list[:, j] = mybatch_ht[:]

            # ht_batch_list.append(mybatch_ht)
            probabilityDistribution_list.append(probabilityDistribution)

        return ht_batch_list, probabilityDistribution_list, S0

    else:
        ht_list = []
        probabilityDistribution_list = []
        S0 = []
        for j in range(len(C_list)):
            Wc = Wc_list[j]
            gamma = gamma_list[j]
            # Compute the probability for each category
            probabilityDistribution = distr(Wc, gamma)
            # Choose a categorical variable at random
            ht = draw(probabilityDistribution)
            ht_list.append(ht)
            probabilityDistribution_list.append(probabilityDistribution)

        return ht_list, probabilityDistribution_list, S0

def DepRound(weights_p, k=1, isWeights=True):
    p = np.array(weights_p)
    K = len(p)
    # Checks
    assert k < K, "Error: k = {} should be < K = {}.".format(k, K)  # DEBUG
    if not np.isclose(np.sum(p), 1):
        p = p / np.sum(p)
    assert np.all(0 <= p) and np.all(p <= 1), "Error: the weights (p_1, ..., p_K) should all be 0 <= p_i <= 1 ...".format(p)  # DEBUG
    assert np.isclose(np.sum(p), 1), "Error: the sum of weights p_1 + ... + p_K should be = 1 (= {}).".format(np.sum(p))  # DEBUG
    # Main loop
    possible_ij = [a for a in range(K) if 0 < p[a] < 1]
    while possible_ij:
        # Choose distinct i, j with 0 < p_i, p_j < 1
        if len(possible_ij) == 1:
            i = np.random.choice(possible_ij, size=1)
            j = i
        else:
            i, j = np.random.choice(possible_ij, size=2, replace=False)
        pi, pj = p[i], p[j]
        assert 0 < pi < 1, "Error: pi = {} (with i = {}) is not 0 < pi < 1.".format(pi, i)  # DEBUG
        assert 0 < pj < 1, "Error: pj = {} (with j = {}) is not 0 < pj < 1.".format(pj, i)  # DEBUG
        assert i != j, "Error: i = {} is different than with j = {}.".format(i, j)  # DEBUG

        # Set alpha, beta
        alpha, beta = min(1 - pi, pj), min(pi, 1 - pj)
        proba = alpha / (alpha + beta)
        if with_proba(proba):  # with probability = proba = alpha/(alpha+beta)
            pi, pj = pi + alpha, pj - alpha
        else:            # with probability = 1 - proba = beta/(alpha+beta)
            pi, pj = pi - beta, pj + beta

        # Store
        p[i], p[j] = pi, pj
        # And update
        possible_ij = [a for a in range(K) if 0 < p[a] < 1]
        if len([a for a in range(K) if np.isclose(p[a], 0)]) == K - k:
            break
    # Final step
    subset = [a for a in range(K) if np.isclose(p[a], 1)]
    if len(subset) < k:
        subset = [a for a in range(K) if not np.isclose(p[a], 0)]
    assert len(subset) == k, "Error: DepRound({}, {}) is supposed to return a set of size {}, but {} has size {}...".format(weights_p, k, k, subset, len(subset))  # DEBUG
    return subset


def estimate_alpha(batch_size, gamma, Wc, C):

    def single_evaluation(alpha):
        denominator = sum([alpha if val > alpha else val for idx, val in enumerate(Wc)])
        rightside = (1 / batch_size - gamma / C) / (1 - gamma)
        output = np.abs(alpha / denominator - rightside)

        return output

    x_tries = np.random.uniform(0, np.max(Wc), size=(100, 1))
    y_tries = [single_evaluation(val) for val in x_tries]
    # find x optimal for init
    # print(f'ytry_len={len(y_tries)}')
    idx_min = np.argmin(y_tries)
    x_init_min = x_tries[idx_min]

    res = minimize(single_evaluation, x_init_min, method='BFGS', options={'gtol': 1e-6, 'disp': False})
    if isinstance(res, float):
        return res
    else:
        return res.x

def distr(weights, gamma=0.0):
    theSum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)


def with_proba(epsilon):
    assert 0 <= epsilon <= 1, "Error: for 'with_proba(epsilon)', epsilon = {:.3g} has to be between 0 and 1 to be a valid probability.".format(epsilon)  # DEBUG
    return random() < epsilon  # True with proba epsilon


# Drawing choice
def draw(weights):
    choice = uniform(0, sum(weights)) #random.
    #    print(choice)
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1