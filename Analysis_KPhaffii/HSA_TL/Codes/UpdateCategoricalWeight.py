
#==========================================
#CoCaBO Algorithm - Function to faciitate sampling of categorical variable
#==========================================
import math
import collections
import pickle
import random

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


def compute_reward_for_all_cat_variable(ht_next_batch_list,C_list,data, result, batch_size):
    # Obtain the reward for each categorical variable: B x len(self.C_list)
    ht_batch_list_rewards = np.zeros((batch_size, len(C_list)))
    if batch_size > 1:
        for b in range(batch_size):
            ht_next_list = ht_next_batch_list[b, :]
            for i in range(len(ht_next_list)):
                idices = np.where(data[:, i] == ht_next_list[i])
                ht_result = result[idices]
                ht_reward = np.mean(ht_result)  #/3+1
                ht_batch_list_rewards[b, i] = ht_reward
    else:
        ht_batch_list_rewards = []
        ht_next_list = ht_next_batch_list
        for i in range(len(ht_next_list)):
            idices = np.where(data[:, i] == ht_next_list[i])
            ht_result = result[idices]
            ht_reward = np.mean(ht_result)  #/3+1
            ht_batch_list_rewards.append(ht_reward)

    return ht_batch_list_rewards

def update_weights_for_all_cat_var(C_list, Gt_ht_list, ht_batch_list, Wc_list, gamma_list,
                                   probabilityDistribution_list, batch_size, S0=None):
    Wc_list_updated = [None]*len(C_list)
    for j in range(len(C_list)):
        Wc = Wc_list[j].copy()
        C = C_list[j]
        gamma = gamma_list[j]
        probabilityDistribution = probabilityDistribution_list[j]
        # print(f'cat_var={j}, prob={probabilityDistribution}')

        if batch_size > 1:
            ht_batch_list = ht_batch_list.astype(int)
            # print(Gt_ht_list[:, j])
            Gt_ht = Gt_ht_list[:, j]
            mybatch_ht = ht_batch_list[:, j]  # 1xB
            for ii, ht in enumerate(mybatch_ht):
                Gt_ht_b = Gt_ht[ii]
                estimatedReward = 1.0 * Gt_ht_b / probabilityDistribution[ht]
                Wc[ht] *= np.exp(estimatedReward * gamma / C) #batch_size *
                # if ht not in S0:
                #     Wc[ht] *= np.exp(estimatedReward * gamma / C) #batch_size *
        else:
            Gt_ht = Gt_ht_list[j]
            ht = ht_batch_list[j]  # 1xB
            estimatedReward = 1.0 * Gt_ht / probabilityDistribution[ht]
            Wc[ht] *= np.exp(estimatedReward * gamma / C)

        # print(Wc)
        Wc_list_updated[j] = Wc
        # print(Wc_list_updated)
    return Wc_list_updated