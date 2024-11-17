# =============================================================================
# Generate experimental design
# =============================================================================

# 1. C - List of number of categroies corresponding to each categorical variable
# 2. Nc - Number of categorical variables
# 3. Nx - Continuous variables
# 4. h - categorical
# 5. x - continuous
# 6. z - [h, x]
#
# 7. nDim - Nc + Nx
# 8. bounds - Lower and Upper bounds of both Catergorical variables and continuous variables
#
# 9. n_iter - Number of iterations to run the algorithm
# 10. initN - Number of intial data points
# 11. batch_size (b) - Number of experiments to be generated in each iteration
#
# 12. acq_type - Acquisition Type
import argparse
import math
import collections
import pickle
import random
import scipy
import json
from matplotlib import pyplot as plt
import GPy
import numpy as np
import pandas as pd
from tqdm import tqdm
from methods.InitialData_Gen import initialize
# from methods.AcquisitionFunctions import EI, PI, UCB, AcquisitionOnSubspace
# from methods.optimization import sample_then_minimize
from methods.AskTell import ask_tell
from scipy.optimize import minimize
from typing import Union, Tuple
from paramz.transformations import Logexp


def design_initial_experiments(data_param, initN, output_file_name, random_seed, background_model_parameter):
    budget = 100
    batch_size = data_param['batch_size']
    bestUpperBoundEstimate = 2 * budget / 3
    gamma_list = [np.sqrt(C * math.log(C / batch_size) / ((math.e - 1) * batch_size * bestUpperBoundEstimate))
                  for C in data_param['C']]

    gamma_list = [g if not np.isnan(g) else 1 for g in gamma_list]

    Wc_list_init = [np.ones(C) for C in data_param['C']]
    Wc_list = Wc_list_init
    

    Exp_0, result_0 = initialize(initN, data_param, random_seed)
    pd.DataFrame(Exp_0).to_csv(output_file_name + '.csv')
    Model_background = {}
    Model_background = {'gamma_list': gamma_list, 'budget': budget,
                        'bestUpperBoundEstimate': bestUpperBoundEstimate, 'Wc_list_init': Wc_list_init,
                        'Wc_list': Wc_list, 'data_param': data_param}

    with open(background_model_parameter, 'wb') as output:
        # Pickle dictionary using protocol 0.
        pickle.dump(Model_background, output)
    return Exp_0

def design_experiments(data, result, batch_size, method, background_file_name, updated_background_file_name ,output_file_name):
    background_file = background_file_name
    with open(background_file, "rb") as fp:
        ModelBackground_0 = pickle.load(fp)

    z_next, Categorical_dist_param, gp_actual = ask_tell(data, result, ModelBackground_0['data_param'], 'Matern52', method,  batch_size,
                                           ModelBackground_0['Wc_list'],  ModelBackground_0['gamma_list'])

    ModelBackground_1 = {}
    ModelBackground_1 = {'gamma_list': ModelBackground_0['gamma_list'], 'budget': ModelBackground_0['budget'],
                         'bestUpperBoundEstimate': ModelBackground_0['bestUpperBoundEstimate'],
                         'Wc_list_init': ModelBackground_0['Wc_list_init'],
                         'Wc_list': Categorical_dist_param['Wc_list'], 'data_param': ModelBackground_0['data_param'],
                         'Categorical_dist_param': Categorical_dist_param}


    with open(updated_background_file_name, 'wb') as output:
        # Pickle dictionary using protocol 0.
        pickle.dump(ModelBackground_1, output)

    pd.DataFrame(z_next).to_csv(output_file_name + '.csv')


    return z_next, gp_actual