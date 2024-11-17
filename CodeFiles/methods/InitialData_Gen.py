# =============================================================================
# Function to initialize data 
# =============================================================================
import sys
# sys.path.append('../bayesopt')
# sys.path.append('../ml_utils')
import argparse
import os
import math
from pyDOE import lhs
import GPy
import numpy as np
import pandas as pd
from tqdm import tqdm
import GPyOpt
from typing import Union, Tuple
from paramz.transformations import Logexp
import scipy
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

def initialize(initN, data_param, seed = 37):
    """Get NxN intial points"""
    data = []
    result = []
    if data_param['approach_type'] == 'CoCa':
        np.random.seed(seed)
        hinit = np.hstack(
            [np.random.randint(0, C, initN)[:, None] for C in data_param['C']])
        # print(hinit.shape)
        Xinit = generateInitialPoints(data_param, initN, data_param['bounds'][len(data_param['C']):], data_param['prob_type'],hinit) #'UnConstrained'

        Zinit = np.hstack((hinit, Xinit))
        yinit = np.zeros([Zinit.shape[0], 1])

        for j in range(initN):
            ht_list = list(hinit[j])
            yinit[j] = 100* np.random.uniform(low=0.0, high=1.0) # The objective function is a real experiment.
    #         # print(ht_list, Xinit[j], yinit[j])

        init_data = {}
        init_data['Z_init'] = Zinit
    #     init_data['y_init'] = yinit

    #     with open(init_fname, 'wb') as init_data_file:
    #         pickle.dump(init_data, init_data_file)

        data.append(Zinit)
        result.append(yinit) # have to be collected later in the lab
    else:
        hinit = []
        Zinit = generateInitialPoints(data_param, initN,
                                      data_param['bounds'], data_param['prob_type'], hinit) #'Constrained'
        #
        yinit = np.zeros([initN, 1])
        for j in range(initN):
            yinit[j] = 100 * np.random.uniform(low=0.0, high=1.0)

    return Zinit, yinit

def generateInitialPoints(data_param, initN, bounds, prob_type, hinit): # Based on uniform number generator
    if prob_type == 'Constrained':
        x_bounds = np.array([d['domain'] for d in bounds if d['type'] == 'continuous'])
        lower_bound = np.asarray(x_bounds)[:, 0].reshape(1, len(x_bounds))
        upper_bound = np.asarray(x_bounds)[:, 1].reshape(1, len(x_bounds))

        feasible_region = GPyOpt.Design_space(space=bounds, constraints=data_param['Constrains'])
        initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 6)
        Xinit = initial_design
    else:
        x_bounds = np.array([d['domain'] for d in bounds if d['type'] == 'continuous'])
        lower_bound = np.asarray(x_bounds)[:, 0].reshape(1, len(x_bounds))
        upper_bound = np.asarray(x_bounds)[:, 1].reshape(1, len(x_bounds))
        diff = upper_bound - lower_bound


        X_design_aux = lhs(data_param['Nx'], initN)
        I = np.ones((X_design_aux.shape[0], 1))
        X_design = np.dot(I, lower_bound) + X_design_aux * np.dot(I, diff)
        Xinit = X_design

    return Xinit