# =============================================================================
#  CoCaBO Algorithms - Initial Data
# =============================================================================
import sys
# sys.path.append('../bayesopt')
# sys.path.append('../ml_utils')
import argparse
import os
import math

import GPy
import numpy as np
import pandas as pd
from tqdm import tqdm

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

        nDim = len(x_bounds)
        Xinit_0 = np.zeros((initN, len(x_bounds)))
        Xinit = np.zeros((initN, len(x_bounds)))

        # def min_obj(X):
        #     return 0  # [0,0]

        # for i in range(initN):
            # Xinit[i, :] = np.array([np.random.uniform(bounds[b]['domain'][0],
            #                        bounds[b]['domain'][1], 1)[0] for b in range(nDim)])
        # Xinit_0 = np.random.uniform(lower_bound, upper_bound, size=(initN, nDim))

        from pyDOE import lhs
        diff = upper_bound - lower_bound
        X_design_aux = lhs(data_param['Nx'], initN*10)
        I = np.ones((X_design_aux.shape[0], 1))
        X_design = np.dot(I, lower_bound) + X_design_aux * np.dot(I, diff)
        Xinit_0 = X_design

        min_val = 1
        cnt = 0
        for i in range(initN*10):
            if cnt < initN:
                if data_param['approach_type'] == 'CoCa':
                    Input = np.concatenate(( hinit[cnt,:],Xinit_0[i,:]))
                    if data_param['Const_func'](Input) < data_param['Const_ub'] and data_param['Const_func'](Input) > data_param['Const_lb']:
                        Xinit[cnt,:] = Xinit_0[i, :]
                        cnt = cnt + 1
                else:
                    Input = Xinit_0[i, :]
                    if data_param['Const_func'](Input) < data_param['Const_ub'] and data_param['Const_func'](Input) > \
                            data_param['Const_lb']:
                        Xinit[cnt, :] = Xinit_0[i, :]
                        cnt = cnt + 1
            # res = minimize(min_obj, x0= Xinit_0[i, :], method='trust-constr', bounds=x_bounds,
            #                constraints=data_param['Constrains'], options={'verbose': 1})
            # if res.fun < min_val:
            #     min_val = res.fun
            #     Xinit[i, :] = res.x
    else:
        x_bounds = np.array([d['domain'] for d in bounds if d['type'] == 'continuous'])
        lower_bound = np.asarray(x_bounds)[:, 0].reshape(1, len(x_bounds))
        upper_bound = np.asarray(x_bounds)[:, 1].reshape(1, len(x_bounds))
        diff = upper_bound - lower_bound

        from pyDOE import lhs
        X_design_aux = lhs(data_param['Nx'], initN)
        I = np.ones((X_design_aux.shape[0], 1))
        X_design = np.dot(I, lower_bound) + X_design_aux * np.dot(I, diff)
        Xinit = X_design

    return Xinit