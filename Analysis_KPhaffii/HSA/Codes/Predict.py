
#==========================================
#CoCaBO Algorithm - Acquisition Function Definition
#==========================================

import math

import GPy
import numpy as np
import pandas as pd
import random

from typing import Union, Tuple
from paramz.transformations import Logexp

from Kernel import MixtureViaSumAndProduct, CategoryOverlapKernel
from SamplingCategorical import compute_prob_dist_and_draw_hts
from InitialData_Gen import initialize
from AcquisitionFunctions import EI, PI, UCB, AcquisitionOnSubspace
from matplotlib import pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint


def predict_internal(data, result, data_param, cont_kernel_name, data_ts): #
    #Scaling the data
    mu_x, std_x, mu_y, std_y, data_norm, result_norm = Scaling_data(data, result)
    data_norm[:, 0:data_param['Nc']] = data[:, 0:data_param['Nc']]
    # define kernel
    default_cont_lengthscale = [1] * data_param['Nx']  # cont lengthscale
    default_variance = 1
    if data_param['approach_type'] == 'CoCa':
        continuous_dims = list(range(data_param['Nc'], data_param['nDim']))
        categorical_dims = list(range(data_param['Nc']))
    else:
        continuous_dims = list(range(0, data_param['Nx']))

    mix_value = 0.2
    fix_mix_in_this_iter = False
    if cont_kernel_name == 'Matern52':
        k_cont = GPy.kern.Matern52(data_param['Nx'], variance=default_variance,
                                   active_dims=continuous_dims, ARD=True)  # continuous kernel
    elif cont_kernel_name == 'Matern32':
        k_cont = GPy.kern.Matern32(data_param['Nx'], variance=default_variance,
                                   active_dims=continuous_dims, ARD=True)  # continuous kernel
    else:
        k_cont = GPy.kern.RBF(data_param['Nx'], variance=default_variance,
                                   active_dims=continuous_dims, ARD=True)  # continuous kernel

    if data_param['approach_type'] == 'CoCa':
        bounds = data_param['bounds']
        C_list = data_param['C']

        k_cat = CategoryOverlapKernel(data_param['Nc'], active_dims=categorical_dims)  # categorical kernel

        my_kernel_v0 = MixtureViaSumAndProduct(data_param['nDim'], k_cat, k_cont, mix=mix_value, fix_inner_variances=True,
                                            fix_mix=fix_mix_in_this_iter)
        white_noise = GPy.kern.White(data_param['nDim'], variance=data_param['Meas_Noise'])
        white_noise.variance.fix()
        my_kernel = my_kernel_v0 + white_noise
        
        # build a GP  model
        gp =  GPy.models.GPRegression(data_norm, result_norm, my_kernel)#GPy.core.gp.GP(data, result, my_kernel, )
        # gp.set_XY(data[:-2, :], result[:-2,])
        gp.optimize(max_iters=2000)
        data_ts_norm = (data_ts - mu_x)/ std_x
        data_ts_norm[:, 0:data_param['Nc']] = data_ts[:, 0:data_param['Nc']]

        Y_ts = gp.posterior_samples_f(data_ts_norm, 10000)
        result_ts = Y_ts

    elif data_param['approach_type'] == 'Co':
        my_kernel = k_cont

        # build a GP  model
        gp = GPy.models.GPRegression(data_norm, result_norm, my_kernel)
        gp.set_XY(data_norm, result_norm)
        gp.optimize(max_iters=2000)

        gp_actual = gp
        Yp = gp.predict(data_norm)[0]
        plt.scatter(result_norm, Yp)
        plt.plot(result_norm, result_norm)

        print(gp)

    return gp, result_ts
        


def Scaling_data(data, result):
    mu_x = np.mean(data, 0)
    std_x = np.std(data, 0)
    mu_y = np.mean(result, 0)
    std_y = np.std(result, 0)
    data_norm = (data - mu_x) / std_x
    result_norm = (result - mu_y) / std_y

    return mu_x, std_x, mu_y, std_y, data_norm, result_norm

