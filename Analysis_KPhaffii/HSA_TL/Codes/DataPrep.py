import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold

import scipy
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import NonlinearConstraint


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.model_selection import train_test_split

from InitialData_Gen import initialize

import GPy
import GPyOpt
from numpy.random import seed
from Predict import predict_internal

import random
import pickle

def data_prep(main_file_path, N_round_input, Rounds_bfr_check, cut_off_round, property_name):
    data = {}
    result = {}
    Design = {}
    data_ts = {}
    result_ts = {}
    RMSEP = {}
    Pred = {}
    Pred_Std = {}
    Train = {}
    Train_Std = {}
    if property_name == 'Specific Productivity':
        alias_property_name = 'SP'
    elif property_name == 'Titer':
        alias_property_name = 'Titer'
    alt_path = '/Users/harini/Dropbox (MIT)/Harini Narayanan’s files/1_Home/Biotech/MediaOptimization/BO_MediaOptimization/Molecules/0_DataAnalysis/Prod_CoCaBO/'
    AllData_prev = pd.read_csv(alt_path + 'AllMolecule/HSA_CoCaBO_Prod_AllData.csv')
    result_prev = AllData_prev[alias_property_name].values.reshape(-1, 1)

    data_prev = np.concatenate((AllData_prev.iloc[:, 1:5].values,
                                1e-5 * np.ones((AllData_prev.shape[0], 4)),
                                0.999 * np.ones((AllData_prev.shape[0], 1))), axis=1)

    for nr in range(0, N_round_input ): #+ 1
        if nr == 0:
            background_file = main_file_path + "Codes/Round" + str(nr) + "/" + str(5) + "_ModelBackground_HSA_Prod.pkl"
            with open(background_file, "rb") as fp:
                ModelBackground_3 = pickle.load(fp)
        else:
            background_file = main_file_path + "Codes/Round" + str(nr - 1) + "/" + str(nr - 1) + "_ModelBackground.pkl"
            with open(background_file, "rb") as fp:
                ModelBackground_3 = pickle.load(fp)

        ModelBackground_3['data_param']['Meas_Noise'] = 0.1
        data_param = ModelBackground_3['data_param']
        cont_kernel_name = 'Matern52'

        if nr == 0:
            data[nr], result[nr], Design[nr] = data_prev, result_prev, data_prev
        else:
            data[nr], result[nr], Design[nr] = exp_data_collection(main_file_path, nr, property_name)

        data_ts[nr], result_ts[nr] = get_test_data(main_file_path, nr, N_round_input, Rounds_bfr_check, property_name)


        if nr ==0:
            tr_data = data_prev
            tr_y = result_prev
            mu_y = np.mean(tr_y, 0)
            std_y = np.std(tr_y, 0)

            gp_actual, result_pred_ts = predict_internal(tr_data, tr_y, data_param,
                                                         cont_kernel_name, data_ts[nr])

            gp_actual, result_pred_tr = predict_internal(tr_data, tr_y, data_param,
                                                         cont_kernel_name, tr_data)  # data[nr]

        elif nr > cut_off_round:
            tr_data = np.concatenate((data_prev,data[cut_off_round]))
            tr_y = np.concatenate((result_prev, result[cut_off_round]))

            mu_y = np.mean(tr_y, 0)
            std_y = np.std(tr_y, 0)

            gp_actual, result_pred_ts = predict_internal(tr_data, tr_y, data_param,
                                                         cont_kernel_name, data_ts[nr])

            gp_actual, result_pred_tr = predict_internal(tr_data, tr_y, data_param,
                                                         cont_kernel_name, tr_data) #data[cut_off_round]

        else:
            tr_data = np.concatenate((data_prev, data[nr]))
            tr_y = np.concatenate((result_prev, result[nr]))
            mu_y = np.mean(tr_y, 0)
            std_y = np.std(tr_y, 0)
            
            gp_actual, result_pred_ts = predict_internal(tr_data, tr_y, data_param,
                                                         cont_kernel_name, data_ts[nr])

            gp_actual, result_pred_tr = predict_internal(tr_data, tr_y, data_param,
                                                         cont_kernel_name, tr_data) #data[nr]

        Yp = np.multiply(np.mean(result_pred_ts, 2), std_y) + mu_y
        Std_ts = np.std(np.multiply(result_pred_ts, std_y) + mu_y, 2)
        Pred[nr] = Yp
        Pred_Std[nr] = Std_ts

        Yp_tr = np.multiply(np.mean(result_pred_tr, 2), std_y) + mu_y
        Train[nr] = Yp_tr
        Std_tr = np.std(np.multiply(result_pred_tr, std_y) + mu_y, 2)
        Train_Std[nr] = Std_tr

        if nr != N_round_input:
            if nr== N_round_input:
                RMSEP[nr] = np.sqrt(np.mean(np.square((Pred[nr] - result_ts[nr].reshape(-1, 1)))))
            else:
                RMSEP[nr] = np.sqrt(np.mean(np.square((Pred[nr] - result_ts[nr][:-1].reshape(-1, 1)))))

        if nr != N_round_input+1:
            if nr > cut_off_round:
                if nr == N_round_input:
                    plt.scatter(result_ts[nr], Pred[nr], s=60)  # [:-1]
                    plt.scatter(result[cut_off_round], Train[cut_off_round][-result[cut_off_round].shape[0]:], s=60)
                    plt.plot(result[cut_off_round], result[cut_off_round], '--k')
                    plt.xlabel('Observed -'+property_name)
                    plt.ylabel('Predicted - '+property_name)
                    plt.legend(['Round' + str(nr), 'Round 0-' + str(nr - 1)], loc='upper center',
                               bbox_to_anchor=(0.5, 1.2),
                               ncol=3, fancybox=True, shadow=True)
                    plt.savefig('ObservedPredicted' + str(nr) + '.png', dpi=72, bbox_inches='tight')
                    plt.show()
                else:
                    plt.scatter(result_ts[nr][:-1], Pred[nr], s=60)  # [:-1]
                    plt.scatter(result[cut_off_round], Train[cut_off_round][-result[cut_off_round].shape[0]:], s=60)
                    plt.plot(result[cut_off_round], result[cut_off_round], '--k')
                    plt.xlabel('Observed -' + property_name)
                    plt.ylabel('Predicted - ' + property_name)
                    plt.legend(['Round' + str(nr), 'Round 0-' + str(nr - 1)], loc='upper center',
                               bbox_to_anchor=(0.5, 1.2),
                               ncol=3, fancybox=True, shadow=True)
                    plt.savefig('ObservedPredicted' + str(nr) + '.png', dpi=72, bbox_inches='tight')
                    plt.show()
            else:
                if nr == N_round_input:
                    plt.scatter(result_ts[nr], Pred[nr], s=60)  # [:-1]
                    plt.scatter(result[nr], Train[nr][-result[nr].shape[0]:], s=60)
                    plt.plot(result[nr], result[nr], '--k')
                    plt.xlabel('Observed -' + property_name)
                    plt.ylabel('Predicted -' + property_name)
                    plt.legend(['Round' + str(nr), 'Round 0-' + str(nr - 1)], loc='upper center',
                               bbox_to_anchor=(0.5, 1.2),
                               ncol=3, fancybox=True, shadow=True)
                    plt.savefig('ObservedPredicted' + str(nr) + '.png', dpi=72, bbox_inches='tight')
                    plt.show()

                else:
                    plt.scatter(result_ts[nr][:-1], Pred[nr], s=60)  #
                    plt.scatter(result[nr], Train[nr][-result[nr].shape[0]:], s=60)
                    plt.plot(result[nr], result[nr], '--k')
                    plt.xlabel('Observed -' + property_name)
                    plt.ylabel('Predicted -' + property_name)
                    plt.legend(['Round' + str(nr), 'Round 0-' + str(nr - 1)], loc='upper center',
                               bbox_to_anchor=(0.5, 1.2),
                               ncol=3, fancybox=True, shadow=True)
                    plt.savefig('ObservedPredicted' + str(nr) + '.png', dpi=72, bbox_inches='tight')
                    plt.show()

    return Train, Train_Std, Pred, Pred_Std, RMSEP, result_ts, Design, gp_actual

def exp_data_collection(main_file_path, N_round, property_name):
    Design = {}
    Result_df = {}
    for nr in range(N_round):
        file_name = main_file_path + 'Codes/Round' + str(nr) + '/Reconstructed_Round' + str(nr) + '.csv'
        Design[nr] = pd.read_csv(file_name)

        Column_Names = pd.read_csv(file_name).columns

        file_name_res = main_file_path + 'Exp/Round' + str(nr) + '/Round' + str(nr) + '_Result_Summary_final.csv'
        Result_df[nr] = pd.read_csv(file_name_res)

        fac = int(Result_df[nr].shape[0] / Design[nr].shape[0])
        #         print(fac)
        data_init = np.repeat(Design[nr].iloc[:, 1:].values, fac, axis=0)
        if nr == 0:
            result_init = Result_df[nr][property_name].iloc[:-1, ].values.reshape(-1, 1)
            data_modified = []
            result_modified = []
            for nc in range(data_init.shape[0]):
                if data_init[nc, 1] == 0:
                    temp = np.concatenate((np.arange(0, 19).reshape(-1, 1), np.zeros((19, 1)),
                                           np.multiply(data_init[nc, 2:], np.ones((19, 7)))), axis=1)

                    temp_res = result_init[nc] * np.ones((19, 1))
                    data_modified + temp.tolist()
                    result_modified + temp_res.tolist()
                else:
                    data_modified.append(data_init[nc, :])
                    result_modified.append(result_init[nc])

            data_modified_np = np.array(data_modified)
            des_bench = np.concatenate((np.arange(0, 19).reshape(-1, 1),
                                        np.zeros((19, 1)), 0.4 * np.ones((19, 1)),
                                        0.15 * np.ones((19, 1)), 1e-5 * np.ones((19, 4)),
                                           0.999 * np.ones((19, 1))), axis=1)

            data = np.concatenate((data_modified, des_bench), axis=0)

            result = np.concatenate((np.array(result_modified),
                                     Result_df[nr][property_name].iloc[-1,] * np.ones((19, 1))),
                                    axis=0)  # , 0.0* np.ones((19,1)), ,  0.4*np.ones((19*4,1))

        else:
            result_init = Result_df[nr][property_name].iloc[:-1, ].values.reshape(-1, 1)
            data_modified = []
            result_modified = []
            for nc in range(data_init.shape[0]):
                if data_init[nc, 1] == 0:
                    temp = np.concatenate((np.arange(0, 19).reshape(-1, 1), np.zeros((19, 1)),
                                           np.multiply(data_init[nc, 2:], np.ones((19, 7)))), axis=1)

                    data_modified + temp.tolist()

                    temp_res = result_init[nc] * np.ones((19, 1))
                    result_modified + temp_res.tolist()
                else:
                    data_modified.append(data_init[nc, :])
                    result_modified.append(result_init[nc])

            data_modified_np = np.array(data_modified)

            des_bench = np.concatenate((np.arange(0, 19).reshape(-1, 1), np.zeros((19, 1)),
                                        0.4 * np.ones((19, 1)), 0.15 * np.ones((19, 1)), 1e-5 * np.ones((19, 4)),
                                           0.999 * np.ones((19, 1))), axis=1)

            data = np.concatenate((data, data_modified_np, des_bench), axis=0)  #

            result_mod_array = np.array(result_modified)
            result = np.concatenate((result, result_mod_array, Result_df[nr][property_name].iloc[-1,] * np.ones((19, 1))),
                                    axis=0)  #

    return data, result, Design

def get_test_data(main_file_path, N_round, N_round_input, Rounds_bfr_check, property_name):
    if (N_round < Rounds_bfr_check) and (N_round != N_round_input+ (N_round_input-Rounds_bfr_check)):
        file_name = main_file_path + 'Codes/Round' + str(N_round) + '/Reconstructed_Round' + str(N_round) + '.csv'
        Design = pd.read_csv(file_name)
        file_name = main_file_path + 'Exp/Round' + str(N_round) + '/Round' + str(
            N_round) + '_Result_Summary_final.csv'
        res = pd.read_csv(file_name)
        fac2 = int(res.shape[0] / Design.shape[0])
        data_init = np.repeat(Design.iloc[:, 1:].values, fac2, axis=0)

        result = res[property_name].values
        # print(result)

    elif N_round < Rounds_bfr_check and N_round == N_round_input+ (N_round_input-Rounds_bfr_check):  # and
        print("I am here")
        file_name = main_file_path + 'Codes/Round' + str(N_round) + '/Reconstructed_Round' + str(N_round) + '.csv'
        Design = pd.read_csv(file_name)
        fac2 = 1
        data_init = np.repeat(Design.iloc[:, 1:].values, fac2, axis=0)
        result = []
    else:
        print("Why did I come here")
        file_name = main_file_path + 'Codes/Checks/Reconstructed_Checks.csv'
        Design = pd.read_csv(file_name)
        file_name = main_file_path + 'Exp/Checks/Check_Result_Summary_final.csv'
        res = pd.read_csv(file_name)

        fac2 = int(res.shape[0] / Design.shape[0])
        fac2 = 1
        data_init = np.repeat(Design.iloc[:, 1:].values, fac2, axis=0)
        result = res[property_name].values

    return data_init, result