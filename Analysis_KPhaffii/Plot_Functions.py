# =============================================================================
#Funcitons for plotting and data analysis
# =============================================================================
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

import GPy
import GPyOpt
from numpy.random import seed

def CollectData(root_path, carbon_source_filepath, Molecule_list, N_round_list):
    Stock_solid = pd.read_excel(carbon_source_filepath + 'CarbonSourceInfo.xlsx', 'Stocks_solid')
    Stock_liquid = pd.read_excel(carbon_source_filepath + 'CarbonSourceInfo.xlsx', 'Stocks_liquid')

    Carbon_Names = Stock_solid['Carbon Source'].values.tolist()
    Carbon_Names.append(Stock_liquid['Carbon Source'][1])
    Carbon_Names.append(Stock_liquid['Carbon Source'][2])

    #N_round = 5
    Design = {}
    Result = {}
    Design_id = {}

    Result_org = {}
    Design_id_org = {}

    res_bench = {}

    Titer_all = {}
    SP_all = {}
    OD_og_all = {}
    OD_prod_all = {}
    Design_all = {}
    res_bench_all = {}
    for Molecule_Name in Molecule_list:
        main_file_path = root_path + '/' + Molecule_Name + '/' + Molecule_Name+'/'
        # analysis_name = Molecule_Name #+ '_Prod_CoCaBO'

        Result_org[Molecule_Name] = {}
        Design_id_org[Molecule_Name] = {}

        res_bench[Molecule_Name] = {}
        Design[Molecule_Name] = {}
        Result[Molecule_Name] = {}
        Design_id[Molecule_Name] = {}

        N_round = N_round_list[Molecule_list.index(Molecule_Name)]
        for nr in range(N_round):
            file_name = root_path + Molecule_Name +  '/Exp/Round' + str(nr) + '/Round' + str(
                nr) + '_Design_Summary.csv'
            Design[Molecule_Name][nr] = pd.read_csv(file_name).iloc[:, 1:].values
            Design_id_org[Molecule_Name][nr] = Design[Molecule_Name][nr]
            Column_Names = pd.read_csv(file_name).columns
            file_name_res = root_path + Molecule_Name +  '/Exp/Round' + str(nr) + '/Round' + str(
                nr) + '_Result_Summary_final.csv'
            Result_org[Molecule_Name][nr] = pd.read_csv(file_name_res).iloc[:, 2:].values
            # Titer - 0, SP - 1, Prod_OD - 2, OG_OG - 3
            data_modified_v1 = []
            result_modified = []
            for k in range(Design[Molecule_Name][nr].shape[0]):
                Design_id_org[Molecule_Name][nr][k, 0] = Carbon_Names.index(Design[Molecule_Name][nr][k, 0])
                if Design_id_org[Molecule_Name][nr][k, 1] == 0:
                    temp = np.concatenate((np.arange(0, 19).reshape(-1, 1),
                                           np.zeros((19, 1)), Design_id_org[Molecule_Name][nr][k, 2] * np.ones((19, 1)),
                                           Design_id_org[Molecule_Name][nr][k,3] * np.ones((19, 1))), axis=1)
                    temp_res = np.multiply(Result_org[Molecule_Name][nr][k,:], np.ones((19, 4)))
                    data_modified_v1 + temp.tolist()
                    result_modified + temp_res.tolist()
                else:
                    data_modified_v1.append(Design_id_org[Molecule_Name][nr][k, :])
                    result_modified.append(Result_org[Molecule_Name][nr][k,:])

            data_modified = np.array(data_modified_v1)
            des_bench = np.concatenate((np.arange(0, 19).reshape(-1, 1), np.zeros((19, 1)), 4 * np.ones((19, 1)),
                                        1.5 * np.ones((19, 1))), axis=1)

            Design_id[Molecule_Name][nr] = np.concatenate((data_modified, des_bench), axis=0)
            Result[Molecule_Name][nr] = np.concatenate(
                (np.array(result_modified), Result_org[Molecule_Name][nr][-1,:] * np.ones((19, 1))),
                axis=0)
            res_bench[Molecule_Name][nr] = Result_org[Molecule_Name][nr][-1:,:]
        if Molecule_Name in ['RBDJ', 'HSA', 'Trastuzumab']:

            file_name = root_path + Molecule_Name +  '/Exp/Checks/Check_Design_Summary.csv'
            Design[Molecule_Name][N_round] = pd.read_csv(file_name).iloc[:, 1:].values
            Design_id_org[Molecule_Name][N_round] = Design[Molecule_Name][N_round]
            file_name_res = root_path + Molecule_Name  + '/Exp/Checks/Check_Result_Summary_final.csv'
            Result_org[Molecule_Name][N_round] = pd.read_csv(file_name_res).iloc[:, 2:].values
            data_modified_v1 = []
            result_modified = []
            for k in range(Design[Molecule_Name][N_round].shape[0]):
                Design_id_org[Molecule_Name][N_round][k, 0] = Carbon_Names.index(Design[Molecule_Name][N_round][k, 0])
                if Design_id_org[Molecule_Name][N_round][k, 1] == 0:
                    temp = np.concatenate((np.arange(0, 19).reshape(-1, 1),
                                           np.zeros((19, 1)), Design_id_org[Molecule_Name][N_round][k, 2] * np.ones((19, 1)),
                                           Design_id_org[Molecule_Name][N_round][k, 3] * np.ones((19, 1))), axis=1)

                    data_modified_v1 + temp.tolist()
                    temp_res = np.multiply(Result_org[Molecule_Name][N_round][k, :], np.ones((19, 4)))
                    result_modified + temp_res.tolist()
                else:
                    data_modified_v1.append(Design_id_org[Molecule_Name][N_round][k,:])
                    result_modified.append(Result_org[Molecule_Name][N_round][k,:])
                    
            data_modified = np.array(data_modified_v1)
            des_bench = np.concatenate((np.arange(0, 19).reshape(-1, 1), np.zeros((19, 1)), 4 * np.ones((19, 1)),
                                        1.5 * np.ones((19, 1))), axis=1)
            Design_id[Molecule_Name][N_round]= np.concatenate((data_modified, des_bench), axis=0)
            Result[Molecule_Name][N_round] = np.concatenate(
                (np.array(result_modified), Result_org[Molecule_Name][N_round][-1, :] * np.ones((19, 1))),
                axis=0)

            res_bench[Molecule_Name][N_round] = Result_org[Molecule_Name][N_round][-1:, :]

        Titer_all[Molecule_Name] = []
        SP_all[Molecule_Name] = []
        OD_og_all[Molecule_Name] = []
        OD_prod_all[Molecule_Name] = []
        res_bench_all[Molecule_Name] = []
        if Molecule_Name in ['RBDJ', 'HSA','Trastuzumab']:
            for nr in range(N_round+ 1): #
                Titer_all[Molecule_Name] = Titer_all[Molecule_Name] + Result[Molecule_Name][nr][:,0].tolist()
                SP_all[Molecule_Name] = SP_all[Molecule_Name] + Result[Molecule_Name][nr][:,1].tolist()
                OD_og_all[Molecule_Name] = OD_og_all[Molecule_Name] + Result[Molecule_Name][nr][:,3].tolist()
                OD_prod_all[Molecule_Name] = OD_prod_all[Molecule_Name] + Result[Molecule_Name][nr][:,2].tolist()

                if nr == 0:
                    Design_all[Molecule_Name] = Design_id[Molecule_Name][nr]
                    res_bench_all[Molecule_Name] = res_bench[Molecule_Name][nr]
                elif nr > 0:
                    Design_all[Molecule_Name] = np.concatenate(
                        (Design_all[Molecule_Name], Design_id[Molecule_Name][nr]),axis=0)  # 'Ribose'
                    res_bench_all[Molecule_Name] = np.concatenate((res_bench_all[Molecule_Name], res_bench[Molecule_Name][nr]), axis =0)
        else:
            for nr in range(N_round): #+ 1
                Titer_all[Molecule_Name] = Titer_all[Molecule_Name] + Result[Molecule_Name][nr][:,0].tolist()
                SP_all[Molecule_Name] = SP_all[Molecule_Name] + Result[Molecule_Name][nr][:,1].tolist()
                OD_og_all[Molecule_Name] = OD_og_all[Molecule_Name] + Result[Molecule_Name][nr][:,3].tolist()
                OD_prod_all[Molecule_Name] = OD_prod_all[Molecule_Name] + Result[Molecule_Name][nr][:,2].values.tolist()

                if nr == 0:
                    Design_all[Molecule_Name] = Design_id[Molecule_Name][nr]
                    res_bench_all[Molecule_Name] = res_bench[Molecule_Name][nr]
                elif nr > 0:
                    Design_all[Molecule_Name] = np.concatenate(
                        (Design_all[Molecule_Name], Design_id[Molecule_Name][nr]),axis=0)  # 'Ribose'
                    res_bench_all[Molecule_Name] = pd.concat(
                        (res_bench_all[Molecule_Name], res_bench[Molecule_Name][nr]), axis=0)


    return Design, Design_all, Result, Titer_all, SP_all, OD_prod_all, OD_og_all, res_bench, res_bench_all

