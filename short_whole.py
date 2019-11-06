# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:31:03 2019

@author: Remy
"""
import os
main_folder_l = ['C:','Users','remyp','Research',
                 'Venezuela Situational Awareness','MultiObj']

main_folder = os.sep.join(main_folder_l)
os.chdir(main_folder)
import glob
from isoweek import Week
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import situ_fn
# import json
# import csv
import importlib
import datetime as dt

os.chdir(main_folder + os.sep + 'optimization')
import situational_awareness as sa
importlib.reload(sa)
import problem
importlib.reload(problem)
import filter_selection as fs
importlib.reload(fs)
os.chdir(main_folder)
from sklearn.model_selection import KFold
from sklearn import linear_model, metrics

np.set_printoptions(linewidth=130)
pd.set_option('display.width', 130)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:,.8f}'.format
pd.set_option('precision', -1)

###############################################################################
###############################################################################
def lin_reg(y,X,lin_reg_intercept=True):
    y_reg = np.array(y.iloc[:,0])
    X_reg = np.array(X) #transpose
    
    # With intercept: y = A*x + b
    if lin_reg_intercept:
        reg = linear_model.LinearRegression()
        reg.fit(X_reg, y_reg)
        intercept = reg.intercept_
        coefficients = reg.coef_
        
    # Without intercept: y = A*x
    else:
        intercept = 0
        coefficients = np.linalg.lstsq(X_reg, y_reg,rcond=-1)[0]
        
    return [intercept,coefficients]
###############################################################################
def lin_pred(X, coefficients):
    intercept, coef = coefficients
    X_reg = np.array(X)
    pred_series = np.dot(X_reg, coef) + intercept
    return pred_series
###############################################################################
def pred_CV_quick(training_goal, sources_df, n_folds=1,lin_reg_intercept=False):
    if n_folds > 1:
        # kf = KFold(dm.length(goal_datum), n_folds)
        kf_p = KFold(n_folds)
        kf = list(kf_p.split(range(len(training_goal))))
    else:
        v = range(len(training_goal))
        kf = [(v,v)]
    
    forecast_ts_CV_list = []
    for train, test in kf:
        # Split data between training and testing
        training_goal_train_k = training_goal.iloc[train,:]
        sources_df_train_k = sources_df.iloc[train,:]
        # training_goal_test_k = training_goal.iloc[test,:]
        sources_df_test_k = sources_df.iloc[test,:]
        
        # Do regression then forecasting with coefficients
        coefficients = lin_reg(training_goal_train_k,sources_df_train_k,lin_reg_intercept=True)
        forecast_ts = lin_pred(sources_df_test_k, coefficients)
        forecast_ts_CV_list.extend(forecast_ts)
        
    return forecast_ts_CV_list
###############################################################################
def R_squared_quick(actual_ts,forecast_ts):
    numerator = scipy.stats.tvar(actual_ts - forecast_ts)
    denominator = float(scipy.stats.tvar(actual_ts))
    rsquared = 1 - numerator/denominator
    return rsquared
###############################################################################
def forward_selection_algo(training_goal,training_predictor,
                           testing_goal,testing_predictor,
                           n_folds,lin_reg_intercept,r_squared_threshold,
                           normalized=False):
    # Outputs
    optimum_predictors = []
    optimum_R_squared = []
    optimum_OOS_R_squared = []
    
    # Normalizing if necessary
    X_train = training_predictor.copy()
    X_test = testing_predictor.copy()
    y_train = training_goal.copy()
    y_test = testing_goal.copy()
    
    if normalized:
        # Training
        for i in range(X_train.shape[1]):
            x = X_train.iloc[:,i]
            X_test.iloc[:,i] = (X_test.iloc[:,i]-np.mean(x)) / np.std(x)
            X_train.iloc[:,i] = (X_train.iloc[:,i]-np.mean(x)) / np.std(x)
            
            # x = X_test.iloc[:,i]
            # X_test.iloc[:,i] = (X_test.iloc[:,i]-np.mean(x)) / np.std(x)
        
        # Testing
        y_train_mean, y_train_std = np.mean(y_train)[0], np.std(y_train)[0]
        y_train = (y_train-y_train_mean)/y_train_std
    else:
        y_train_mean, y_train_std = 0.0, 1.0
        
    
    candidates = X_train.columns.tolist()
    # Loop through predictors in set to find optimal combination using a 
    # forward selection approach
    for i in range(len(candidates)):
        objective_values = []
        for c in candidates:
            temp_optimum = optimum_predictors + [c]
            sources_df = X_train.loc[:,temp_optimum]
            forecast_ts_CV_list = pred_CV_quick(y_train,sources_df,
                                                n_folds,lin_reg_intercept)
            r_squared = R_squared_quick(np.array(y_train.iloc[:,0]),
                                        forecast_ts_CV_list)
            objective_values.append(r_squared)
        # if i == 0:
        #     objective_values_single_datum = objective_values
        argmax = np.argmax(objective_values)
        interim_optimum = candidates.pop(argmax)
        
        # Stopping criteria
        if (i>0) and (max(objective_values)>0.15):
            if max(objective_values) < optimum_R_squared[-1] + r_squared_threshold:
                break
        
        optimum_predictors = optimum_predictors + [interim_optimum]
        optimum_R_squared.append(max(objective_values))
        
        # Test set of optimum series out of sample
        #print(problem.test_OOS(optimum))
        # Get OOS R squared
        sources_df = X_train.loc[:,optimum_predictors]
        OOS_coef = lin_reg(y_train,sources_df,lin_reg_intercept=lin_reg_intercept)
        sources_df = X_test.loc[:,optimum_predictors]
        OOS_forecast_ts = lin_pred(sources_df, OOS_coef)*y_train_std + y_train_mean
        r_squared_OOS = R_squared_quick(np.array(y_test.iloc[:,0]),
                                        OOS_forecast_ts)
        optimum_OOS_R_squared.append(r_squared_OOS)
    
    return optimum_predictors,optimum_R_squared,optimum_OOS_R_squared
###############################################################################
def forward_selection_multi_obj(training_goal,training_predictor,
                           testing_goal,testing_predictor,
                           n_folds,lin_reg_intercept,r_squared_threshold,
                           weights=None,
                           normalized=False):
    """ Here training goal and testing goal dataframes contain several columns
    for the different objectives.
        weights: list of weights to give to each of the columns in 
    training_goal (in the same order), they will be normalized to sum up to 1.
    If None then all weights are set equal to one.    
    """
    # Outputs
    optimum_predictors = []
    optimum_R_squared = [] # for the multi-objective
    optimum_R_squared_all = [] # for each gold standard in objective
    optimum_OOS_R_squared_all = []
    
    # Normalizing if necessary
    X_train = training_predictor.copy()
    X_test = testing_predictor.copy()
    y_train = training_goal.copy()
    y_test = testing_goal.copy()
    
    # Multiple objective time series
    nb_goal_ts = y_train.shape[1]
    
    if normalized:
        # Training
        for i in range(X_train.shape[1]):
            x = X_train.iloc[:,i]
            X_test.iloc[:,i] = (X_test.iloc[:,i]-np.mean(x)) / np.std(x)
            X_train.iloc[:,i] = (X_train.iloc[:,i]-np.mean(x)) / np.std(x)
            
            # x = X_test.iloc[:,i]
            # X_test.iloc[:,i] = (X_test.iloc[:,i]-np.mean(x)) / np.std(x)
        
        # Testing
        y_train_mean, y_train_std  = [0.0]*nb_goal_ts, [1.0]*nb_goal_ts
        for i in range(y_train.shape[1]):
            y = y_train.iloc[:,i]
            y_train_mean[i] = np.mean(y)
            y_train_std[i] = np.std(y)
            y_test.iloc[:,i] = (y_test.iloc[:,i]-y_train_mean[i]) / y_train_std[i]
            y_train.iloc[:,i] = (y_train.iloc[:,i]-y_train_mean[i]) / y_train_std[i]
    else:
        y_train_mean, y_train_std = [0.0]*nb_goal_ts, [1.0]*nb_goal_ts
    
    # Normalize weights for objective
    if weights is None:
        weights = [1] * y_train.shape[1]
    
    sum_weights = sum(weights)
    weights_norm = [x / sum_weights for x in weights]
    
    candidates = X_train.columns.tolist()
    # Loop through predictors in set to find optimal combination using a 
    # forward selection approach
    for i in range(min(len(candidates),40)):
#        print('\n        i = ' + np.str(i))
        objective_values = np.empty(shape=(len(candidates),nb_goal_ts))
        for j in range(nb_goal_ts):
#            print('j = ' + np.str(j))
            for i_c in range(len(candidates)):
                c = candidates[i_c]
                temp_optimum = optimum_predictors + [c]
                sources_df = X_train.loc[:,temp_optimum]
                forecast_ts_CV_list = pred_CV_quick(y_train.iloc[:,[j]],sources_df,
                                                    n_folds,lin_reg_intercept)
                r_squared = R_squared_quick(np.array(y_train.iloc[:,j]),
                                            forecast_ts_CV_list)
                objective_values[i_c,j] = r_squared
        # if i == 0:
        #     objective_values_single_datum = objective_values
        multi_obj_vals = np.sum(objective_values * np.array(weights_norm),axis=1)
        argmax = np.argmax(multi_obj_vals)
        interim_optimum = candidates.pop(argmax)
        
        # Stopping criteria
        if (i>0) and (max(multi_obj_vals)>0.15):
            if max(multi_obj_vals) < optimum_R_squared[-1] + r_squared_threshold:
                break
        
        optimum_predictors = optimum_predictors + [interim_optimum]
        optimum_R_squared.append(max(multi_obj_vals))
        optimum_R_squared_all.append(objective_values[argmax,:])
        
        # Test set of optimum series out of sample
        #print(problem.test_OOS(optimum))
        # Get OOS R squared for each objective time series
        OOS_R_sq = [-1] * nb_goal_ts
        for j in range(nb_goal_ts):
            sources_df = X_train.loc[:,optimum_predictors]
            OOS_coef = lin_reg(y_train.iloc[:,[j]],sources_df,
                               lin_reg_intercept=lin_reg_intercept)
            sources_df = X_test.loc[:,optimum_predictors]
            OOS_forecast_ts = lin_pred(sources_df, OOS_coef)*y_train_std[j] + y_train_mean[j]
            r_squared_OOS = R_squared_quick(np.array(y_test.iloc[:,j]),
                                            OOS_forecast_ts)
            OOS_R_sq[j] = r_squared_OOS
        optimum_OOS_R_squared_all.append(OOS_R_sq)
    
    # Compute average OOS R squared
    optimum_OOS_R_squared = np.sum(optimum_OOS_R_squared_all * \
                                   np.array(weights_norm),axis=1)
    
    return optimum_predictors,optimum_R_squared,optimum_OOS_R_squared,\
        np.array(optimum_OOS_R_squared_all),np.array(optimum_R_squared_all)
###############################################################################
###############################################################################
###############################################################################
### Simple implementation of logic for testing
# Folders
gold_standard_folder='GoldStandard'
candidate_folder='SourcesToOptimize'
save_folder = 'OptimizationResults'
exclude_predictors_path = 'Parameters' + os.sep + 'Predictors_Excluded.csv'

# Parameters
train_dates = ['1/2/2005','12/30/2012']
test_dates = ['1/6/2013','12/28/2014']
normalized = False
r_squared_threshold = 0.01
lin_reg_intercept = True


## Get predictors to exclude, if any
to_exclude = pd.read_csv(os.getcwd() + os.sep + exclude_predictors_path)
if 'Unnamed: 0' in to_exclude.columns:
    to_exclude.drop(columns='Unnamed: 0',inplace=True)
exclude_predictors = to_exclude.to_dict('list')
for k,v in exclude_predictors.items():
    new_v = [x.replace('-', '_') for x in v if pd.notnull(x)]
    exclude_predictors[k] = new_v

# Gold standard data
gold_standard_path = os.sep.join([os.getcwd(),gold_standard_folder])
gold_standard_files = glob.glob(os.path.join(gold_standard_path, '*'))

gold_standard_list = []
for g in gold_standard_files:
    gold_standard_list.append(g.split(os.sep)[-1].split('.')[0])

df_goal = pd.read_csv(gold_standard_files[0])
goal_name = [gold_standard_files[0].split(os.sep)[-1].split('.')[0]]
for g in gold_standard_files[1:]:
    df_g = pd.read_csv(g)
    df_goal = pd.merge(df_goal,df_g,left_on='year/week', 
                       right_on='year/week',how='left')
    goal_name.append(g.split(os.sep)[-1].split('.')[0])
df_goal.rename(columns={'year/week':'Date'},inplace=True)
df_goal.set_index('Date',inplace=True)


# Sources data
candidates_path = os.sep.join([os.getcwd(),candidate_folder])
candidates_files = glob.glob(os.path.join(candidates_path, '*'))  
candidates_data = {}
candidates_list = []
for c in candidates_files:
    df_c = pd.read_csv(c)
    df_c.rename(columns={'year/week':'Date'},inplace=True)
    df_c.set_index('Date',inplace=True)
    source_name = c.split(os.sep)[-1].split('.')[0]
    candidates_data.update({source_name:df_c})
    candidates_list.append(c.split(os.sep)[-1].split('.')[0])

## Loop through predictor sources, gold standards and number of folds
# Choose predictors source
#data_source = 'Colombia' #'Colombia', 'ColombiaBorderPlusGT', 'ColombiaPlusGT', 
# 'ColombiaPlusGTByState', 'ColombiaPlusGTBySymptom', 'DengueGT_CO', 
# 'GTByStateVenAndCol', 'GTVenezuela'
# Gold standard
#goal_data_id = 'AMAZONAS-VE'
 # 'AMAZONAS-VE', 'ANZOATEGUI-VE', 'APURE-VE', 'ARAGUA-VE', 'BARINAS-VE', 'BOLIVAR-VE',
 # 'CARABOBO-VE', 'COJEDES-VE', 'DELTAAMACURO-VE', 'DTTOMETRO-VE', 'FALCON-VE',
 # 'GUARICO-VE', 'LARA-VE', 'MERIDA-VE', 'MIRANDA-VE', 'MONAGAS-VE', 'NUEVAESPARTA-VE',
 # 'PORTUGUESA-VE', 'SUCRE-VE', 'TACHIRA-VE', 'TOTAL-VE', 'TRUJILLO-VE',
 # 'VARGAS-VE', 'YARACUY-VE', 'ZULIA-VE'
###############################################################################
###############################################################################
# Predictor sets
for data_source in candidates_list:
    # Gold standards
    for goal_data_id in gold_standard_list:
        # Number of folds to divide training data in
        for n_folds in [1,2,4,8]:
            # Load data
            goal_df_single = df_goal.loc[:,[goal_data_id]].copy()
            predictor_df = candidates_data[data_source].copy()
            
            # Replace - by _ in column names of predictors
            new_cols = [x.replace('-', '_') for x in predictor_df.columns]
            predictor_df.columns = new_cols
            
            # Exclude some predictors
            exclude_ = exclude_predictors[data_source]
            keep_cols = [x for x in predictor_df.columns if x not in exclude_]
            predictor_df = predictor_df.loc[:,keep_cols]
            
            # Filter training and testing data
            training_goal = goal_df_single.loc[train_dates[0]:train_dates[1]]
            testing_goal = goal_df_single.loc[test_dates[0]:test_dates[1]]
            training_predictor = predictor_df.loc[train_dates[0]:train_dates[1]]
            testing_predictor = predictor_df.loc[test_dates[0]:test_dates[1]]
            
            ## Forward selection algorithm
            optimum_predictors,optimum_R_squared,optimum_OOS_R_squared = \
                forward_selection_algo(training_goal,training_predictor,
                                       testing_goal,testing_predictor,
                                       n_folds,lin_reg_intercept,r_squared_threshold,
                                       normalized=normalized)
            
            optimum = {'id':optimum_predictors,'objective_values':optimum_R_squared,
                       'OOS_R_squared':optimum_OOS_R_squared}
            df_optimum = pd.DataFrame(optimum)
            #print(df_optimum)
            
            ## Save results
            # Make sure folder exists
            results_folder = save_folder + os.sep + np.str(n_folds)
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
                
            # Save results dataframe
            path = os.path.join(results_folder,
                '{}_{}_objective_optimum.csv'.format(goal_data_id, data_source))
            df_optimum.to_csv(path)
###############################################################################
# Aggregate results in single dataframe and keep final numbers only
for n_folds in [1,2,4]: #[8,16,24,32,40,48]:
    folder = 'OptimizationResults' + os.sep + np.str(n_folds)
    # Aggregate main function results per gold standard
    situ_fn.aggregate_results(data_folder=folder)
    folder_in = os.sep.join(['OptimizationResults',np.str(n_folds),'Summary'])
    situ_fn.write_summary_results(folder=folder_in,
                                  out_folder='OptimizationResults//SummaryAll',
                                  n_folds=n_folds)
###############################################################################
###############################################################################
### Multi objective: in algorithm predictor time series are added based on the
# increase of the average (potentially weighted) R squared across all regions
# (excluding the total country time series)
save_folder_multi = 'OptimizationResultsMulti'
# Predictor sets
for data_source in candidates_list:
    print(data_source)
    if data_source in ['Colombia','ColombiaBorderPlusGT','ColombiaPlusGT',\
                       'ColombiaPlusGTByState','ColombiaPlusGTBySymptom',\
                       'DengueGT_CO','GTByStateVenAndCol']:
        continue
    # Gold standards
#    for goal_data_id in gold_standard_list:
    goal_df_single = df_goal.loc[:,:].copy()
    training_goal = goal_df_single.loc[train_dates[0]:train_dates[1]]
    testing_goal = goal_df_single.loc[test_dates[0]:test_dates[1]]
    weights = [1 if x != 'TOTAL-VE' else 0 for x in goal_df_single.columns]
    
    # Number of folds to divide training data in
    for n_folds in [1,2,4]:
        print('Number of folds: ' + np.str(n_folds))
        if (data_source == 'GTVenezuela') and (n_folds in [1,2]):
            continue
        # Load data
        predictor_df = candidates_data[data_source].copy()
        
        # Replace - by _ in column names of predictors
        new_cols = [x.replace('-', '_') for x in predictor_df.columns]
        predictor_df.columns = new_cols
        
        # Exclude some predictors
        exclude_ = exclude_predictors[data_source]
        keep_cols = [x for x in predictor_df.columns if x not in exclude_]
        predictor_df = predictor_df.loc[:,keep_cols]
        
        # Filter training and testing data
        training_predictor = predictor_df.loc[train_dates[0]:train_dates[1]]
        testing_predictor = predictor_df.loc[test_dates[0]:test_dates[1]]
        
        ## Forward selection algorithm
        optimum_predictors,optimum_R_squared,optimum_OOS_R_squared, \
            optimum_OOS_R_squared_all,optimum_R_squared_all = \
            forward_selection_multi_obj(training_goal,training_predictor,
                                        testing_goal,testing_predictor,
                                        n_folds,lin_reg_intercept,r_squared_threshold,
                                        weights=weights,
                                        normalized=normalized)
        
        optimum = {'id':optimum_predictors,'objective_values':optimum_R_squared,
                   'OOS_R_squared':optimum_OOS_R_squared}
        df_optimum = pd.DataFrame(optimum)
        #print(df_optimum)
        
        ## Save results
        # Make sure folder exists
        results_folder = save_folder_multi + os.sep + np.str(n_folds)
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
            
        # Save results dataframe
        # Multi objective data
        goal_data_id = 'MultiObjective'
        path = os.path.join(results_folder,
            '{}_{}_objective_optimum.csv'.format(goal_data_id, data_source))
        df_optimum.to_csv(path)
        
        # Each region
        for j in range(len(goal_df_single.columns)):
            goal_data_id = goal_df_single.columns[j]
            optimum = {'id':optimum_predictors,
                       'objective_values':optimum_R_squared_all[:,j],
                       'OOS_R_squared':optimum_OOS_R_squared_all[:,j]}
            df_optimum = pd.DataFrame(optimum)
            path = os.path.join(results_folder,
                '{}_{}_objective_optimum.csv'.format(goal_data_id, data_source))
            df_optimum.to_csv(path)
###############################################################################
# Aggregate results in single dataframe and keep final numbers only
for n_folds in [1,2,4]: #[8,16,24,32,40,48]:
    folder = 'OptimizationResultsMulti' + os.sep + np.str(n_folds)
    # Aggregate main function results per gold standard
    situ_fn.aggregate_results(data_folder=folder)
    folder_in = os.sep.join(['OptimizationResultsMulti',np.str(n_folds),'Summary'])
    situ_fn.write_summary_results(folder=folder_in,
                                  out_folder='OptimizationResultsMulti//SummaryAll',
                                  n_folds=n_folds)
###############################################################################
###############################################################################

    
    
    
    