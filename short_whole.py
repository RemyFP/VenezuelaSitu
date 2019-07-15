# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:31:03 2019

@author: Remy
"""
import os
main_folder_l = ['C:','Users','Remy','Desktop','UT Austin','Meyers Lab',
                 'Venezuela Situational Awareness','RemySituAwareness']
main_folder = os.sep.join(main_folder_l)
os.chdir(main_folder)
import glob
from isoweek import Week
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
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
###############################################################################
###############################################################################
### Simple implementation of logic for testing
gold_standard_folder='GoldStandard'
candidate_folder='SourcesToOptimize'
normalized = True
# Gold standard data
gold_standard_path = os.sep.join([os.getcwd(),gold_standard_folder])
gold_standard_files = glob.glob(os.path.join(gold_standard_path, '*'))

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
for c in candidates_files:
    df_c = pd.read_csv(c)
    df_c.rename(columns={'year/week':'Date'},inplace=True)
    df_c.set_index('Date',inplace=True)
    source_name = c.split(os.sep)[-1].split('.')[0]
    candidates_data.update({source_name:df_c})


# Choose dates for training and testing, choose predictors source and gold standard
train_dates = ['1/2/2005','12/30/2012']
test_dates = ['1/6/2013','12/28/2014']
data_source = 'ColombiaPlusGTBySymptom' #'Colombia', 'ColombiaBorderPlusGT', 'ColombiaPlusGT', 
# 'ColombiaPlusGTByState', 'ColombiaPlusGTBySymptom', 'DengueGT_CO', 
# 'GTByStateVenAndCol', 'GTVenezuela'
goal_data_id = 'APURE-VE'
 # 'AMAZONAS-VE', 'ANZOATEGUI-VE', 'APURE-VE', 'ARAGUA-VE', 'BARINAS-VE', 'BOLIVAR-VE',
 # 'CARABOBO-VE', 'COJEDES-VE', 'DELTAAMACURO-VE', 'DTTOMETRO-VE', 'FALCON-VE',
 # 'GUARICO-VE', 'LARA-VE', 'MERIDA-VE', 'MIRANDA-VE', 'MONAGAS-VE', 'NUEVAESPARTA-VE',
 # 'PORTUGUESA-VE', 'SUCRE-VE', 'TACHIRA-VE', 'TOTAL-VE', 'TRUJILLO-VE',
 # 'VARGAS-VE', 'YARACUY-VE', 'ZULIA-VE'

# Choose number of folds to divide training data in
n_folds = 8
r_squared_threshold = 0.0001
lin_reg_intercept = True

# Load data
goal_df_single = df_goal.loc[:,[goal_data_id]]
predictor_df = candidates_data[data_source]

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
    
optimum = {'predictors':optimum_predictors,'R_squared':optimum_R_squared,
           'R_squared_OOS':optimum_OOS_R_squared}
df_optimum = pd.DataFrame(optimum)
print(df_optimum)

df_optimum_non_normalized = df_optimum.copy()


# Current
X_train = training_predictor.loc[:,optimum_predictors]
X_test = testing_predictor.loc[:,optimum_predictors]
y_train = training_goal
y_test = testing_goal
OOS_coef = lin_reg(y_train,X_train,lin_reg_intercept=lin_reg_intercept)
OOS_forecast_ts = lin_pred(X_test, OOS_coef)
r_squared_OOS = R_squared_quick(np.array(y_test.iloc[:,0]),OOS_forecast_ts)

## Comparing methods
## Regression
y_reg = np.array(y_train.iloc[:,0])
X_reg = np.array(X_train) #transpose
reg = linear_model.LinearRegression()
reg.fit(X_reg, y_reg)
intercept = reg.intercept_
coefficients = reg.coef_
# Forecast
X_fcst = np.array(X_test)
pred_series = np.dot(X_fcst, coefficients) + intercept
r_squared_OOS = R_squared_quick(np.array(y_test.iloc[:,0]),
                                        pred_series)
# Different code for forecast and R squared
test_pred = reg.predict(X_fcst)
test_mse = metrics.mean_squared_error(np.array(y_test.iloc[:,0]), test_pred)
test_r2 = metrics.r2_score(np.array(y_test.iloc[:,0]), test_pred)
denominator = float(scipy.stats.tvar(np.array(y_test.iloc[:,0])))
1-test_mse/denominator

plt.plot(pred_series)
plt.show()
plt.plot(test_pred)
plt.show()
plt.plot(y_test)
plt.show()



plt.plot(y_test)
plt.plot(OOS_forecast_ts)

plt.plot(y_test)
plt.plot(OOS_forecast_ts_from_norm)

plt.plot(OOS_forecast_ts)
plt.plot(OOS_forecast_ts_from_norm)



## With normalization
y_reg = np.array(y_train.iloc[:,0])
X_reg = np.array(X_train) #transpose
reg = linear_model.LinearRegression(normalize=True)
reg.fit(X_reg, y_reg)
intercept = reg.intercept_
coefficients = reg.coef_
# Forecast
X_fcst = np.array(X_test)
pred_series_normed = np.dot(X_fcst, coefficients) + intercept
r_squared_OOS = R_squared_quick(np.array(y_test.iloc[:,0]),
                                        pred_series)
# Different code for forecast and R squared
test_pred_normed = reg.predict(X_fcst)
test_mse = metrics.mean_squared_error(np.array(y_test.iloc[:,0]), test_pred)

plt.plot(pred_series_normed)
plt.show()
plt.plot(test_pred_normed)
plt.show()


## With manual normalization
## Data
X_train = training_predictor.loc[:,optimum_predictors]
X_test = testing_predictor.loc[:,optimum_predictors]
y_train = training_goal
y_test = testing_goal

## Normalize
# Training
X_train_norm = X_train.copy()
X_test_norm = X_test.copy()

for i in range(X_train.shape[1]):
    plt.plot(range(len(X_train)),X_train.iloc[:,i])
    plt.plot(range(len(X_train),len(X_train)+len(X_test)),X_test.iloc[:,i])
    plt.show()
    #print('\n',X_train.columns[i])
    x = X_train.iloc[:,i]
    #print('Training: ',np.round(np.mean(x),2),np.round(np.std(x),2))
    X_test_norm.iloc[:,i] = (X_test_norm.iloc[:,i]-np.mean(x)) / np.std(x)
    X_train_norm.iloc[:,i] = (X_train.iloc[:,i]-np.mean(x)) / np.std(x)
    #x = X_test_norm.iloc[:,i]
    #print('Testing: ',np.round(np.mean(x),2),np.round(np.std(x),2))
    #X_test_norm.iloc[:,i] = (X_test_norm.iloc[:,i]-np.mean(x)) / np.std(x)
# Testing
y_train_mean, y_train_std = np.mean(y_train)[0], np.std(y_train)[0]
y_train_norm = (y_train-y_train_mean)/y_train_std

OOS_coef_norm = lin_reg(y_train_norm,X_train_norm,lin_reg_intercept=lin_reg_intercept)
OOS_forecast_ts_norm = lin_pred(X_test_norm, OOS_coef_norm)
OOS_forecast_ts_from_norm = OOS_forecast_ts_norm * y_train_std + y_train_mean
r_squared_OOS_norm = R_squared_quick(np.array(y_test.iloc[:,0]),OOS_forecast_ts_from_norm)

plt.plot(OOS_forecast_ts_from_norm)
plt.show()
plt.plot(test_pred)
plt.show()
plt.plot(y_test)
plt.plot(OOS_forecast_ts_from_norm)
plt.show()

## Comparing methods
## Regression
y_reg = np.array(y_train.iloc[:,0])
X_reg = np.array(X_train) #transpose
reg = linear_model.LinearRegression()
reg.fit(X_reg, y_reg)
intercept = reg.intercept_
coefficients = reg.coef_
# Forecast
X_fcst = np.array(X_test)
pred_series = np.dot(X_fcst, coefficients) + intercept
r_squared_OOS = R_squared_quick(np.array(y_test.iloc[:,0]),
                                        pred_series)
# Different code for forecast and R squared
test_pred = reg.predict(X_fcst)
test_mse = metrics.mean_squared_error(np.array(y_test.iloc[:,0]), test_pred)
test_r2 = metrics.r2_score(np.array(y_test.iloc[:,0]), test_pred)
denominator = float(scipy.stats.tvar(np.array(y_test.iloc[:,0])))
1-test_mse/denominator

plt.plot(pred_series)
plt.show()
plt.plot(test_pred)
plt.show()



