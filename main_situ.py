# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 20:22:18 2019

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
import situ_fn

os.chdir(main_folder + os.sep + 'optimization')
import situational_awareness as sa
importlib.reload(sa)
import problem
importlib.reload(problem)
import filter_selection as fs
importlib.reload(fs)
os.chdir(main_folder)
from sklearn.model_selection import KFold
from sklearn import linear_model
#import pdb


# if os.getcwd()[-12:] != 'optimization':
#     os.chdir(os.getcwd() + '\\optimization\\')
# import problem
# import filter_selection as fs
# import situational_awareness as sa
# os.chdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

def _to_json(_id, _times, _values):
    """
    Generate a dict
    """
    data_source = {'id': _id,
                   'data':
                   {'times': _times,
                    'values': _values}
                  }
    return data_source

def csv_to_json(file_path):
    """
    Convert data from csv format to Json format
    """
    # read csv
    csv_data = pd.read_csv(file_path)
    
    # check value in column 'year/week' is date or year/week
    if len(str(csv_data.loc[0, 'year/week']).split('/')) > 1:
        csv_data.loc[:, 'date'] = csv_data.loc[:, 'year/week'].apply(pd.to_datetime)
    else:
        # convert year /week to date
        csv_data.loc[:, 'year'] = [i[:4] for i in csv_data.loc[:, 'year/week'].apply(str)]
        csv_data.loc[:, 'week'] = [i[-2:] for i in csv_data.loc[:, 'year/week'].apply(str)]
        csv_data.loc[:, 'date'] = [pd.to_datetime(Week(int(csv_data.loc[i, 'year']),
                                        int(csv_data.loc[i, 'week'])).sunday()) for i in csv_data.index.values]
    
    # write into json
    column_names = csv_data.columns
    all_data = []
    for i in range(csv_data.shape[1]):
        if column_names[i] in ['year/week', 'year', 'week', 'date']:
            pass
        else:
            _id = column_names[i].replace('-', '_')
            _times = csv_data.loc[:, 'date'].tolist()
            _values = csv_data.loc[:, column_names[i]].tolist()
            data_source = {'id': _id,
                           'data':
                           {'times': _times,
                            'values': _values}
                          }
            all_data.append(data_source)
    return all_data


def filter_data(data,
                date_start='2010-11-22',
                date_end='2016-09-19',
                copy=True):
    """
    Filter data to make sure all data are within the same date range
    """
    if copy == True:
        data_f = data[:]
    else:
        data_f = data
    
    for i in range(len(data_f)):
        bool_mask = fs.filter_on_times(data_f[i], after=date_start, before=date_end)
        bool_mask_list = list(bool_mask) # for Python 3 map is one time only
        d = fs.select(data_f[i], bool_mask_list)
        data_f[i] = d
    return data_f


def optimization(gold_standard,
                 candidate_data_sources,
               objective='R_squared',
               n_folds=15,
               output_size=141,
               OOS_data=None,
               threshold_optim=0.0001,
               lin_reg_intercept=False):
    """
    Run optimization for situational awareness.
    
    candidate_data_sources: list
    gold_standard: dict
    OOS_data: list (gold standard and candidate_date_sources for OOS dates)
    lin_reg_intercept: whether to include an intercept when doing linear
        regressions (ie: y = A*x + b instead of y = A*x)
    """
    # optimization
    sa_optimization = problem.FS_problem(gold_standard,
                                         candidate_data_sources,
                                         req_data=[],
                                         objective=objective,
                                         n_folds=n_folds,
                                         OOS_data=OOS_data,
                                         lin_reg_intercept=lin_reg_intercept)
    optimum, objective_value, objective_single, OOS_R_squared = \
        sa_optimization.optimize('forward_selection', choose=output_size,
                                 threshold_optim=threshold_optim)
    # predictions based on optimum
    prediction = sa.pred_CV(gold_standard, optimum)['data']
    return optimum, objective_value, objective_single, prediction, OOS_R_squared


def main(gold_standard_folder,
         candidate_folder,
         date_start='2010-11-22',
         date_end='2016-09-19',
         objective='R_squared',
         n_folds=15,
         output_size=141,
         save_folder='OptimizationResults',
         save_start_date=False,
         OOS_testing_dates=None, # = ['1/5/2014', '1/31/2015']
         threshold_optim=None,
         lin_reg_intercept=False,
         exclude_predictors_path=None):
    # Get predictors to exclude, if any
    if exclude_predictors_path is not None:
        to_exclude = pd.read_csv(os.getcwd() + os.sep + exclude_predictors_path)
        if 'Unnamed: 0' in to_exclude.columns:
            to_exclude.drop(columns='Unnamed: 0',inplace=True)
        exclude_predictors = to_exclude.to_dict('list')
        for k,v in exclude_predictors.items():
            new_v = [x.replace('-', '_') for x in v if pd.notnull(x)]
            exclude_predictors[k] = new_v
        
    
    # retrieve all csv files in each folder
    gold_standard_paths = glob.glob(os.path.join(gold_standard_folder, '*'))
    candidate_paths = glob.glob(os.path.join(candidate_folder, '*'))
    
    for g_path in gold_standard_paths:
        # load gold standard
        print('\n\nGold standard: {}'.format(g_path.split('/')[-1]))
        gold_standard = csv_to_json(g_path)
        # print('Done loading gold standard.')
        
        for c_path in candidate_paths:
            # load each csv file in SourceToOptimize
            print('Candidate data source: {}'.format(c_path.split('/')[-1]))
            candidate_data_sources = csv_to_json(c_path)
            # print('Done loading candidate data sources.')
            set_name = c_path.split(os.sep)[-1].split('.')[0]
            exclude_from_c = exclude_predictors[set_name]
            candidate_data_sources = [x for x in candidate_data_sources if
                                      x['id'] not in exclude_from_c]
            
            
            ## Filter gold standard and candidate data sources
            # Testing data
            if OOS_testing_dates is not None:
                start_OOS, end_OOS = OOS_testing_dates
                gold_standard_OOS = filter_data(gold_standard,
                                            date_start=start_OOS,
                                            date_end=end_OOS)
                candidate_data_sources_OOS = filter_data(candidate_data_sources,
                                                     date_start=start_OOS,
                                                     date_end=end_OOS)
            # Training data
            gold_standard_train = filter_data(gold_standard,
                                        date_start=date_start,
                                        date_end=date_end)
            candidate_data_sources_train = filter_data(candidate_data_sources,
                                                 date_start=date_start,
                                                 date_end=date_end)
            
            # If output_size is equal to 'max' we set it equal to the number
            # of candidate data sources
            if output_size == 'max':
                output_size_int = len(candidate_data_sources)
            else:
                output_size_int = output_size
            
            # print('Done filtering data.')
            
            # optimization
            optimum, objective_value, objective_single, prediction, OOS_R_squared = \
                 optimization(gold_standard_train[0],
                              candidate_data_sources_train,
                              objective=objective,
                              n_folds=n_folds,
                              output_size=output_size_int,
                              OOS_data=[gold_standard_OOS[0],candidate_data_sources_OOS],
                              threshold_optim=threshold_optim,
                              lin_reg_intercept=lin_reg_intercept)
            
            # get name of gold standard csv and candidate data csv names.
            # g_name = g_path.split('/')[-1].split('.')[0]
            # c_name = c_path.split('/')[-1].split('.')[0]
            g_name = g_path.split(os.sep)[-1].split('.')[0]
            c_name = c_path.split(os.sep)[-1].split('.')[0]
            
            # write to csv
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                
            # save performance of each single data source in candidate
            ids = [d['id'] for d in candidate_data_sources_train]
            df = pd.DataFrame(columns=['id', 'objective_values'])
            df['id'] = ids
            df['objective_values'] = objective_single
            if save_start_date:
                path = os.path.join(save_folder,
                    '{}_{}_objective_single_{}.csv'.format(g_name, c_name,date_start))
            else:
                path = os.path.join(save_folder,
                    '{}_{}_objective_single.csv'.format(g_name, c_name))
            df.to_csv(path)

            # save performance of optimal combination of data sources.
            id_optimum = [i['id'] for i in optimum]
            df = pd.DataFrame(columns=['id', 'objective_values','OOS_R_squared'])
            df['id'] = id_optimum
            df['objective_values'] = objective_value[1:]
            if len(OOS_R_squared) > 0:
                df['OOS_R_squared'] = OOS_R_squared
            if save_start_date:
                path = os.path.join(save_folder,
                    '{}_{}_objective_optimum_{}.csv'.format(g_name, c_name,date_start))
            else:
                path = os.path.join(save_folder,
                    '{}_{}_objective_optimum.csv'.format(g_name, c_name))
            df.to_csv(path)

            # save prediction results
            df = pd.DataFrame(columns=['times', 'values'])
            df['times'] = prediction['times']
            df['values'] = prediction['values']
            if save_start_date:
                path = os.path.join(save_folder,
                    '{}_{}_prediction_{}.csv'.format(g_name, c_name,date_start))
            else:
                path = os.path.join(save_folder,
                    '{}_{}_prediction.csv'.format(g_name, c_name))
            df.to_csv(path)
                                
            # print('Done optimization using gold standard from {} and candidate data from {}.\n'.format(g_path.split('/')[-1], c_path.split('/')[-1]))
    
    # print('Done all optimizations!')
###############################################################################
### Run main
run_main_functions = False
if run_main_functions:
    n_folds=24 # 8 ################
    results_folder = 'OptimizationResults' + os.sep + np.str(n_folds)
    date_start = '1/2/2005'
    date_end = '1/6/2013'
    OOS_testing_dates = ['1/6/2013','1/31/2015']
    gold_standard_folder='GoldStandard'
    candidate_folder='SourcesToOptimize'
    objective='R_squared'
    output_size='max'
    save_start_date=False
    threshold_optim=0.01
    lin_reg_intercept=True
    exclude_predictors_path = 'Parameters' + os.sep + 'Predictors_Excluded.csv'
    
    main(gold_standard_folder='GoldStandard',
                candidate_folder='SourcesToOptimize',
                date_start=date_start,
                date_end=date_end,
                #    date_start='2010-11-22',
                #    date_end='2013-09-19',
                objective='R_squared',
                n_folds=n_folds,
                output_size='max',
                save_folder=results_folder,
                save_start_date=False,
                OOS_testing_dates=OOS_testing_dates,
                threshold_optim=0.0001,
                lin_reg_intercept=lin_reg_intercept,
                exclude_predictors_path=exclude_predictors_path)
########################################
    n_folds=48 # 8 ################
    results_folder = 'OptimizationResults' + os.sep + np.str(n_folds)
    date_start = '1/2/2005'
    date_end = '1/6/2013'
    OOS_testing_dates = ['1/6/2013','1/31/2015']
    gold_standard_folder='GoldStandard'
    candidate_folder='SourcesToOptimize'
    objective='R_squared'
    output_size='max'
    save_start_date=False
    threshold_optim=0.0001
    lin_reg_intercept=True
    exclude_predictors_path = 'Parameters' + os.sep + 'Predictors_Excluded.csv'
    
    main(gold_standard_folder='GoldStandard48',
                candidate_folder='SourcesToOptimize',
                date_start=date_start,
                date_end=date_end,
                #    date_start='2010-11-22',
                #    date_end='2013-09-19',
                objective='R_squared',
                n_folds=n_folds,
                output_size='max',
                save_folder=results_folder,
                save_start_date=False,
                OOS_testing_dates=OOS_testing_dates,
                threshold_optim=0.0001,
                lin_reg_intercept=lin_reg_intercept,
                exclude_predictors_path=exclude_predictors_path)
########################################
###############################################################################
# Aggregate main function results per gold standard
situ_fn.aggregate_results(data_folder='OptimizationResults//8')
situ_fn.aggregate_results(data_folder='OptimizationResults//16')
situ_fn.aggregate_results(data_folder='OptimizationResults//24')
situ_fn.aggregate_results(data_folder='OptimizationResults//32')
situ_fn.aggregate_results(data_folder='OptimizationResults//40')
situ_fn.aggregate_results(data_folder='OptimizationResults//48')
# Aggregate results in single dataframe and keep final numbers only
for n_folds in [8,16,24,32,40,48]:
    folder_in = os.sep.join(['OptimizationResults',np.str(n_folds),'Summary'])
    situ_fn.write_summary_results(folder=folder_in,
                                  out_folder='OptimizationResults//SummaryAll',
                                  n_folds=n_folds)
###############################################################################
### Debugging
results_folder = 'OptimizationResultsDummy'
date_start = '1/2/2005'
date_end = '1/6/2013'
OOS_testing_dates = ['1/6/2013','1/31/2015']
gold_standard_folder='GoldStandard'
candidate_folder='SourcesToOptimize'
objective='R_squared'
n_folds=8
output_size='max'
save_start_date=False
threshold_optim=0.0001
lin_reg_intercept=True

save_folder=results_folder
gold_standard_paths = glob.glob(os.path.join(gold_standard_folder, '*'))
candidate_paths = glob.glob(os.path.join(candidate_folder, '*'))
g_path = gold_standard_paths[20]
c_path = candidate_paths[3]
gold_standard = csv_to_json(g_path)
candidate_data_sources = csv_to_json(c_path)

main(gold_standard_folder='GoldStandard',
            candidate_folder='SourcesToOptimize',
            date_start=date_start,
            date_end=date_end,
            #    date_start='2010-11-22',
            #    date_end='2013-09-19',
            objective='R_squared',
            n_folds=n_folds,
            output_size='max',
            save_folder=results_folder,
            save_start_date=False,
            OOS_testing_dates=OOS_testing_dates,
            threshold_optim=0.0001,
            lin_reg_intercept=True)
###############################################################################
### Manually test
file_path = 'TOTAL-VE_ColombiaPlusGTByState_objective_optimum.csv'
df = pd.read_csv(file_path)

gold_standard = csv_to_json(g_path)
candidate_data_sources = csv_to_json(c_path)

# OOS data
start_OOS, end_OOS = OOS_testing_dates
gold_standard_OOS = filter_data(gold_standard,
                            date_start=start_OOS,
                            date_end=end_OOS)
candidate_data_sources_OOS = filter_data(candidate_data_sources,
                                     date_start=start_OOS,
                                     date_end=end_OOS)
# Training data
gold_standard_train = filter_data(gold_standard,
                            date_start=date_start,
                            date_end=date_end)
candidate_data_sources_train = filter_data(candidate_data_sources,
                                     date_start=date_start,
                                     date_end=date_end)

goal_f = gold_standard_train[0]
list_s = [x for x in df.iloc[:,1]]
subset = [x for x in candidate_data_sources_train if x['id'] in list_s]
subset = [x for x in candidate_data_sources_train]

coef = sa.lin_reg(goal_f,subset,lin_reg_intercept=True)
pred_series_in_sample = sa.lin_pred(subset,coef)
goal_series_in_sample = goal_f['data']['values']
numerator = scipy.stats.tvar(goal_series_in_sample - pred_series_in_sample)
denominator = float(scipy.stats.tvar(goal_series_in_sample))
rsquared_in_sample = 1 - numerator/denominator
plt.plot(goal_series_in_sample)
plt.plot(pred_series_in_sample)
plt.show()
print(rsquared_in_sample)

# Get predicted time series out of sample
subset_ids = [x['id'] for x in subset] # subset_all = subset_ids[:]
OOS_subset = [datum for datum in candidate_data_sources_OOS if datum['id'] in subset_ids]
pred_series_OOS = sa.lin_pred(OOS_subset,coef)

# Compute R-squared for out of sample data
goal_series_OOS = gold_standard_OOS[0]['data']['values']
numerator_OOS = scipy.stats.tvar(goal_series_OOS - pred_series_OOS)
denominator_OOS = float(scipy.stats.tvar(goal_series_OOS))
rsquared_OOS = 1 - numerator_OOS/denominator_OOS
plt.plot(goal_series_OOS)
plt.plot(pred_series_OOS)
plt.show()
print(rsquared_OOS)


for x in subset: #candidate_data_sources_train:
    plt.plot(x['data']['times'],x['data']['values'])
    plt.title(x['id'])
    plt.show()
    


people = ['Jim', 'Pam', 'Micheal', 'Dwight']
ages = [27, 25, 4, 9]

people = np.array(people)
ages = np.array(ages)
inds = ages.argsort()
sortedPeople = people[inds]


subset = sorted(subset, key = lambda x: x['id']) 

###############################################################################
###############################################################################
###############################################################################
### Simple implementation of logic for testing
gold_standard_folder='GoldStandard'
candidate_folder='SourcesToOptimize'
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
data_source = 'ColombiaPlusGT' #'Colombia', 'ColombiaBorderPlusGT', 'ColombiaPlusGT', 
# 'ColombiaPlusGTByState', 'ColombiaPlusGTBySymptom', 'DengueGT_CO', 
# 'GTByStateVenAndCol', 'GTVenezuela'
goal_data_id = 'AMAZONAS-VE'
 # 'AMAZONAS-VE', 'ANZOATEGUI-VE', 'APURE-VE', 'ARAGUA-VE', 'BARINAS-VE', 'BOLIVAR-VE',
 # 'CARABOBO-VE', 'COJEDES-VE', 'DELTAAMACURO-VE', 'DTTOMETRO-VE', 'FALCON-VE',
 # 'GUARICO-VE', 'LARA-VE', 'MERIDA-VE', 'MIRANDA-VE', 'MONAGAS-VE', 'NUEVAESPARTA-VE',
 # 'PORTUGUESA-VE', 'SUCRE-VE', 'TACHIRA-VE', 'TOTAL-VE', 'TRUJILLO-VE',
 # 'VARGAS-VE', 'YARACUY-VE', 'ZULIA-VE'

# Choose number of folds to divide training data in
n_folds = 8
r_squared_threshold = 0.0001
intercept = True

# Load data
goal_df_single = df_goal.loc[:,[goal_data_id]]
predictor_df = candidates_data[data_source]

# Filter training and testing data
training_goal = goal_df_single.loc[train_dates[0]:train_dates[1]]
testing_goal = goal_df_single.loc[test_dates[0]:test_dates[1]]
training_predictor = predictor_df.loc[train_dates[0]:train_dates[1]]
testing_predictor = predictor_df.loc[test_dates[0]:test_dates[1]]

# Forward selection algorithm

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
                           n_folds,intercept,r_squared_threshold):
    # Outputs
    optimum_predictors = []
    optimum_R_squared = []
    optimum_OOS_R_squared = []
    
    candidates = training_predictor.columns.tolist()
    # Loop through predictors in set to find optimal combination using a 
    # forward selection approach
    for i in range(len(candidates)):
        objective_values = []
        for c in candidates:
            temp_optimum = optimum_predictors + [c]
            sources_df = training_predictor.loc[:,temp_optimum]
            forecast_ts_CV_list = pred_CV_quick(training_goal,sources_df,
                                                n_folds,intercept)
            r_squared = R_squared_quick(np.array(training_goal.iloc[:,0]),
                                        forecast_ts_CV_list)
            objective_values.append(r_squared)
        # if i == 0:
        #     objective_values_single_datum = objective_values
        argmax = np.argmax(objective_values)
        interim_optimum = candidates.pop(argmax)
        optimum_predictors = optimum_predictors + [interim_optimum]
        optimum_R_squared.append(max(objective_values))
        
        # Test set of optimum series out of sample
        #print(problem.test_OOS(optimum))
        # Get OOS R squared
        sources_df = training_predictor.loc[:,optimum_predictors]
        OOS_coef = lin_reg(training_goal,sources_df,lin_reg_intercept=intercept)
        sources_df = testing_predictor.loc[:,optimum_predictors]
        OOS_forecast_ts = lin_pred(sources_df, OOS_coef)
        r_squared_OOS = R_squared_quick(np.array(testing_goal.iloc[:,0]),
                                        OOS_forecast_ts)
        optimum_OOS_R_squared.append(r_squared_OOS)
        
        # Stopping criteria
        if (i>0) and (optimum_R_squared[-1]>0.15):
            if optimum_R_squared[-1] < optimum_R_squared[-2] + r_squared_threshold:
                break
    
    return optimum_predictors,optimum_R_squared,optimum_OOS_R_squared
###############################################################################
optimum_predictors,optimum_R_squared,optimum_OOS_R_squared = \
    forward_selection_algo(training_goal,training_predictor,
                           testing_goal,testing_predictor,
                           n_folds,intercept,r_squared_threshold)
    
optimum = {'predictors':optimum_predictors,'R_squared':optimum_R_squared,
           'R_squared_OOS':optimum_OOS_R_squared}
df_optimum = pd.DataFrame(optimum)
print(df_optimum)

####################

####################

####################

####################

###############################################################################


