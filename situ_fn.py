# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:37:37 2019

@author: Remy
"""
import os
import numpy as np
import pandas as pd
import glob
from sklearn import linear_model
import scipy
###############################################################################
def write_summary_results(folder='OptimizationResults//8//Summary',
                          out_folder='OptimizationResults//SummaryAll',
                          n_folds=8):
    """ Gets all results from function aggregate_results in a single file and
    aggregates the final results in a single file.
    """
    summary_results_folder = os.sep.join([os.getcwd(),folder])
    file_paths = glob.glob(os.path.join(summary_results_folder, '*'))
    df_all = None
    
    # Loop through files and aggregate them in a single folder
    for f in file_paths:
        # Get file and region name
        df = pd.read_csv(f)
        filename = f.split(os.sep)[-1]
        region = filename.split('_')[0].split('-')[0]
        df.set_index(df.columns[0],inplace=True)
        df.loc['Region',:] = region
        
        # Rename columns to add the region's name
        new_cols = [region + '|' + x for x in df.columns]
        df.columns = new_cols
        
        # Add this region's data to 
        if df_all is not None:
            df_all = df_all.join(df,how='outer')
        else:
            df_all = df.copy()
    
    # Add number of folds to all results
    df_all.loc['NbFolds',:] = n_folds
    # Reorder rows
    df_all.loc[:,'Sort'] = df_all.index
    df_all.loc[:,'Sort'] = df_all.apply(lambda x: np.int(x['Sort']) + 3 if
           x['Sort'].isdigit() else 0,axis=1)
    df_all.sort_values(by='Sort',inplace=True)
    df_all.drop(columns=['Sort'],inplace=True)
    
    
    # Get last value of each row
    df = df_all.copy()
    df.loc['Last',:] = df.ffill(axis=0).iloc[-1,:]
    # Only keep last value and source, column, region of each column
    rows = ['SourcesSet','Column','Region','Last']
    agg = pd.pivot_table(df.loc[rows,:].T, values='Last', index=['Region','Column'],
                         columns=['SourcesSet'], aggfunc='last')
    agg.reset_index(inplace=True)
    agg = agg.loc[~(agg.Column == 'id'),:]
    
    ## Rename columns and some elements
    # Data sources
    colnames = ['Column', 'Colombia', 'ColombiaBorderPlusGT', 'ColombiaPlusGT',
                'ColombiaPlusGTByState','ColombiaPlusGTBySymptom', 'DengueGT_CO',
                'GTByStateVenAndCol', 'GTVenezuela']
    newnames = ['ScoreType', 'Colombia', 'Colombia Border & GT', 'Colombia & GT',
                'Colombia & GT by State','Colombia & GT by Symptom', 'Dengue GT CO',
                'GT by State - Ven & Col', 'GT Venezuela']
    
    # Type of scores
    agg.rename(columns=dict(zip(colnames,newnames)),inplace=True)
    agg.replace(to_replace={'objective_values':'InSample','OOS_R_squared':'OOS'},
                inplace=True)
    # Regions, to match GPS data
    agg.loc[:,'Region'] = list(map(str.title,agg.loc[:,'Region']))
    region_rename = {'Deltaamacuro':'Delta Amacuro','Dttometro':'Distrito Federal',
                     'Nuevaesparta':'Nueva Esparta'}
    agg.replace(to_replace={'Region':region_rename},inplace=True)
    
    # Get max value for each region
    agg.loc[:,newnames[1:]] = agg.loc[:,newnames[1:]].apply(pd.to_numeric)
    agg.loc[:,'BestSource'] = agg.loc[:,newnames[1:]].apply(lambda x:
        newnames[1+np.array(x).argmax()],axis=1)
    agg.loc[:,'Best'] = agg.loc[:,newnames[1:]].max(axis=1)
    
    # For OOS best corresponds to best in sample R squared
    best_source = agg.loc[agg.ScoreType == 'InSample',['Region','BestSource']].\
        rename(columns={'BestSource':'BestSource_InSample'})
    agg = pd.merge(agg,best_source,on='Region',how='left')
    agg.drop(columns='BestSource',inplace=True)
    agg.rename(columns={'BestSource_InSample':'BestSource'},inplace=True)
    
    # Convert to long version
    agg_long = pd.melt(agg, id_vars=['Region','ScoreType','Best','BestSource'],
                       value_name='Value')
    agg_long.loc[:,'IsBest'] = (agg_long['BestSource'] == agg_long['SourcesSet'])
    agg_long.drop(columns=['Best'],inplace=True)
    
    # Add number of folds to summary results
    agg_long.loc[:,'NbFolds'] = n_folds
    
    # Save results in csv files
    n_folds_str = 'nfolds' + '-' + np.str(n_folds)
    path_agg = out_folder + os.sep + 'SummaryAggregate' + '_' + n_folds_str + '.csv'
    agg_long.to_csv(path_agg)
    path_all = out_folder + os.sep + 'AggregateAll' + '_' + n_folds_str + '.csv'
    df_all.to_csv(path_all)
    
    return agg_long, df_all
###############################################################################
def aggregate_results(data_folder='OptimizationResults'):
    """ Aggregates results of the main function in situ_main in a new folder 
    called Summary.
    """
    # List of all files in subfolders of save folder
    file_paths = glob.glob(os.path.join(data_folder, '*'))
    
    # Save results in dictionary, each entry corresponding to a gold standard
    dict_all = {}
    
    for p in file_paths:
        if p[-3:] != 'csv':
            continue
        #p = file_paths[0]
        # Get details of file (number of intervals, sources)
        
        _,filename = p.split(os.sep)
        filename_split = filename.split('_')
        gold_standard,sources,_ = [filename_split[i] for i in [0,1,-1]]
        if sources == 'DengueGT':
            sources = 'DengueGT_CO'
        
        # Load file and parse only if it contains scores (optimum in name)
        if 'optimum.csv' not in filename_split:
            continue
        df = pd.read_csv(p).iloc[:,[1,2,3]]
        
        # Save relevant information
        df.loc['Column',:] = list(df.columns)
        df.loc['SourcesSet',:] = sources
        
        # Rename columns so they will be unique
        for c in list(df.columns):
            new_name = '_'.join(list(df[c].loc[
                    ['SourcesSet','Column']]))
            df.rename(columns={c:new_name},inplace=True)
        
        # Save result in dictionary
        if gold_standard in dict_all.keys():
            df_g = dict_all[gold_standard]
            df_g = df_g.join(df,how='outer')
            dict_all[gold_standard] = df_g
        else:
            dict_all[gold_standard] = df
    
    # Write results in csv files, each corresponding to a gold standard
    output_folder = data_folder + os.sep + 'Summary'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    info_rows = ['Column','SourcesSet']
    for k in dict_all.keys():
        df_out = dict_all[k]
        nb_rows = [x for x in df_out.index if x not in info_rows]
        df_out = df_out.reindex(info_rows + nb_rows,copy=False)
        path = os.path.join(output_folder,'{}_Summary.csv'.format(k))
        df_out.to_csv(path)
    
    return
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
def R_squared_quick(actual_ts,forecast_ts):
    numerator = scipy.stats.tvar(actual_ts - forecast_ts)
    denominator = float(scipy.stats.tvar(actual_ts))
    rsquared = 1 - numerator/denominator
    return rsquared
###############################################################################
    

