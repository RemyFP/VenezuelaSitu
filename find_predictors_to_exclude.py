# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 17:12:19 2019

@author: Remy
"""
import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
###############################################################################
def get_summary_results(folder='OptimizationResults//Summary'):
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
    
    return agg_long, df_all
###############################################################################
### Find predictors that cause OOS R squared to jump to very negative values
## Get results for all
results_folder = 'OptimizationResults' + os.sep + '8' + os.sep + 'Summary'
agg, df_all = get_summary_results(folder=results_folder)
OOS_R_squared_threshold = -5

# Loop through all results
issue_predictors = []
issue_predictors_w_region = []
issues_per_set = {}
for i in range(len(df_all.columns)):
    col = df_all.iloc[:,i]
    if col['Column'] == 'OOS_R_squared':
        for j in range(pd.notnull(col).sum()-3):
            if np.float(col[np.str(j)]) < OOS_R_squared_threshold:
                source_set = col['SourcesSet']
                predictor = df_all.iloc[:,i-2][np.str(j)]
                region = col['Region']
                issue_predictors.append(source_set + '|' + predictor)
                issue_predictors_w_region.append(source_set + '|' + predictor +\
                    '|' + region + '|' + np.str(np.round(np.float(col[np.str(j)]),1)))
                
                if source_set in issues_per_set.keys():
                    issues_per_set[source_set].append(predictor)
                else:
                    issues_per_set.update({source_set:[predictor]})
                
                # Go to next column, we found the predictor causing a jump
                break

# Get list of unique issues     
issues_unique = np.unique(issue_predictors)
len(issues_unique)
for k in issues_per_set.keys():
    issues_per_set[k] = list(np.unique(issues_per_set[k]))

## Get predictors data and plot them
candidate_folder='SourcesToOptimize'

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

# Loop through issues and plot them
for p in issues_unique:
    source_set,predictor = p.split('|')
    source_df = candidates_data[source_set]
    predictor_df = source_df.loc[:,predictor.replace('_','-')]
    plt.plot(range(len(predictor_df)),predictor_df)
    plt.title(predictor)
    plt.show()



### Identify sources to exclude based on some criteria
# Sources data
candidates_path = os.sep.join([os.getcwd(),candidate_folder])
candidates_files = glob.glob(os.path.join(candidates_path, '*'))  

exclude = {}
total_nb_sources = 0
nb_per_set = {}
for c_file in candidates_files:
    exclude_c = []
    df_c = pd.read_csv(c_file)
    df_c.rename(columns={'year/week':'Date'},inplace=True)
    df_c.set_index('Date',inplace=True)
    source_name = c_file.split(os.sep)[-1].split('.')[0]
    total_nb_sources += len(df_c.columns)
    
    for col in df_c.columns:
        # Round to 2 decimals
        ts = df_c[col]
        ts = ts.apply(lambda x: np.round(x,1))
        
        # Count number of consecutive days with the same value
        changes_count = (ts.diff(1) != 0).astype('int').cumsum()
        unique, counts = np.unique(changes_count, return_counts=True)
        if max(counts) > 90:
            exclude_c.append(col)
            
        # Exclude if most values in time series are identical (1/3 of total)
        # ts = np.array(df_c[col])
        # unique, counts = np.unique(ts, return_counts=True)
        # if max(counts) > sum(counts)/3:
        #     exclude_c.append(col)
    
    nb_per_set.update({source_name:[len(exclude_c),len(df_c.columns)]})
    exclude.update({source_name:exclude_c})
    
    for s in exclude_c:
        plt.plot(np.array(df_c[s]))
        plt.title(s)
        plt.show()


issues_unique

# Don't exclude from Colombia (index 0)
# Exclude from ColombiaBorderPlusGT (index 1)
# Exclude from ColombiaPlusGT (index 2) except those in Colombia (index 0)
# Exclude from ColombiaPlusGTByState (index 3) except those in Colombia (index 0)
# Exclude from ColombiaPlusGTBySymptom (index 4) except those in Colombia (index 0)
# Exclude from DengueGT_CO (index 5)
# Exclude from GTByStateVenAndCol (index 6)
# Exclude from GTVenezuela (index 7)

# Graph predictors identified in selected source set
index = 7
c = list(exclude.keys())[index] # len(exclude.keys()) # = 8
print(c)
c_file = candidates_files[index]
df_c = pd.read_csv(c_file)
df_c.rename(columns={'year/week':'Date'},inplace=True)
df_c.set_index('Date',inplace=True)
exclude_c = exclude[c]
for s in exclude_c:
    plt.plot(np.array(df_c[s]))
    plt.title(s)
    plt.show()

## Check all the predictors that cause issues in the regression are found
# in this method to identify problem time series
not_found = []
for p in issues_unique:
    source_set,predictor = p.split('|')
    predictor = predictor.replace('_','-')
    exclude_set = exclude[source_set]
    if predictor not in exclude_set:
        not_found.append(p)
        index = list(exclude.keys()).index(source_set)
        c_file = candidates_files[index]
        df_c = pd.read_csv(c_file)
        df_c.rename(columns={'year/week':'Date'},inplace=True)
        df_c.set_index('Date',inplace=True)
        plt.plot(np.array(df_c[predictor]))
        plt.title(predictor)
        plt.show()
    
### Export list of predictors to exclude per source set
# Do not exclude time series from source set Colombia
colombia_predictors = pd.read_csv(candidates_files[0]).columns.tolist()
for c in exclude.keys():
    exclude_c = exclude[c]
    to_remove = []
    for predictor in exclude_c:
        if predictor in colombia_predictors:
            to_remove.append(predictor)
    for r in to_remove:
        exclude_c.remove(r)
    exclude[c] = exclude_c
    

# Add the ones found when running regressions
#'ColombiaPlusGT|Dengue_CundinamarcaGT_CO',
# 'DengueGT_CO|Dengue_CundinamarcaGT_CO',
# 'GTByStateVenAndCol|Boyaca_CO_GT'
for n in not_found:
    source_set,predictor = n.split('|')
    predictor = predictor.replace('_','-')
    exclude_c = exclude[source_set]
    print(exclude_c)
    exclude_c.append(predictor)
    exclude[source_set] = exclude_c


# Dataframe to export
df_exclude = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in exclude.items() ]))

export_filename = 'Predictors_Excluded.csv'
df_exclude.to_csv(os.getcwd() + os.sep + export_filename)
