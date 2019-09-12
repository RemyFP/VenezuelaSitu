# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import glob
import geopandas as gpd
# import json
# from bokeh.io import output_notebook, output_file
# from bokeh.io import show
# from bokeh.plotting import figure
# from bokeh.models import GeoJSONDataSource,LinearColorMapper,ColorBar
# from bokeh.models import ColumnDataSource,FactorRange,Panel
# from bokeh.palettes import brewer, Category20, Viridis256, Category10
# from bokeh.models import FixedTicker,NumeralTickFormatter,HoverTool
# from bokeh.plotting import show as show_inline
from bokeh.models.widgets import Tabs#,RadioButtonGroup, Div
# from bokeh.layouts import column,row, widgetbox,WidgetBox
from bokeh.io import curdoc
# import situ_fn
import visu_tab_fit
import visu_tab_nfolds
import visu_tab_map
import visu_tab_table
import visu_tab_tabledetails
import visu_tab_source_comparison
import visu_tab_region_comparison

### Inputs
## Folders
results_folder = 'OptimizationResults' + os.sep + 'SummaryAll'
gold_standard_folder='GoldStandard'
candidate_folder='SourcesToOptimize'
map_shapefile = os.sep.join([os.getcwd(),'MapData','Regions']) + os.sep +\
    'ven_admbnda_adm1_20180502.shp'
map_shapefile_country = os.sep.join([os.getcwd(),'MapData','Country']) + os.sep +\
    'ven_admbnda_adm0_20180502.shp'

## Parameters used
# n_folds = 8 # value of nfolds to use in data displayed in fit tab
train_dates = ['1/2/2005','12/30/2012']
test_dates = ['1/6/2013','12/28/2014']

### Names mappings
sourceset_names_old = ['Column', 'Colombia', 'ColombiaBorderPlusGT', 'ColombiaPlusGT',
        'ColombiaPlusGTByState','ColombiaPlusGTBySymptom', 'DengueGT_CO',
        'GTByStateVenAndCol', 'GTVenezuela','ClimateData']
newnames = ['ScoreType', 'Colombia', 'Colombia Border & GT', 'Colombia & GT',
        'Colombia & GT by State','Colombia & GT by Symptom', 'Dengue GT CO',
        'GT by State - Ven & Col', 'GT Venezuela','Climate']
old_to_new_sources = dict(zip(sourceset_names_old,newnames))
new_to_old_sources = dict(zip(newnames,sourceset_names_old))


### Load all aggregate data
# Get n_folds values for which data exists
folder_path = os.sep.join([os.getcwd(),results_folder])
existing_file_paths = glob.glob(os.path.join(folder_path, '*'))
n_folds_list_all = []
for f in existing_file_paths:
    n_str = f.split(os.sep)[-1].split('_')[-1].split('.')[0].replace('nfolds-','')
    n_folds_list_all.append(np.int(n_str))
n_folds_list = np.unique(n_folds_list_all).tolist()
n_folds_list.sort()
n_folds = n_folds_list[0]

## Aggregate results data
agg_all_nfolds = None
for n in n_folds_list:
    agg_n_filename = 'SummaryAggregate_nfolds-' + np.str(n) + '.csv'
    agg_n_path = os.sep.join([os.getcwd(),results_folder,agg_n_filename])
    agg_n = pd.read_csv(agg_n_path)
    
    if agg_all_nfolds is None:
        agg_all_nfolds = agg_n.copy()
    else:
        agg_all_nfolds = agg_all_nfolds.append(agg_n)

## All detailed results
df_all_nfolds = None
for n in n_folds_list:
    # Fetch data
    df_all_n_filename = 'AggregateAll_nfolds-' + np.str(n) + '.csv'
    df_all_n_path = os.sep.join([os.getcwd(),results_folder,df_all_n_filename])
    df_all_n = pd.read_csv(df_all_n_path)
    
    # Add n folds value in each column name to diffentiate between files
    cols_old = df_all_n.columns.tolist()[1:]
    cols_new = [x + '|' + np.str(n) for x in cols_old]
    cols_rename = dict(zip(cols_old,cols_new))
    df_all_n.rename(columns=cols_rename,inplace=True)
    
    # Set first column as index
    if 'Unnamed: 0' in df_all_n.columns:
        df_all_n.rename(columns={'Unnamed: 0':'RowNames'},inplace=True)
        df_all_n.set_index(df_all_n['RowNames'],inplace=True)
    
    # Aggregate different files
    if df_all_nfolds is None:
        df_all_nfolds = df_all_n.iloc[:,1:].copy()
    else:
        df_all_nfolds = pd.merge(df_all_nfolds,df_all_n.iloc[:,1:],
                                 left_index=True,right_index=True,how='outer')    

### Original data: gold standards and predictors
# Original Gold Standard data
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

# Original Sources data
candidates_path = os.sep.join([os.getcwd(),candidate_folder])
candidates_files = glob.glob(os.path.join(candidates_path, '*'))  
candidates_data = {}
for c in candidates_files:
    df_c = pd.read_csv(c)
    df_c.rename(columns={'year/week':'Date'},inplace=True)
    df_c.set_index('Date',inplace=True)
    source_name = c.split(os.sep)[-1].split('.')[0]
    candidates_data.update({source_name:df_c})


# Map for gold standard names
region_rename = {'Deltaamacuro':'Delta Amacuro','Dttometro':'Distrito Federal',
                 'Nuevaesparta':'Nueva Esparta'}
region_names_old =  np.unique(df_all_nfolds.loc['Region',:].values)
region_names_new = [region_rename[x.title()]  if \
    (x.title() in region_rename.keys()) else x.title() for x in region_names_old]
old_to_new_regions = dict(zip(region_names_old,region_names_new))
new_to_old_regions = dict(zip(region_names_new,region_names_old))

    
### Data for map
## GPS data
gdf = gpd.read_file(map_shapefile)[['ADM1_ES', 'geometry']]
gdf.columns = ['Region', 'geometry']
gdf_country = gpd.read_file(map_shapefile_country)['geometry']
## Optimization data
# agg = agg_all_nfolds.loc[agg_all_nfolds.NbFolds == n_folds,:]
score_types_list = np.unique(agg_all_nfolds.ScoreType.tolist())
sources_list = list(np.unique(agg_all_nfolds.SourcesSet.tolist() + ['Best']))
# Put Climate at the end
if 'Climate' in sources_list:
    c_idx = sources_list.index('Climate')
    sources_list.pop(c_idx)
    sources_list.append('Climate')

### Call tab functions
# Create each of the tabs
tab_fit = visu_tab_fit.fit_tab(agg_all_nfolds,df_all_nfolds,df_goal,
    candidates_data,train_dates,test_dates,old_to_new_sources,new_to_old_sources,
    old_to_new_regions,n_folds_list)
tab_nfolds = visu_tab_nfolds.nfolds_tab(old_to_new_sources,new_to_old_sources,
    old_to_new_regions,n_folds_list,agg_all_nfolds)
tab_map = visu_tab_map.map_tab(gdf,gdf_country,agg_all_nfolds,score_types_list,
                               sources_list,n_folds_list)
tab_table = visu_tab_table.table_tab(agg_all_nfolds)
tab_tabledetails = visu_tab_tabledetails.tabledetails_tab(df_all_nfolds,
    old_to_new_regions,new_to_old_regions,old_to_new_sources,new_to_old_sources)
tab_source_comparison = visu_tab_source_comparison.source_comparison_tab(\
                          old_to_new_sources,new_to_old_sources,
                          old_to_new_regions,n_folds_list,agg_all_nfolds)
tab_region_comparison = visu_tab_region_comparison.region_comparison_tab(old_to_new_sources,new_to_old_sources,
                          old_to_new_regions,n_folds_list,agg_all_nfolds)

# Put all the tabs into one application
tabs = Tabs(tabs = [tab_map,tab_fit,tab_source_comparison,tab_region_comparison,
                    tab_nfolds,tab_table,tab_tabledetails])

# Put the tabs in the current document for display
curdoc().add_root(tabs)
curdoc().title = 'Venezuela Situational Awareness'

# To run from Spyder (radio buttons won't work)
# output_file('foo.html')
# show(column(regions_button_plt,source_set_button_plot,div,p_ts),browser="chrome")

# To run from command (in the folder of the file) using the command
# bokeh serve --show main_visu.py
# curdoc().add_root(column(regions_button_plt,source_set_button_plot,div,p_ts))
# curdoc().title = "Venezuela Situational Awareness"