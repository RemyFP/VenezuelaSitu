# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import glob
import geopandas as gpd
import json
from bokeh.io import output_notebook, output_file
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource,LinearColorMapper,ColorBar
from bokeh.models import ColumnDataSource,FactorRange
from bokeh.palettes import brewer, Category20, Viridis256, Category10
from bokeh.models import FixedTicker,NumeralTickFormatter,HoverTool
from bokeh.plotting import show as show_inline
from bokeh.models.widgets import RadioButtonGroup, Div
from bokeh.layouts import column,row, widgetbox
from bokeh.io import curdoc
import situ_fn

np.set_printoptions(linewidth=130)
pd.set_option('display.width', 130)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:,.8f}'.format
pd.set_option('precision', -1)

# import webbrowser    
# urL='https://www.google.com'
# chrome_path="C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
# webbrowser.register('chrome', None,webbrowser.BackgroundBrowser(chrome_path),1)
# Map from original source sets to formatted ones

## Results to fetch
results_folder = 'OptimizationResults' + os.sep + 'SummaryAll'
n_folds = 8

## Gold standard and candidate sources orginal data, with dates used
train_dates = ['1/2/2005','12/30/2012']
test_dates = ['1/6/2013','12/28/2014']
gold_standard_folder='GoldStandard'
candidate_folder='SourcesToOptimize'

# Names mappings
sourceset_names_old = ['Column', 'Colombia', 'ColombiaBorderPlusGT', 'ColombiaPlusGT',
        'ColombiaPlusGTByState','ColombiaPlusGTBySymptom', 'DengueGT_CO',
        'GTByStateVenAndCol', 'GTVenezuela']
newnames = ['ScoreType', 'Colombia', 'Colombia Border & GT', 'Colombia & GT',
        'Colombia & GT by State','Colombia & GT by Symptom', 'Dengue GT CO',
        'GT by State - Ven & Col', 'GT Venezuela']
old_to_new_sources = dict(zip(sourceset_names_old,newnames))
new_to_old_sources = dict(zip(newnames,sourceset_names_old))
###############################################################################
def set_stype(figure,  xlabel="", ylabel=""):
    #figure.title = 
    figure.title.align ='center'
    
    figure.xaxis.axis_label=xlabel
    figure.yaxis.axis_label =ylabel
    figure.xaxis.axis_label_text_font="times"
    
    figure.yaxis.axis_label_text_font="times"
    figure.xaxis.axis_label_text_font_style ="bold"
    figure.yaxis.axis_label_text_font_style ="bold"
    
    figure.title.text_font = "times"
    figure.title.text_font_style = "bold"
    
    return
###############################################################################
def get_graph_data(df_all,agg,df_goal,candidates_data,gs_name,source_set,
                   train_dates,test_dates,old_to_new_regions):
    # Get data of chosen gold standard and source set
    results = df_all.loc[:,(df_all.loc['Region'] == gs_name[:-3]) & \
                           (df_all.loc['SourcesSet'] == source_set) & \
                           (df_all.loc['Column'] == 'id')].iloc[:,0]
    predictors_list = results.loc[results.index.isin(\
        [np.str(i) for i in range(pd.notnull(results).sum()-4)])].values.tolist()
    predictors_list = [x.replace('_','-') for x in predictors_list]
    predictors_ts = candidates_data[source_set].loc[:,predictors_list]
    gs_ts = df_goal.loc[:,[gs_name]]
    
    
    ## Split between training and testing data, do regression
    # Split data
    X_train = predictors_ts.loc[train_dates[0]:train_dates[1]]
    X_test = predictors_ts.loc[test_dates[0]:test_dates[1]]
    y_train = gs_ts.loc[train_dates[0]:train_dates[1]]
    y_test = gs_ts.loc[test_dates[0]:test_dates[1]]
    dates_train = y_train.index.tolist()
    dates_test = y_test.index.tolist()
    
    # Do regression, forecast in and out of sample
    OOS_coef = situ_fn.lin_reg(y_train,X_train,lin_reg_intercept=True)
    in_sample_forecast_ts = situ_fn.lin_pred(X_train, OOS_coef)
    OOS_forecast_ts = situ_fn.lin_pred(X_test, OOS_coef)
    # in_sample_forecast_df = pd.DataFrame({'Date':dates_train,
    #                                       'InSample':in_sample_forecast_ts})
    # in_sample_forecast_df.set_index('Date',inplace=True)
    # OOS_forecast_df = pd.DataFrame({'Date':dates_test,'OOS':OOS_forecast_ts})
    # OOS_forecast_df.set_index('Date',inplace=True)
    
    # Compure R squared
    # r_squared_in_sample = situ_fn.R_squared_quick(np.array(y_train.iloc[:,0]),
    #                                               in_sample_forecast_ts)
    # r_squared_OOS = situ_fn.R_squared_quick(np.array(y_test.iloc[:,0]),OOS_forecast_ts)
    agg_gs_source = agg.loc[(agg.Region == old_to_new_regions[gs_name[:-3]]) & \
            (agg.SourcesSet == old_to_new_sources[source_set])]
    r_squared_in_sample = agg_gs_source.loc[\
        agg_gs_source.ScoreType == 'InSample','Value'].iloc[0]
    r_squared_OOS = agg_gs_source.loc[agg_gs_source.ScoreType == 'OOS','Value'].iloc[0]
    
    return dates_train, dates_test, gs_ts,in_sample_forecast_ts,OOS_forecast_ts,\
        r_squared_in_sample, r_squared_OOS
###############################################################################
## Optimization data
filename = 'SummaryAggregate_nfolds-' + np.str(n_folds) + '.csv'
agg_path = os.sep.join([os.getcwd(),results_folder,filename])
agg = pd.read_csv(agg_path)
df_all_filename = 'AggregateAll_nfolds-' + np.str(n_folds) + '.csv'
df_all_path = os.sep.join([os.getcwd(),results_folder,df_all_filename])
df_all = pd.read_csv(df_all_path)
df_all.rename(columns={'Unnamed: 0':'RowNames'},inplace=True)
df_all.set_index('RowNames',inplace=True)


# Map for gold standard names
region_rename = {'Deltaamacuro':'Delta Amacuro','Dttometro':'Distrito Federal',
                 'Nuevaesparta':'Nueva Esparta'}
region_names_old =  np.unique(df_all.loc['Region',:].values)
region_names_new = [region_rename[x.title()]  if \
    (x.title() in region_rename.keys()) else x.title() for x in region_names_old]
old_to_new_regions = dict(zip(region_names_old,region_names_new))

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


## Pick gold standard and data source
gs_name = 'AMAZONAS-VE'
source_set = 'Best'
if source_set == 'Best':
    agg_source_best = agg.loc[(agg.Region == old_to_new_regions[gs_name[:-3]]),
                              'BestSource'].values[0]
    source_set = new_to_old_sources[agg_source_best]

dates_train, dates_test, gs_ts,in_sample_forecast_ts,OOS_forecast_ts,\
        r_squared_in_sample, r_squared_OOS = get_graph_data(\
            df_all,agg,df_goal,candidates_data,gs_name,source_set,
            train_dates,test_dates,old_to_new_regions)

### Create Bokeh graph
all_dates = pd.to_datetime(gs_ts.index.tolist())
dates_train_plot = pd.to_datetime(dates_train)
dates_test_plot = pd.to_datetime(dates_test)
#xs = [dates_train,dates_test,dates_train,dates_test]
xs = [all_dates,dates_train_plot,dates_test_plot]
ys = [np.array(gs_ts),in_sample_forecast_ts,OOS_forecast_ts]
source = ColumnDataSource(data=dict(
     x = xs,
     y = ys,
     color = (Category10[3])[0:len(xs)],
     group = ['Gold Standard','Forecast In Sample','Forecast OOS']))
TOOLS="pan,wheel_zoom,box_zoom,reset,hover,save"
p_ts = figure(plot_width=800, plot_height=500,x_axis_type='datetime',
            title='', tools=TOOLS,)
p_ts.multi_line(
     xs='x',
     ys='y',
     legend='group',
     source=source,
     line_color='color')

p_ts.legend.location = (0,350)

## Radio buttons
# Define buttons
sourceset_display = ['Best'] + sourceset_names_old[1:]
source_set_button_plot = RadioButtonGroup(labels=sourceset_display, active=0)
regions_button_plt = RadioButtonGroup(labels=region_names_new, active=0)

# Text above graph
div_text = 'In and Out of Sample fit <br>Gold Standard: ' + \
    old_to_new_regions[gs_name[:-3]] + ' - Predictor Set: ' + \
    old_to_new_sources[source_set] + '<br>' + \
    'In Sample R<sup>2</sup>: ' + np.str(np.round(r_squared_in_sample,2)) + \
    ' - ' + 'Out of Sample R<sup>2</sup>: ' + np.str(np.round(r_squared_OOS,2))
div = Div(text=div_text,width=700, height=100)


# Update function
def plot_callback(attr, old, new):
    # Get new selected value
    #new_score = score_types[new.active]
    gs_name_selected = region_names_new[regions_button_plt.active]
    if gs_name_selected == 'Distrito Federal':
        gs_name = 'DTTOMETRO-VE'
    else:
        gs_name = gs_name_selected.replace(' ','').upper() + '-VE'
        
    source_set = sourceset_display[source_set_button_plot.active]
    if source_set == 'Best':
        agg_source_best = agg.loc[(agg.Region == old_to_new_regions[gs_name[:-3]]),
                                  'BestSource'].values[0]
        source_set = new_to_old_sources[agg_source_best]
    
    # Get data to update graph
    dates_train, dates_test, gs_ts,in_sample_forecast_ts,OOS_forecast_ts,\
        r_squared_in_sample, r_squared_OOS = get_graph_data(\
            df_all,agg,df_goal,candidates_data,gs_name,source_set,
            train_dates,test_dates,old_to_new_regions)

    ### Create Bokeh graph
    all_dates = pd.to_datetime(gs_ts.index.tolist())
    dates_train_plot = pd.to_datetime(dates_train)
    dates_test_plot = pd.to_datetime(dates_test)
    #xs = [dates_train,dates_test,dates_train,dates_test]
    xs = [all_dates,dates_train_plot,dates_test_plot]
    ys = [np.array(gs_ts),in_sample_forecast_ts,OOS_forecast_ts]
    #new_data = pd.DataFrame({'x':xs,'y':ys,'color':(Category10[3])[0:len(xs)]})
    
    new_source = ColumnDataSource(data=dict(
          x = xs,
          y = ys,
          color = (Category10[3])[0:len(xs)],
          group = ['Gold Standard','Forecast In Sample','Forecast OOS']))
    source.data = new_source.data
    # p_ts.source = new_source
    
    new_text = 'In and Out of Sample fit <br>Gold Standard :' + \
        old_to_new_regions[gs_name[:-3]] + ' - Predictor Set: ' + \
        old_to_new_sources[source_set] + '<br>' + \
        'In Sample R<sup>2</sup>: ' + np.str(np.round(r_squared_in_sample,2)) + \
        '<br>Out of Sample R<sup>2</sup>: ' + np.str(np.round(r_squared_OOS,2))
    div.text = new_text

source_set_button_plot.on_change('active', plot_callback)
regions_button_plt.on_change('active', plot_callback)


# To run from Spyder (radio buttons won't work)
# output_file('foo.html')
# show(column(regions_button_plt,source_set_button_plot,div,p_ts),browser="chrome")

# To run from command (in the folder of the file) using the command
# bokeh serve --show visu_ts.py
# curdoc().add_root(column(regions_button_plt,source_set_button_plot,div,p_ts))
# curdoc().title = "Venezuela Situational Awareness"



### Histogram showing the impact of n_folds
# Get data
## Results to fetch
# n_folds values for which data exists
folder_path = os.sep.join([os.getcwd(),results_folder])
existing_file_paths = glob.glob(os.path.join(folder_path, '*'))
n_folds_list_all = []
for f in existing_file_paths:
    n_str = f.split(os.sep)[-1].split('_')[-1].split('.')[0].replace('nfolds-','')
    n_folds_list_all.append(np.int(n_str))
    
n_folds_list = np.unique(n_folds_list_all).tolist()
n_folds_list.sort()
score_types = ['In Sample','OOS']
agg_all_nfolds = None
for n in n_folds_list:
    agg_n_filename = 'SummaryAggregate_nfolds-' + np.str(n) + '.csv'
    agg_n_path = os.sep.join([os.getcwd(),results_folder,agg_n_filename])
    agg_n = pd.read_csv(agg_n_path)
    
    if agg_all_nfolds is None:
        agg_all_nfolds = agg_n.copy()
    else:
        agg_all_nfolds = agg_all_nfolds.append(agg_n)

# All regions and sources
all_regions = np.unique(agg_all_nfolds.Region).tolist()
all_sources = np.unique(agg_all_nfolds.SourcesSet).tolist()

# Data to display
gs_name_bar = 'Amazonas'
agg_n_gs = agg_all_nfolds.loc[agg_all_nfolds.Region == gs_name_bar]

source_set_bar = 'Best'
if source_set_bar == 'Best':
    agg_n_gs_source = agg_n_gs.loc[agg_n_gs.IsBest,['ScoreType','NbFolds','Value']]
else:
    agg_n_gs_source = agg_n_gs.loc[agg_n_gs.SourcesSet == source_set_bar,['ScoreType','NbFolds','Value']]


agg_n_gs_source.sort_values(by='NbFolds',ascending=True,inplace=True)
in_sample_vals = agg_n_gs_source.loc[agg_n_gs_source.ScoreType == 'InSample',\
                                     'Value'].values.tolist()
OOS_vals = agg_n_gs_source.loc[agg_n_gs_source.ScoreType == 'OOS',\
                                     'Value'].values.tolist()
data_dict = {'Number Folds' : n_folds_list,
             'In Sample' : in_sample_vals,
             'OOS' : OOS_vals}
x = [ (np.str(n), score) for n in n_folds_list for score in score_types ]

counts = sum(zip(data_dict['In Sample'], data_dict['OOS']), ()) # like an hstack
source_bar = ColumnDataSource(data=dict(x=x, counts=counts))

p_bar = figure(x_range=FactorRange(*x), plot_height=250, plot_width=700,
           title='R squared as a function of number of intervals for training',
           toolbar_location=None, tools="")

p_bar.vbar(x='x', top='counts', width=0.9, source=source_bar)

p_bar.y_range.start = -1
p_bar.y_range.end = 1
p_bar.x_range.range_padding = 0.1
p_bar.xaxis.major_label_orientation = 1
p_bar.xgrid.grid_line_color = None

#show(p_bar,browser="chrome")

## Radio buttons for bar graph
# Define buttons
regions_button_bar = RadioButtonGroup(labels=region_names_new, active=0)
source_set_button_bar = RadioButtonGroup(labels=sourceset_display, active=0)

# Separator between the 2 graphs
div_separator = Div(text='',width=700, height=50)
# Text above graph
div_text_bar = 'Gold Standard: ' + gs_name_bar + \
               '<br>Predictor Set: ' + source_set_bar
div_bar = Div(text=div_text_bar,width=700, height=50)

# Update function
def bar_callback(attr, old, new):
    # Get new selected value
    #new_score = score_types[new.active]
    gs_name_bar = region_names_new[regions_button_bar.active]
    source_set_bar = sourceset_display[source_set_button_bar.active]
    
    # Update data to display
    agg_n_gs = agg_all_nfolds.loc[agg_all_nfolds.Region == gs_name_bar]
    if source_set_bar == 'Best':
        agg_n_gs_source = agg_n_gs.loc[agg_n_gs.IsBest,['ScoreType','NbFolds','Value']]
    else:
        source_set_bar = old_to_new_sources[source_set_bar]
        agg_n_gs_source = agg_n_gs.loc[agg_n_gs.SourcesSet == source_set_bar,['ScoreType','NbFolds','Value']]
    
    agg_n_gs_source.sort_values(by='NbFolds',ascending=True,inplace=True)
    in_sample_vals = agg_n_gs_source.loc[agg_n_gs_source.ScoreType == 'InSample',\
                                         'Value'].values.tolist()
    OOS_vals = agg_n_gs_source.loc[agg_n_gs_source.ScoreType == 'OOS',\
                                         'Value'].values.tolist()
    data_dict = {'Number Folds' : n_folds_list,
                 'In Sample' : in_sample_vals,
                 'OOS' : OOS_vals}
    x = [ (np.str(n), score) for n in n_folds_list for score in score_types ]
    
    counts = sum(zip(data_dict['In Sample'], data_dict['OOS']), ()) # like an hstack
    new_source = ColumnDataSource(data=dict(x=x, counts=counts))
    source_bar.data = new_source.data
    
    new_text_bar = 'Gold Standard: ' + gs_name_bar + \
                   '<br>Predictor Set: ' + source_set_bar
    div_bar.text = new_text_bar

source_set_button_bar.on_change('active', bar_callback)
regions_button_bar.on_change('active', bar_callback)


# Show graphs in browser
curdoc().add_root(column(regions_button_plt,source_set_button_plot,div,p_ts,\
                         div_separator,regions_button_bar,source_set_button_bar,\
                         div_bar,p_bar))
curdoc().title = "Venezuela Situational Awareness"


