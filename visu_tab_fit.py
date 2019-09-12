# -*- coding: utf-8 -*-
# import os
import numpy as np
import pandas as pd
# import glob
# import geopandas as gpd
# import json
# from bokeh.io import output_notebook, output_file
# from bokeh.io import show
from bokeh.plotting import figure
# from bokeh.models import GeoJSONDataSource,LinearColorMapper,ColorBar
from bokeh.models import ColumnDataSource,Panel#,FactorRange
from bokeh.palettes import Category10#,brewer, Category20, Viridis256
# from bokeh.models import FixedTicker,NumeralTickFormatter,HoverTool
# from bokeh.plotting import show as show_inline
from bokeh.models.widgets import RadioButtonGroup, Div
from bokeh.layouts import column,WidgetBox#,row,widgetbox
# from bokeh.io import curdoc
import situ_fn
###############################################################################
def get_graph_data(df_all_n,agg,df_goal,candidates_data,gs_name,source_set,
                   train_dates,test_dates,old_to_new_regions,old_to_new_sources):
    # Get data of chosen gold standard and source set
    results = df_all_n.loc[:,(df_all_n.loc['Region'] == gs_name[:-3]) & \
                             (df_all_n.loc['SourcesSet'] == source_set) & \
                             (df_all_n.loc['Column'] == 'id')].iloc[:,0]
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
def fit_tab(agg_all_nfolds,df_all_nfolds,df_goal,candidates_data,train_dates,
            test_dates,old_to_new_sources,new_to_old_sources,old_to_new_regions,
            n_folds_list):
    ## Get region names and list of n_folds values
    region_names_new = list(old_to_new_regions.values())
    region_names_new.sort()
    n_folds_display = [np.str(x) for x in n_folds_list]
    
    ## Pick gold standard, data source and n_folds values to start with
    gs_name = 'AMAZONAS-VE'
    source_set_data = 'Best'
    n_folds_fit = n_folds_list[0]
    agg_n = agg_all_nfolds.loc[agg_all_nfolds.NbFolds == n_folds_fit,:]
    df_all_n = df_all_nfolds.loc[:,df_all_nfolds.loc['NbFolds'] == np.str(n_folds_fit)]
    if source_set_data == 'Best':
        agg_source_best = agg_n.loc[(agg_n.Region == old_to_new_regions[gs_name[:-3]]),
                                  'BestSource'].values[0]
        source_set_data = new_to_old_sources[agg_source_best]
    
    dates_train, dates_test, gs_ts,in_sample_forecast_ts,OOS_forecast_ts,\
            r_squared_in_sample, r_squared_OOS = get_graph_data(\
                df_all_n,agg_n,df_goal,candidates_data,gs_name,source_set_data,
                train_dates,test_dates,old_to_new_regions,old_to_new_sources)
    
    ### Create Bokeh graph
    all_dates = pd.to_datetime(gs_ts.index.tolist())
    dates_train_plot = pd.to_datetime(dates_train)
    dates_test_plot = pd.to_datetime(dates_test)
    #xs = [dates_train,dates_test,dates_train,dates_test]
    # min_y = [np.max(np.concatenate((np.array(gs_ts.T)[0],in_sample_forecast_ts,OOS_forecast_ts),axis=0))]
    # max_y = [np.min(np.concatenate((np.array(gs_ts.T)[0],in_sample_forecast_ts,OOS_forecast_ts),axis=0))]
    min_y,max_y = [np.min(np.array(gs_ts.T)[0])],[np.max(np.array(gs_ts.T)[0])]
    xs = [all_dates,dates_train_plot,dates_test_plot,all_dates[:1],all_dates[:1]]
    ys = [np.array(gs_ts),in_sample_forecast_ts,OOS_forecast_ts,min_y,max_y]
    source = ColumnDataSource(data=dict(
         x = xs,
         y = ys,
         color = (Category10[3])[0:len(xs)] +['#ffffff','#ffffff'],
         group = ['Gold Standard','Forecast In Sample','Forecast OOS','','']))
    TOOLS="pan,wheel_zoom,box_zoom,reset,hover,save"
    p_ts = figure(plot_width=1000, plot_height=500,x_axis_type='datetime',
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
    sourceset_display = ['Best'] + list(new_to_old_sources.keys())[1:]
    #sourceset_display = ['Best'] + sourceset_names_old[1:]
    n_folds_button = RadioButtonGroup(labels=n_folds_display, active=0)
    source_set_button_plot = RadioButtonGroup(labels=sourceset_display, active=0)
    regions_button_plt = RadioButtonGroup(labels=region_names_new, active=0)
    
    # Text above graph
    div_text = 'In and Out of Sample fit <br>Gold Standard: ' + \
        old_to_new_regions[gs_name[:-3]] + '<br>Predictor Set: ' + \
        old_to_new_sources[source_set_data] + '<br>' + \
        'In Sample R<sup>2</sup>: ' + np.str(np.round(r_squared_in_sample,2)) + \
        ' - ' + 'Out of Sample R<sup>2</sup>: ' + np.str(np.round(r_squared_OOS,2))
    div = Div(text=div_text,width=700, height=100)
    
    ###########################################################################
    # Update function
    def plot_callback(attr, old, new):
        # Get new selected value
        n_folds_fit = np.int(n_folds_display[n_folds_button.active])
        agg_n = agg_all_nfolds.loc[agg_all_nfolds.NbFolds == n_folds_fit,:]
        df_all_n = df_all_nfolds.loc[:,df_all_nfolds.loc['NbFolds'] == np.str(n_folds_fit)]
        
        gs_name_selected = region_names_new[regions_button_plt.active]
        if gs_name_selected == 'Distrito Federal':
            gs_name = 'DTTOMETRO-VE'
        else:
            gs_name = gs_name_selected.replace(' ','').upper() + '-VE'
            
        source_set = sourceset_display[source_set_button_plot.active]
        if source_set == 'Best':
            source_set_data = source_set
            agg_source_best = agg_n.loc[(agg_n.Region == old_to_new_regions[gs_name[:-3]]),
                                      'BestSource'].values[0]
            source_set_data = new_to_old_sources[agg_source_best]
        else:
            source_set_data = new_to_old_sources[source_set]
            
        # Get data to update graph
        dates_train, dates_test, gs_ts,in_sample_forecast_ts,OOS_forecast_ts,\
            r_squared_in_sample, r_squared_OOS = get_graph_data(\
                df_all_n,agg_n,df_goal,candidates_data,gs_name,source_set_data,
                train_dates,test_dates,old_to_new_regions,old_to_new_sources)
    
        ### Create Bokeh graph
        all_dates = pd.to_datetime(gs_ts.index.tolist())
        dates_train_plot = pd.to_datetime(dates_train)
        dates_test_plot = pd.to_datetime(dates_test)
        #xs = [dates_train,dates_test,dates_train,dates_test]
        min_y,max_y = [np.min(np.array(gs_ts.T)[0])],[np.max(np.array(gs_ts.T)[0])]
        xs = [all_dates,dates_train_plot,dates_test_plot,all_dates[:1],all_dates[:1]]
        ys = [np.array(gs_ts),in_sample_forecast_ts,OOS_forecast_ts,min_y,max_y]
        
        new_source = ColumnDataSource(data=dict(
              x = xs,
              y = ys,
              color = (Category10[3])[0:len(xs)] +['#ffffff','#ffffff'],
              group = ['Gold Standard','Forecast In Sample','Forecast OOS','','']))
        source.data = new_source.data
        
        new_text = 'In and Out of Sample fit <br>Gold Standard :' + \
            old_to_new_regions[gs_name[:-3]] + ' - Predictor Set: ' + \
            source_set + '<br>' + \
            'In Sample R<sup>2</sup>: ' + np.str(np.round(r_squared_in_sample,2)) + \
            '<br>Out of Sample R<sup>2</sup>: ' + np.str(np.round(r_squared_OOS,2))
        div.text = new_text
        return
    ###########################################################################
    
    n_folds_button.on_change('active', plot_callback)
    source_set_button_plot.on_change('active', plot_callback)
    regions_button_plt.on_change('active', plot_callback)
    
    # Put controls in a single element
    controls = WidgetBox(n_folds_button,regions_button_plt,source_set_button_plot)
	
	# Create a row layout
    layout = column(controls,div,p_ts)
	
	# Make a tab with the layout
    tab = Panel(child=layout, title = 'Regression Fit')
    
    return tab
    # To run from Spyder (radio buttons won't work)
    # output_file('foo.html')
    # show(column(regions_button_plt,source_set_button_plot,div,p_ts),browser="chrome")
    
    # To run from command (in the folder of the file) using the command
    # bokeh serve --show visu_ts.py
    # curdoc().add_root(column(regions_button_plt,source_set_button_plot,div,p_ts))
    # curdoc().title = "Venezuela Situational Awareness"
###############################################################################

