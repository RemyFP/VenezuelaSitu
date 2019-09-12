# -*- coding: utf-8 -*-
# import os
import numpy as np
# import pandas as pd
# import glob
# import geopandas as gpd
# import json
# from bokeh.io import output_notebook, output_file
# from bokeh.io import show
from bokeh.plotting import figure
# from bokeh.models import GeoJSONDataSource,LinearColorMapper,ColorBar
from bokeh.models import ColumnDataSource,FactorRange,Panel
from bokeh.palettes import Category10#, brewer, Category20, Viridis256
# from bokeh.models import FixedTicker,NumeralTickFormatter,HoverTool
# from bokeh.plotting import show as show_inline
from bokeh.models.widgets import RadioButtonGroup, Div
from bokeh.layouts import column,WidgetBox#,row,widgetbox
# from bokeh.io import curdoc
# import situ_fn
###############################################################################
def nfolds_tab(old_to_new_sources,new_to_old_sources,old_to_new_regions,
               n_folds_list,agg_all_nfolds):
    
    ## Get region names and list of sources
    region_names_new = list(old_to_new_regions.values())
    region_names_new.sort()
    sourceset_display = ['Best'] + list(new_to_old_sources.keys())[1:]
    
    # Score types considered
    score_types = ['In Sample','OOS']
    
    # Data to display
    gs_name_bar = 'Amazonas'
    agg_n_gs = agg_all_nfolds.loc[agg_all_nfolds.Region == gs_name_bar]
    
    source_set_bar = 'Best'
    if source_set_bar == 'Best':
        agg_n_gs_source = agg_n_gs.loc[agg_n_gs.IsBest,['ScoreType','NbFolds','Value']].\
                                  sort_values(by='NbFolds',ascending=True)
    else:
        agg_n_gs_source = agg_n_gs.loc[agg_n_gs.SourcesSet == source_set_bar,\
                                  ['ScoreType','NbFolds','Value']].\
                                  sort_values(by='NbFolds',ascending=True)
    
    in_sample_vals = agg_n_gs_source.loc[agg_n_gs_source.ScoreType == 'InSample',\
                                         'Value'].values.tolist()
    OOS_vals = agg_n_gs_source.loc[agg_n_gs_source.ScoreType == 'OOS',\
                                         'Value'].values.tolist()
    data_dict = {'Number Folds' : n_folds_list,
                 'In Sample' : in_sample_vals,
                 'OOS' : OOS_vals}
    x = [ (np.str(n), score) for n in n_folds_list for score in score_types ]
    
    counts = sum(zip(data_dict['In Sample'], data_dict['OOS']), ()) # like an hstack
    colors = (Category10[3])[1:]*len(n_folds_list)
    source_bar = ColumnDataSource(data=dict(x=x, counts=counts, color=colors))
    
    p_bar = figure(x_range=FactorRange(*x), plot_height=350, plot_width=800,
               title='R squared as a function of number of intervals for training',
               toolbar_location=None, tools="")
    
    p_bar.vbar(x='x', top='counts', width=0.9, source=source_bar,color='color')
    
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
    
    # Text above graph
    div_text_bar = 'Gold Standard: ' + gs_name_bar + \
                   '<br>Predictor Set: ' + source_set_bar
    div_bar = Div(text=div_text_bar,width=700, height=50)
    
    # Update function
    ###########################################################################
    def bar_callback(attr, old, new):
        # Get new selected value
        #new_score = score_types[new.active]
        gs_name_bar = region_names_new[regions_button_bar.active]
        source_set_bar = sourceset_display[source_set_button_bar.active]
        
        # Update data to display
        agg_n_gs = agg_all_nfolds.loc[agg_all_nfolds.Region == gs_name_bar]
        if source_set_bar == 'Best':
            agg_n_gs_source = agg_n_gs.loc[agg_n_gs.IsBest,\
                                  ['ScoreType','NbFolds','Value']].\
                                  sort_values(by='NbFolds',ascending=True)
        else:
            agg_n_gs_source = agg_n_gs.loc[agg_n_gs.SourcesSet == source_set_bar,\
                                  ['ScoreType','NbFolds','Value']].\
                                  sort_values(by='NbFolds',ascending=True)
        
        in_sample_vals = agg_n_gs_source.loc[agg_n_gs_source.ScoreType == 'InSample',\
                                             'Value'].values.tolist()
        OOS_vals = agg_n_gs_source.loc[agg_n_gs_source.ScoreType == 'OOS',\
                                             'Value'].values.tolist()
        data_dict = {'Number Folds' : n_folds_list,
                     'In Sample' : in_sample_vals,
                     'OOS' : OOS_vals}
        
        counts = sum(zip(data_dict['In Sample'], data_dict['OOS']), ()) # like an hstack
        # new_source = ColumnDataSource(data=dict(x=x, counts=counts))
        new_source_data = dict(x=x, counts=counts, color=colors)
        source_bar.data = new_source_data
        
        new_text_bar = 'Gold Standard: ' + gs_name_bar + \
                       '<br>Predictor Set: ' + source_set_bar
        div_bar.text = new_text_bar
        return
    ###########################################################################
    source_set_button_bar.on_change('active', bar_callback)
    regions_button_bar.on_change('active', bar_callback)
    
    # Put controls in a single element
    controls = WidgetBox(regions_button_bar,source_set_button_bar)
	
	# Create a row layout
    layout = column(controls,div_bar,p_bar)
	
	# Make a tab with the layout
    tab = Panel(child=layout, title = 'Impact of number of training windows')
    
    return tab
###############################################################################


