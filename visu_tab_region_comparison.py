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
def prep_data(df,regions,source_set_bar,n_folds_bar,sourceset_display,
              score_types,colors):
    # Filter data for chosen source, n_folds value and regions
    if source_set_bar == 'Best':
        agg_source = df.loc[df.IsBest,['Region','ScoreType','NbFolds','Value']]
    else:
        agg_source = df.loc[df.SourcesSet == source_set_bar,\
                                       ['Region','ScoreType','NbFolds','Value']]
    
    agg_n_source = agg_source.loc[agg_source.NbFolds == n_folds_bar,:].\
        sort_values(by='Region',ascending=True)
    
    agg_n_source = agg_n_source.loc[agg_n_source.Region.isin(regions),:]
    
    # Data for bar chart
    in_sample_vals = agg_n_source.loc[agg_n_source.ScoreType == 'InSample',\
                                  'Value'].values.tolist()
    OOS_vals = agg_n_source.loc[agg_n_source.ScoreType == 'OOS',\
                            'Value'].values.tolist()
    data_dict = {'Source Set' : sourceset_display,
                 'In Sample' : in_sample_vals,
                 'OOS' : OOS_vals}
    
    x = [ (r, score) for r in regions for score in score_types ]
    counts = sum(zip(data_dict['In Sample'], data_dict['OOS']), ()) # like an hstack
    source_data = dict(x=x, counts=counts, color=colors)
    
    return source_data, x
###############################################################################
def format_graph(p_bar):
    p_bar.y_range.start = -1
    p_bar.y_range.end = 1
    p_bar.x_range.range_padding = 0.1
    p_bar.xaxis.major_label_orientation = 1
    p_bar.xgrid.grid_line_color = None
    return p_bar
###############################################################################
def region_comparison_tab(old_to_new_sources,new_to_old_sources,
                          old_to_new_regions,n_folds_list,agg_all_nfolds):
    
    ## Get region names and list of sources
    region_names_new = list(old_to_new_regions.values())
    region_names_new.sort()
    # sourceset_display = ['Best'] + list(old_to_new_sources.keys())[1:]
    # sourceset_display = list(old_to_new_sources.keys())[1:]
    sourceset_display = sorted(list(new_to_old_sources.keys())[1:])
    # Put Climate at the end
    if 'Climate' in sourceset_display:
        c_idx = sourceset_display.index('Climate')
        sourceset_display.pop(c_idx)
        sourceset_display.append('Climate')
    n_folds_display = [np.str(x) for x in n_folds_list]
    
    # Split data and display into 2 parts
    mid = np.int(np.ceil(len(region_names_new)/2))
    regions1 = region_names_new[:mid]
    regions2 = region_names_new[mid:]
    
    # Score types considered
    score_types = ['In Sample','OOS']
    
    # Data to display to start: select one region and one value of n_folds
    source_set_bar = 'Best'
    n_folds_bar = n_folds_list[0]
    colors1 = (Category10[3])[1:]*len(regions1)
    colors2 = (Category10[3])[1:]*len(regions2)
    
    # Get source data
    source_data1, x1 = prep_data(agg_all_nfolds,regions1,source_set_bar,
                             n_folds_bar,sourceset_display,score_types,colors1)
    source_data2, x2 = prep_data(agg_all_nfolds,regions2,source_set_bar,
                             n_folds_bar,sourceset_display,score_types,colors2)
    
    source_bar1 = ColumnDataSource(data=source_data1)
    source_bar2 = ColumnDataSource(data=source_data2)

    # Create graphs
    p_bar1 = figure(x_range=FactorRange(*x1), plot_height=350, plot_width=1150,
               title='R squared for a given Source Set',
               toolbar_location=None, tools="")
    p_bar1.vbar(x='x', top='counts', width=0.9, source=source_bar1,color='color')
    
    p_bar2 = figure(x_range=FactorRange(*x2), plot_height=350, plot_width=1150,
               title='R squared for a given Source Set',
               toolbar_location=None, tools="")
    p_bar2.vbar(x='x', top='counts', width=0.9, source=source_bar2,color='color')
    
    # Format graphs
    p_bar1 = format_graph(p_bar1)
    p_bar2 = format_graph(p_bar2)
    
    
    ## Radio buttons for bar graph
    source_sets_button_bar = RadioButtonGroup(labels=sourceset_display, active=0)
    n_folds_button_bar = RadioButtonGroup(labels=n_folds_display, active=0)
    
    # Text above graph
    # div_text_bar = 'Gold Standard: ' + gs_name_bar + \
    #                '<br>Predictor Set: ' + np.str(n_folds_bar)
    div_text_bar = ''
    div_bar = Div(text=div_text_bar,width=700, height=50)
    
    # Update function
    ###########################################################################
    def bar_callback(attr, old, new):
        # Get new selected value
        #new_score = score_types[new.active]
        source_set_bar = sourceset_display[source_sets_button_bar.active]
        n_folds_bar = np.int(n_folds_display[n_folds_button_bar.active])
        
        # Update data to display
        source_data1_new, x1 = prep_data(agg_all_nfolds,regions1,source_set_bar,
                             n_folds_bar,sourceset_display,score_types,colors1)
        source_data2_new, x2 = prep_data(agg_all_nfolds,regions2,source_set_bar,
                             n_folds_bar,sourceset_display,score_types,colors2)
        
        # Update graphs
        source_bar1.data = source_data1_new
        source_bar2.data = source_data2_new
        
        # new_text_bar = 'Gold Standard: ' + gs_name_bar + \
        #                '<br>Predictor Set: ' + np.str(n_folds_bar)
        # div_bar.text = new_text_bar
        return
    ###########################################################################
    source_sets_button_bar.on_change('active', bar_callback)
    n_folds_button_bar.on_change('active', bar_callback)
    
    # Put controls in a single element
    controls = WidgetBox(n_folds_button_bar,source_sets_button_bar)
	
	# Create a row layout
    layout = column(controls,div_bar,p_bar1,p_bar2)
	
	# Make a tab with the layout
    tab = Panel(child=layout, title = 'Comparison of Regions')
    
    return tab
###############################################################################