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
def source_comparison_tab(old_to_new_sources,new_to_old_sources,
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
    
    # Replace long names to shorter ones for display
    shorter_source_names = {'Colombia & GT by State':'CO & GT by State',
                            'Colombia & GT by Symptom':'CO & GT by Symptom',
                            'Colombia Border & GT':'CO Border & GT',
                            'GT by State - Ven & Col':'GT by State - VE & CO'}
    sourceset_display_short = [x if x not in shorter_source_names.keys() else
                               shorter_source_names[x] for x in sourceset_display]
    
    # All regions and sources
    # all_regions = np.unique(agg_all_nfolds.Region).tolist()
    # all_sources = np.unique(agg_all_nfolds.SourcesSet).tolist()
    
    # Score types considered
    score_types = ['In Sample','OOS']
    
    # Data to display: select one region and one value of n_folds
    gs_name_bar = 'Amazonas'
    n_folds_bar = n_folds_list[0]
    agg_gs = agg_all_nfolds.loc[agg_all_nfolds.Region == gs_name_bar]
    agg_n_gs = agg_gs.loc[agg_gs.NbFolds == n_folds_bar,:].\
        sort_values(by='SourcesSet',ascending=True)
    # agg_n_gs.sort_values(by='SourcesSet',ascending=True,inplace=True)
    in_sample_vals = agg_n_gs.loc[agg_n_gs.ScoreType == 'InSample',\
                                  'Value'].values.tolist()
    OOS_vals = agg_n_gs.loc[agg_n_gs.ScoreType == 'OOS',\
                            'Value'].values.tolist()
    data_dict = {'Source Set' : sourceset_display_short,
                 'In Sample' : in_sample_vals,
                 'OOS' : OOS_vals}
    x = [ (s, score) for s in sourceset_display_short for score in score_types ]
    
    counts = sum(zip(data_dict['In Sample'], data_dict['OOS']), ()) # like an hstack
    colors = (Category10[3])[1:]*len(sourceset_display_short)
    source_bar = ColumnDataSource(data=dict(x=x, counts=counts, color=colors))
    p_bar = figure(x_range=FactorRange(*x), plot_height=350, plot_width=1050,
               title='R squared for a given Region',
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
        gs_name_bar = region_names_new[regions_button_bar.active]
        n_folds_bar = np.int(n_folds_display[n_folds_button_bar.active])
        
        # Update data to display
        agg_gs = agg_all_nfolds.loc[agg_all_nfolds.Region == gs_name_bar]
        agg_n_gs = agg_gs.loc[agg_gs.NbFolds == n_folds_bar,:].\
            sort_values(by='SourcesSet',ascending=True)
        # agg_n_gs.sort_values(by='SourcesSet',ascending=True,inplace=True)
        in_sample_vals = agg_n_gs.loc[agg_n_gs.ScoreType == 'InSample',\
                                             'Value'].values.tolist()
        OOS_vals = agg_n_gs.loc[agg_n_gs.ScoreType == 'OOS',\
                                             'Value'].values.tolist()
        data_dict = {'Source Set' : sourceset_display_short,
                     'In Sample' : in_sample_vals,
                     'OOS' : OOS_vals}
        # x = [ (s, score) for s in sourceset_display for score in score_types ]
        
        counts = sum(zip(data_dict['In Sample'], data_dict['OOS']), ()) # like an hstack
        # new_source = ColumnDataSource(data=dict(x=x, counts=counts, color=colors))
        new_source_data = dict(x=x, counts=counts, color=colors)
        source_bar.data = new_source_data
        
        # new_text_bar = 'Gold Standard: ' + gs_name_bar + \
        #                '<br>Predictor Set: ' + np.str(n_folds_bar)
        # div_bar.text = new_text_bar
        return
    ###########################################################################
    regions_button_bar.on_change('active', bar_callback)
    n_folds_button_bar.on_change('active', bar_callback)
    
    # Put controls in a single element
    controls = WidgetBox(n_folds_button_bar,regions_button_bar)
	
	# Create a row layout
    layout = column(controls,div_bar,p_bar)
	
	# Make a tab with the layout
    tab = Panel(child=layout, title = 'Comparison of Source Sets')
    
    return tab
###############################################################################