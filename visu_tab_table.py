# -*- coding: utf-8 -*-
# import os
import numpy as np
# import pandas as pd
# import glob
# import geopandas as gpd
# import json
# from bokeh.io import output_notebook, output_file
# from bokeh.io import show
# from bokeh.plotting import figure
# from bokeh.models import GeoJSONDataSource,LinearColorMapper,ColorBar
from bokeh.models import ColumnDataSource,Panel#,FactorRange
# from bokeh.palettes import brewer, Category20, Viridis256, Category10
# from bokeh.models import FixedTicker,NumeralTickFormatter,HoverTool
# from bokeh.plotting import show as show_inline
from bokeh.models.widgets import CheckboxGroup,Div,Button#,RadioButtonGroup
from bokeh.models.widgets import DataTable,TableColumn#,DateFormatter
from bokeh.models.widgets import NumberFormatter
from bokeh.layouts import row,column#,WidgetBox,widgetbox
# from bokeh.io import curdoc
# import situ_fn
###############################################################################
def data_to_show(agg_all_nfolds,conditions=None):
    df = agg_all_nfolds.copy()
    # Filter data based on passed conditions
    if conditions is not None:
        for k in conditions.keys():
            df = df.loc[df[k].isin(conditions[k]),:]
    
    data = dict(
        NbFolds=df['NbFolds'].values.tolist(),
        ScoreType=df['ScoreType'].values.tolist(),
        Region=df['Region'].values.tolist(),
        SourcesSet=df['SourcesSet'].values.tolist(),
        IsBest=df['IsBest'].values.tolist(),
        Value=df['Value'].values.tolist(),
    )
    
    return data
###############################################################################
def table_tab(agg_all_nfolds):
    
    ## Define checkboxes to select data to display
    # Get possible values for each column
    nb_folds_list = list(np.unique(agg_all_nfolds['NbFolds']))
    nb_folds_list.sort()
    
    score_type_list = list(np.unique(agg_all_nfolds['ScoreType']))
    score_type_list.sort()
    
    region_list = list(np.unique(agg_all_nfolds['Region']))
    region_list.sort()
    
    sources_set_list = list(np.unique(agg_all_nfolds['SourcesSet']))
    sources_set_list.sort()
    
    is_best_list = list(np.unique(agg_all_nfolds['IsBest']))
    is_best_list.sort()

    # Define boxes
    nb_folds_box = CheckboxGroup(labels=[np.str(x) for x in nb_folds_list],
                                 active=[0,1])
    score_type_box = CheckboxGroup(labels=score_type_list, active=[0,1])
    region_box = CheckboxGroup(labels=region_list, active=[0,1])
    sources_set_box = CheckboxGroup(labels=sources_set_list, active=[0,1])
    is_best_box = CheckboxGroup(labels=[np.str(x) for x in is_best_list],
                                active=[0,1])
    
    # Divs to contain each checkbox's name
    nb_folds_div = Div(text='<b>Nb Folds</b>')
    score_type_div = Div(text='<b>Score Type</b>')
    region_div = Div(text='<b>Region</b>')
    sources_set_div = Div(text='<b>Sources Set</b>')
    is_best_div = Div(text='<b>Is Best</b>')
    
    # Buttons to select all/none regions or sources at once
    select_all_regions_button = Button(label="Select All")
    select_all_sources_button = Button(label="Select All")
    select_no_region_button = Button(label="Unselect All")
    select_no_source_button = Button(label="Unselect All")
    
    ###########################################################################
    # Update function
    def boxes_update(attr):#attr, old, new):
        # Get list of selected items for each button
        nb_folds_selected = [nb_folds_list[x] for x in nb_folds_box.active]
        score_type_selected = [score_type_list[x] for x in score_type_box.active]
        region_selected = [region_list[x] for x in region_box.active]
        sources_set_selected = [sources_set_list[x] for x in sources_set_box.active]
        is_best_selected = [is_best_list[x] for x in is_best_box.active]
        
        # Update source data
        conditions = {'NbFolds':nb_folds_selected,
                      'ScoreType':score_type_selected,
                      'Region':region_selected,
                      'SourcesSet':sources_set_selected,
                      'IsBest':is_best_selected}
        new_data = data_to_show(agg_all_nfolds,conditions)
        # new_source = ColumnDataSource(data)
        source.data = new_data
        
    def update_all_regions():
        region_box.active = list(range(len(region_list)))
    
    def update_all_sources():
        sources_set_box.active = list(range(len(sources_set_list)))
    
    def update_no_region():
        region_box.active = []
    
    def update_all_source():
        sources_set_box.active = []
    ###########################################################################
    
    # Callback / Update function events
    nb_folds_box.on_click(boxes_update)
    score_type_box.on_click(boxes_update)
    region_box.on_click(boxes_update)
    sources_set_box.on_click(boxes_update)
    is_best_box.on_click(boxes_update)
    select_all_regions_button.on_click(update_all_regions)
    select_all_sources_button.on_click(update_all_sources)
    select_no_region_button.on_click(update_no_region)
    select_no_source_button.on_click(update_all_source)

    ## Create table and get data to display when opening the page
    starting_cond = {'NbFolds':[nb_folds_list[x] for x in nb_folds_box.active],
                     'Region':[region_list[x] for x in region_box.active],
                     'SourcesSet':[sources_set_list[x] for x in sources_set_box.active]}
    data_start = data_to_show(agg_all_nfolds,conditions=starting_cond)
    source = ColumnDataSource(data_start)
    columns = [
            TableColumn(field="NbFolds", title="Nb Folds"),
            TableColumn(field="ScoreType", title="Score Type"),
            TableColumn(field="Region", title="Region"),
            TableColumn(field="SourcesSet", title="Predictors Set"),
            TableColumn(field="IsBest", title="Is Best"),
            TableColumn(field="Value", title="Value",
                        formatter=NumberFormatter(format='0,0.00')),
        ]
    # Create data table, fit_columns=False ensures we have a scrolling bar
    data_table = DataTable(source=source, columns=columns, width=700, height=900)

    # Put controls in a single element
    controls = row(column(nb_folds_div,nb_folds_box,width=60),
                   column(score_type_div,score_type_box,width=90),
                   column(region_div,select_all_regions_button,
                          select_no_region_button,region_box,width=140),
                   column(sources_set_div,select_all_sources_button,
                          select_no_source_button,sources_set_box,width=200),
                   column(is_best_div,is_best_box,width=90))
	
	# Create a row layout
    layout = row(controls,data_table)
	
	# Make a tab with the layout
    tab = Panel(child=layout, title = 'Summary Results - Table')
    
    return tab
###############################################################################
