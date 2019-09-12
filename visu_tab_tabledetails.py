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
# from bokeh.models.widgets import NumberFormatter
from bokeh.layouts import row,column#,WidgetBox,widgetbox
# from bokeh.io import curdoc
# import situ_fn
###############################################################################
def data_to_show(df_all_t,result_cols,conditions=None):
    df = df_all_t.copy()
    # Filter data based on passed conditions
    if conditions is not None:
        for k in conditions.keys():
            df = df.loc[df[k].isin(conditions[k]),:]
    
    data = dict(
        NbFolds=df['NbFolds'].values.tolist(),
        Region=df['Region'].values.tolist(),
        SourcesSet=df['SourcesSet'].values.tolist(),
        Column=df['Column'].values.tolist(),
        Score=df['Score'].values.tolist()
    )

    # Add the actual result columns (column names are numbers)
    for c in result_cols:
        data.update({np.str(c):df[c].values.tolist()})
    
    return data
###############################################################################
def tabledetails_tab(df_all_nfolds,old_to_new_regions,new_to_old_regions,
                     old_to_new_sources,new_to_old_sources):
    # Transpose dataframe
    df_all_t = df_all_nfolds.copy().T
    
    # Replace values in dataframe
    region_replace = dict(old_to_new_sources)
    del region_replace['Colombia']
    df_all_t.replace(to_replace={'Column':{'id': 'Predictor',
                                           'objective_values':'R2 Training',
                                           'OOS_R_squared':'R2 OOS'},
                                 'SourcesSet':region_replace,
                                 'Region':old_to_new_regions},
                     inplace=True)
    
    # Define columns between selector and results (2nd column is for width)
    select_cols = {'NbFolds':['Nb Folds',60],
               'Region':['Region',100],
               'SourcesSet':['Predictors Set',150],
               'Column':['Result Type',80],
               'Score':['Score',80]}
    result_cols = sorted([x for x in df_all_t.columns.tolist() if x.isdigit()],
                         key=lambda x: float(x))
    
    # Transform number entries into numbers and round them
    nb_rows = ['R2 Training','R2 OOS']
    df_all_t.loc[df_all_t.Column.isin(nb_rows),result_cols] = \
        df_all_t.loc[df_all_t.Column.isin(nb_rows),result_cols].applymap(\
        lambda x: np.float(x)).round(3)
    
    # Get final value for each row
    df_all_t.loc[:,'Score'] = df_all_t.loc[:,result_cols].apply(lambda x:
        x.fillna(method='ffill')[-1],axis=1)
    df_all_t.loc[:,'Score'] = df_all_t.apply(lambda x: x['Score'] if 
                x['Column'] in nb_rows else '',axis=1)
    df_all_t.fillna(value='',inplace=True)
    
    ## Define checkboxes to select data to display
    # Get possible values for each column
    nb_folds_list = sorted(np.unique(df_all_t['NbFolds']), key=lambda x: float(x))
    region_list = sorted(np.unique(df_all_t['Region']))
    sources_set_list = sorted(np.unique(df_all_t['SourcesSet']))
    result_type_list = sorted(np.unique(df_all_t['Column']))


    # Define boxes
    nb_folds_box = CheckboxGroup(labels=[np.str(x) for x in nb_folds_list],
                                 active=[0,1])
    region_box = CheckboxGroup(labels=region_list, active=[0,1])
    sources_set_box = CheckboxGroup(labels=sources_set_list, active=[0,1])
    result_type_box = CheckboxGroup(labels=result_type_list, active=[0,1,2])
    
    # Divs to contain each checkbox's name
    nb_folds_div = Div(text='<b>Nb Folds</b>')
    region_div = Div(text='<b>Region</b>')
    sources_set_div = Div(text='<b>Sources Set</b>')
    result_type_div = Div(text='<b>Result Type</b>')
    
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
        result_type_selected = [result_type_list[x] for x in result_type_box.active]
        region_selected = [region_list[x] for x in region_box.active]
        sources_set_selected = [sources_set_list[x] for x in sources_set_box.active]
        
        # Update source data
        conditions = {'NbFolds':nb_folds_selected,
                      'Column':result_type_selected,
                      'Region':region_selected,
                      'SourcesSet':sources_set_selected}
        new_data = data_to_show(df_all_t,result_cols,conditions)
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
    result_type_box.on_click(boxes_update)
    region_box.on_click(boxes_update)
    sources_set_box.on_click(boxes_update)
    select_all_regions_button.on_click(update_all_regions)
    select_all_sources_button.on_click(update_all_sources)
    select_no_region_button.on_click(update_no_region)
    select_no_source_button.on_click(update_all_source)

    ## Create table and get data to display when opening the page
    starting_cond = {'NbFolds':[nb_folds_list[x] for x in nb_folds_box.active],
                     'Region':[region_list[x] for x in region_box.active],
                     'SourcesSet':[sources_set_list[x] for x in sources_set_box.active]}
    data_start = data_to_show(df_all_t,result_cols,conditions=starting_cond)
    source = ColumnDataSource(data_start)
    
    # Define table columns
    columns = []
    for k in select_cols.keys():
        columns.append(TableColumn(field=k, title=select_cols[k][0],
                                   width=select_cols[k][1]))
    # Add the actual result columns (column names are numbers)
    for c in result_cols:
        columns.append(TableColumn(field=c, title=np.str(np.int(c)+1),
                                   width=200))
    
    # Create data table, fit_columns=False ensures we have a scrolling bar
    data_table = DataTable(source=source, columns=columns,
                           width=1000, height=700, fit_columns=False)

    # Put controls in a single element
    controls = row(column(nb_folds_div,nb_folds_box,width=60),
                   column(region_div,select_all_regions_button,
                          select_no_region_button,region_box,width=140),
                   column(sources_set_div,select_all_sources_button,
                          select_no_source_button,sources_set_box,width=200),
                   column(result_type_div,result_type_box,width=100))
	
	# Create a row layout
    layout = row(controls,data_table)
	
	# Make a tab with the layout
    tab = Panel(child=layout, title = 'Detailed Results - Table')
    
    return tab
###############################################################################