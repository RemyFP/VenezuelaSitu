# import os
import numpy as np
import pandas as pd
# import glob
# import geopandas as gpd
# import json
# from bokeh.io import output_notebook, output_file
# from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import Panel,GeoJSONDataSource,LinearColorMapper,ColorBar
from bokeh.models import ColumnDataSource
from bokeh.palettes import Viridis256#,brewer, Category20
from bokeh.models import NumeralTickFormatter,HoverTool#,FixedTicker
# from bokeh.plotting import show as show_inline
from bokeh.models.widgets import RadioButtonGroup#, Div
from bokeh.layouts import column,WidgetBox#,row,widgetbox
# from bokeh.io import curdoc
###############################################################################
def geopd_prepare(gdf):
    """ Prepare geopandas dataframe to have single polygons, by extracting 
    them from MultiPolygons (if any)
    """
    # Loop through rows to test each entry for polygon or multipolygon
    regions,polygons = [],[]
    for index, row in gdf.iterrows():
        if row['geometry'].geom_type == 'MultiPolygon':
            # polygons = list(row['geometry'])
            for p in row['geometry']:
                regions.append(row['Region'])
                polygons.append(p)
        
        elif row['geometry'].geom_type == 'Polygon':
            regions.append(row['Region'])
            polygons.append(row['geometry'])
    
    # Create dataframe containing polygons only
    new_gdf = pd.DataFrame({'Region':regions,'geometry':polygons})
    return new_gdf 
###############################################################################
def getPolyCoords(row, geom, coord_type):
    """Returns the coordinates ('x' or 'y') of edges of a Polygon exterior
        Source: https://automating-gis-processes.github.io/2016/
        Lesson5-interactive-map-bokeh.html
    """
    # Differentiate between the MultiPolygon and single Polygon cases
    # MultiPolygon: several pieces
    
    # Parse the exterior of the coordinate
    exterior = row[geom].exterior

    if coord_type == 'x':
        # Get the x coordinates of the exterior
        return list( exterior.coords.xy[0] )
    elif coord_type == 'y':
        # Get the y coordinates of the exterior
        return list( exterior.coords.xy[1] )
###############################################################################
def country_adjust(gdf_country):
    # Parameters to move and scale the total map
    y0_new, x0_new = 2., -73.
    y0_old, x0_old = 0., -74.
    scale = 1/4
    
    # Extract from multipolygon and adjust
    adjusted_x, adjusted_y = [], []
    for p in gdf_country[0]:
        adjusted_x.append(list(\
            (np.array(p.exterior.coords.xy[0]) - x0_old) * scale + x0_new))
        adjusted_y.append(list(\
            (np.array(p.exterior.coords.xy[1]) - y0_old) * scale + y0_new))
    
    return adjusted_x, adjusted_y
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
###############################################################################
#def map_tab(map_all_data,score_types_list,sources_list):
def map_tab(gdf,gdf_country,agg_all_nfolds,score_types_list,sources_list,
            n_folds_list):
    # Prepare coordinates data
    gdf.sort_values(by=['Region'], inplace=True)
    gdf.reset_index(drop=True, inplace=True)
    gdf = geopd_prepare(gdf)
    
    # Get lis of coordinates of states limits, and their names
    country_xs = gdf.apply(getPolyCoords, geom='geometry', coord_type='x', axis=1)
    country_ys = gdf.apply(getPolyCoords, geom='geometry', coord_type='y', axis=1)
    states = gdf.Region.tolist()
    
    # Get entire country coordinates and add them to the list
    total_country_x, total_country_y = country_adjust(gdf_country)
    country_xs = country_xs.append(pd.Series(total_country_x),ignore_index=True)
    country_ys = country_ys.append(pd.Series(total_country_y),ignore_index=True)
    states.extend(['Total'] * len(total_country_x))
    
    # List of n_folds values
    n_folds_display = [np.str(x) for x in n_folds_list]
    
    # Filter data we start with
    score_type_filter = 'InSample' #'OOS'
    source_filter = 'Best'
    n_folds_map = n_folds_list[0]
    # t = score_types_list[0]
    # s = sources_list[0]
    df_show = agg_all_nfolds.loc[agg_all_nfolds.NbFolds == n_folds_map,:]
    df_show = df_show.loc[df_show.ScoreType == score_type_filter,:]
    if source_filter == 'Best':
        df_show = df_show.loc[df_show['IsBest'],:]
    else:
        df_show = df_show.loc[df_show['SourcesSet'] == source_filter,:]
    df_show_dict = df_show[['Region','Value']].set_index('Region').T.to_dict()
    scores = [df_show_dict[x]['Value'] for x in states]
    
    # Create bokeh map source data
    col_source = ColumnDataSource(data = dict(xs=country_xs,
                                              ys=country_ys,
                                              Region=states,
                                              Score=scores))
    # geosource = GeoJSONDataSource(geojson = \
    #     map_all_data[score_type_filter + '|' + source_filter].geojson)
    
    ## Map formatting ##################
    palette = Viridis256 # brewer["Spectral"][8] # Category20[20] # brewer["PRGn"][11]
    color_mapper = LinearColorMapper(palette = palette, low=-1,high=1)
    color_bar = ColorBar(color_mapper=color_mapper,width = 650, height = 20,
                         formatter = NumeralTickFormatter(format=",0.00"),
                         border_line_color=None, orientation = 'horizontal',
                         location=(0, 0) )
    hover = HoverTool(tooltips = [('Region','@Region'),('Score', '@Score')])
    
    #Create figure object.
    width = 750
    height = np.int(width*13/15)
    p = figure(title = 'Venezuela Situational Awareness', 
               plot_height = height , plot_width = width, 
               tools = [hover,"pan,wheel_zoom,box_zoom,reset" ], 
               toolbar_location = "left")
    
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    
    #Add patch renderer to figure. 
    p.patches('xs','ys', source = col_source,
              fill_color = {'field' :'Score', 'transform' : color_mapper},
              line_color = 'white', line_width = 1, fill_alpha = 0.7)
    #Specify figure layout.
    p.add_layout(color_bar, 'below')
    
    
    ## Radio buttons
    # Define buttons
    n_folds_button = RadioButtonGroup(labels=n_folds_display, active=0)
    score_types = score_types_list.tolist()
    score_type_button = RadioButtonGroup(labels=score_types, active=0)
    data_sources = sources_list#.tolist()
    sources_button = RadioButtonGroup(labels=data_sources, active=0)

    ###########################################################################
    # Update function
    def score_type_callback(attr, old, new):
        # Get new score data to display
        # Filter data we start with
        n_folds_map_new = np.int(n_folds_display[n_folds_button.active])
        new_score = score_types[score_type_button.active]
        new_source = data_sources[sources_button.active]
        
        df_show = agg_all_nfolds.loc[agg_all_nfolds.NbFolds == n_folds_map_new,:]
        df_show = df_show.loc[df_show.ScoreType == new_score,:]
        if new_source == 'Best':
            df_show = df_show.loc[df_show['IsBest'],:]
        else:
            df_show = df_show.loc[df_show['SourcesSet'] == new_source,:]
        df_show_dict = df_show[['Region','Value']].set_index('Region').T.to_dict()
        new_scores = [df_show_dict[x]['Value'] for x in states]
        
        # Patch source data with new scores
        col_source.patch({'Score': [(slice(len(new_scores)), new_scores)] })
    ###########################################################################  
    # Update events
    n_folds_button.on_change('active', score_type_callback)
    score_type_button.on_change('active', score_type_callback)
    sources_button.on_change('active', score_type_callback)
    
    # Figure formatting
    set_stype(p)
    
    # Put controls in a single element
    controls = WidgetBox(n_folds_button,score_type_button,sources_button)
	
	# Create a row layout
    layout = column(controls,p)
	
	# Make a tab with the layout
    tab = Panel(child=layout, title = 'Map of Scores')
    
    return tab
    
    # In Notebook (without callbacks)
    #output_notebook()
    #show(p)
    
    # In Spyder (without callbacks)
    # output_file('foo.html')
    # show(column(widgetbox(score_type_button),p),browser="chrome")
    
    # To run from command (in the folder of the file) using the command
    # bokeh serve --show situ_map.py
    # curdoc().add_root(column(score_type_button,sources_button,p))
    # curdoc().title = "Venezuela Situational Awareness"
###############################################################################