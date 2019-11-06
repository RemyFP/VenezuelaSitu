# import os
import numpy as np
# import pandas as pd
# import glob
# import geopandas as gpd
# import json
# from bokeh.io import output_notebook, output_file
# from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import Panel,GeoJSONDataSource,LinearColorMapper,ColorBar
from bokeh.palettes import Viridis256#,brewer, Category20
from bokeh.models import NumeralTickFormatter,HoverTool#,FixedTicker
# from bokeh.plotting import show as show_inline
from bokeh.models.widgets import RadioButtonGroup#, Div
from bokeh.layouts import column,WidgetBox#,row,widgetbox
# from bokeh.io import curdoc

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
def map_tab(map_all_data,score_types_list,sources_list):
    # Filter data we start with
    score_type_filter = 'InSample' #'OOS'
    source_filter = 'Best'
    geosource = GeoJSONDataSource(geojson = \
        map_all_data[score_type_filter + '|' + source_filter].geojson)
    
    
    ## Map formatting ##################
    palette = Viridis256 # brewer["Spectral"][8] # Category20[20] # brewer["PRGn"][11]
    color_mapper = LinearColorMapper(palette = palette, low=-1,high=1)
    color_bar = ColorBar(color_mapper=color_mapper,width = 650, height = 20,
                         formatter = NumeralTickFormatter(format=",0.00"),
                         border_line_color=None, orientation = 'horizontal',
                         location=(0, 0) )
    hover = HoverTool(tooltips = [('Region','@Region'),('Value', '@Value')])
    
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
    p.patches('xs','ys', source = geosource,
              fill_color = {'field' :'Value', 'transform' : color_mapper},
              line_color = 'white', line_width = 1, fill_alpha = 0.7)
    #Specify figure layout.
    p.add_layout(color_bar, 'below')
    
    
    ## Radio buttons
    # Define buttons
    score_types = score_types_list.tolist()
    score_type_button = RadioButtonGroup(labels=score_types, active=0)
    data_sources = sources_list.tolist()
    sources_button = RadioButtonGroup(labels=data_sources, active=0)

    
    # Update function
    def score_type_callback(attr, old, new):
        # Get new selected value
        #new_score = score_types[new.active]
        new_score = score_types[score_type_button.active]
        new_source = data_sources[sources_button.active]
        new_key = new_score + '|' + new_source
        geosource_new = map_all_data[new_key]
        geosource.geojson = geosource_new.geojson
    
    # Update events
    score_type_button.on_change('active', score_type_callback)
    sources_button.on_change('active', score_type_callback)
    
    # Figure formatting
    set_stype(p)
    
    # Put controls in a single element
    controls = WidgetBox(score_type_button,sources_button)
	
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


