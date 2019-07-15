import os
import numpy as np
import pandas as pd
import glob
import geopandas as gpd
import json
from bokeh.io import output_notebook, output_file
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
from bokeh.palettes import brewer, Category20, Viridis256
from bokeh.models import FixedTicker,NumeralTickFormatter,HoverTool
from bokeh.plotting import show as show_inline
from bokeh.models.widgets import RadioButtonGroup, Div
from bokeh.layouts import column,row, widgetbox
from bokeh.io import curdoc


#import webbrowser    
#urL='https://www.google.com'
#chrome_path="C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
#webbrowser.register('chrome', None,webbrowser.BackgroundBrowser(chrome_path),1)
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
###############################################################################
## GPS data
shapefile = os.sep.join([os.getcwd(),'MapData','Regions']) + os.sep +\
    'ven_admbnda_adm1_20180502.shp'
gdf = gpd.read_file(shapefile)[['ADM1_ES', 'geometry']]
gdf.columns = ['Region', 'geometry']
gdf.sort_values(by=['Region'], inplace=True)
gdf.reset_index(drop=True, inplace=True)

## Optimization data
results_folder = 'OptimizationResults' + os.sep + 'SummaryAll'
n_folds = 8
filename = 'SummaryAggregate_nfolds-' + np.str(n_folds) + '.csv'
agg_path = os.sep.join([os.getcwd(),results_folder,filename])
agg = pd.read_csv(agg_path)

## Get all possible combinations of data to speed up display
all_data = {}
score_types_list = np.unique(agg.ScoreType.tolist())
sources_list = np.unique(agg.SourcesSet.tolist() + ['Best'])

# Loop through possible score types and sources
for t in score_types_list:
    for s in sources_list:
        # Filter optimization data
        df_show = agg.loc[agg.ScoreType == t,:]
        if s == 'Best':
            df_show = df_show.loc[df_show['IsBest'],:]
        else:
            df_show = df_show.loc[df_show['SourcesSet'] == s,:]
        df_show = df_show[['Region','Value']]
        
        # Add optimization data to geo data
        merged_df = gdf.merge(df_show,left_on='Region',right_on='Region',how='left')
        merged_json = json.loads(merged_df.to_json())
        json_data = json.dumps(merged_json)
        geosource_t_s = GeoJSONDataSource(geojson = json_data)
        
        # Save in dictionary
        k = t + '|' + s
        all_data.update({k:geosource_t_s})


# Filter data we start with
score_type_filter = 'InSample' #'OOS'
source_filter = 'Best'
geosource = GeoJSONDataSource(geojson = \
    all_data[score_type_filter + '|' + source_filter].geojson)


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

# For testing only
#div_text = 'OOS to start' # For testing only
#div = Div(text=div_text,width=200, height=50) # For testing only

# Update function
def score_type_callback(attr, old, new):
    # Get new selected value
    #new_score = score_types[new.active]
    new_score = score_types[score_type_button.active]
    new_source = data_sources[sources_button.active]
    new_key = new_score + '|' + new_source
    geosource_new = all_data[new_key]
    geosource.geojson = geosource_new.geojson
    
    #div.text = new_score # For testing only

score_type_button.on_change('active', score_type_callback)
sources_button.on_change('active', score_type_callback)

#Display figure.
set_stype(p)

# In Notebook (without callbacks)
#output_notebook()
#show(p)

# In Spyder (without callbacks)
# output_file('foo.html')
# show(column(widgetbox(score_type_button),p),browser="chrome")

# To run from command (in the folder of the file) using the command
# bokeh serve --show situ_map.py
curdoc().add_root(column(score_type_button,sources_button,p))
curdoc().title = "Venezuela Situational Awareness"


