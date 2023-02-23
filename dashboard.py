# ##############################              ############################## #
# 								import package                               #
# ##############################              ############################## #
import base64
import bridge  					# local file where data is processed
import datetime
import dash.dependencies as dd
import numpy as np 				# pip install numpy
import pandas as pd             # pip install pandas
import plotly.express as px
import plotly.graph_objs as go

from dash import Dash, html, dcc, Input, Output                               # pip install dash
from dash_extensions.enrich import Trigger, FileSystemStore, ServersideOutput # pip install dash_extensions
from io import BytesIO          
from PIL import Image                                                         # pip install Pillow
from wordcloud import WordCloud                                               # pip install wordcloud

import classify


# ##############################                   ############################## #
# 								dashboard functions                               #
# ##############################                   ############################## #
# Generation of the Table which contains real information 
# maximum rows is currently fixed to 5
# TODO : slider can be created to show more or less informations
def generate_info_table(dataframe, max_rows=5):
	return html.Table(children=[
			html.Thead(
				html.Tr(children=[
					html.Th(children="Date"),
					html.Th(children="Source"),
					html.Th(children="Content")
					], className="table.GeneratedTable thead")), # end of html.Thread
			html.Tbody([
				html.Tr([
					html.Td(dataframe.iloc[i][0]),
					html.Td(dataframe.iloc[i][1]),
					html.Td(dataframe.iloc[i][2])
					]) for i in range(min(len(dataframe), max_rows)) # end of html.Tr
				]) # end of html.Tbody
		], className="")# end of html.Table

# Generation the WordCloud
# Visualization is based on major words used to determine the disInformation
def generate_wordcloud(dataframe):
	d = {a: x for a, x in dataframe.values}
	wc = WordCloud(background_color='white', width=720, height=360)
	wc.fit_words(d)
	return wc.to_image()


# Generation Line Graphic
# Visualization is based on all data from csv.
fig_line = go.Figure(data=[go.Scatter(x=bridge.df_counts['unique_date'], y=bridge.df_counts['counts_date'], name="Total info"),
	go.Scatter(x=bridge.dfFake_counts['unique_date'], y=bridge.dfFake_counts['counts_date'], name="Fake info"),
	go.Scatter(x=bridge.dfReal_counts['unique_date'], y=bridge.dfReal_counts['counts_date'], name="Real info"),
	], layout={"legend" : {"title" : "legend"}, "title" : "Graphic of Number of messages per days (total, Fake, Real)"})


# ##############################             ############################## #
# 				                app Dash info                               #
# ##############################             ############################## #
# Dashboard app entry
app = Dash(__name__)

app.layout = html.Div(id="top-layer", children=[
	# Header Layout
	html.Div(id="sub-layer-1", children=[
		html.Img(src=app.get_asset_url('top banner-200h.png'), className=""),
		html.H1(children="DISInformation Analyser : Disinformation Dashboard", className=""),
		], className=""), # endof html.Div#sub-layer-1

	# Info and Research Layout
	html.Div(id="sub-layber-2", children=[
		html.H2(children="Disinformation Analyzer"),
		html.Br(),
		html.Span(children="Insert some text here!"),
		html.Br(),
		dcc.Textarea(id="input-textarea-research",
			value="Input text to analyzed here", 
			className=""),
		html.Br(),
		html.Span(id="input-from-user"),
		html.Br(),
		html.Button('Submit', id='submit-text', n_clicks=0, className="")
		],className=""), # endof html.Div#sub-layer-2

	# Graphic and Word Cloud
	html.Div(id="sub-layer-3", children=[
		dcc.Graph(id="line-graph-visu", figure=fig_line),
		html.Img(id="image_wc"),
	], style={'display':'flex', 'flex-direction':'row', 'padding':10}), # endof html.Div#sub-layer-3
	
	# Table Layout
	html.Div(id="sub-layer-4", children=[
		html.Table(id="table-text-information", children=[
			html.Thead(children=[
				html.Tr(children=[
					html.Th(children=["Fake News"]),
					html.Th(children=["Real News"])
				]),
			]),
			html.Tbody(children=[
				html.Tr(children=[
					html.Td(children=[generate_info_table(bridge.dfFake)]),
					html.Td(children=[generate_info_table(bridge.dfReal)])
				])
			]),
		]),
	]),#style={'display':'flex', 'flex-direction':'row', 'padding':10}), # endof html.Div#sub-layer-4

	# Bottom Layout
	html.Div(id="sub-layer-5", children=[
		html.Img(src=app.get_asset_url('bottom banner-200h.png'), className="")
	],className=""), # endof html.Div#sub-layer-5

	# background layout for update
	dcc.Interval(id="update-csv-data", interval=30*1000, n_intervals=0),
	dcc.Store(id="store-dataframe"),
])# endof html.Div#top-layer

# ##############################                            ############################## #
#                               dashboard callback functions                               #
# ##############################                            ############################## #
# callback which is used to send data to the ML/AI 
@app.callback(dd.Output('input-from-user', 'children'), dd.Input('submit-text', 'n_clicks'), [dd.State("input-textarea-research","value")])
def btn_submit_text(n_clicks, value):

	label = classify.news_classifier(value)
	label = label[0]['label']
	if label == 'LABEL_0':
		label = 'Disinformation'
	else:
		label = 'Verified Information'


	# Use this block for gpt
	# real_info = classify.generate( value )
	# return 'Classification of "{}" is ************{}.************ The truth about this subject is: {}.'.format(value, label, real_info)


	# Simply classification
	return 'The input text of {} is classified as ***{}*** '.format(value, label)




# callback to create and update WordCloud
@app.callback(dd.Output('image_wc', 'src'),[dd.Input('update-csv-data', 'n_intervals')])
def make_image_wc_update(b):
	df = pd.read_csv("dis_results.csv", error_bad_lines=False, engine="python")
	# Dataframe for WordCloud
	dfWords_tmps=df['wordSentences'].str.split(r" ", expand=False)
	dfWords = pd.DataFrame({"words" : []})
	ndfWords = np.array([])
	for i in range(len(dfWords_tmps)):
		for j in range(len(dfWords_tmps[i])):
			ndfWords = np.append(ndfWords, dfWords_tmps[i][j])
	dfWords = pd.DataFrame(ndfWords, columns=['Words'])
	dfWords_counts = pd.DataFrame(dfWords['Words'].value_counts(dropna=True, sort=True))
	dfWords_counts = dfWords_counts.reset_index()
	dfWords_counts.columns = ['unique_words', 'counts_words']
	for i in range(len(bridge.REMOVED_TEXT)):
		print()
		dfWords_counts = dfWords_counts[dfWords_counts.unique_words != bridge.REMOVED_TEXT[i]]

	img = BytesIO()
	generate_wordcloud(dataframe=bridge.dfWords_counts).save(img, format='PNG')
	return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


# callback to create and update figure
@app.callback(dd.Output("line-graph-visu", "figure"), dd.Input("update-csv-data", "n_intervals"))
def update_figure_visualization(b):
	df = pd.read_csv("dis_results.csv", error_bad_lines=False, engine="python")
	# split value between Fake & Real
	dfFake = df[df['disInformationStatus'] == "Fake"].reset_index()
	dfFake = dfFake.drop(columns=['index'])
	dfFake = dfFake.sort_values(by=['datePublication'], ascending=False)

	dfReal = df[df['disInformationStatus'] == "Real"].reset_index()
	dfReal = dfReal.drop(columns=['index'])
	dfReal = dfReal.sort_values(by=['datePublication'],ascending=False)

	fig_line = go.Figure(data=[go.Scatter(x=bridge.df_counts['unique_date'], y=bridge.df_counts['counts_date'], name="Total info"),
		go.Scatter(x=bridge.dfFake_counts['unique_date'], y=bridge.dfFake_counts['counts_date'], name="Fake info"),
		go.Scatter(x=bridge.dfReal_counts['unique_date'], y=bridge.dfReal_counts['counts_date'], name="Real info"),
		], layout={"legend" : {"title" : "legend"}, "title" : "Graphic of Number of messages per days (total, Fake, Real)"})

	return fig_line

# update data regarding Interval / timer 
@app.callback(dd.Output("store-dataframe", "data"),Trigger("update-csv-data", "n_intervals"), memoize=True)
def update_dataframe(b):
	print("update dataframe")
	#df = pd.read_csv("/home/kali/TIDE/Dashboard/data/dis_results.csv", error_bad_lines=False, engine="python")
	return datetime.datetime.now() #


# @app.callback(dd.Output("log-events", "children"), dd.Input("store-dataframe", "data"))
# def show_log_events(b):
# 	return f"Data were collected at {b}, with a current time is {datetime.datetime.now()}"

# ##############################                     ############################## #
#                               dashboard entry point                               #
# ##############################                     ############################## #
if __name__ == '__main__':
	app.run_server(debug=True, port=8050, host='0.0.0.0')


# ##############################                  ############################## #
#                               weblink references                               #
# ##############################                  ############################## #
# https://github.com/PrashantSaikia/Wordcloud-in-Plotly
# https://plotly.com/python/line-charts/
# https://dash.plotly.com/