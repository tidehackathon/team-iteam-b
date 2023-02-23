# import package
import base64
import dash.dependencies as dd
import numpy as np                                                            # pip install numpy
import pandas as pd                                                           # pip install pandas
import plotly.express as px
import plotly.graph_objs as go

from dash import Dash, html, dcc, Input, Output                               # pip install dash
from dash_extensions.enrich import Trigger, FileSystemStore, ServersideOutput # pip install dash_extensions
from io import BytesIO          
from PIL import Image           # pip install PIL
from wordcloud import WordCloud # pip install worldcloud

# Adding machine learning compenents
import training_own, classify, data_work

# global variables 
ALLOWED_TYPES = ("text", )
REMOVED_TEXT = ("a", "as", "at",
				"b", "be", "by",
				"c", "can"
				"d", "do",
				"e",
				"f", "for",
				"g",
				"h", "here",
				"i", "in", "into", "is", "it", "its",
				"j",
				"k",
				"l",
				"m",
				"n", "not", "now"
				"o", "on", "of",
				"p",
				"q",
				"r",
				"s",
				"t", "that", "the", "The", "their", "to",
				"u",
				"v",
				"was", "were", 'while', 'will',
				"x",
				"you",
				"z")

# create server side stor to hold data.

# fss = FileSystemStore(cache_dir="/home/kali/TIDE/Dashboard/data", default_timeout=10)


# csv file from ML/AI 
df = pd.read_csv("dis_results.csv", error_bad_lines=False, engine="python")
# split value between Fake & Real
dfFake = df[df['disInformationStatus'] == "Fake"].reset_index()
dfFake = dfFake.drop(columns=['index'])
dfFake = dfFake.sort_values(by=['datePublication'], ascending=False)

dfReal = df[df['disInformationStatus'] == "Real"].reset_index()
dfReal = dfReal.drop(columns=['index'])
dfReal = dfReal.sort_values(by=['datePublication'],ascending=False)


# split value regarding nbrMsg depending on dates
df_counts = pd.DataFrame(df['datePublication'].value_counts(dropna=True, sort=True))
df_counts = df_counts.reset_index()
df_counts.columns = ['unique_date', 'counts_date']
df_counts = df_counts.sort_values(by=['unique_date'])

dfFake_counts = pd.DataFrame(dfFake['datePublication'].value_counts(dropna=True, sort=True))
dfFake_counts = dfFake_counts.reset_index()
dfFake_counts.columns = ['unique_date', 'counts_date']
dfFake_counts = dfFake_counts.sort_values(by=['unique_date'])

dfReal_counts = pd.DataFrame(dfReal['datePublication'].value_counts(dropna=True, sort=True))
dfReal_counts = dfReal_counts.reset_index()
dfReal_counts.columns = ['unique_date', 'counts_date']
dfReal_counts = dfReal_counts.sort_values(by=['unique_date'])


# Dataframe for WordCloud
dfWords_tmps=df['wordSentences'].str.split(r" ", expand=False)
dfWords = pd.DataFrame({"words" : []})
ndfWords = np.array([])
for i in range(len(dfWords_tmps)):
	for j in range(len(dfWords_tmps[i])):
		ndfWords = np.append(ndfWords, dfWords_tmps[i][j])
# print(ndfWords)
# print(len(ndfWords))
dfWords = pd.DataFrame(ndfWords, columns=['Words'])
# print(dfWords)
# print(dfWords.value_counts())
dfWords_counts = pd.DataFrame(dfWords['Words'].value_counts(dropna=True, sort=True))
dfWords_counts = dfWords_counts.reset_index()
dfWords_counts.columns = ['unique_words', 'counts_words']
# print(dfWords_counts)
for i in range(len(REMOVED_TEXT)):
	# print()
	dfWords_counts = dfWords_counts[dfWords_counts.unique_words != REMOVED_TEXT[i]]

# Function which generate part of the dashboard layout
def generate_disinfo_real_table(dataframe, max_rows=10):
	return html.Div(children=[
		html.H2(children=["Information"]),
		html.Table(children=[
			html.Thead(
				html.Tr(children=[html.Th(col) for col in dataframe.columns]),
				style={'border':'1px solid black'}), # end of html.Thread
			html.Tbody([
				html.Tr([
					html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
					]) for i in range(min(len(dataframe), max_rows)) # end of html.Tr
				]) # end of html.Tbody
		])# end of html.Table
	], style={'textAlign':'center'}) # end of html.Div


def generate_disinfo_fake_table(dataframe, max_rows=100):
	return html.Div(children=[
		html.H2(children=["DisInformation"]),
		html.Table(children=[
			html.Thead(
				html.Tr(children=[html.Th(col) for col in dataframe.columns])
				,style={'border':'1px solid black'}), # end of html.Thread
			html.Tbody([
				html.Tr([
					html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
					], style={'backgroundColor':'#67768b','border':'1px solid black'}) for i in range(min(len(dataframe), max_rows)) # end of html.Tr
				], style= {
					'borderColor':'#616161',
					'border' : '1px solid black',
					'border-style': 'dotted'

			}) # end of html.Tbody
		], style={
			'border': "1px solid #616161",
		}) # end of html.Table
	], style={'textAlign': 'center', 'padding': 20})# end of html.Div


# Generate Line Graphic
fig_line = go.Figure(data=[go.Scatter(x=df_counts['unique_date'], y=df_counts['counts_date'], name="Total info"),
	go.Scatter(x=dfFake_counts['unique_date'], y=dfFake_counts['counts_date'], name="Fake info"),
	go.Scatter(x=dfReal_counts['unique_date'], y=dfReal_counts['counts_date'], name="Real info"),
	], layout={"legend" : {"title" : "legend"}, "title" : "Graphic of Number of messages per days (total, Fake, Real)"})


# Generate the WordCloud schema
def generate_wordcloud(dataframe):
	#dataframe = pd.DataFrame({'word': ['apple', 'pear', 'orange'], 'freq': [1,3,9]})

	d = {a: x for a, x in dataframe.values}
	wc = WordCloud(background_color='white', width=1080, height=360)
	wc.fit_words(d)
	return wc.to_image()

# app Dash info
app = Dash(__name__)

# dashboard layout
app.layout = html.Div(children=[
	#html.Img(src=app.get_asset_url('2023_TIDE_Hackathon_Logo_Horizontal.jpg'), alt='image', width="1920"), # end of html.Img
	html.H1(
		children='DisInformation Analyzer Challenge',
		style={
			'textAlign': 'center'
		}),
	html.Div(children="i Team B",
		style={
			'textAlign': 'center'
		}),
	html.Br(),
	
	# input from user 
	html.Div(children=[
		dcc.Input(id="input_{}".format(_),
			type=_,
			placeholder="input type{}".format(_),) for _ in ALLOWED_TYPES
		] + [html.Div(id="input-from-user"),
		html.Button('Submit', id='submit-text', n_clicks=0)
	]), # end of html.div # input from user

	# generate the graphic visulatization
	html.Div(children=[
		#generate_graphic_nbrs(df),
		dcc.Graph(id="line_graph_disinfo", figure=fig_line)
		],style={}), # end of html.Div # graph

	html.Br(),html.Br(),
	# generate the wordcloud view
	html.Img(id="image_wc"),

	# generate the table of fake & real news
	html.Div(children=[
		generate_disinfo_real_table(dfReal),
		generate_disinfo_fake_table(dfFake)
	], style={'display':'flex', 'flex-direction':'row', 'padding':10}), # html.Div # Table

	# timer which update dataframe uodate every minute
	# Original value is 1000
	dcc.Interval(id='update-csv-data', interval=50*1000, n_intervals=0),
	dcc.Store(id="store-dataframe")

])# app.layout


# callback regarding button which submit text
@app.callback(dd.Output('input-from-user', 'children'), dd.Input('submit-text', 'n_clicks'), [dd.State('input_{}'.format(_),"value") for _ in ALLOWED_TYPES])
def btn_submit_text(n_clicks, value):

	ret_val = classify.news_classifier( value )
	ret_val = ret_val[0]['label']



	if ret_val == 'LABEL_0': ret_val = 'Disinformation'
	else: ret_val = 'Verified Information'




	# Use this line for GPT return
	# real_info = classify.generate( value )
	# return 'Classification of "{}" is  ************{}.************ The truth about this subject is: {}.'.format(value, ret_val, real_info)


# callback regarding line graphic disinfo
@app.callback(dd.Output('line_graph_disinfo', 'figure'), [dd.Input('line_graph_disinfo', 'id')])
def make_line_graph_disinfo(b):
	df = px

# callback regarding wordcloud
@app.callback(dd.Output('image_wc', 'src'),[dd.Input('image_wc', 'id')])
def make_image_wc(b):
	#img = Image.open("/home/kali/TIDE/Dashboard/image_wc.png")
	img = BytesIO()
	#generate_wordcloud(dataframe=df).save(img, format='PNG')
	generate_wordcloud(dataframe=dfWords_counts).save(img, format='PNG')
	return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

# update data regarding Interval / timer 
@app.callback(dd.Output("store-dataframe", "data"),Trigger("update-csv-data", "n_intervals"), memoize=True)
def update_dataframe():
	print("update dataframe")

# entry point
if __name__ == '__main__':
	app.run_server(debug=True, port=8050, host='0.0.0.0')


# References : 
# https://github.com/PrashantSaikia/Wordcloud-in-Plotly
# https://plotly.com/python/line-charts/
# https://dash.plotly.com/