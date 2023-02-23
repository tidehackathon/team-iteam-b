'''
Entry point used mostly for development, testing, and training models
This file is not necessary for any of the function calls in the other supporting files.
'''

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, TrainingArguments, Trainer, pipeline, RobertaForSequenceClassification
import pandas as pd
from pathlib import Path
import data_work, training_own, classify

# Use for fine-tuning a model with twitter data
# training_own.train_twitter(5)


# print( classify.generate("biological weapons in ukraine") )

# Use for fine-tuning a news model
# training_own.train(2000)

'''
# News Classifier
# Testing by classifying the EU vs Disinfo dataset
# Get the EU vs Disinfo dataset
dis_df = data_work.get_vs()
true_df = data_work.get_vs()

# Extract the Information and Disinformation columns
dis_text = dis_df['Disinformation'].head(400).tolist()
true_text = true_df['Information'].head(400).tolist()

# Testing on the NY Times articles
nyt_df = data_work.get_nytimes()
nyt_text = nyt_df['articles']

# Classify each one
dis_results = pd.DataFrame(classify.classify_news( dis_text ))
true_results = pd.DataFrame(classify.classify_news( true_text ))
nyt_results = pd.DataFrame(classify.classify_news( true_text ))

# Post-process the results, changing LABEL_0 and LABEL_1 to show Real vs Fake news
dis_results['label'] = dis_results.label.map({'LABEL_0':'Fake', 'LABEL_1':'Real'} )
true_results['label'] = true_results.label.map({'LABEL_0':'Fake', 'LABEL_1':'Real'} )
nyt_results['label'] = nyt_results.label.map({'LABEL_0':'Fake', 'LABEL_1':'Real'} )

output_format = ['datePublication' ,' originPublication' , 'textPublication' , 'disInformationStatus', 'wordSentences']

print(dis_results)

# Write out to a CSV
dis_results = dis_df.head(400).join(dis_results)[['Date', 'Outlets', 'Disinformation', 'label', 'Title']]
dis_results = dis_results.set_axis( output_format, axis=1)
true_results = true_df.head(400).join(true_results)[['Date', 'Outlets', 'Information', 'label', 'Title']]
true_results = true_results.set_axis( output_format, axis=1)
nyt_results = nyt_df.head(400).join(nyt_results)[['published', 'headlines', 'articles', 'label']]

print(nyt_results)

nyt_results = nyt_results.set_axis( ['datePublication', 'wordSentences', 'textPublication', 'disInformationStatus'], axis=1)

print(dis_results)
filepath = Path('dis_results.csv')
dis_results.to_csv(filepath, index = False)

print (true_results)
filepath = Path('inf_results.csv')
true_results.to_csv(filepath, index = False)

print (nyt_results)
filepath = Path('nyt_results.csv')
nyt_results.to_csv(filepath, index = False)
'''



'''
# Quantification of how well we did
# Using Guardian, NYTimes, and eu vs disinfo databases

dis_results = pd.read_csv('dis_results.csv')
inf_results = pd.read_csv('inf_results.csv')
nyt_results = pd.read_csv('nyt_results.csv')

print( dis_results['disInformationStatus'].value_counts() )
print( inf_results['disInformationStatus'].value_counts() )
print( nyt_results['disInformationStatus'].value_counts() )

inf_results = inf_results[inf_results['disInformationStatus'] == 'Fake']['textPublication']
filepath = Path('false_negative.csv')
inf_results.to_csv(filepath, index = False)
'''



# LABEL_0: Fake news
# LABEL_1: Real news

# Token:  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

'''
To integrate with dashboard:
# Adding machine learning compenents
import training_own, classify, data_work

def btn_submit_text(n_clicks, value):

	ret_val = classify.news_classifier( value )
	ret_val = ret_val[0]['label']
	if ret_val == 'LABEL_0': ret_val = 'Disinformation'
	else: ret_val = 'Verified Information'
	# ret_val = ret
	# if ret_val[0]
	#
	# [{'label': 'LABEL_0', 'score': 0.9999428987503052}]


	return 'Classification of "{}" is {}. The button has been clicked {} times.'.format(value, ret_val, n_clicks)
	

'''