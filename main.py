from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, TrainingArguments, Trainer, pipeline, RobertaForSequenceClassification
import pandas as pd
import numpy as np
import evaluate
from pathlib import Path
import data_work, training_own, classify


# News Classifier
# Testing by classifying the EU vs Disinfo dataset
# Get the EU vs Disinfo dataset
dis_df = data_work.get_vs()
true_df = data_work.get_vs()

# Extract the Information and Disinformation columns
dis_text = dis_df['Disinformation'].head(400).tolist()
true_text = true_df['Information'].head(400).tolist()

# Classify each one
dis_results = pd.DataFrame(classify.classify_news( dis_text ))
true_results = pd.DataFrame(classify.classify_news( true_text ))

# Relabel some of the columns
dis_results['label'] = dis_results.label.map({'LABEL_0':'Fake', 'LABEL_1':'Real'} )
true_results['label'] = true_results.label.map({'LABEL_0':'Fake', 'LABEL_1':'Real'} )
# output_cols = ['Date', 'Outlets', 'Disinformation', 'label', 'Title']
output_format = ['datePublication' ,' originPublication' , 'textPublication' , 'disInformationStatus', 'wordSentences']

print(dis_results)

# Write out to a CSV
dis_results = dis_df.head(400).join(dis_results)[['Date', 'Outlets', 'Disinformation', 'label', 'Title']]
dis_results = dis_results.set_axis( output_format, axis=1)
true_results = true_df.head(400).join(true_results)[['Date', 'Outlets', 'Information', 'label', 'Title']]
true_results = true_results.set_axis( output_format, axis=1)

print(dis_results)
filepath = Path('dis_results.csv')
dis_results.to_csv(filepath, index = False)

print (true_results)
filepath = Path('inf_results.csv')
true_results.to_csv(filepath, index = False)

# Quantification of how well we did
dis_results = pd.read_csv('dis_results.csv')
inf_results = pd.read_csv('inf_results.csv')

print( dis_results['disInformationStatus'].value_counts() )
print( inf_results['disInformationStatus'].value_counts() )

inf_results = inf_results[inf_results['disInformationStatus'] == 'Fake']['textPublication']
filepath = Path('false_negative.csv')
inf_results.to_csv(filepath, index = False)






# print( classify.classify_news("Ukraine is the aggressor in the conflict") )


# LABEL_0: Fake news
# LABEL_1: Real news