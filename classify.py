# Functions for classification

import training_own, data_work
from pathlib import Path
from transformers import pipeline
import pandas as pd

# Pipelines are made of:
# A tokenizer in charge of mapping raw textual input to token.
# A model to make predictions from the inputs.
# Some (optional) post-processing for enhancing model’s output.

# "image-to-text": will return a ImageToTextPipeline.
# "token-classification" (alias "ner" available): will return a TokenClassificationPipeline.

# Initialize:  get model from local or online and set up a pipeline for classification
news_model = training_own.get_model('trained_models/trained_02_21_2023_size_400')
tokenizer = training_own.get_tokenizer()
news_classifier = pipeline("text-classification", tokenizer=tokenizer, model=news_model)

# Pipeline for twitter classification
twitter_model = training_own.get_model('trained_models/twitter_trained_02_22_2023_size_20')
tokenizer = training_own.get_tokenizer()
twitter_classifier = pipeline("text-classification", tokenizer=tokenizer, model=twitter_model)


# Given that something is news, run it through the news classifier
# LABEL_0: Fake news
# LABEL_1: Real news
def classify_news( text ):
    results = news_classifier(text, truncation=True)

    # Batch processing
    if type(text) is list:
        return results

    # Single processing
    else:
        if results == "LABEL_0": return 'fake'
        else: return 'true'

# Given that something is a tweet, run it through the tweet classifier
# LABEL_0: Fake news
# LABEL_1: Real news
def classify_tweet( text ):

    # Run through classifier
    results = twitter_classifier( text, truncation=True )

    # If batch, return batch
    if type(text) is list:
        return results

    # If single string, return string
    else:
        if results == "LABEL_0": return 'fake'
        else: return 'true'



'''
PROTOTYPE NEWS CLASSIFIER:
# News Classifier
    # Classifying the EU vs Disinfo dataset
    # Get the EU vs Disinfo dataset
    dis_df = data_work.get_vs()
    true_df = data_work.get_vs()

    dis_text = dis_df['Disinformation'].head(10).tolist()
    true_text = true_df['Information'].head(10).tolist()

    # Run through classifier
    dis_results = pd.DataFrame(news_classifier(dis_text, truncation=True)).label.map({'LABEL_0': 'Fake', 'LABEL_1': 'Real'})
    true_results = pd.DataFrame(news_classifier(true_text, truncation=True)).label.map({'LABEL_0': 'Fake', 'LABEL_1': 'Real'})

    output_cols = ['Date', 'Outlets', 'Disinformation', 'label', 'Title']
    output_format = ['datePublication', ' originPublication', 'textPublication', 'disInformationStatus',
                     'wordSentences']
    dis_results = dis_df.head(10).join(dis_results)[output_cols].set_axis(output_format, axis=1)
    true_results = true_df.head(10).join(true_results)[output_cols].set_axis(output_format, axis=1)
    
    @TODO: OUTPUT WITHOUT INDEX
    
    # datePublication,originPublication,textPublication,disInformationStatus,wordSentences
    print(dis_results)
    filepath = Path('dis_results.csv')
    dis_results.to_csv(filepath)
    
    print (true_results)
    filepath = Path('inf_results.csv')
    true_results.to_csv(filepath)
'''




'''
PROTOTYPE for Guardian classification (used for initial validation)
# Import some dataset
text = data_work.get_guardian()['articles'].head(20).tolist()

# Run through classifier
retvals = clf( text, truncation=True )

# Export
retvals = pd.DataFrame(retvals).label.map({'LABEL_0':'Fake', 'LABEL_1':'Real'} )
# retvals['label'] = retvals
filepath = Path('classified_news.csv')
retvals.to_csv(filepath)

'''