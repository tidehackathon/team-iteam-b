'''
Script for running classifiers

Takes advantage of huggingface's open source Transformers class (https://huggingface.co/docs/transformers/main_classes/pipelines)

All of our models are availble on HuggingFace!  The below code will download, cache, and read them in!
Models are available in both of these locations:
    HuggingFace:  https://huggingface.co/brianthehardy
    Azure:  https://tidedatastore.file.core.windows.net/tide-data/DisInformation-Challenge-Data/iteam_b_models
    (put the whole folder into the 'trained_models' directory to use)
    File tree should be trained_models/trained_02_22_2023_size_400, for example


The workhorse here is the Pipeline class, which requires a tokenizer, model, and post-processing.
This allows for flexibility when training, testing, and deploying models
From the Pipeline documentation:
Pipelines are made of:
    A tokenizer in charge of mapping raw textual input to token.
    A model to make predictions from the inputs.
    Some (optional) post-processing for enhancing model’s output.

'''

import training_own, data_work
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelWithLMHead
import pandas as pd


'''
This block runs at initialization, as it takes a while to read in the models.
This includes reading in the model and tokenizer and initializing the pipeline
Models are saved in the 'trained_models' directory
'''

# Initialize:  get model from local or online and set up a pipeline for classification

# From local
news_model = training_own.get_model('trained_models/trained_02_22_2023_size_400')

tokenizer = training_own.get_tokenizer()
news_classifier = pipeline("text-classification", tokenizer=tokenizer, model=news_model)

# Pipeline for twitter classification
twitter_model = training_own.get_model('trained_models/twitter_trained_02_22_2023_size_20')
tokenizer = training_own.get_tokenizer()
twitter_classifier = pipeline("text-classification", tokenizer=tokenizer, model=twitter_model)

# To pull models from remote server
news_model = training_own.get_model('brianthehardy/ROBERTA_ukrwar')
twitter_model = training_own.get_model('brianthehardy/ukr_twitter')


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
# Prototype for generative text:

This takes in a news article, and then returns a factual summary that provides additional context

An example is below:

Input Article: Western efforts to arm Ukraine are useless, the new technologically advanced 
and expensive weapons are immediately captured by Russia and studied in a retro-engineering 
effort or sent to the front to fight against Ukraine. Recently, two French Caesar howitzers were captured and sent to 
Uralvagonzavod; a Ural's weapon factory, to be studied and used in Ukraine against Ukrainian forces. 
It was confirmed in France.Western countries should think twice before sending new equipment to Ukraine.  

article:  The use of "lethal" force against civilians is not allowed in the Geneva Conventions. 
This is a matter for the international community, but the UK should not send a message to Europe that its own armed 
forces will not be used to prevent the fighting and destruction in eastern Ukraine.
'''

'''
gen_model = 'ktrapeznikov/gpt2-medium-topic-news'
# gen_model = 'addy88/t5-qa-genrate-explain-context'
gen_model = training_own.get_model(gen_model)
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model)
model = AutoModelWithLMHead.from_pretrained(gen_model)



def generate( text ):


    topic = text
    prompt = gen_tokenizer(f"topic: {topic} article:", return_tensors="pt")
    out = model.generate(prompt["input_ids"], do_sample=True, max_length=500, early_stopping=True, top_p=.9)
    out = gen_tokenizer.decode(list(out.cpu()[0]))
    print(out)

    return out
'''




'''
PROTOTYPE NEWS CLASSIFIER:
This code was used for some early-on testing and is saved here only for reference
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
Only saved here for reference.
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