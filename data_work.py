'''

Script for reading and wrangling data

Supports the following datasets:
- The Guardian news articles, provided by Hackathon Sponsors
- NYTimes news articles, provided by Hackathon Sponsors
- LIAR dataset, an open-source dataset derived from a POLITIFACT.COM fact-checking database.  https://paperswithcode.com/dataset/liar
- LIAR ONE/ZERO, which modifies
- Vox Disinformation Library, an open-source dataset of Vox Media's coverage of Ukraine war disinformation.
    Discussion:  https://voxukraine.org/en/voxcheck-presents-propaganda-diary-a-database-of-russian-propaganda-in-the-italian-and-german-media/
    Location:  https://docs.google.com/spreadsheets/d/1j5JuUDCpc7T9cAXqHC7MOe8mBsOJiw1SH5JzoUxFIYk/edit#gid=0
- EU vs Disinformation Dataset, an EU-provided dataset debunking Russian disinformation about the war. https://euvsdisinfo.eu/disinformation-cases/
- Russian Twitter Propganda Dataset, an open-source dataset of tweets from known Russian actors and Western journalists.  https://www.kaggle.com/datasets/dariusalexandru/russian-propaganda-tweets-vs-western-tweets-war?resource=download&select=western_analysts_tweets.csv
    As of now, this dataset is NOT supported, due to some encoding issues in the original files
'''

import pandas as pd
from pathlib import Path

# Imports all the Guardian articles (provided by hackathon sponsors)
def get_guardian():
    loc = Path('data/Guardians_Russia_Ukraine.csv')
    ret_df = pd.read_csv(loc)
    return ret_df

# Imports all the NYTimes articles (provided by hackathon sponsors)
def get_nytimes():
    loc = Path('data/NYT_Russia_Ukraine.csv')
    ret_df = pd.read_csv(loc)
    return ret_df

# Imports the LIAR Dataset
# Available here:  https://paperswithcode.com/dataset/liar
def get_liar():
    loc = Path('data/LIAR_data/liar.csv')
    ret_df = pd.read_csv(loc)
    return ret_df

# Converts the above dataset into a binary true/false encoding
# versus the 6-point scale used by the original data set
def liar_one_zero():
    loc = Path('data/LIAR_data/liar_one_zero.csv')
    if Path.is_file(loc):
        return pd.read_csv(loc)
    all_data = get_liar()
    all_data['label'] = all_data.label.map(
        {'pants-fire': 0, 'false': 0, 'barely-true': 0, 'half-true': 1, 'mostly-true': 1, 'true': 1})
    all_data.to_csv(filepath)

    return all_data

# Imports the Russian Twitter Troll dataset, provided by hackathon sponsors
def get_rus_trolls():
    ret_df = pd.read_csv("data/rus_troll.csv", index_col=False, sep=',')
    return ret_df

# Imports the EU vs Disinformation dataset
# Further discussion, and full dataset, available here:
# https://euvsdisinfo.eu/disinformation-cases/
def get_vs():
    ret_df = pd.read_csv('data/euvsdisinfo_all_texts_rename.csv')
    return ret_df

# Database from Vox media "Propoganda Diary"
# Discussion:  https://voxukraine.org/en/voxcheck-presents-propaganda-diary-a-database-of-russian-propaganda-in-the-italian-and-german-media/
# Location:  https://docs.google.com/spreadsheets/d/1j5JuUDCpc7T9cAXqHC7MOe8mBsOJiw1SH5JzoUxFIYk/edit#gid=0
def get_vox_disinfo():
    ret_df = pd.read_csv('data/vox_disinfo.csv')
    return ret_df

# Creates the pre-training/fine-tuning dataset
# Limited to an overall size of 800 (400 truth, 400 disinfo) to maintain good split
def get_pretraining_dataset( size ):
    if size > 400: size = 400
    columns = ['text', 'label']

    # Add in EU vx Disinfo database?
    vs_df = get_vs()

    # Extract the Information and Disinformation columns
    dis_text = vs_df['Disinformation'].head(size)
    dis_text['label'] = 0
    inf_text = vs_df['Information'].head(size)
    inf_text['label'] = 1

    # Read in the Vox Disinformation Database, clean, and label
    fake_df = get_vox_disinfo()
    fake_df['label'] = 0
    fake_df = fake_df[['Disinfo_cases_en', 'label']].dropna()
    fake_df.columns = columns

    # Add in the NYTimes and Guardian databases, clean, and label
    true_df = pd.concat( [get_guardian(), get_nytimes()] ).dropna()
    true_df['label'] = 1
    true_df = true_df[['articles', 'label']]
    true_df.columns = columns

    ret_df = pd.concat( [fake_df.head(size), true_df.head(size), fake_df.head(size), true_df.head(size)] ).reset_index(drop=True)
    return ret_df

# Reads in information from Twitter
# Rus_troll is a dataset
# Twitter dataset needs to be reformatted, as it contains emojis and characters
# not compatible with UFT-8
# @TODO:  Reformat Twitter Data so that Pandas can read it in
def get_twitter_training_data( size = 400 ):

    raise Exception("Not implimented yet.  Sorry!")

    # id,conversation_id,created_at,date,time,timezone,user_id,username,name,place,tweet,language,mentions,urls,photos,replies_count,retweets_count,likes_count,hashtags,cashtags,link,retweet,quote_url,video,thumbnail,near,geo,source,user_rt_id,user_rt,retweet_id,reply_to,retweet_date,translate,trans_src,trans_dest
    twitter_dis_df = pd.read_csv('data/russian_propaganda_tweets2.csv', nrows=size)[['created_at', 'username', 'tweet']]
    twitter_dis_df = twitter_dis_df.set_axis( ['date', 'user', 'text'], axis=1)

    # id,conversation_id,created_at,date,time,timezone,user_id,username,name,place,tweet,language,mentions,urls,photos,replies_count,retweets_count,likes_count,hashtags,cashtags,link,retweet,quote_url,video,thumbnail,near,geo,source,user_rt_id,user_rt,retweet_id,reply_to,retweet_date,translate,trans_src,trans_dest
    twitter_true_df = pd.read_csv('data/western_analysts_tweets.csv', nrows=size).dropna().head(size)[['created_at', 'username', 'tweet']]
    twitter_true_df = twitter_true_df.set_axis(['date', 'user', 'text'], axis=1)

    # Add the labels
    twitter_dis_df['label'] = 0
    twitter_true_df['label'] = 1

    ret_df = pd.concat( [twitter_true_df, twitter_true_df] )

    return ret_df
    # LABEL_0: Fake news
    # LABEL_1: Real news

'''
# Spaghetti code for processing the liar dataset.  Not relevant to hackathon
# Import dataset
df_headers = [ 'idx', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'bt', 'f', 'ht', 'mt', 'pof', 'context']
train_df = pd.read_csv("data/LIAR_data/train.tsv", index_col=False, header = None, sep='\t', names = df_headers)[['label', 'statement']]
test_df = pd.read_csv("data/LIAR_data/test.tsv", index_col=False, header = None, sep='\t', names = df_headers)[['label', 'statement']]
validate_df = pd.read_csv("data/LIAR_data/valid.tsv", index_col=False, header = None, sep='\t', names = df_headers)[['label', 'statement']]
all_df = pd.concat([train_df, test_df, validate_df])

all_df['label'] = all_df.label.map({'pants-fire':0, 'false':1, 'barely-true':2, 'half-true':3, 'mostly-true':4, 'true':5})

# Print out dataset
filepath = Path('data/LIAR_data/all_data.csv')

if not Path.is_file(filepath):
    all_df.to_csv(filepath, index=False, index_label=False)

# all_df.to_csv(filepath, index=False, index_label=False)

'''

# Export as datePublication,originPublication,textPublication,disInformationStatus,wordSentences