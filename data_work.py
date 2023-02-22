# File for reading and wrangling data
import pandas as pd
from pathlib import Path

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

#
def get_data_df():
    return all_df

#
def get_data_path():
    return filepath

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
    dis_text = vs_df['Disinformation'].head(size).tolist()
    dis_text['label'] = 0
    inf_text = vs_df['Information'].head(size).tolist()
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

    ret_df = pd.concat( [fake_df.head(size), true_df.head(size)] ).reset_index(drop=True)
    return ret_df

# Reads in information from Twitter
# Rus_troll is a dataset
def get_twitter_training_data( size = 400 ):

    # id,conversation_id,created_at,date,time,timezone,user_id,username,name,place,tweet,language,mentions,urls,photos,replies_count,retweets_count,likes_count,hashtags,cashtags,link,retweet,quote_url,video,thumbnail,near,geo,source,user_rt_id,user_rt,retweet_id,reply_to,retweet_date,translate,trans_src,trans_dest
    twitter_dis_df = pd.read_csv('data/russian_propaganda_tweets.csv', delimiter='\t', nrows=400)[['created_at', 'username', 'tweet']].head(size)
    twitter_dis_df = twitter_dis_df.set_axis( ['date', 'user', 'text'], axis=1)

    # id,conversation_id,created_at,date,time,timezone,user_id,username,name,place,tweet,language,mentions,urls,photos,replies_count,retweets_count,likes_count,hashtags,cashtags,link,retweet,quote_url,video,thumbnail,near,geo,source,user_rt_id,user_rt,retweet_id,reply_to,retweet_date,translate,trans_src,trans_dest
    twitter_true_df = pd.read_csv('data/western_analysts_tweets.csv', nrows=400).dropna().head(size)[['created_at', 'username', 'tweet']]
    twitter_dis_df = twitter_dis_df.set_axis(['date', 'user', 'text'], axis=1)

    # Add the labels
    twitter_dis_df['label'] = 0
    twitter_true_df['label'] = 1

    ret_df = pd.concat( [twitter_true_df, twitter_true_df] )

    return ret_df
    # LABEL_0: Fake news
    # LABEL_1: Real news

# Export as datePublication,originPublication,textPublication,disInformationStatus,wordSentences