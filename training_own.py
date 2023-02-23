'''
Library for training Classifiers.  Takes advantage of huggingface's open source Trainer Class (https://huggingface.co/docs/transformers/main_classes/trainer)
Here is where we pull the Disinformatoin ROBERTA model from huggingface and fine-tune it to War news and Tweets

This will automatically read in the appropriate data from data_work.py and train.

The below hyperparameters have not been optimized, and provide a way forward for better training, especially on small
data sets in a time-constrained environment
'''

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, TrainingArguments, Trainer, RobertaForSequenceClassification
import numpy as np
import evaluate
import data_work
from datetime import datetime

# Initiate parameters for which pre-trained model we want to use
MODEL = "jy46604790/Fake-News-Bert-Detect"
bert_model = RobertaForSequenceClassification.from_pretrained(MODEL)
tokenizer = RobertaTokenizer.from_pretrained(MODEL)

def tokenize (to_tokenize):
    return tokenizer( to_tokenize, padding="max_length", truncation=True, max_length=512)

# Training a news model on NYTimes, Guardian, and the Vox Disinformation archive
# Maximum size is about 400 records from each source, for about 600 Real and 600 Fake news
# @TODO:  Refactor train, train_twitter, and train_news to DRYer code.  Lots of copy-and-pasting during development
def train( size ):
    # Dataset for downstream finetuning.  Compilation of NYTimes, Guardian, and Fake news
    path_to_write = 'trained_models/trained_' + datetime.now().strftime("%m_%d_%Y_size_") + str(size)
    finetuning_df = data_work.get_pretraining_dataset(size)
    finetuning_df['tokenized'] = finetuning_df['text'].apply( tokenize )
    finetuning_dataset = finetuning_df['tokenized'].values
    finetuning_labels = finetuning_df['label'].values

    for ii in range(len(finetuning_dataset)):
        finetuning_dataset[ii]['labels'] = finetuning_labels[ii]

    # Split into a test and train dataset
    train_dataset, test_dataset = train_test_split( finetuning_dataset, test_size = 0.2 )
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(output_dir=path_to_write, evaluation_strategy="epoch")

    trainer = Trainer(
        model = bert_model ,
        args = training_args,
        eval_dataset=test_dataset,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()

# Train a news model, utilizing above API
def train_news():
    input_data_size = 400
    output_name = 'trained_models/trained_' + datetime.now().strftime("%m_%d_%Y_size_") + str(input_data_size)
    train(output_name, input_data_size)

# Used for training a model on twitter data
# Limit is about 400 tweets, which is very small
# @TODO:  Find a larger dataset and spend the additional time training a more robust model
def train_twitter(size = 400):
    path_to_write = 'trained_models/trained_' + datetime.now().strftime("%m_%d_%Y_size_") + str(size)
    finetuning_df = data_work.get_twitter_training_data( size )
    finetuning_df['tokenized'] = finetuning_df['text'].apply(tokenize)
    finetuning_dataset = finetuning_df['tokenized'].values
    finetuning_labels = finetuning_df['label'].values

    for ii in range(len(finetuning_dataset)):
        finetuning_dataset[ii]['labels'] = finetuning_labels[ii]

    # Split into a test and train dataset
    train_dataset, test_dataset = train_test_split(finetuning_dataset, test_size=0.2)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(output_dir=path_to_write, evaluation_strategy="epoch")

    trainer = Trainer(
        model=bert_model,
        args=training_args,
        eval_dataset=test_dataset,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()

# Returns the model.  Used for reading in the models.
# @TODO:  Refactor this and move to the classify.py module
def get_model( location = None ):
    if location is not None:
        MODEL = location
        bert_model = RobertaForSequenceClassification.from_pretrained(MODEL)
        return bert_model
    else:
        MODEL = "jy46604790/Fake-News-Bert-Detect"
        bert_model = RobertaForSequenceClassification.from_pretrained(MODEL)
        return bert_model

# Returns the tokenizer.  Of note, the tokenizer must remain constant throughout pre-training, fine-tuning, and deployment
# @TODO:  Refactor this and move to the classify.py module
def get_tokenizer():
    return RobertaTokenizer.from_pretrained('jy46604790/Fake-News-Bert-Detect')



