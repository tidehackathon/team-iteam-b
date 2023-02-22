from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, TrainingArguments, Trainer, pipeline, RobertaForSequenceClassification
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
def train( path_to_write, size ):
    # Dataset for downstream finetuning.  Compilation of NYTimes, Guardian, and Fake news
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

# Used for training a model on twitter data
def train_twitter(path_to_write, size = 20):
    # Dataset for downstream finetuning.  Compilation of NYTimes, Guardian, and Fake news
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



def get_model( location = None ):
    if location is not None:
        MODEL = location
        bert_model = RobertaForSequenceClassification.from_pretrained(MODEL)
        return bert_model
    else:
        MODEL = "jy46604790/Fake-News-Bert-Detect"
        bert_model = RobertaForSequenceClassification.from_pretrained(MODEL)
        return bert_model
    return

def get_tokenizer():
    return RobertaTokenizer.from_pretrained('jy46604790/Fake-News-Bert-Detect')

# Train a news model
def train_news():
    input_data_size = 400
    output_name = 'trained_models/trained_' + datetime.now().strftime("%m_%d_%Y_size_") + str(input_data_size)
    train( output_name, input_data_size )

# Train a model for Twitter Data
def train_twitter():
    input_data_size = 20
    output_name = 'trained_models/twitter_trained_' + datetime.now().strftime("%m_%d_%Y_size_") + str(input_data_size)
    train( output_name, input_data_size )

