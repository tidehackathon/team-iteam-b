i-TEAM-B:  Disinformation Analyser
A tool for identifying and classifying disinformation related to the War in Ukraine.  We provide a solution for identifying disinformation in news and twitter sources.

************************************
Getting Started
************************************
To get everything up and running, you'll need python3!  Everything else will depend on your environment and which packages you have installed, but at the very least, you'll need the following:

To run the dashboard:
Pandas, HuggingFace Transformers, dash, dash_extensions, wordcloud, and plotly
Trained models.  As the code is written now, they will automatically download the models from our repo on HuggingFace. If you cannot access them, following the directions below under "Discussion of engineering and design/Our models are hosted at:"


To train additional models and do additional data analytics:
All of the above, PLUS sklearn, numpy and the below data analytics instructions
You'll also need to download the data we've posted in the data file on Azure (https://tidedatastore.file.core.windows.net/tide-data/DisInformation-Challenge-Data/iteam_b_models)
Put this in a folder in your base directory called 'data'.  These files are very big, some over 500 MB, so you'll probably want 

Our entry point is dashboard.py
Running this will bring up the web server at 0.0.0.0:8050

The remainder of the APIs are standalone.


************************************
Discussion of engineering and design
************************************
Back-end:  
Python, with supporing JSON, csv, and binaries.  
Significant dependencies include pytorch, pandas, and the hugging face Transformers API (https://huggingface.co/docs/transformers/index)
_____________
AI/ML Models:  
Retrained open-source, pre-trained ROBERTA model for disinformation (https://huggingface.co/jy46604790/Fake-News-Bert-Detect).  
We then adapted this by re-training (sometimes called fine-tuning) this model on provided and open-sources information/disinformation data sets.  This produced a Twitter and a News model, which can then be used for classification.

Our models are hosted at:
	- HuggingFace: https://huggingface.co/brianthehardy
	- Azure:  https://tidedatastore.file.core.windows.net/tide-data/DisInformation-Challenge-Data/iteam_b_models
To load them, include something like the following.  Our codebase will automatically download these!  No need to download them yourself from huggingface or Azure!
	news_model = training_own.get_model('brianthehardy/ROBERTA_ukrwar')
	twitter_model = training_own.get_model('brianthehardy/ukr_twitter')
	If this doesn't work, or you want offline-only models, download the folders and place them in a folder called 'trained_models' in the same directory as main.py/dashboard.py
	At the minimum, your filepath will need to include trained_models/trained/trained_02_22_2023_size_400 and rained_models/trained/twitter_trained_02_22_2023_size_20
_________
Frontend:  
HTML and CSS, developed using dash.  Incorporates wordcloud and plotly extensions


************************************
Futher Discussion of ROBERTA:
************************************
	ROBERTA is a Natural Language Processing model based on BERT.  Its primary innovation is the use of Transformers, which are deep-learning models similar to Recurrent Neural Networks, but with a few key advantages.  They are able to process texts of up to 512 word at a time.  This is key for understanding how sentences unfold, and allows the model to train on context--not just the contained words.  It's also able to draw links between sentences, even if they are separated by many paragraphs, which gives it a huge advantage over Long Short Term Memory (LSTM) and .  As news is written in a manner where the first 2-3 paragraphs are most important, 512 words is more than sufficient.  
