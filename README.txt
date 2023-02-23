i-TEAM-B:  Disinformation Analyser
A tool for identifying and classifying disinformation related to the War in Ukraine.  We provide a solution for identifying disinformation in news and twitter sources.


Overall Schematic:
Back-end:  
Python, with supporing JSON, csv, and binaries.  
Significant dependencies include pytorch, pandas, and the hugging face Transformers API (https://huggingface.co/docs/transformers/index)

AI/ML Models:  
Retrained open-source, pre-trained ROBERTA model for disinformation (https://huggingface.co/jy46604790/Fake-News-Bert-Detect).  
We then adapted this by re-training (sometimes called fine-tuning) this model on provided and open-sources information/disinformation data sets.  This produced a Twitter and a News model, which can then be used for classification.

Our models are hosted at:
	- HuggingFace: https://huggingface.co/brianthehardy
	- Azure:  https://tidedatastore.file.core.windows.net/tide-data/DisInformation-Challenge-Data/iteam_b_models
To load them, include something like the following.  Our codebase will automatically download these!  No need to download them yourself from huggingface or Azure!
	news_model = training_own.get_model('brianthehardy/ROBERTA_ukrwar')
	twitter_model = training_own.get_model('brianthehardy/ukr_twitter')

Frontend:  HTML and CSS

Futher Discussion of ROBERTA:
	ROBERTA is a Natural Language Processing model based on BERT.  Its primary innovation is the use of Transformers, which are deep-learning models similar to Recurrent Neural Networks, but with a few key advantages.  They are able to process texts of up to 512 word at a time.  This is key for understanding how sentences unfold, and allows the model to train on context--not just the contained words.  It's also able to draw links between sentences, even if they are separated by many paragraphs, which gives it a huge advantage over Long Short Term Memory (LSTM) and .  As news is written in a manner where the first 2-3 paragraphs are most important, 512 words is more than sufficient.  
