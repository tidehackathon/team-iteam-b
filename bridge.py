import pandas as pd
import numpy as np

REMOVED_TEXT = ("a", "and", "are", "as", "at",
				"b", "be", "by",
				"c", "can"
				"d", "do",
				"e",
				"f", "for",
				"g",
				"h", "here",
				"i", "in", "into", "is", "it", "its",
				"j",
				"k",
				"l",
				"m",
				"n", "not", "now"
				"o", "on", "of",
				"p",
				"q",
				"r",
				"s",
				"t", "that", "the", "The", "their", "to",
				"u",
				"v",
				"was", "were", "what", 'while', 'will',
				"x",
				"you",
				"z")

# csv file from ML/AI 
df = pd.read_csv("dis_results.csv", error_bad_lines=False, engine="python")
# split value between Fake & Real
dfFake = df[df['disInformationStatus'] == "Fake"].reset_index()
dfFake = dfFake.drop(columns=['index'])
dfFake = dfFake.sort_values(by=['datePublication'], ascending=False)

dfReal = df[df['disInformationStatus'] == "Real"].reset_index()
dfReal = dfReal.drop(columns=['index'])
dfReal = dfReal.sort_values(by=['datePublication'],ascending=False)


# split value regarding nbrMsg depending on dates
df_counts = pd.DataFrame(df['datePublication'].value_counts(dropna=True, sort=True))
df_counts = df_counts.reset_index()
df_counts.columns = ['unique_date', 'counts_date']
df_counts = df_counts.sort_values(by=['unique_date'])

dfFake_counts = pd.DataFrame(dfFake['datePublication'].value_counts(dropna=True, sort=True))
dfFake_counts = dfFake_counts.reset_index()
dfFake_counts.columns = ['unique_date', 'counts_date']
dfFake_counts = dfFake_counts.sort_values(by=['unique_date'])

dfReal_counts = pd.DataFrame(dfReal['datePublication'].value_counts(dropna=True, sort=True))
dfReal_counts = dfReal_counts.reset_index()
dfReal_counts.columns = ['unique_date', 'counts_date']
dfReal_counts = dfReal_counts.sort_values(by=['unique_date'])


# Dataframe for WordCloud
dfWords_tmps=df['wordSentences'].str.split(r" ", expand=False)
dfWords = pd.DataFrame({"words" : []})
ndfWords = np.array([])
for i in range(len(dfWords_tmps)):
	for j in range(len(dfWords_tmps[i])):
		ndfWords = np.append(ndfWords, dfWords_tmps[i][j])
print(ndfWords)
print(len(ndfWords))
dfWords = pd.DataFrame(ndfWords, columns=['Words'])
print(dfWords)
print(dfWords.value_counts())
dfWords_counts = pd.DataFrame(dfWords['Words'].value_counts(dropna=True, sort=True))
dfWords_counts = dfWords_counts.reset_index()
dfWords_counts.columns = ['unique_words', 'counts_words']
print(dfWords_counts)
for i in range(len(REMOVED_TEXT)):
	print()
	dfWords_counts = dfWords_counts[dfWords_counts.unique_words != REMOVED_TEXT[i]]