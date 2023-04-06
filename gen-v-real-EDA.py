# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import regex as re
import nltk
#nltk.download('stopwords')
#nltk.download('cmudict')
from nltk.corpus import stopwords
from nltk.corpus import cmudict
from sklearn.feature_extraction.text import CountVectorizer
import os
import random

# set
sns.set_theme(style="whitegrid")
IMGPATH = '/home/ubuntu/Capstone/Final_Plots'

# import training dataset of real songs
df_train = pd.read_csv('training.csv', index_col=0)
df_train = df_train[['lyrics']] # remove other information, keep lyrics only
df_train = df_train.rename(columns={'lyrics': 'song'})
df_train['gen_type'] = 'None' # denotes not generated

# import songs predicted by the bert base model
df_bert = pd.read_csv('bert-lstm-gen.csv', index_col=0)

# concat into one df
df = df_train.append(df_bert)
df.reset_index(drop=True, inplace=True)


# define function to clean the data
# 1) replace the ' ##' in BERT word-part tokenization with ''
# 2) strip <SONGBREAK>, [SEP], [BOS], [EOS], <newline>
# 3) create column that contains count of [EOS] tokens in generated songs, to see how many stopped from learning and not design
# 4) from cleaned column, also create column that's a list of the tokens in each song

def helper1(row):
    row = str(row)
    remove_items = ['<SONGBREAK>', '[SEP]', '[BOS]', '[EOS]', '<newline>']
    for item in remove_items:
        row = row.replace(item, '')
    return row

def helper2(row):
    row_split = row.split()
    eos = [item for item in row_split if item == '[EOS]']
    return len(eos)

def wordCount(verses):
    '''
    word count for a song (unique words)
    :param verses: single phrase verse
    :return: length Counter dict for a song
    '''
    verse_counter = Counter([item.lower() for item in verses])
    verse_words = len(verse_counter)
    return verse_words

def CleanData(df):
    df['stripped_song'] = df['song'].apply(helper1)
    df['EOS_count'] = df['song'].apply(helper2)
    df['tokenized_song'] = df['stripped_song'].apply(lambda x:x.split())
    df['song_length'] = df['tokenized_song'].apply(lambda x:len(x))
    df['word_count'] = df['tokenized_song'].apply(wordCount)
    return df

df = CleanData(df)

# 4: Scatter Histogram
fig = plt.figure()
ax = sns.histplot(
    df, x="song_length", y="word_count", hue="gen_type",
    bins=50, discrete=(False, False), log_scale=(False, False),
    cbar=True, cbar_kws=dict(shrink=.75)
)
ax.set_ylabel('Word Count')
ax.set_xlabel('Length')
ax.set(title='Song Length and Word Count Distribution')
#fig.savefig('Song_Length_and_Word_Count_Distribution.png', bbox_inches='tight', dpi=200)
plt.show()