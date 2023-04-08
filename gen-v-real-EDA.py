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
    '''
    # define function to clean the data
    # 1) replace the ' ##' in BERT word-part tokenization with ''
    # 2) strip <SONGBREAK>, [SEP], [BOS], [EOS], <newline>
    # 3) create column that contains count of [EOS] tokens in generated songs, to see how many stopped from learning and not design
    # 4) from cleaned column, also create column that's a list of the tokens in each song
    :param df:
    :return: df
    '''
    df['stripped_song'] = df['song'].apply(helper1)
    df['EOS_count'] = df['song'].apply(helper2)
    df['tokenized_song'] = df['stripped_song'].apply(lambda x:x.split())
    df['song_length'] = df['tokenized_song'].apply(lambda x:len(x))
    df['word_count'] = df['tokenized_song'].apply(wordCount)
    return df

def get_top_n_ngram(corpus, n=None, ngram=1):
    '''
    adapted from https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
    :param corpus:
    :param n: int, how many words to return
    :param ngram: int, how long of an ngram to search for. Default = 1
    :return: tuple of ngrams and frequencies
    '''
    vec = CountVectorizer(ngram_range=(ngram, ngram), min_df=1).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

df = CleanData(df)
os.chdir(IMGPATH) # change directory to where images are

# 1: Scatter Histogram
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

# 2: most common words, with and without stop words (unigrams)

# with stop words
punctuation = ['?', '!', '-', ',', '.', '(', ')', '']
stop_words = stopwords.words('english') + punctuation

all_words_train = [item.lower() for item in df[df['gen_type'] == 'None']['tokenized_song'].explode() if item.lower() not in punctuation] # get all words
all_words_bert = [item.lower() for item in df[df['gen_type'] == 'lstm-bert']['tokenized_song'].explode() if item.lower() not in punctuation]

len_words_train = len(all_words_train) # get # of total words
len_words_bert = len(all_words_bert)

top_words_train  = Counter(all_words_train)
top_words_bert  = Counter(all_words_bert)

train_x = [item[0] for item in top_words_train.most_common(20)]
train_y = [item[1]/len_words_train for item in top_words_train.most_common(20)] #normalize

bert_x = [item[0] for item in top_words_bert.most_common(20)]
bert_y = [item[1]/len_words_bert for item in top_words_bert.most_common(20)]

train_top_20 = pd.DataFrame()
train_top_20['words'] = train_x
train_top_20['freq'] = train_y

bert_top_20 = pd.DataFrame()
bert_top_20['words'] = bert_x
bert_top_20['freq'] = bert_y

merged_top_20 = pd.merge(train_top_20, bert_top_20, on='words', how='outer', )
merged_top_20.fillna(value=0, inplace=True)
df_train = df_train.rename(columns={'freq_x': 'None', 'freq_y': 'lstm-bert'})
merged_melt = merged_top_20.melt('words', var_name='cols', value_name='vals')

fig = plt.figure()
ax = sns.barplot(x=merged_melt['words'], y=merged_melt['vals'], hue=merged_melt['cols'])
sns.set(rc={"figure.figsize":(6, 7)})
ax.set(title='20 Most Common Words')
ax.set_ylabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
fig.savefig('20_Most_Common_Words.png', bbox_inches='tight', dpi=200)
plt.show()


# without stop words
all_words_train = [item.lower() for item in df[df['gen_type'] == 'None']['tokenized_song'].explode() if item.lower() not in stop_words] # get all words
all_words_bert = [item.lower() for item in df[df['gen_type'] == 'lstm-bert']['tokenized_song'].explode() if item.lower() not in stop_words]

len_words_train = len(all_words_train) # get # of total words
len_words_bert = len(all_words_bert)

top_words_train  = Counter(all_words_train)
top_words_bert  = Counter(all_words_bert)

train_x = [item[0] for item in top_words_train.most_common(20)]
train_y = [item[1]/len_words_train for item in top_words_train.most_common(20)] #normalize

bert_x = [item[0] for item in top_words_bert.most_common(20)]
bert_y = [item[1]/len_words_bert for item in top_words_bert.most_common(20)]

train_top_20 = pd.DataFrame()
train_top_20['words'] = train_x
train_top_20['freq'] = train_y

bert_top_20 = pd.DataFrame()
bert_top_20['words'] = bert_x
bert_top_20['freq'] = bert_y

merged_top_20 = pd.merge(train_top_20, bert_top_20, on='words', how='outer', )
merged_top_20.fillna(value=0, inplace=True)
merged_top_20 = merged_top_20.rename(columns={'freq_x': 'None', 'freq_y': 'lstm-bert'})
merged_melt = merged_top_20.melt('words', var_name='cols', value_name='vals')

fig = plt.figure()
ax = sns.barplot(x=merged_melt['words'], y=merged_melt['vals'], hue=merged_melt['cols'])
sns.set(rc={"figure.figsize":(6, 7)})
ax.set(title='20 Most Common Words (Stop Words Removed)')
ax.set_ylabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
fig.savefig('20_Most_Common_Words.png', bbox_inches='tight', dpi=200)
plt.show()

# 3 most common bigrams
# train_bigrams = get_top_n_ngram(df[df['gen_type'] == 'None']['tokenized_song'].explode() , 20, 2)
# bert_bigrams = get_top_n_ngram(df[df['gen_type'] == 'lstm-bert']['tokenized_song'].explode() , 20, 2)
#
# x = []
# y = []
# type = []
#
# for word, freq in train_bigrams:
#     x.append(word), y.append(freq), type.append('None')
#
# for word, freq in bert_bigrams:
#     x.append(word), y.append(freq), type.append('bert-lstm')
#
# bigram_df = pd.DataFrame()
# bigram_df['word'] = x
# bigram_df['freq'] = y
# bigram_df['type'] = type
#
# fig = plt.figure()
# ax = sns.barplot(x=bigram_df['word'], y=bigram_df['freq'], hue=bigram_df['type'])
# sns.set(rc={"figure.figsize":(6, 7)})
# ax.set(title='20 Most Common Bigrams')
# ax.set_ylabel('')
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
# ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
# fig.savefig('20_Most_Common_Bigrams.png', bbox_inches='tight', dpi=200)
# plt.show()