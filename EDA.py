# imports
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
from collections import Counter
import regex as re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

# helper functions
def cleanData(df):
    '''
    function cleans the original pandas dataframe imported from .csv and returns a clean frame
    :param df:
    :return: cleaned pandas dataframe
    '''
    for col in ['verses', 'verse_types', 'verses_transformed']:
        df[col] = df[col].apply(lambda x:eval(x, {'__builtins__': None}, {}))
    df['EDA_verses'] = df['verses'].apply(versesTransform)
    return df

def sentencePipe(sent):
    pat = re.compile(r"([.()!?,:;/-])")
    new_sentence = pat.sub(" \\1 ", sent)
    new_sentence = re.sub(r'\s+', ' ', new_sentence)
    return new_sentence.split(" ")

def versesTransform(verses):
    full_list = []
    for verse in verses:
        sentence_list = [word for sentence in verse for word in sentencePipe(sentence)]
        [full_list.append(word) for word in sentence_list]
    return full_list

def wordCount(verses):
    verse_counter = Counter([item.lower() for item in verses])
    verse_words = len(verse_counter)
    return verse_words

# clean df
df = pd.read_csv('df_cleaned.csv', index_col=0)
df = cleanData(df)

# create stats:

# 1: verse type counts
x = [item.capitalize() for item in df['verse_types'].explode().value_counts().index]
y = df['verse_types'].explode().value_counts()
ax = sns.barplot(x=x, y=y)
sns.set(rc={"figure.figsize":(6, 7)})
ax.set(title='Song Component Occurences')
ax.set_ylabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.show()

# 2: word count histogram
df['word_count'] = df['EDA_verses'].apply(wordCount)
ax = sns.histplot(data=df, x='word_count')
ax.set_ylabel('')
ax.set_xlabel('Word Count')
ax.set(title='Song Unique Word Count Distribution')
plt.show()

# 3: length histogram
df['length'] = df['EDA_verses'].apply(lambda x:len(x))
ax = sns.histplot(data=df, x='length', kde=True)
ax.set_ylabel('')
ax.set_xlabel('Length')
ax.set(title='Song Length Distribution')
plt.show()

# 4: for fun, both plotted together as a scatter histogram
ax = sns.histplot(
    df, x="length", y="word_count",
    bins=50, discrete=(False, False), log_scale=(False, False),
    cbar=True, cbar_kws=dict(shrink=.75)
)
ax.set_ylabel('Word Count')
ax.set_xlabel('Length')
ax.set(title='Song Length and Word Count Distribution')
plt.show()

# 5: most common words, with and without stop words
punctuation = ['?', '!', '-', ',', '.', '(', ')', '']
stop_words = stopwords.words('english') + punctuation

# with stop words
top_words = Counter([item.lower() for item in df['EDA_verses'].explode() if item.lower() not in punctuation])

x = [item[0] for item in top_words.most_common(20)]
y = [item[1] for item in top_words.most_common(20)]
ax = sns.barplot(x=x, y=y)
sns.set(rc={"figure.figsize":(6, 7)})
ax.set(title='20 Most Common Words')
ax.set_ylabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.show()


# without stop words
top_words_rem = Counter([item.lower() for item in df['EDA_verses'].explode() if item.lower() not in stop_words])
x = [item[0] for item in top_words_rem.most_common(20)]
y = [item[1] for item in top_words_rem.most_common(20)]
ax = sns.barplot(x=x, y=y)
sns.set(rc={"figure.figsize":(6, 7)})
ax.set(title='20 Most Common Words (Stopwords Removed')
ax.set_ylabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.show()

#6: unigrams

#7: bigrams

#8: trigrams

#9: rhyme distribution 