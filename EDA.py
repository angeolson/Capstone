# imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
from collections import Counter
import regex as re
import nltk
#nltk.download('stopwords')
#nltk.download('cmudict')
from nltk.corpus import stopwords
from nltk.corpus import cmudict
from sklearn.feature_extraction.text import CountVectorizer

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
    '''
    pipeline for lines. removes select punctuation and excess whitespace, tokenizes by word (space)
    :param sent: line
    :return: word tokenized line
    '''
    pat = re.compile(r"([.()!?,:;/-])")
    new_sentence = pat.sub(" \\1 ", sent)
    new_sentence = re.sub(r'\s+', ' ', new_sentence)
    return new_sentence.split(" ")

def versesTransform(verses):
    '''
    runs each line of a verse through the pipeline, then appends each word individually to one phrase per song
    :param verses:
    :return: one phrase per song
    '''
    full_list = []
    for verse in verses:
        sentence_list = [word for sentence in verse for word in sentencePipe(sentence)]
        [full_list.append(word) for word in sentence_list]
    return full_list

def wordCount(verses):
    '''
    word count for a song (unique words)
    :param verses: single phrase verse
    :return: length Counter dict for a song
    '''
    verse_counter = Counter([item.lower() for item in verses])
    verse_words = len(verse_counter)
    return verse_words

def get_top_n_ngram(corpus, n=None, ngram=1):
    '''
    adapted from https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
    :param corpus:
    :param n: int, how many words to return
    :param ngram: int, how long of an ngram to search for. Default = 1
    :return: tuple of ngrams and frequencies
    '''
    vec = CountVectorizer(ngram_range=(ngram, ngram)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def rhyme(inp, level):
    '''
    adapted from https://stackoverflow.com/questions/25714531/find-rhyme-using-nltk-in-python
    :param inp:
    :param level:
    :return:
    '''
    entries = cmudict.entries()
    syllables = [(word, syl) for word, syl in entries if word == inp]
    rhymes = []
    for (word, syllable) in syllables:
        rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]
    return set(rhymes)

def doTheyRhyme(word1, word2, level=3):
    '''
    adapted from https://stackoverflow.com/questions/25714531/find-rhyme-using-nltk-in-python
    :param word1: string
    :param word2: string
    :param level: nltk rhyme level, default is 3 (includes slant rhymes)
    :return: boolean
    '''
    return word1 in rhyme(word2, level)

def versesRhymeTransform(verses_transformed):
    '''
    splits a verse line by line while removing excess space, punctuation, for use in determining rhyme structure
    :param verses_transformed: 'verses_transformed' column of dataframe
    :return: song split line by line, not by verses.
    '''
    typedict = {'verse': '<VERSE>',
                'chorus': '<CHORUS>',
                'pre chorus': '<PRECHORUS>',
                'bridge': '<BRIDGE>',
                'outro': '<OUTRO>',
                'intro': '<INTRO>',
                'refrain': '<REFRAIN>',
                'hook': '<HOOK>',
                'post chorus': '<POSTCHORUS>',
                'other': '<OTHER>'
                }
    punctuation_spaces = ['(', '[', '.', '(', ')', '!', '?', ',', ':', ';', '/', '-', ']', ')', ' ', '']
    result = [item for item in verses_transformed if item not in typedict.values()]
    result = split_list([item for item in result if item not in punctuation_spaces],'<NEWLINE>')
    return result

def getSongRhyme(verses_transformed, level, option='AA'):
    '''
    gets the rhyme score of a song by comparing the last words of each line to the line proceeding it (when available).
    Divides the total number of rhymes by the total number of possible rhymes, returning a score between 0 (no rhymes) and 1 (either follows AA BB .. pattern perfectly or ABAB pattern perfectly)
    :param verses_transformed:
    :param level:
    :return:
    '''
    verses = [item[-1] for item in versesRhymeTransform(verses_transformed)]
    if option == 'AA':
        max_rhymes = len(verses) - 1
        list_ = []
        for i in range(1, len(verses)):
            list_.append(doTheyRhyme(verses[i-1], verses[i], level=level))
        return sum(list_)/max_rhymes
    else:
        max_rhymes = (len(verses)/2) - 1
        list_ = []
        for i in range(2, len(verses)):
            list_.append(doTheyRhyme(verses[i-2], verses[i], level=level))
        return sum(list_)/max_rhymes

def split_list(input_list,seperator):
    '''
    taken from https://stackoverflow.com/questions/30538436/how-to-to-split-a-list-at-a-certain-value
    :param input_list:
    :param seperator:
    :return:
    '''
    outer = []
    inner = []
    for elem in input_list:
        if elem == seperator:
            if inner:
                outer.append(inner)
            inner = []
        else:
            inner.append(elem)
    if inner:
        outer.append(inner)
    return outer

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

# 5: most common words, with and without stop words (unigrams)
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
ax.set(title='20 Most Common Words (Stopwords Removed)')
ax.set_ylabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.show()


#7: bigrams
bigrams = get_top_n_ngram(df['EDA_verses'].explode(), 20, 2)
x = []
y = []
for word, freq in bigrams:
    x.append(word), y.append(freq)
ax = sns.barplot(x=x, y=y)
sns.set(rc={"figure.figsize":(6, 7)})
ax.set(title='20 Most Common Bigrams')
ax.set_ylabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
plt.show()

#8: trigrams
# trigrams = get_top_n_ngram(df['EDA_verses'].explode(), 20, 3)
# x = []
# y = []
# for word, freq in trigrams:
#     x.append(word), y.append(freq)
# ax = sns.barplot(x=x, y=y)
# sns.set(rc={"figure.figsize":(6, 7)})
# ax.set(title='20 Most Common Trigrams')
# ax.set_ylabel('')
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
# ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
# plt.show()

#9: rhyme distribution

df['rhymescore_AA'] = df['verses_transformed'].apply(getSongRhyme, args=(2, 'AA'))
df['rhymescore_AB'] = df['verses_transformed'].apply(getSongRhyme, args=(2, 'AB'))
print('done!')

ax = sns.histplot(data=df, x='rhymescore_AA', kde=True)
ax.set_ylabel('')
ax.set_xlabel('Length')
ax.set(title='Song Rhyme Score Distribution')
plt.show()

ax = sns.histplot(data=df, x='rhymescore_AB', kde=True)
ax.set_ylabel('')
ax.set_xlabel('Length')
ax.set(title='Song Rhyme Score Distribution')
plt.show()
