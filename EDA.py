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
import os
import random

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

def Convert(tup, di):
    '''
    from https://www.geeksforgeeks.org/python-convert-a-list-of-tuples-into-dictionary/
    :param tup:
    :param di:
    :return:
    '''
    for a, b in tup:
        di.setdefault(a, []).append(b)
    return di

def minimum(a, b):
    if a <= b:
        return a
    else:
        return b

def rhymeCheck(word1, word2, syllable_list, level=3):
    '''
    helper function for checking for a rhyme. Takes two words, a list of words/syllables from a word dictionary, and the level (number of syllables) that need to rhyme
    :param word1: str
    :param word2: str
    :param syllable_list: list
    :param level: int
    :return: boolean, do the words rhyme or not
    '''
    syllables = syllable_list
    dict_ = {}
    rhyme_dict = Convert(syllables, dict_) # create dict
    word1 = word1.lower().strip()
    word2 = word2.lower().strip()
    if (word1 in rhyme_dict.keys()) and (word2 in rhyme_dict.keys()):
        word1_syllables = rhyme_dict[word1][0]
        word2_syllables = rhyme_dict[word2][0]
        min_len = minimum(len(word1_syllables), len(word2_syllables))
        if min_len < level:
            level = min_len # in cases where one word is shorter and is shorter than set level of syllables, change level to smaller number
        if word1_syllables[-level:] == word2_syllables[-level:]:
            return True
        else:
            return False
    else:
        return False

def doTheyRhyme(word1, word2, level=3):
    '''
    adapted from https://stackoverflow.com/questions/25714531/find-rhyme-using-nltk-in-python
    :param word1: string
    :param word2: string
    :param level: nltk rhyme level, default is 3 (includes slant rhymes)
    :return: boolean
    '''
    return word1 in rhyme(word2, level)

def safeDivision(n,d):
    return n/d if d > 0 else 0

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

def getSongRhyme(verses_transformed, level, syllable_list, option='AA'):
    '''
    gets the rhyme score of a song by comparing the last words of each line to the line proceeding it (when available).
    Divides the total number of rhymes by the total number of possible rhymes, returning a score between 0 (no rhymes) and 1 (either follows AA BB .. pattern perfectly or ABAB pattern perfectly)
    :param verses_transformed:
    :param level:
    :return:
    '''
    verses = [item[-1] for item in versesRhymeTransform(verses_transformed)]
    if option == 'AA':
        max_rhymes = len(verses) // 2
        list_ = []
        i = 1 # init i
        while i < len(verses):
            #list_.append(doTheyRhyme(verses[i-1], verses[i], level=level))
            list_.append(rhymeCheck(verses[i - 1], verses[i], syllable_list, level=level)) # check if the line rhymes with the preceeding
            i += 2 # skip ahead to the next look-back lines

        return safeDivision(sum(list_), max_rhymes)
    else:
        max_rhymes = (len(verses) // 2) - 1
        list_ = []
        i = 2
        while i < len(verses):
            #list_.append(doTheyRhyme(verses[i-2], verses[i], level=level))
            list_.append(rhymeCheck(verses[i - 2], verses[i], syllable_list, level=level))
            i += 1  # skip ahead to the next look-back lines
            list_.append(rhymeCheck(verses[i - 2], verses[i], syllable_list, level=level))
            i += 3 # skip ahead to the next look-back lines
        return safeDivision(sum(list_), max_rhymes)

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

# vars
DATAPATH = '/home/ubuntu/Capstone/'
IMGPATH = '/home/ubuntu/Capstone/Plots'
entries = cmudict.entries()
syllable_list = [(word, syl) for word, syl in entries]
SEED = 48
random.seed(48)

# clean df
os.chdir(DATAPATH)
df = pd.read_csv('df_cleaned.csv', index_col=0)
df = cleanData(df)
df.reset_index(drop=True, inplace=True)
os.chdir(IMGPATH)
# create stats:

# 1: verse type counts
x = [item.capitalize() for item in df['verse_types'].explode().value_counts().index]
y = df['verse_types'].explode().value_counts()

fig = plt.figure()
ax = sns.barplot(x=x, y=y)
sns.set(rc={"figure.figsize":(6, 7)})
ax.set(title='Song Component Occurences')
ax.set_ylabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
fig.savefig('Song_Component_Occurences.jpg', bbox_inches='tight', dpi=150)
plt.show()

# 2: word count histogram
df['word_count'] = df['EDA_verses'].apply(wordCount)
fig = plt.figure()
ax = sns.histplot(data=df, x='word_count')
ax.set_ylabel('')
ax.set_xlabel('Word Count')
ax.set(title='Song Unique Word Count Distribution')
fig.savefig('Song_Unique_Word_Count_Distribution.jpg', bbox_inches='tight', dpi=150)
plt.show()

# 3: length histogram
df['length'] = df['EDA_verses'].apply(lambda x:len(x))
fig = plt.figure()
ax = sns.histplot(data=df, x='length', kde=False)
ax.set_ylabel('')
ax.set_xlabel('Length')
ax.set(title='Song Length Distribution')
fig.savefig('Song_Length_Distribution.jpg', bbox_inches='tight', dpi=150)
plt.show()

# 4: for fun, both plotted together as a scatter histogram
fig = plt.figure()
ax = sns.histplot(
    df, x="length", y="word_count",
    bins=50, discrete=(False, False), log_scale=(False, False),
    cbar=True, cbar_kws=dict(shrink=.75)
)
ax.set_ylabel('Word Count')
ax.set_xlabel('Length')
ax.set(title='Song Length and Word Count Distribution')
fig.savefig('Song_Length_and_Word_Count_Distribution.jpg', bbox_inches='tight', dpi=150)
plt.show()

# 5: most common words, with and without stop words (unigrams)
punctuation = ['?', '!', '-', ',', '.', '(', ')', '']
stop_words = stopwords.words('english') + punctuation

# with stop words
top_words = Counter([item.lower() for item in df['EDA_verses'].explode() if item.lower() not in punctuation])

x = [item[0] for item in top_words.most_common(20)]
y = [item[1] for item in top_words.most_common(20)]
fig = plt.figure()
ax = sns.barplot(x=x, y=y, color='blue')
sns.set(rc={"figure.figsize":(6, 7)})
ax.set(title='20 Most Common Words')
ax.set_ylabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
fig.savefig('20_Most_Common_Words.jpg', bbox_inches='tight', dpi=150)
plt.show()


# without stop words
top_words_rem = Counter([item.lower() for item in df['EDA_verses'].explode() if item.lower() not in stop_words])
x = [item[0] for item in top_words_rem.most_common(20)]
y = [item[1] for item in top_words_rem.most_common(20)]
fig = plt.figure()
ax = sns.barplot(x=x, y=y, color='blue')
sns.set(rc={"figure.figsize":(6, 7)})
ax.set(title='20 Most Common Words (Stopwords Removed)')
ax.set_ylabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
fig.savefig('20_Most_Common_Words_No_Stopwords.jpg', bbox_inches='tight', dpi=150)
plt.show()


#7: bigrams
bigrams = get_top_n_ngram(df['EDA_verses'].explode(), 20, 2)
x = []
y = []
for word, freq in bigrams:
    x.append(word), y.append(freq)
fig = plt.figure()
ax = sns.barplot(x=x, y=y, color='blue')
sns.set(rc={"figure.figsize":(6, 7)})
ax.set(title='20 Most Common Bigrams')
ax.set_ylabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
fig.savefig('20_Most_Common_Bigrams.jpg', bbox_inches='tight', dpi=150)
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
# df['rhymescore_AA'] = df['verses_transformed'].apply(getSongRhyme, args=(2, 'AA'))
# df['rhymescore_AB'] = df['verses_transformed'].apply(getSongRhyme, args=(2, 'AB'))

sample_df = df.sample(n=300, random_state=SEED)

sample_df['rhymescore_AA'] = sample_df['verses_transformed'].apply(getSongRhyme, args=(2, syllable_list, 'AA'))
print('done with AA!')
sample_df['rhymescore_AB'] = sample_df['verses_transformed'].apply(getSongRhyme, args=(2, syllable_list, 'AB'))
print('done with AB!')

# export df
os.chdir(DATAPATH)
df.to_csv('df_EDA.csv')
sample_df.to_csv('rhyme_sample.csv')
print('exported')
print('done')

os.chdir(IMGPATH)
fig = plt.figure()
ax = sns.histplot(data=sample_df, x='rhymescore_AA', kde=True)
ax.set_ylabel('')
ax.set_xlabel('Length')
ax.set(title='Song AA Rhyme Score Distribution')
fig.savefig('Song_AA_Rhyme_Score_Distribution.jpg', bbox_inches='tight', dpi=150)
plt.show()

fig = plt.figure()
ax = sns.histplot(data=sample_df, x='rhymescore_AB', kde=True)
ax.set_ylabel('')
ax.set_xlabel('Length')
ax.set(title='Song AB Rhyme Score Distribution')
fig.savefig('Song_AB_Rhyme_Score_Distribution.jpg', bbox_inches='tight', dpi=150)
plt.show()

