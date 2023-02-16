# imports
import pandas as pd
import regex as re

# helper functions
def cleanData(df):
    '''
    function cleans the original pandas dataframe imported from .csv and returns a clean frame
    :param df:
    :return: cleaned pandas dataframe
    '''
    df = df.iloc[: , 1:] # remove uneeded index column
    df['verses'] = df['verses'].apply(lambda x:eval(x, {'__builtins__': None}, {}))
    df['verse_types'] = df['verse_types'].apply(lambda x: eval(x, {'__builtins__': None}, {}))
    df = df[['artist', 'title', 'verses', 'verse_types']]
    return df

def removeX(item):
    if item is None:
        return item
    if item.lower() == 'x':
        return ''
def concatType(item, type='chorus'):
    if item is None:
        return item
    if item.startswith(type):
        return type

typelist = ['verse', 'chorus', 'prechorus', 'bridge', 'outro', 'intro', 'refrain', 'hook']
def verseTypeCleaner(verse_type):
    new_types = [item.split(':')[0] for item in verse_type] # only keep verse title before [:]
    new_types = [re.sub(r'[^A-Za-z\s]', '', item) for item in new_types] # remove numbers, punctuation
    new_types = [item.strip() for item in new_types] # remove trailing/leading whitespace
    new_types = [item.lower() for item in new_types]
    new_types = [removeX(item) for item in new_types]
    for t in typelist:
        new_types = [concatType(item, type=t) for item in new_types]
    return new_types

# read in data
df = pd.read_csv('data_delineated.csv')

# cleaning pipeline for dataframe
df = cleanData(df)

# step one: what are the types of verses in the songs?
print(df.verse_types.explode().dropna().value_counts())

# a lot of verse names seem to have extra information after the colon, or the '[' and ']' is missing and causing duplicates. We also probably don't need verse numbers.
df['verse_types'] = df['verse_types'].apply(verseTypeCleaner)

# observe as .csv file
df.verse_types.explode().dropna().value_counts().reset_index().to_csv('verse_types.csv')

# To do:
# replace x with blank
# want anything that's not chorus, verse, prechorus, bridge, outro, intro, refrain, hook to be blank
# turn Chorus x into Chorus
# want Chorus I = Chorus II = Chorus
# want verse = VERSE = Verse
# remove non-english songs from frame
