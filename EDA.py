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

def verseTypeCleaner(verse_type):
    new_types = [item.split(':')[0] for item in verse_type]
    new_types = [re.sub(r'[^A-Za-z\s]', '', item) for item in new_types]
    new_types = [item.strip() for item in new_types]
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
# turn 'Intrumental...' into 'Intstrumental'
# turn Chorus x into Chorus
# want Guitar Solo = Guitar solo = guitar solo
# want Chorus I = Chorus II = Chorus
# want verse = VERSE = Verse
# remove non-english songs from frame
