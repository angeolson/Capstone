# imports
import pandas as pd
import regex as re

# init dict of verse types to keep
typedict = {'verse': 1,
            'chorus': 2,
            'pre chorus': 3,
            'bridge': 4,
            'outro': 5,
            'intro': 6,
            'refrain': 7,
            'hook': 8,
            'instrumental': 9,
            'post chorus': 10
}

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
    elif item.lower() == 'x':
        return ''
    else:
        return item
def concatType(item, type='chorus'):
    if item is None:
        return item
    elif item.startswith(type):
        return type
    else:
        return item

def keepTypes(item):
    if item in typedict.keys():
        return item
    else:
        return ''

def verseTypeCleaner(verse_type):
    new_types = [item.split(':')[0] for item in verse_type] # only keep verse title before [:]
    new_types = [re.sub(r'[\d\[\].!?\\]', '', item) for item in new_types] # remove numbers, punctuation exept -
    new_types = [re.sub(r'\-', ' ', item) for item in new_types]
    new_types = [item.strip() for item in new_types] # remove trailing/leading whitespace
    new_types = [item.lower() for item in new_types]
    new_types = [removeX(item) for item in new_types]
    for t in typedict.keys():
        new_types = [concatType(item, type=t) for item in new_types]
    new_types = [keepTypes(item) for item in new_types]
    return new_types

# read in data
df = pd.read_csv('data_delineated.csv')

# cleaning pipeline for dataframe
df = cleanData(df)

# step one: what are the types of verses in the songs?
print(df.verse_types.explode().dropna().value_counts())

# a lot of verse names seem to have extra information after the colon, or the '[' and ']' is missing and causing duplicates. We also probably don't need verse numbers.
df['verse_types'] = df['verse_types'].apply(verseTypeCleaner)

print(df.verse_types.explode().dropna().value_counts())

# observe as .csv file
# df.verse_types.explode().dropna().value_counts().reset_index().to_csv('verse_types.csv')


# To do:
# replace x with blank
# want anything that's not chorus, verse, prechorus, bridge, outro, intro, refrain, hook to be blank
# turn Chorus x into Chorus
# want Chorus I = Chorus II = Chorus
# want verse = VERSE = Verse
# remove non-english songs from frame
