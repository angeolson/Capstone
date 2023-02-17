# imports
import pandas as pd
import regex as re
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

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
    '''
    function removes 'x' from verse type

    :param item: string
    :return: item without 'x'
    '''
    if item is None:
        return item
    elif item.lower() == 'x':
        return ''
    else:
        return item
def concatType(item, type='chorus'):
    '''
    function removes words that come after the specified verse type to get it down to its 'root type'

    :param item: string
    :param type: string
    :return: item where item == type or None
    '''
    if item is None:
        return item
    elif item.startswith(type):
        return type
    else:
        return item

def keepTypes(item):
    '''
    function limits the different type names kept in the dataframe

    :param item: string
    :return: item in list typedict.keys()
    '''
    if item in typedict.keys():
        return item
    else:
        return 'other'

def verseTypeCleaner(verse_type):
    '''
    this function splits any verse type at the ':' delimeter and keeps the first hald, removes numbers and select punctuation,
    removes trailing/leading whitespace, lowercases words, gets verse types down to their root (e.g. "chorus" instead of "chorus repeated"),
    and removes the 'x' label. Lastly, this function removes types that are uncommon and replaces them with blank space.

    :param verse_type:
    :return: cleaned verse type list
    '''
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

def get_lang_detector(nlp, name):
    return LanguageDetector()

def isEnglish(song):
    '''
    determines if a song is englihs using spacy english model

    :param song: verse column of dataframe
    :return: boolean if the song is english or not
    '''
    sentence_list = [sentence for verse in song for sentence in verse]
    text = " ".join(sentence_list)
    doc = nlp(text)
    detect_language = doc._.language
    if detect_language['language'] == 'en':
        return True
    else:
        return False

def VerseCleaner(df):
    '''
    cleans the verses of the dataframe; removes songs that are not english

    :param df: pandas dataframe
    :return: df
    '''
    df['english'] = df['verses'].apply(isEnglish)
    df = df[ df['english'] is True]
    return df

# init dict of verse types to keep
typedict = {'verse': '<VERSE>',
            'chorus': '<CHORUS>',
            'pre chorus': '<PRECHORUS>',
            'bridge': '<BRIDGE>',
            'outro': '<OUTRO>',
            'intro': '<INTRO>',
            'refrain': '<REFRAIN>',
            'hook': '<HOOK>',
            'instrumental': '<INSTRUMENTAL>',
            'post chorus': '<POSTCHORUS>',
            'other': '<OTHER>'
}

# init spacy
nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)

# read in data
df = pd.read_csv('data_delineated.csv')

# cleaning pipeline for dataframe
df = cleanData(df)
# step one: what are the types of verses in the songs?
# print(df.verse_types.explode().dropna().value_counts())

# cleaning pipeline for verse types
df['verse_types'] = df['verse_types'].apply(verseTypeCleaner)
# print(df.verse_types.explode().dropna().value_counts())
# observe as .csv file
# df.verse_types.explode().dropna().value_counts().reset_index().to_csv('verse_types.csv')

# cleaning pipeline for verses
# to do:
# 1: insert spaces between punctuation marks and letters
# 2: determine if the song is in english: if it is, keep it. Otherwise, delete song from dataframe
# 3: create a new column for each song, which will be the verses in the format we want them: add in special tokens for verse type, line breaks, but otherwise unpack lists

# os.system('pip3 install markupsafe==2.0.1') # last compatible version of markupsafe to use with language detector



test = df.iloc[0]


# this loop adds whitespace after punctuation, removes instances of doublewhitespace
pat = re.compile(r"([.()!?,:;/-])")
sentence_list = [sentence for verse in test.verses for sentence in verse]
for sentence in sentence_list:
    new_sentence = pat.sub(" \\1 ", sentence)
    new_sentence = re.sub(r'\s+', ' ', new_sentence)
    print(new_sentence)


# this loop inserts special tokens and creates a list for each verse
for i in range(len(test.verses)):
    list_ = []
    list_.append(typedict[test.verse_types[i]])
    sentence_list = [sentence for sentence in test.verses[i]]
    for item in sentence_list:
        list_.append('<NEWLINE>')
        for token in item.split(" "):
            list_.append(token)
    print(list_)


