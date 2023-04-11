# imports
import pandas as pd
import regex as re
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
# os.system('pip3 install markupsafe==2.0.1') # last compatible version of markupsafe to use with language detector

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
    df = VerseCleaner(df)
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
    new_types = [item for item in new_types if item != 'instrumental']
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
        return 'Yes'
    else:
        return 'No'

def VerseCleaner(df):
    '''
    cleans the verses and verse types of the dataframe; removes songs that are not english

    :param df: pandas dataframe
    :return: df
    '''
    df['verse_types'] = df['verse_types'].apply(verseTypeCleaner)
    df['english'] = df['verses'].apply(isEnglish)
    df = df[ df['english'] == 'Yes']
    df['verses_transformed'] = df.apply(versesTransform, axis=1)
    return df

def sentencePipe(sent):
    '''
    adds spaces between certain punctuation marks and words, so that those will be counted as separate
    tokens. removes leading/trailing whitespace. adds a special token denoted a newline to the front of each sentence.
    :param sent:
    :return:
    '''
    pat = re.compile(r"([.()!?,:;/-])")
    new_sentence = pat.sub(" \\1 ", sent)
    new_sentence = re.sub(r'\s+', ' ', new_sentence)
    new_sentence = '<NEWLINE> ' + new_sentence
    return new_sentence.split(" ")

def versesTransform(df):
    '''
    either returns cleaned sentences within each verse of a song, or denotes if the # of verses
    is not equal to the # of verse markers, suggesting this song should not be part of the final
    dataset.

    :param df:
    :return:
    '''
    full_list = []
    if len(df.verses) == len(df.verse_types):
        for i in range(len(df.verses)):
            list_ = []
            list_.append(typedict[df.verse_types[i]])
            sentence_list = [word for sentence in df.verses[i] for word in sentencePipe(sentence)]
            [list_.append(word) for word in sentence_list]
            full_list.append(list_)
        return [word for sentence in full_list for word in sentence]
    else:
        return 'Lengths are not equal'

# init dict of verse types to keep
typedict = {'verse': '<VERSE>',
            'chorus': '<CHORUS>',
            'pre chorus': '<PRECHORUS>',
            'bridge': '<BRIDGE>',
            'outro': '<OUTRO>',
            'intro': '<INTRO>',
            'refrain': '<REFRAIN>',
            'hook': '<HOOK>',
            # 'instrumental': '<INSTRUMENTAL>',
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
print('Done!')
df = df[df['verses_transformed'] != 'Lengths are not equal']
df.to_csv('df_cleaned.csv')







