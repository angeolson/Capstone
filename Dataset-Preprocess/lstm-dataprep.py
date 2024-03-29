# imports
import pandas as pd
import regex as re
import argparse
import os

# argparse vars
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("-dt_pth", "--input_path", default = 'None', type=str, help = "path to the data files", required=True)
parser.add_argument("-ex_pth", "--export_path", default = 'None', type=str, help = "path for final dataset", required=True)
args = vars(parser.parse_args())

def cleanData(df):
    '''
    function cleans the original pandas dataframe imported from .csv and returns a clean frame
    :param df:
    :return: cleaned pandas dataframe
    '''
    for col in ['verses', 'verse_types', 'verses_transformed', 'EDA_verses']:
        df[col] = df[col].apply(lambda x:eval(x, {'__builtins__': None}, {}))
    return df

contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                    "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                    "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                    "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
                    "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                    "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                    "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                    "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                    "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                    "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                    "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
                    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                    "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                    "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                    "here's": "here is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                    "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                    "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
                    "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is",
                    "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                    "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                    "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
                    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                    "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                    "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
                    "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

reverse_typedict = {'<verse>': '<SONGBREAK>',
            '<chorus>': '<SONGBREAK>',
            '<prechorus>': '<SONGBREAK>',
            '<bridge>': '<SONGBREAK>',
            '<outro>': '<SONGBREAK>',
            '<intro>': '<SONGBREAK>',
            '<refrain>': '<SONGBREAK>',
            '<hook>': '<SONGBREAK>',
            # 'instrumental': '<INSTRUMENTAL>',
            '<postchorus>': '<SONGBREAK>',
            '<other>': '<SONGBREAK>'
}


def _get_fromdict(dict):
    re_ = re.compile('(%s)' % '|'.join(dict.keys()))
    return dict, re_


contractions, contractions_re = _get_fromdict(contraction_dict)
songbreaks, songbreaks_re = _get_fromdict(reverse_typedict)

def replace_fromdict(text, dict, re_):
    def replace(match):
        return dict[match.group(0)]

    return re_.sub(replace, text)

os.chdir(args['input_path'])
df = pd.read_csv('df_EDA.csv', index_col=0)
df = cleanData(df)
df.reset_index(drop=True, inplace=True)

def cleanVerses(verses):
    punctuation_spaces = ['(', '[', ']', ')', ' ', '']
    verses = [item.lower() for item in verses]
    verses = [item for item in verses if item not in punctuation_spaces]
    lyrics = " ".join(verses)
    lyrics = replace_fromdict(lyrics, contractions, contractions_re)
    lyrics= replace_fromdict(lyrics, songbreaks, songbreaks_re)
    return lyrics

os.chdir(args['output_path'])
df['lyrics'] = df['verses_transformed'].apply(cleanVerses)
df.to_csv('df_LSTM.csv')


