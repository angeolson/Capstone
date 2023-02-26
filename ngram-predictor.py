'''
procedure and select helper functions adapted from https://towardsdatascience.com/text-generation-using-n-gram-model-8d12d9802aa0

'''

# imports
import pandas as pd
import string
import random
import time
from typing import List

def cleanData(df):
    '''
    function cleans the original pandas dataframe imported from .csv and returns a clean frame
    :param df:
    :return: cleaned pandas dataframe
    '''
    for col in ['verses', 'verse_types', 'verses_transformed', 'EDA_verses']:
        df[col] = df[col].apply(lambda x:eval(x, {'__builtins__': None}, {}))
    return df


def get_ngrams(n: int, tokens: list) -> list:
    """
    :param n: n-gram size
    :param tokens: tokenized sentence
    :return: list of ngrams
    ngrams of tuple form: ((previous wordS!), target word)
    """
    tokens = (n-1)*['<START>']+tokens
    l = [(tuple([tokens[i-p-1] for p in reversed(range(n-1))]), tokens[i]) for i in range(n-1, len(tokens))]
    return l


class NgramModel(object):

    def __init__(self, n):
        self.n = n

        # dictionary that keeps list of candidate words given context
        self.context = {}

        # keeps track of how many times ngram has appeared in the text before
        self.ngram_counter = {}

    def update(self, sentence: str) -> None:
        """
        Updates Language Model
        :param sentence: input text
        """
        n = self.n
        ngrams = get_ngrams(n, sentence)
        for ngram in ngrams:
            if ngram in self.ngram_counter:
                self.ngram_counter[ngram] += 1.0
            else:
                self.ngram_counter[ngram] = 1.0

            prev_words, target_word = ngram
            if prev_words in self.context:
                self.context[prev_words].append(target_word)
            else:
                self.context[prev_words] = [target_word]

    def random_token(self, context):
        """
        Given a context we "semi-randomly" select the next word to append in a sequence
        :param context:
        :return:
        """
        r = random.random()
        map_to_probs = {}
        token_of_interest = self.context[context]
        for token in token_of_interest:
            map_to_probs[token] = self.prob(context, token)

        summ = 0
        for token in sorted(map_to_probs):
            summ += map_to_probs[token]
            if summ > r:
                return token

    def generate_text(self, token_count: int, start_option: str):
        """
        :param token_count: number of words to be produced
        :param option: either 'random' or a starting prompt word
        :return: generated text
        """
        n = self.n
        if start_option == 'random':
            context_queue = (n - 1) * ['<START>']
            result = []
        else:
            context_queue = (n - 1) * [start_option]
            result = [start_option]
        for _ in range(token_count):
            obj = self.random_token(tuple(context_queue))
            result.append(obj)
            if n > 1:
                context_queue.pop(0)
                if obj == '<NEWLINE>':
                    context_queue = (n - 1) * ['<START>']
                else:
                    context_queue.append(obj)
        return ' '.join(result)

    def prob(self, context, token):
        """
        Calculates probability of a candidate token to be generated given a context
        :return: conditional probability
        """
        try:
            count_of_token = self.ngram_counter[(context, token)]
            count_of_context = float(len(self.context[context]))
            result = count_of_token / count_of_context

        except KeyError:
            result = 0.0
        return result


def create_ngram_model(n, song):
    m = NgramModel(n)
    for sentence in song:
        m.update(sentence)
    return m

def create_ngram_model_from_df(df, n):
    m = NgramModel(n)
    for i in range(len(df)):
        song = df['verses_transformed'][i]
        song_transformed = versesTransform(song)
        for sentence in song_transformed:
            m.update(sentence)
    return m

# load data
df = pd.read_csv('df_EDA.csv', index_col=0)
df = cleanData(df)
df.reset_index(drop=True, inplace=True)

# clean verses_transformed column, which will be used to generate text

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
def versesTransform(verses_transformed):
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
    result = [[token.lower() for token in item] for item in result]
    for item in result:
        item = item.append('<NEWLINE>')
    return result


# run

m_2 = create_ngram_model_from_df(df, 2)
print(m_2.generate_text(202, option='strong'))