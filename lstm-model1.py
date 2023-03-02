# imports
import pandas as pd

def cleanData(df):
    '''
    function cleans the original pandas dataframe imported from .csv and returns a clean frame
    :param df:
    :return: cleaned pandas dataframe
    '''
    for col in ['verses', 'verse_types', 'verses_transformed', 'EDA_verses']:
        df[col] = df[col].apply(lambda x:eval(x, {'__builtins__': None}, {}))
    return df

# load data
df = pd.read_csv('df_EDA.csv', index_col=0)
df = cleanData(df)
df.reset_index(drop=True, inplace=True)