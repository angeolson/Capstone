# imports
import pandas as pd

# init df
df = pd.DataFrame()

# loop through all files
for dataframe in ['train_df_chunk_one.csv', 'train_df_chunk_two.csv', 'train_df_chunk_three.csv', 'test_df.csv']:
    frame = pd.read_csv(dataframe)
    frame = frame.iloc[: , 1:]
    df = df.append(frame)

# see how many have non-blank verses
df = df[df['lyrics'] != 'song not found']

# 10029 lyrics instances; export
df.to_csv('data.csv')

df['verse_num'] = df['verse_types'].apply(lambda x:len(x))
df_delin_verses = df[ df['verse_num'] > 2]

# 3658 contain separated out verses, chorus; export
df_delin_verses.to_csv('data_delineated.csv')
