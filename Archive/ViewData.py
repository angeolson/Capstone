import pandas as pd

df = pd.DataFrame()

# loop through all files
for dataframe in ['train_df_chunk_one.csv', 'train_df_chunk_two.csv', 'train_df_chunk_three.csv', 'test_df.csv']:
    frame = pd.read_csv('train_df_chunk_one.csv')
    frame = frame.iloc[: , 1:]
    df = df.append(frame)

# see how many have non-blank verses
df = df[df['lyrics'] != 'song not found']

# 8992 lyrics instances

df['verse_num'] = df['verse_types'].apply(lambda x:len(x))
df_delin_verses = df[ df['verse_num'] > 2]

# 3852 contain separated out verses, chorus

