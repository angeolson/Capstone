from sklearn.model_selection import train_test_split
import pandas as pd
import random

SEED = 48
random.seed(48)

df = pd.read_csv('df_LSTM.csv', index_col=0)
df_copy = df.copy()
df_copy.reset_index(drop=True, inplace=True)

# split data
train_, val_ = train_test_split(df_copy, train_size=0.8, random_state=SEED)

# export
train_.to_csv(f'training.csv')
val_.to_csv(f'validation.csv')