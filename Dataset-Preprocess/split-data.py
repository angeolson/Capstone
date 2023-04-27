from sklearn.model_selection import train_test_split
import pandas as pd
import random
import os
import argparse


# argparse vars
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("-dt_pth", "--input_path", default = 'None', type=str, help = "path to the data files", required=True)
parser.add_argument("-ex_pth", "--export_path", default = 'None', type=str, help = "path for final dataset", required=True)
args = vars(parser.parse_args())

SEED = 48
random.seed(48)

os.chdir(args['input_path'])
df = pd.read_csv('df_LSTM.csv', index_col=0)
df_copy = df.copy()
df_copy.reset_index(drop=True, inplace=True)

# split data
train_, val_ = train_test_split(df_copy, train_size=0.8, random_state=SEED)

# export
os.chdir(args['output_path'])
train_.to_csv(f'training.csv')
val_.to_csv(f'validation.csv')