"""
last update: 4/11
purpose: remove songs without lyrics; split into two dataframes, one with delineated verses and the other without
author: Ange Olson

*note: in the original format, I needed to create chunks of dataframes (data-extract.py takes time to run)
in this update, I am also assuming the user will need to import dataframes as some sort of list.
"""

# imports
import pandas as pd
import argparse
import os

# argparse vars
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("-dt_pth", "--input_path", default = 'None', type=str, help = "path to the data files", required=True)
parser.add_argument("-ex_pth", "--export_path", default = 'None', type=str, help = "path for final dataset", required=True)
parser.add_argument("-f", "--files", default = 'None', type=str, nargs='+', help = "files to read in", required=True)
args = vars(parser.parse_args())

# set environment vars
INPUT_PATH_ = args['input_path']
DATAPATH_ = args['export_path']
FILENAMES_ = args['files']

# init df
df = pd.DataFrame()

# loop through all files
# for dataframe in ['train_df_chunk_one.csv', 'train_df_chunk_two.csv', 'train_df_chunk_three.csv', 'test_df.csv']:
os.chdir(INPUT_PATH_)
for dataframe in FILENAMES_:
    frame = pd.read_csv(dataframe)
    frame = frame.iloc[: , 1:]
    df = df.append(frame)

# see how many have non-blank verses
df = df[df['lyrics'] != 'song not found']

# 10029 lyrics instances; export
os.chdir(DATAPATH_)
df.to_csv('data.csv')

df['verse_num'] = df['verse_types'].apply(lambda x:len(x))
df_delin_verses = df[ df['verse_num'] > 2]

# 3658 contain separated out verses, chorus; export
df_delin_verses.to_csv('data_delineated.csv')
