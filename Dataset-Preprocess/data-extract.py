"""
last update: 4/11, added arguments to run file at the command line.
purpose: create dataframe from .zip file with json data
author: Ange Olson

before running, have the train and/or test dataset downloaded already and unzipped
data comes from http://millionsongdataset.com/lastfm/
"""

import os
import json
import pandas as pd 
import regex as re 
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import argparse

# argparse vars
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("-dt_pth", "--input_path", default = 'None', type=str, help = "path to the data files", required=True)
parser.add_argument("-ex_pth", "--export_path", default = 'None', type=str, help = "path for final dataset", required=True)
parser.add_argument("-ex_nme", "--export_name", default = 'None.csv', type=str, help = "name for final dataset", required=True)
parser.add_argument("-type", "--type", default = "train", type=str, help = "type, either train or test", required=True)
args = vars(parser.parse_args())

# set environment vars
INPUT_PATH_ = args['input_path']
DATAPATH_ = args['export_path']
FILENAME_ = args['export_name']
TYPE_ = args['type']

# extract data from zip file
# for file in filepath_list:
#     with zipfile.ZipFile(file, 'r') as zip:
#         zip.extractall(DATAPATH_)

# Helper Functions

def getDataframe(type='train'):
    '''
    :param type: either train or test split
    :return: pandas dataframe
    '''
    filepath = INPUT_PATH_ + f'/lastfm_{type}'
    files = os.listdir(filepath)
    from itertools import product
    combos = list(product(files, files, files))
    string_list = []
    for item in combos:
        i,j,k = item
        string = f'/{i}/{j}/{k}/'
        string_list.append(string)

    df = pd.DataFrame(columns=['artist', 'title', 'tags']) # init df
    for file in string_list:
        if os.path.exists(filepath+file):
            for subfile in os.listdir(filepath+file):
                with open(filepath+file+subfile) as json_file:
                    data = json.load(json_file)
                    fields = {}
                    fields['artist'] = data['artist']
                    fields['title'] = data['title']
                    fields['tags'] = ", ".join([item[0] for item in data['tags']])
                    if 'indie' in fields['tags']:
                        file_df = pd.DataFrame(fields, index=[0])
                        df = pd.concat([df,file_df])
    df.reset_index(drop=True)
    return df

def cleanTitle(tit):
    '''
    this function takes a song title and converts it to a format that can be used as part of a URL
    to scrape lyrics off the Genius website

    input: song title (string)
    output: song title (string) that can be used in a URL 
    '''
    pattern = re.compile(r'\(.*?\)') # pattern to remove parenthesis, anything in parenthesis 
    tit = re.sub(pattern, "", tit) # sub pattern
    tit = re.sub(r'\s+$', '', tit) # remove trailing whitespace
    tit = tit.replace("'","") # remove apostrophes
    tit = "-".join(tit.split(" ")) # join each word with a dash
    return tit

def getLyrics(artist,song):
    '''
    this function takes a URL-friendly artist and song name and returns the raw lyrics from the Genius site
    for that song

    inputs: artist (string), song(string)
    output: raw lyrics from genius site 
    '''
    try:
        URL = f'https://genius.com/{artist}-{song}-lyrics'
        req = Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
        web_byte = urlopen(req).read()
        soup = BeautifulSoup(web_byte, "html.parser")
    except: return "song not found"
    # Scrape the song lyrics from the HTML
    try:
        lyrics = soup.find("div", class_="Lyrics__Container-sc-1ynbvzw-6 YYrds").get_text()
    except: return "lyrics not found"
    return lyrics 

def getVerseTypes(lyrics):
    '''
    this function tries to parse out the names of the individual verses in a song (Verse 1, Verse 2, etc.)
    this only works when the verse names are enclosed in square brackets ([ ])

    input: raw lyrics (string)
    output: list of verse types (strings) present in the song 
    '''
    verse_types = re.findall(r'\[.*?\]', lyrics)
    return verse_types

def getVerses(lyrics):
    '''
    this function tries to separate each verse in a song
    this only works when the verse names are enclosed in square brackets ([ ])
    it then splits each found verse line by line, where lines are divided by an uppercase
    letter immediately following a lowercase letter (e.g. "byMe" --> ['....by', 'Me....'])

    input: raw lyrics (string)
    output: list of verses (strings) present in the song, then sublists broken down line by line
    '''
    pattern = re.compile(r'\[.*?\]')
    split_lyrics = [item for item in re.split(pattern, lyrics) if len(item) > 0]
    split_lyrics_delimited = [re.sub(r'(?<![A-Z\W])(?=[A-Z])', '-', item) for item in split_lyrics]
    split_lyrics_lines = [verse.split("-") for verse in split_lyrics_delimited]
    split_lyrics_lines_clean = [[item for item in verse if len(item) > 0] for verse in split_lyrics_lines]
    return split_lyrics_lines_clean

def dataframeLyrics(df):
    '''
    this function wraps around getLyrics() to be applied to a pandas dataframe

    input: dataframe
    output: column of raw lyrics, when using apply with axis=1 
    '''
    artist = df['artist_lookup']
    song = df['title_lookup']
    lyrics = getLyrics(artist, song)
    return lyrics 

def dataframePipeline(type='train'):
    '''
    run through the pipeline based on a defined split
    :param type: train/test split
    :return: dataframe
    '''
    df = getDataframe(type=type)
    print('dataframe!')
    print(df.head())
    df['artist_lookup'] = df['artist'].apply(lambda x: "-".join(x.split(" ")))  # makes artist name URL friendly
    df['title_lookup'] = df['title'].apply(cleanTitle)
    df['lyrics'] = df.apply(dataframeLyrics, axis=1)
    print('lyrics!')
    df['verses'] = df['lyrics'].apply(getVerses)
    df['verse_types'] = df['lyrics'].apply(getVerseTypes)
    print('done!')
    print(df.size)
    return df

# Run

if __name__ == "__main__":
    os.chdir(DATAPATH_)
    df = dataframePipeline(type=TYPE_)
    df.to_csv(FILENAME_)