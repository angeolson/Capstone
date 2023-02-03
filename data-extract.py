'''
last update: 1/31
purpose: create dataframe from .zip file with json data 
author: Ange Olson 

data comes from http://millionsongdataset.com/lastfm/ 
'''

import os 
import zipfile
import json
import pandas as pd 
import regex as re 
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

# set environment vars 
PATH_ = os.getcwd()
FILEPATH_ = PATH_ + os.sep + 'lastfm_subset.zip'
DATAPATH_ = PATH_ + os.sep + 'Data'

# extract data from zip file 
# with zipfile.ZipFile(FILEPATH_, 'r') as zip:
#     zip.extractall(DATAPATH_)

# test
# file = 'TRAAAAW128F429D538.json'

filepath = DATAPATH_ + '/lastfm_subset/A/A/A/'
files = os.listdir(filepath)
df = pd.DataFrame(columns=['artist', 'title', 'tags'])
for file in files:
    with open(filepath+file) as json_file:
        data = json.load(json_file)
        fields = {}
        fields['artist'] = data['artist']
        fields['title'] = data['title']
        fields['tags'] = ", ".join([item[0] for item in data['tags']])
        file_df = pd.DataFrame(fields, index=[0])
        df = pd.concat([df,file_df])

df.reset_index(drop=True)

def cleanTitle(tit):
    pattern = re.compile(r'\(.*?\)') # pattern to remove parenthesis, anything in parenthesis 
    tit = re.sub(pattern, "", tit) # sub pattern
    tit = re.sub(r'\s+$', '', tit) # remove trailing whitespace
    tit = tit.replace("'","") # remove apostrephes 
    tit = "-".join(tit.split(" ")) # join each word with a dash
    return tit


df['artist_lookup'] = df['artist'].apply(lambda x:"-".join(x.split(" ")))
df['title_lookup'] = df['title'].apply(cleanTitle)
# artist = "-".join(art.split(" "))
# song = cleanTitle(tit)

def getLyrics(artist,song):
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
    verse_types = re.findall(r'\[.*?\]', lyrics)
    return verse_types

def getVerses(lyrics):
    pattern = re.compile(r'\[.*?\]')
    split_lyrics = [item for item in re.split(pattern, lyrics) if len(item) > 0]
    split_lyrics_delimited = [re.sub(r'(?<![A-Z\W])(?=[A-Z])', '-', item) for item in split_lyrics]
    split_lyrics_lines = [verse.split("-") for verse in split_lyrics_delimited]
    split_lyrics_lines_clean = [[item for item in verse if len(item) > 0] for verse in split_lyrics_lines]
    return split_lyrics_lines_clean

def dataframeLyrics(df):
    artist = df['artist_lookup']
    song = df['title_lookup']
    lyrics = getLyrics(artist, song)
    return lyrics 

df['lyrics'] = df.apply(dataframeLyrics, axis=1)
df['verses'] = df['lyrics'].apply(getVerses)
df['verse_types'] = df['lyrics'].apply(getVerseTypes)
