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

# set environment vars 
PATH_ = os.getcwd()
FILEPATH_ = PATH_ + os.sep + 'lastfm_subset.zip'
DATAPATH_ = PATH_ + os.sep + 'Data'

# extract data from zip file 
# with zipfile.ZipFile(FILEPATH_, 'r') as zip:
#     zip.extractall(DATAPATH_)

# test
file = 'TRAAAAW128F429D538.json'
filepath = DATAPATH_ + '/lastfm_subset/A/A/A/' + file
with open(filepath) as json_file:
    data = json.load(json_file)
    fields = {}
    fields['artist'] = data['artist']
    fields['title'] = data['title']
    fields['tags'] = ", ".join([item[0] for item in data['tags']])

df = pd.DataFrame(fields, index=[0])


from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

art = df['artist'][0]
tit = df['title'][0]

def cleanTitle(tit):
    tit = tit.replace("'","")
    tit = "-".join(tit.split(" "))
    return tit

artist = "-".join(art.split(" "))
song = cleanTitle(tit)

URL = f'https://genius.com/{artist}-{song}-lyrics'
req = Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
web_byte = urlopen(req).read()
soup = BeautifulSoup(web_byte, "html.parser")
# page = requests.get(URL)
# html = BeautifulSoup(page.text, "html.parser") # Extract the page's HTML as a string

# Scrape the song lyrics from the HTML
lyrics = soup.find("div", class_="Lyrics__Container-sc-1ynbvzw-6 YYrds").get_text()

import regex as re 
results = re.findall(r'[a-z]+[A-Z][a-z]+', lyrics) # finds all occurences of words that should be split up 
pattern = re.compile(r'[a-z]+[A-Z][a-z]+')

" ".join(re.sub( r"([A-Z])", r" \1", lyrics).split()) # works 
