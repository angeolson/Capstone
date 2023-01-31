'''
last update: 1/31
purpose: create dataframe from .zip file with json data 
author: Ange Olson 

data comes from http://millionsongdataset.com/lastfm/ 
'''

import os 
import zipfile

with zipfile.ZipFile('/Users/datagy/Archive.zip', 'r') as zip:
    zip.extract('file1.txt', '/Users/datagy/')