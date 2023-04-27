# Capstone: Data Preprocessing

This folder contains project files associated with my George Washington Univeristy MS in Data Science Captstone, specifically in data preprocessing. 

To recreate or build off of the results from my project, please first download and unzip both the train and test dat splits from the MSD here: http://millionsongdataset.com/lastfm/ 

Then, run the files in the following order:

* data-extract.py: uses the artist/song/genre information from the MSD to scrape lyric data from the web. 
* create-dataset.py: remove songs where lyrics were not found from the dataset. Note: webscraping lyrics takes time. If you need to break up this process, this file can also be used to merge different dataframes.
* clean-dataset.py: performs initial data cleaning 
* lstm-datprep.py: prepares the dataset specifically for LSTM modeling (does not need to be run if you are only interested in using the bigram generator)
* split-data.py: splits the dataset into train and test samples (does not need to be run if you are only interested in using the bigram generator). 


