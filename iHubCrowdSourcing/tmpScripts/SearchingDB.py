
import pandas as pd
from pandas.io.data import DataReader
from datetime import datetime

import sqlite3 
import pandas.io.sql as psql
import csv

import timeit
from django.utils.encoding import force_unicode


pd.set_printoptions(max_colwidth = 400)

path = "/home/phcostello/Documents/Data/iHub/S3_RawData/"
dbfile = "CrowdSourcingData.sqlite"


con = sqlite3.connect(path+ dbfile, detect_types=sqlite3.PARSE_DECLTYPES)

#get rows with polling in text
sqlPolling = 'SELECT [twitter.text], Newsworthy \
            FROM HT_Annotated WHERE ([twitter.text] LIKE "% polling %"  )'

polling = psql.read_frame(sqlPolling , con)
polling.head()
polling = pd.DataFrame(polling)

#Get rows with wordcount > 12
func_wordcounts = lambda x: len(x.strip().split(" "))
wordcounts = polling['twitter.text'].apply(func_wordcounts)
wcGT12 = polling[wordcounts > 12]

wcGT12[ wcGT12['Newsworthy']=='t']
wcGT12
wcGT12.head()

#Get rows with counts of polling greater than 2     
import nltk

def countWord(sentence, text):
    fd = nltk.FreqDist()
    
    for word in nltk.word_tokenize(sentence.lower()):
        
        fd.inc(word)
        
    try:
        return dict(fd.items())[text]
    except:
        return 0
        

numberPolling = polling['twitter.text'].apply(countPolling)
polling['numberPolling'] = numberPolling
polling[polling['Newsworthy']=='t']
#number polling are all equal 1 so that means can just use polling exists


#get rows with police in text
sqlPolice = 'SELECT [twitter.text], Newsworthy \
            FROM HT_Annotated WHERE ([twitter.text] LIKE "% police %"  )'

police = psql.read_frame(sqlPolice , con)
police.head()
police = pd.DataFrame(police)
(police['Newsworthy']=='t').sum()



#get rows with police in text
sqlPolice = 'SELECT [twitter.text], Newsworthy \
            FROM HT_Annotated WHERE ([twitter.text] LIKE "% killed %"  )'

police = psql.read_frame(sqlPolice , con)
police.head()
police = pd.DataFrame(police)
(police['Newsworthy']=='t').sum()



#get all rows
sqlPolling = 'SELECT match_rowid,[twitter.text], Newsworthy \
            FROM HT_Annotated WHERE ([twitter.text] LIKE "% polling %"  )'



