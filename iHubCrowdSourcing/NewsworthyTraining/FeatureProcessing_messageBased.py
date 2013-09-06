'''
Created on Jun 18, 2013

@author: phcostello
'''
import sys
import os

import pandas as pd
import numpy as np
from pandas.io.data import DataReader
from datetime import datetime
import sqlite3 
import pandas.io.sql as psql
import PSQLUtils as pu


from sklearn.feature_extraction.text import CountVectorizer
os.getcwd()
os.chdir('/home/phcostello/Documents/workspace/iHubCrowdSourcing')

#if __name__ == '__main__':
        
path = "/home/phcostello/Documents/Data/iHub/S3_RawData/"
dbfile = "CrowdSourcingData.sqlite"
con = sqlite3.connect(path+ dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
#pu.showTables(con)

startDate = "2011-03-03"
endDate = "2015-03-05"
#First make vocab from reduced features with which we'll train
dfreduced = pu.readDB(con, 'FeaturesReduced2500T10000F', startDate, endDate)
#maintwitter = pu.readDB(con,'MasterData',startDate,endDate, fields=['match_rowid','twitter.text'])
#dfreduced_withtwitter = pd.merge(dfreduced,maintwitter,on='match_rowid')
#pu.toDB(con, dfreduced_withtwitter, 'FeaturesHT')

fin = open('Results/unique.txt')
text_train = fin.readlines()
fin.close()

#This is fitting vocab
vectoriser = CountVectorizer(min_df=1,stop_words='english')
vect_train = vectoriser.fit_transform(text_train)
#Change from sparse matrix to dense matrix
vect_train = vect_train.todense()
vectoriser.vocabulary_


text_test = dfreduced['twitter.text']
text_test = text_test.apply( lambda x: x.lower())
text_test = text_test.apply( lambda x: x.replace('#',''))
text_test = text_test.apply( lambda x: x.replace('@',''))
vect_test = vectoriser.transform(list(text_test))
vect_test = vect_test.todense()
vect_test


#Make into df
colnames = vectoriser.vocabulary_
df_features_reduced = pd.DataFrame(vect_test, columns = colnames)
df_features_reduced.head()

#Merge back to original df
dfreduced_added = pd.merge(dfreduced,df_features_reduced,how='inner',left_index=True,right_index=True)
dfreduced_added.info()

len(dfreduced_added.columns)
#write to db - has problem. Have to recomple sqlite with higher col number
dfreduced_added = dfreduced_added.drop('twitter.text',1)
pu.toDB(con, dfreduced_added, 'FeaturesReduced3000T10000FwithWords')
dfreduced_added.columns
pu.to_weka(dfreduced_added, outfile='features_reduced.csv')








