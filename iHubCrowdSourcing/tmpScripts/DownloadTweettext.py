import sys
import os

import pandas as pd
import numpy as np
from pandas.io.data import DataReader
from datetime import datetime
import sqlite3 
import pandas.io.sql as psql
import PatUtils.PSQLUtils as pu
reload(pu)

from sklearn.feature_extraction.text import CountVectorizer
os.getcwd()
#os.chdir('/home/phcostello/Documents/workspace/iHubCrowdSourcing')


#if __name__ == '__main__':
path = "/home/phcostello/Documents/Data/iHub/S3_RawData/"
dbfile = "CrowdSourcingData.sqlite"
con = sqlite3.connect(path+ dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
#fix outfile path
pathout = '/home/phcostello/Documents/weka-3-7-9/Crowdsourcing/ResultsPredictions/'

fout = open(pathout + 'outfile.txt','w')
cur = con.cursor()

#sql = 'SELECT [twitter.text] FROM MasterData AS md JOIN FeatureswithWords as fwd \
sql = 'SELECT [twitter.text] FROM FeaturesHT' 
#"SELECT [twitter.user.statuses_count], [twitter.text] FROM FeatureswithWords"
for row in cur.execute(sql):
    outrow = [ str(it) for it in row]
    outrow = '\t'.join(outrow)
    outrow += '\n'
    fout.write(outrow)

fout.close()

