import sys
import os

import pandas as pd
import numpy as np
from pandas.io.data import DataReader
from datetime import datetime
import sqlite3 
import pandas.io.sql as psql
import PSQLUtils as pu

#Make some random sets
import random
#Selection random rows from dataframe
featursFalse = features[features['Newsworthy']=='f']
len(featursFalse)
sampleFalseRows = [random.randrange(start=0, stop=len(featursFalse))  for i in range(0,10000)]
sampleFalseRows
sampledFalseData = featursFalse.iloc[sampleFalseRows]

featureTrue = features[features['Newsworthy']=='t']
allT = len(featureTrue)
sampleTrueRows = [random.randrange(start=0, stop=len(featureTrue))  for i in range(0,2500)]
sampleTrueRows
sampledTrueData = featureTrue.iloc[sampleTrueRows]

featuresReduced = sampledTrueData.append(sampledFalseData)
#featureTrue.info()
#sampledFalseData.info()
#featuresReduced.info()
cur = con.cursor()
sql = 'DROP TABLE "main"."FeaturesReduced"'
cur.execute(sql)
featuresReduced['twitter.text']

psql.write_frame(featuresReduced, "FeaturesReduced2500T10000F" , con)