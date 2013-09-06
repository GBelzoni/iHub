'''
Created on May 2, 2013

@author: phcostello
'''

import pandas as pd
import numpy as np
from pandas.io.data import DataReader
from datetime import datetime
import sqlite3 
import pandas.io.sql as psql
import csv
import timeit
from pandas.core.common import isnull
import PSQLUtils as put
pd.set_printoptions(max_colwidth = 400)


dbpath = "/home/phcostello/Documents/Data/iHub/S3_RawData/"
dbfile = "CrowdSourcingData.sqlite"
con = sqlite3.connect(dbpath+ dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
outpath = '/home/phcostello/Documents/workspace/iHubCrowdSourcing/'
out_table_name= 'Features_true'
output_type = 'return'

#feature creators these are applied to columns elementwise
def countlist(x):
    if x == None:
        return 0
    else:
        return len(x.split(','))
    
def isTF(x):
    #Have to take care of two case, where x is None and where x in NaN
    if x == None or ( isinstance(x , np.float64) and np.isnan(x)):
        return False
    else:
        return True

def wordcount(x):
    if x == None:
        return 0
    else:
        return len(x.split(' '))

def replaceNullwithZero( x): 
    if np.isnan(x): 
        return 0
    else:
        return x
    
#message based feature creator function


def read_raw_featurs_from_DB(con):
    
    fields = ['twitter.links',\
    'twitter.user.verified',\
    'twitter.user.listed_count',\
    'twitter.text',\
    'twitter.mentions',\
    'twitter.mention_ids',\
    'klout.score',\
    'twitter.hashtags',\
    'twitter.user.statuses_count',\
    'twitter.user.followers_count',\
    'twitter.user.friends_count',\
    'twitter.user.geo_enabled',\
    'language.confidence',\
    'twitter.user.lang',\
    'twitter.created_at',\
    'twitter.user.created_at',\
    'Newsworthy']
    
    fieldsConc = '[' + '],['.join(fields) + ']'
    
    sqlQuery = "SELECT md.match_rowid , {} FROM MasterData AS md JOIN AnnotationsBatch4 AS an \
    ON md.match_rowid = an.match_rowid \
    WHERE an.Newsworthy IS NOT NULL".format(fieldsConc)
    data = psql.read_frame(sqlQuery, con)
    
    return data


def create_user_based_features(data, output_type = 'return', outputpath = None, out_table_name ='features_user_based'):    
    if output_type not in set(['weka', 'sqlite', 'return']):
        
        raise ValueError('output_type {} not supported '.format(output_type))
    
    #count items in string lists
    fieldstocount = ['twitter.links',\
    'twitter.mentions',\
    'twitter.hashtags']
    counts = data[fieldstocount].applymap(countlist)
    counts['match_rowid'] = data['match_rowid']
    
    #Check fields that are t/f
    fieldsTF = ['twitter.links' ,\
                'twitter.user.verified',\
                'twitter.mentions',\
                'twitter.user.geo_enabled']
    data[fieldsTF].head()
    truefalse = data[fieldsTF].applymap(isTF)
    truefalse['match_rowid']=data['match_rowid']
    
    
    #Create twitter age
    twitterAgeFields = ['twitter.created_at' ,\
                'twitter.user.created_at']
    #Convert to unicode string to datetime
    data[twitterAgeFields].head()
    data[twitterAgeFields].ix[0,0]
    dt = pd.to_datetime(data[twitterAgeFields].ix[0,0])
    #dt = data[twitterAgeFields].applymap(pd.to_datetime)
    #Date are read in in following unicode string format u'2013-03-19 13:00:16+00:00'
    dt = data[twitterAgeFields].applymap( lambda x:  int(x.split('-')[0])) #convert datestring to int year
    dt.ix[0,0]                                     
    twitterage = pd.DataFrame()
    twitterage= dt['twitter.created_at'] - dt['twitter.user.created_at'] 
    twitterage = pd.DataFrame(twitterage)
    twitterage.columns = ['twitterage']
    twitterage['match_rowid']=data['match_rowid']
    #print len(twitterage)
    #twitterage.head()
    
    #Word count
    wordcounts = pd.DataFrame(data['twitter.text']).applymap(wordcount)
    wordcounts['match_rowid'] = data['match_rowid']
    wordcounts.columns= ['wordcounts','match_rowid']
    #print len(wordcounts)
    #wordcounts.head()
    
    rawfields = ['match_rowid',
    'twitter.user.listed_count',\
    'klout.score',\
    'twitter.user.statuses_count',\
    'twitter.user.followers_count',\
    'twitter.user.friends_count',\
    'language.confidence',\
    'twitter.user.lang',\
    'Newsworthy']
    
    untouchedfields = data[ rawfields]
    features = pd.DataFrame()
    features = pd.merge(counts,truefalse ,on= 'match_rowid')
    features = pd.merge(features,twitterage ,on= 'match_rowid')
    features = pd.merge(features,wordcounts ,on= 'match_rowid')
    features = pd.merge(features,untouchedfields ,on= 'match_rowid')
    print len(features)
    
    #Add column names
    colnames = ['links_number','mentions_number','Hashtags_number',\
                        'match_rowid','links_exist','user_verified','mentions_exist',\
                        'geo_location_exist', 'twitter_age','wordcounts']
    
    colnames += rawfields[1:] #Drop match_rowid from rawfields names, 1st element
    features.columns = colnames
    
    #Replace empty value in numeric data with zero, e.g for count vars
    
    features[rawfields[:(len(rawfields)-2)]] = features[rawfields[:(len(rawfields)-2)]].applymap(replaceNullwithZero)
    
    #Add twitter text at the end
    features = pd.merge(features,data[['twitter.text','match_rowid']] ,on= 'match_rowid')
    
    if output_type == 'weka':
        con.close()
        #Write to weka friendly csv using custom function above
        outfile = outpath + 'CSV/{}.csv'.format(out_table_name)
        put.to_weka(features, outfile )
        
        
    elif output_type == 'sqlite':
        #Write to db
        cur = con.cursor()
        sql = 'DROP TABLE "main"."{}"'.format(out_table_name)
        try:
            cur.execute(sql)
        except sqlite3.OperationalError as e:
            print 'got error {}'.format(e)
        con.commit()
        psql.write_frame(features, out_table_name , con)#, append='replace')
        con.commit()
        con.close()
        
    elif output_type == 'return':
        con.close()
        return features

#def create_messag_based_features

if __name__ == '__main__':
    
   features = create_user_based_features(con, output_type ='return')
   print features.info
    


