
import pandas as pd
import pandas.io.sql as psql
import sqlite3

path = '/home/phcostello/Documents/workspace/iHubCrowdSourcing/'

#Read File
new_annotations = pd.read_csv(path + '/CSV/randomSample_pc.csv',sep=',')
new_annotations.info()

new_annotations.pop('twitter.text')


#Check is ok
set(new_annotations.columns) == set(['match_rowid','Newsworthy']) #Check if columns match, use set as order not important
set(new_annotations['Newsworthy'].values) == set(['t','f']) #Validate Newsworthy items

#Read annotations from db
dbpath = "/home/phcostello/Documents/Data/iHub/S3_RawData/"
dbfile = "CrowdSourcingData.sqlite"
con = sqlite3.connect(dbpath + dbfile)
sql = "SELECT * FROM Annotations WHERE NEWSWORTHY IS NOT NULL"
existing_annotations = psql.read_frame(sql, con)
existing_annotations.info()

#compare match_rowids and row numbers for only new non-existing annotations
new_annotations['match_rowid']
new_annotation_rows = set(new_annotations['match_rowid']).difference(set(existing_annotations['match_rowid']))
new_annotation_rows = list(new_annotation_rows)
type(new_annotation_rows)
len(new_annotation_rows)


#make df of only new rows, by setting index to match_rowid filtering by rows and resetting index
#do this by setting index 
new_annotations_filtered = new_annotations.set_index('match_rowid').loc[new_annotation_rows]
new_annotations_filtered = new_annotations_filtered.reset_index()
new_annotations_filtered.head()
new_annotations_filtered.info()

new_annotations_filtered.columns = ['match_rowid','Newsworthy']
#append to Annotations
psql.write_frame(new_annotations_filtered,'Annotations', con ,if_exists='append')
con.commit()
con.close()

