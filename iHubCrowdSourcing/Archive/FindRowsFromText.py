'''
Created on Jun 17, 2013

@author: phcostello
'''



import pandas as pd
import pandas.io.sql as psql
pd.set_printoptions(max_colwidth = 400)
import sqlite3
import os, sys

path = "/home/phcostello/Documents/workspace/iHubCrowdSourcing/Archive/"
os.chdir(path)

#if __name__ == '__main__':

#Open DB
pathdb = "/home/phcostello/Documents/Data/iHub/S3_RawData/"
dbfile = "CrowdSourcingData.sqlite"
con = sqlite3.connect(pathdb+ dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor() #Set cursor for reading records


#Open up file with twitter text. Note I manually removed Newsworthy column, and header row in Excel
#as only wanted the text 
fin = open( "Master_True_just_text.csv")
lines = fin.readlines()
fin.close()
print lines[0:2] #just check it's read ok 

#Now want to build query where I select all rows where twitter.text is in
#thes lines read
linesConc = ",".join(lines) #This is simple way to concat lines with comma seperator
print linesConc
#Build query - Note: USE SQL IN command, rather than == on each text field. V quick
sql = "SELECT rowid, [twitter.text] FROM MasterData WHERE [twitter.text] in ({})".format(linesConc)

#Run query and read retrieved row into list
matchingRows = [ row for row in cur.execute(sql)]

print matchingRows
print len(matchingRows) #Check length looks right

#Make into df to add trues and get rid of text easily
df = pd.DataFrame(matchingRows, columns=['match_rowid','text'])
df['Newsworthy']='t'

df.to_csv(path +'master_true_rowid_inDB.csv')
#
#dfout = df[['match_rowid','Newsworthy']]
#dfout.head()
#
#psql.write_frame(dfout, "tmpAnnotated", con, if_exists = 'append')
#con.commit()
#
##Update Annotation file for new values
##Note sqlite doesn't support FULL OUTER JOINS, or UPDATE JOINS :(
#
##table to be updated = Annotations
##table with new data = test
#
##First replace existing rowids
#sqlReplaceOld = 'UPDATE Annotations\
#SET\
#Newsworthy = ( SELECT test.Newsworthy \
#FROM test\
#WHERE test.match_rowid = Annotations.match_rowid)\
#WHERE\
#EXISTS (\
#        SELECT *\
#        FROM test\
#        WHERE test.match_rowid = Annotations.match_rowid\
#)'
#
##Rename table to be extended
#sqlRename='ALTER TABLE "main"."Annotations" RENAME TO "Annotations_tmp"'
#
##Create table to insert back into
#sqlNewEmpty = 'CREATE TABLE "Annotations"(  match_rowid INT,  Newsworthy TEXT)'
#
#
##Union together new file with old renamed file 
#sqlPopulateUnion = 'INSERT INTO Annotations\
#SELECT * FROM (SELECT * FROM Annotated_tmp\
#UNION\
#SELECT * FROM test)'
#
#
##Drop old table
#sqlDropOld='DROP TABLE "main"."AnnotatedSmall'
#
#con.close()
    