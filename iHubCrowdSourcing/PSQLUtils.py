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



pd.set_printoptions(max_colwidth = 400)
path = "/home/phcostello/Documents/Data/iHub/S3_RawData/"
dbfile = "CrowdSourcingData.sqlite"


def showTables(con, display = True):

    cur = con.cursor()
    cur.execute("SELECT name, type FROM sqlite_master")
    tbls = pd.DataFrame(cur.fetchall())
    if(display):
        print tbls
    return tbls
    

def readDB(con, table, startDate, endDate, DateField=None, fields = None):  

    #Read parts of table
    #Default if fields = None is to read rowid, *
    if fields == None:
        fieldTxt = '*'
    else:
        fieldTxt = '[' + '],['.join(fields) + ']'

    #Default if DateField = None then reads all dataRange
    if DateField == None:
        sqlText = "select rowid, {1} from {0}".format(table,fieldTxt)
    else:
        sqlText = "select rowid, {1} from {0} where"\
        "(date({2}) > date('{3}') and "\
        "date({2}) < date('{4}') )".format(table,fieldTxt,DateField,startDate,endDate)

    print sqlText
    data = psql.read_frame(sqlText, con )
    return data

def toDB(con, table, tableName):
    
    #Drop table if it exists
    if psql.table_exists(tableName, con, flavor='sqlite'):
        cur = con.cursor()
        sql = 'DROP TABLE "main"."{}"'.format(tableName)
        cur.execute(sql)
        con.commit()
    
    #Write to db    
    psql.write_frame(table, tableName , con)
    con.commit()
    
    

def to_weka(data,outfile):
    
    data.to_csv(outfile,index=False, quoting=csv.QUOTE_NONNUMERIC)
#    
#def displayTable(df):  
#    
#    import PyQt4  
#    df  = read_csv(filename, index_col = 0,header = 0)
#    datatable = QtGui.QTableWidget(parent=self)
#    datatable.setColumnCount(len(df.columns))
#    datatable.setRowCount(len(df.index))
#    for i in range(len(df.index)):
#        for j in range(len(df.columns)):
#            self.datatable.setItem(i,j,QtGui.QTableWidgetItem(str(df.iget_value(i, j))))
