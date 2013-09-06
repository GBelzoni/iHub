import pandas as pd
path='/home/phcostello/Documents/workspace/iHubCrowdSourcing/CSV/'
infile = path+'master_search_kill_no_text.csv'

dfan = pd.read_csv(infile,sep='\t')
dfan.columns

df2an = pd.read_csv(path + 'master_search_combined_just_rows.csv',sep = '\t')
df2an.columns
merged =pd.merge(dfan,df2an, how='inner', on='match_rowid')

rmvrows = merged['rownumber']
rmvrows = list(rmvrows.values)
rmvrows[0:5]

merged['match_rowid'].head()

df2an.head(30)

fin = open(path + 'master_search_combined.csv')
fout = open(path + 'master_search_combined_anrem.csv','w')
i=0

skip = rmvrows.pop(0)
skip
for line in fin:
    i+=1
    if i == skip:
        if len(rmvrows) !=0: 
            skip = rmvrows.pop(0)
        continue    
    fout.write(line)
    
len(df2an) - len(rmrows)

path + 'master_search_combined_anrem.csv'
df3 = pd.read_csv(path + 'master_search_combined_anrem.csv')