import pandas as pd
import random 

path='/home/phcostello/Documents/workspace/iHubCrowdSourcing/CSV/'


df = pd.read_csv(path+'master_search_combined_anrem.csv')

range(0,4)
randrows = random.sample(range(0,len(df)),4000)

dfsample = df.iloc[randrows]
len(dfsample)

dfsample.to_csv(path + 'randomSample.csv')