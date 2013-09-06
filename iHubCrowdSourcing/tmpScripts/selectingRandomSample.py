from BoilerPlate import *
#Get Annotations
pred = pd.read_csv('Results/predictions_batch4.csv')

pred.drop_duplicates(cols=['match_rowid'], inplace=True)
pred.info()
tweets = pred[['match_rowid','twitter.text']]
import random
tweets.shape
sample = random.sample(range(0,tweets.shape[0]),2000)

tweets_sample = tweets.iloc[sample]
tweets_sample.info()

tweets_sample.to_csv('CSV/sample_batch4_2000.csv',index=False)

for_chris = tweets_sample.iloc[1000:]

for_chris = for_chris.to_html()

fout = open("CSV/sample_batch4_chris.html",'w')
fout.write(for_chris)
fout.close()