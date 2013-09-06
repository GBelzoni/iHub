import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

import sqlite3 
import pandas.io.sql as psql
import PSQLUtils as pu

from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from sklearn.cluster import KMeans, MiniBatchKMeans


import time
import pickle

import os

os.getcwd()
os.chdir('/home/phcostello/Documents/workspace/iHubCrowdSourcing')

#Get Annotations

dbpath = "/home/phcostello/Documents/Data/iHub/S3_RawData/"
dbfile = "CrowdSourcingData.sqlite"
#con = sqlite3.connect(dbpath+ dbfile, detect_types=sqlite3.PARSE_DECLTYPES)

pred = pd.read_csv('Results/predictions_batch4.csv')
pred.info()
pred = pred.drop_duplicates(['match_rowid'])
pred.info()

tweets = pred[['match_rowid','twitter.text']].values.tolist()

tweets


from ttp import ttp
p = ttp.Parser()
tweeteg = tweets[198]
type(tweeteg)
import re
res = p.parse(tweeteg)
res.urls
ndun = re.compile('\\.')
tweeteg.decode("utf-8")
unicode(tweeteg).encode("utf-8")
pattern = "\\\\x"
re.findall(pattern, string=tweeteg)
i=0
type(cleanedTweets[0])

#Really rough way to clean tweets for hashtags and usernames
import codecs

tweets[0][1]
cleanedTweets = []
for tweetrow in tweets:
    tweet = tweetrow[1]
    try:
        result = p.parse(tweet)
        toremove = result.users + result.urls + result.tags + result.tags 
        for rm in toremove:
            tweet = tweet.replace(rm,'')
        tweet = re.sub('[@#\n]',"",tweet)
        codecs.utf_8_decode(tweet) #this gives issue in vectoriser below so skip on error
        cleanedTweets.append([tweetrow[0],str(tweet)])
    except:
        pass
    finally:
        print tweet,i
        i+=1

#Check if codecs will have problem
for tweet in cleanedTweets:
    try:
       codecs.utf_8_decode(tweet[1])
    except Exception as e:
        print cleanedTweets.index(tweet)
        print e




###Example from http://scikit-learn.org/dev/auto_examples/document_clustering.html
max_featurs =10
use_idf = True
vectorizer = TfidfVectorizer(max_df=0.5, stop_words='english', use_idf=use_idf, min_n=1, max_n=3)

dfCleanedTweets = pd.DataFrame(cleanedTweets, columns =[ 'match_rowid','twitter.text'])
dfCleanedTweets.info()
X = vectorizer.fit_transform(dfCleanedTweets['twitter.text'].values)
X.shape

#from sklearn.decomposition import PCA, RandomizedPCA
#3000 components to many
#2000 components gets 88% of info 
#pca = RandomizedPCA(n_components=2000)
#pca.fit(X)
#sum(pca.explained_variance_)
#pickle.dump(pca, open('pickle_jar/Xpcs.pkl','w'))?
#pca = pickle.load(open('pickle_jar/Xpcs.pkl'))
#Xpca = pca.transform(X)

cluster_file = pd.read_csv('Results/ComparativeAnalysis1.csv')
cluster_file.head()
cluster_file.info()
clusters_initial = cluster_file['Incidences'].values
clusters_initial.shape

#import difflib
#difflib.get_close_matches(clusters_initial[0], tweets,n=2)
#orig_text = []
#probs = []
#for it in clusters_initial:
#    try:
#        ans = difflib.get_close_matches(it, tweets,n=1, cutoff=0.4)
#        print ans
#        orig_text.append(ans[0])
#    except:
#        print 'problem with {}', it
#        probs.append(it)
#pickle.dump(orig_text,open('pickle_jar/clustersinData.pkl','w'))

vec_clust_init = vectorizer.transform(clusters_initial)
dftext = dfCleanedTweets
dftext.shape

orig_text = pickle.load(open('pickle_jar/clustersinData.pkl'))
mask = dftext['twitter.text'].isin(orig_text)
dftextuniques = dftext[['twitter.text','match_rowid']][mask]
dftextuniques.drop_duplicates(cols=['twitter.text'],inplace=True)
clusters_initial_orig = dftextunique['twitter.text'].tolist()
clusters_initial_centroid = vectorizer.transform(clusters_initial_orig)

clusters_initial_centroid = vec_clust_init.todense()
clusters_initial_centroid.shape
#Can pca to truncate if needed
n_clusters = 97
if False:
    #minibatch working in sklearn version 1.4
    km = MiniBatchKMeans(n_clusters, init='k-means++')
else:
#    km = KMeans(n_clusters, init='k-means++', max_iter=100, n_init=1,
#                verbose=False)
    km = KMeans(n_clusters, init=clusters_initial_centroid, max_iter=100, n_init=1, tol=1e-6,
                verbose=False)

#
t0 = time.time()
km.fit(X)
print "done in {} secs".format(time.time()-t0)
pickle.dump(km,open('pickle_jar/km_clusters_given_centroids.pkl','w'))
km = pickle.load(open('pickle_jar/km_clusters_given_centroids.pkl'))


labels = km.labels_
#from sklearn import metrics
##sil_score = metrics.silhouette_score(X, labels, metric='euclidean')
#from sklearn.cluster import DBSCAN
#dbs = DBSCAN()
#from scipy.spatial import distance
#Xdense = Xpca
#D = distance.squareform(distance.pdist(Xdense))
#S = 1 - (D / np.max(D))
#predsDBS = dbs.fit(S)

#from sklearn.cluster import AffinityPropagation
#af = AffinityPropagation()
#
#predsAF = af.fit(Xpca)

preds = km.predict(X)
dftext['clusters']=preds
preds_initial_clusters = km.predict(vec_clust_init)
cluster_file['clusters'] = preds_initial_clusters
np.unique(preds_initial_clusters).shape
cluster_file.to_csv('Results/kmeansOn_orig.csv')

pd.merge(dftext,cluster_file, how = 'left', on = 'clusters')





preds_centroid = km.predict(vec_clust_init)
preds_centroid
len(clusters_initial)
np.unique(preds_centroid).shape
len(clusters_initial_orig)
dfcentroids = pd.DataFrame({'clust_text':clusters_initial_orig,'clusters':preds_centroid})
dfcentroids.info()
dfcentroids.shape
dfcentroids.head()

dfcentroid_unique = dfcentroids.drop_duplicates(cols='clusters')
dfcentroid_unique

pd.merge(dftext,dfcentroid_unique,how='left',on='clusters')

dftext.to_csv('Results/clustering_100_cls_3grams.csv')

outdf = dftext[['match_rowid','clusters']] 
outdf.head()
outdf.to_csv('just_cluster.csv')



html = dftext.to_html()
fout = open('Results/clustering_pred_compAnal_3gram.html','w')
fout.write(html)
fout.close()
