#
#import sys
#import os
#
#import pandas as pd
#import numpy as np
#from pandas.io.data import DataReader
#from datetime import datetime
#import sqlite3 
#import pandas.io.sql as psql
#import PSQLUtils as pu
#
#import time
#import pickle

from BoilerPlate import *
from sklearn.feature_extraction.text import CountVectorizer

#Annotate

#Get Annotations
import FeatureExtraction as fex

dbpath = "/home/phcostello/Documents/Data/iHub/S3_RawData/"
dbfile = "CrowdSourcingData.sqlite"
con = sqlite3.connect(dbpath+ dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
outpath = '/home/phcostello/Documents/workspace/iHubCrowdSourcing/'

from FeatureExtraction import read_raw_featurs_from_DB

features = read_raw_featurs_from_DB(con)


#Load Annotated featurs from saved pickle
features = pickle.load(open('sklearn_objects/full_user_features_pkl','rb'))
features.info()

#Choose random subset
#Training a vectoriser on message based features
tweets_training = features['twitter.text']
text_training = tweets_training.tolist()
target_training = features['Newsworthy'].values

#Make large word features
#uses sklearn CountVectoriser/bag of words
#This is fitting vocab
vectoriser_training = CountVectorizer(min_df=1,stop_words='english',strip_accents='unicode')
t = time.time()
features_msg_training = vectoriser_training.fit_transform(text_training) 
print "training text to word vector took", time.time()-t, "seconds"
features_msg_training.shape
#Change from sparse matrix to dense matrix
#vect_train = features_msg_training.todense() #BIG memory usage 2GB, see how to use the sparse for training

print features_msg_training

#fit vectoriser to select most importand features from X
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
X= features_msg_training
y = target_training

t = time.time()
clf=LinearSVC(C=1,penalty ='l1',dual=False).fit(X,y)
print "training took", time.time()-t, "seconds"

#Check performance
training_predicted = clf.predict(X)

from sklearn import metrics
cm_training = metrics.confusion_matrix(y,training_predicted)
print "Confusion Matrix on training data"
print cm_training

from sklearn import cross_validation

Xcsr = X.tocsr()
row = Xcsr[0].todense()
skf = cross_validation.StratifiedKFold(y=y, n_folds=3)
for train_index, test_index in skf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = Xcsr[train_index], Xcsr[test_index]
    y_train, y_test = y[train_index], y[test_index]


#Cross validation not straightforward with sparse matrix
#as iterating over k-folds requires X to dense as has to slice
#on ranges. When doing the fitting this kill computer memory

#
#for train_index, test_index in skf:
#    print clf.fit(Xcsr[train_index],y[train_index]).score(Xcsr[test_index],y[test_index])

from sklearn.metrics import precision_score, recall_score    
clfCV=LinearSVC(C=1,penalty ='l1',dual=False)

score = cross_validation.cross_val_score(clfCV, Xcsr, y, cv=skf, n_jobs=1)

#GridSearch
#Having issues with this and sparsity as well
from sklearn.grid_search import GridSearchCV
Cs = np.arange(0.0001,1,.05)
Cs
clfGrid = GridSearchCV(estimator=clfCV, param_grid=dict(C=Cs), n_jobs=1)
clfGrid.fit(X, y)
clfGrid.best_score_ #0.65
clfGrid.best_estimator_.C
clfGrid.score(X,y)
predicted = clfGrid.predict(X)
metrics.confusion_matrix(y_true = y,y_pred=predicted)
ymap = {'t':1,'f':0}
y_true = [ymap[it] for it in y]
y_pred = [ymap[it] for it in predicted]
print metrics.classification_report(y_true, y_pred)


#confusion matrix on training data
#    [92812,    50],
#     [   45,  3936]

#How does it go on the bigger set

#Use to predict on full set
#sql = "SELECT [match_rowid],[twitter.text] FROM MasterData"
#allTweets = psql.read_frame(sql, con)
#pickle.dump(allTweets, open('sklearn_objects/all_tweets.pkl','wb'))

allTweets = pickle.load(open('sklearn_objects/all_tweets.pkl','rb'))
allTweets.info()

text_test = allTweets['twitter.text'].values
target_test = None

#full features
t = time.time()
X = vectoriser_training.transform(text_test)
print 'time to vectorise all tweets in db for 90000 features is', t - time.time()
#pickle.dump(X, open('sklearn_objects/all_tweets_to90000_features.pkl','wb'))
X = pickle.load(open('sklearn_objects/all_tweets_to90000_features.pkl','rb'))
X.shape

t = time.time()
predicted_test = clf.predict(X)
print 'time to predict is', t - time.time()
tfcount = pd.Series(predicted_test).value_counts()
tfcount
allTweets['predicted'] = predicted_test

result = pd.merge(allTweets, features[['match_rowid', 'Newsworthy']],how='left',on='match_rowid')
resultTrues = result[result['predicted']=='t']
resultTrues.to_csv('Results/predictions_batch4.csv',index=False)

#Can we reduce features
from sklearn.feature_selection import SelectKBest,chi2
X = features_msg_training
y= target_training
clf_select = SelectKBest(score_func=chi2, k=5000)
clf_select.fit(X,y)

msg_feature_names = np.array(vectoriser_training.vocabulary_.keys())
msg_feature_names
support_indices = clf_select.get_support(indices = True)
#support_indices = support_indices -1
features_reduced_selected_text = msg_feature_names[support_indices]
features_reduced_selected_text
len(features_reduced_selected_text)

#Fit vectoriser for just selected features
vectoriser_selected = CountVectorizer(min_df=1,stop_words='english',strip_accents='unicode')
vectoriser_selected.fit(features_reduced_selected_text)
len(vectoriser_selected.vocabulary_.keys())

features_training_selected = vectoriser_selected.transform(text_training)
features_training_selected.shape
X = features_training_selected
X.shape
y= target_training
clf_selected = LinearSVC(C=1,penalty ='l1',dual=False)
clf_selected.fit(X,y)
predicted_selected_training = clf_selected.predict(X)

from sklearn import metrics
cm_rs = metrics.confusion_matrix(y,predicted_selected_training)
cm_rs

#How does the reduced classifier go in the test set
features_test_selected = vectoriser_selected.transform(text_test)
features_test_selected.shape
X = features_test_selected
X.shape
predicted_selected_test = clf_selected.predict(X)
tfcount = pd.Series(predicted_selected_test).value_counts()
tfcount

allTweets['predicted'] = predicted_selected_test
result = pd.merge(allTweets, features[['match_rowid', 'Newsworthy']],how='left',on='match_rowid')
resultTrues = result[result['predicted']=='t']
resultTrues.to_csv('Results/predictions_90000_redfeatures.csv',index=False)

    
