
import sys
import os

import pandas as pd
import numpy as np
from pandas.io.data import DataReader
from datetime import datetime
import sqlite3 
import pandas.io.sql as psql
import PSQLUtils as pu
from sklearn.feature_extraction.text import CountVectorizer
import time
import pickle


os.getcwd()
os.chdir('/home/phcostello/Documents/workspace/iHubCrowdSourcing')

#Annotate

#Get Annotations
import FeatureExtraction as fex


dbpath = "/home/phcostello/Documents/Data/iHub/S3_RawData/"
dbfile = "CrowdSourcingData.sqlite"
con = sqlite3.connect(dbpath+ dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
outpath = '/home/phcostello/Documents/workspace/iHubCrowdSourcing/'
out_table_name= 'Features_true'
output_type = 'return'

#Get annoations
#Need to fix the annotating step
#data = fex.read_raw_featurs_from_DB(con)
#features = fex.create_user_based_features(data)
#features.info()
#fout = open('full_user_features_pkl','w')
#pickle.dump(features, fout)
#fout.close()

features = pickle.load(open('sklearn_objects/full_user_features_pkl','rb'))
features.info()

#Choose random subset

#Make some random sets
import random
#Selection random rows from dataframe
featursFalse = features[features['Newsworthy']=='f']
len(featursFalse)
sampleFalseRows = [random.randrange(start=0, stop=len(featursFalse))  for i in range(0,8000)]
sampleFalseRows
sampledFalseData = featursFalse.iloc[sampleFalseRows]
featureTrue = features[features['Newsworthy']=='t']
allT = len(featureTrue)
sampleTrueRows = [random.randrange(start=0, stop=len(featureTrue))  for i in range(0,1500)]
sampleTrueRows
sampledTrueData = featureTrue.iloc[sampleTrueRows]

features_training = sampledTrueData.append(sampledFalseData)
features_test = features.loc[features.index - features_training.index]

features_training = features
features_test = features
#Training a vectoriser on message based features
tweets_training = features_training['twitter.text']
text_training = tweets_training.tolist()
target_training = features_training['Newsworthy'].values

#Make large word features
#uses sklearn CountVectoriser/bag of words
#This is fitting vocab
vectoriser_training = CountVectorizer(min_df=1,stop_words='english',strip_accents='unicode')
features_msg_training = vectoriser_training.fit_transform(text_training) 
features_msg_training.shape
##Change from sparse matrix to dense matrix
##vect_train = vect_train.todense() #BIG memory usage 2GB, see how to use the sparse for training
colnames = vectoriser_training.vocabulary_.keys()
colnames
len(colnames)

##fit vectoriser to select most importand features from X
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.ensemble import ExtraTreesClassifier

X= features_msg_training
y = target_training

t0 = time.time()
clf1 = LogisticRegression(C=1)
clf1.fit(X, y)
t1 = time.time() - t0
t1
clf2 = None
clf2 = SelectKBest(score_func=chi2, k=5000)
clf2.fit(X,y)
clf2.get_support()
t2 = time.time() - t0 - t1
t2
#Can't use tree on sparse data
#cls3 = ExtraTreesClassifier(compute_importances=True, random_state = 0).fit_transform(X,y) #no good for dense data
#t3 = time.time() - t0 - t2
t3=t2
clf4=LinearSVC(C=1,penalty ='l1',dual=False).fit(X,y)
t4 = time.time() - t0 - t3

print 'fitting time', t1,t2,t3,t4

#Check performance
training_predicted = clf4.predict(X)

from sklearn import metrics
cm_training = metrics.confusion_matrix(y,training_predicted)
cm_training

#confusion matrix on training data
#    [9994,    6]
#    [   1, 2999]

#How does it go on the bigger set

#Training a vectoriser on message based features
tweets_test = features_test['twitter.text']
text_test = tweets_test.tolist()
target_test = features_test['Newsworthy'].values

#full features
X = vectoriser_training.transform(text_test)
X.shape
y = target_test
predicted_test = clf4.predict(X)
cm_test = metrics.confusion_matrix(y,predicted_test)
cm_test

#confusion matrix on training data
#    [82592,   773],
#    [   95,  1752]

#Can we reduce the features and still get good performance
#select features using SelectKBest above

X = features_msg_training
y= target_training
clf_select = SelectKBest(score_func=chi2, k=20000)
clf_select.fit(X,y)

msg_feature_names = np.array(vectoriser_training.vocabulary_.keys())
msg_feature_names
support_indices = clf_select.get_support(indices = True)
#support_indices = support_indices -1
features_reduced_selected_text = msg_feature_names[support_indices]
features_reduced_selected_text

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

#confusion matrix on training data, 5000 top features included
#    [7980,   20],
#    [  96, 1404]


#How does the reduced classifier go in the test set
features_test_selected = vectoriser_selected.transform(text_test)
features_test_selected.shape
X = features_test_selected
X.shape
y = target_test
y.shape
predicted_selected_test = clf_selected.predict(X)

cm_stest = metrics.confusion_matrix(y,predicted_selected_test)
cm_stest


##Run vectoriser on X full features to get new features

#Evaluate
from sklearn import metrics
from sklearn import cross_validation
clf = clf1
clf.score(X,y)

#Cross validation not straightforward with sparse matrix
#as iterating over k-folds requires X to dense as has to slice
#on ranges. When doing the fitting this kill computer memory

#Xdense = X.todense()
#y = np.array(y)
#k_fold = cross_validation.KFold(len(Xdense),n_folds=10)
#rn = None
#for i,j in k_fold:
#    clf.fit(Xdense[i],y[i]).score(Xdense[j],y[j])
#    break
#y[rn]
#cross_validation.cross_val_score(clf, Xdense, y,cv=k_fold, n_jobs=-1)

#GridSearch
#Having issues with this and sparsity as well
from sklearn.grid_search import GridSearchCV
Cs = np.arange(0,1,.05)
Cs
clfGrid = GridSearchCV(estimator=clf, param_grid=dict(Cs=Cs), n_jobs=-1)
clfGrid.fit(X, y)
clfGrid.best_score_
clfGrid.best_estimator_.Cs
clfGrid.score(X,y)
predicted = clf.predict(X)
metrics.confusion_matrix(y_true = y,y_pred=predicted)
ymap = {'t':1,'f':0}
y_true = [ymap[it] for it in y]
y_pred = [ymap[it] for it in predicted]
print metrics.classification_report(y_true, y_pred)





sql = "SELECT [twitter.text] FROM MasterData"
allTweets = psql.read_frame(sql, con)
allTweets.info()
text_all = allTweets['twitter.text'].values
tstart = time.time()
vect_all_test = vectoriser.transform(text_all)
trun = tstart - time.time()
trun/60
vect_all_test.shape

predict_all = selected_features_vectoriser.predict(vect_all_test)
allf = ['f' for j in range(0,len(predict_all))]


cm = metrics.confusion_matrix(predict_all,allf)
cm
float(cm[1,0])/float(cm[0,0])



len
trues = allTweets.iloc[predict_all =='t']
featout = features[['Newsworthy','match_rowid']]
featout = featout.set_index('match_rowid')
trues.index = trues.index + 1
results = pd.merge(trues,featout,how='left', left_index=True, right_index=True)

results.to_csv('Results/allTrues_fullModel_fullFeatures.csv')

#Apply smaller features to full text to generate text based features

#tweets_all = features['twitter.text'].

#TRAIN_FOLDER





#Apply to new ds


#Extract from bigger set the same a little
    
text_test = dfreduced['twitter.text']
text_test = text_test.apply( lambda x: x.lower())
text_test = text_test.apply( lambda x: x.replace('#',''))
text_test = text_test.apply( lambda x: x.replace('@',''))
vect_test = vectoriser.transform(list(text_test))
vect_test = vect_test.todense()
vect_test


#Make into df
colnames = vectoriser.vocabulary_
df_features_reduced = pd.DataFrame(vect_test, columns = colnames)
df_features_reduced.head()

#Merge back to original df
dfreduced_added = pd.merge(dfreduced,df_features_reduced,how='inner',left_index=True,right_index=True)
dfreduced_added.info()

len(dfreduced_added.columns)
#write to db - has problem. Have to recomple sqlite with higher col number
dfreduced_added = dfreduced_added.drop('twitter.text',1)
pu.toDB(con, dfreduced_added, 'features_training3000T10000FwithWords')
dfreduced_added.columns
pu.to_weka(dfreduced_added, outfile='features_reduced.csv')
    


#Evaluate and look at predicted tf

    #Output just match_rowid + text + t|f
    
    
    
    
