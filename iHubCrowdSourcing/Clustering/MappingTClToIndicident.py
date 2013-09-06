from BoilerPlate import *

from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

#Get Annotations
import os

dfTLClusters = pd.read_csv('Results/ClustersWithDuplicates_With_Meta_tlTrue.csv')
dfTLClusters.info()

clusters = np.unique(dfTLClusters['cluster'].values)

#Get tweets top level clusters
thisCluster = dfTLClusters[dfTLClusters['cluster'] == 'Kisumu']
#e.g. Kisumu


#Get sub-clusters
dfTL2Sub = pd.read_csv('Results/Comparative Analysis_Patrick.csv')
subClusters = dfTL2Sub['Incidences'][dfTL2Sub['TopLevelCluster'] == 'Kisumu']

#Train classifier with these categories

#Make vectorisor for training set
use_idf = True
vectorizer = TfidfVectorizer(max_df=0.5, stop_words='english', use_idf=use_idf, min_n=1, max_n=3)

X_train = vectorizer.fit_transform(subClusters)
indicator = range(0,len(subClusters))

#fit vectoriser to select most importand features from X
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
X= X_train
y = indicator

t = time.time()
clf=LinearSVC(C=1,penalty ='l1',dual=False).fit(X,y)
print "training took", time.time()-t, "seconds"
#Check performance
training_predicted = clf.predict(X)
np.unique(training_predicted).shape

from sklearn.svm import SVC
X = X_train.todense()
t =time.time()
clf2= SVC().fit(X_train,y)
print "training took", time.time()-t, "seconds"
#Check performance
training_predicted = clf2.predict(X)
np.unique(training_predicted).shape
training_predicted




X_pred = vectorizer.transform(thisCluster['twitter.text'].values)
predictions = clf2.predict(X_pred)
thisCluster['prediction']= subClusters.values[predictions]
thisCluster.to_csv('Results/thisCluster.csv')



from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
clfnb = GradientBoostingClassifier().fit(X, y)
#Check performance
training_predicted = clfnb.predict(X)
np.unique(training_predicted).shape

