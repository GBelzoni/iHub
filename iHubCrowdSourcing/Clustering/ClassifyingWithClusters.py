from BoilerPlate import *

from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

#Get Annotations
pred = pd.read_csv('Results/predictions_batch4.csv')
pred.info()
pred = pred.drop_duplicates(['match_rowid'])
pred.info()
tweets = pred[['match_rowid','twitter.text']].values.tolist()



#Really rough way to clean tweets for hashtags and usernames
from ttp import ttp
import codecs
import re

p = ttp.Parser()
i=0
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

#Check if codecs will have problem - should have empty output
for tweet in cleanedTweets:
    try:
       codecs.utf_8_decode(tweet[1])
    except Exception as e:
        print cleanedTweets.index(tweet)
        print e


#Make vectorisor for training set
use_idf = True
vectorizer = TfidfVectorizer(max_df=0.5, stop_words='english', use_idf=use_idf, min_n=1, max_n=3)
#vectorizer = CountVectorizer(max_df=0.5, stop_words='english', min_n=1, max_n=3,lowercase=True)

#Make Training set
#dfcluster = pd.read_csv('Results/ComparativeAnalysis1.csv')
dfcluster = pd.ExcelFile('Results/Comparative Analysis1.xlsx').parse('TopLevelClusters')
dfcluster.head()
dfcluster.info()
clusters_initial = dfcluster['Incidences'].values
clusters_initial.shape
vec_clust_init = vectorizer.fit_transform(clusters_initial)
vec_clust_init.shape

#Regex match - tweets = { (match_rowid, tweet)}
regexs_mbsa = ['.*[Mm]ombasa.*', '.*MRC.*', '.*[Ch]angamwe.*', '.*[Mm]iritini].*']
regexs_kisumu = ['.*[Kk]isumu.*'] 
regexs = regexs_kisumu
filtered = [ tweet for rgx in regexs for tweet in tweets if re.match(rgx,tweet[1])]
len(filtered)
dfFiltered = pd.DataFrame(filtered, columns = [ 'match_rowid', 'twitter.text'])
dfFiltered = dfFiltered.drop_duplicates(cols='match_rowid')
dfFiltered.shape


X_train = vec_clust_init
targetlabels = np.unique(dfcluster['TopLevelCluster']).values.tolist()
targetmapping = dict(zip(targetlabels,range(0,len(targetlabels))))
target = [ targetmapping[lb] for lb in dfcluster['TopLevelCluster'].values.tolist()]
dfcluster['clusters']=target

dfCleanedTweets = pd.DataFrame(cleanedTweets, columns =[ 'match_rowid','twitter.text'])
dfCleanedTweets.info()

dfTweets = pd.DataFrame(tweets, columns =[ 'match_rowid','twitter.text'])

#X_test = vectorizer.transform(dfTweets['twitter.text'].values)
X_test = vectorizer.transform(dfCleanedTweets['twitter.text'].values)
X_test.shape

#fit vectoriser to select most importand features from X
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
X= X_train
y = target

t = time.time()
clf=LinearSVC(C=1,penalty ='l1',dual=False).fit(X,y)
print "training took", time.time()-t, "seconds"
#Check performance
training_predicted = clf.predict(X)
np.unique(training_predicted).shape

from sklearn.svm import SVC
Xdense = X.todense()
t =time.time()
clf2= SVC().fit(Xdense,y)
print "training took", time.time()-t, "seconds"
#Check performance
training_predicted = clf2.predict(Xdense)
np.unique(training_predicted).shape
training_predicted

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
clfnb = GradientBoostingClassifier().fit(Xdense, y)
#Check performance
training_predicted = clfnb.predict(Xdense)
np.unique(training_predicted).shape
training_predicted


from sklearn import metrics
cm_training = metrics.confusion_matrix(target,training_predicted)
print zip(target,training_predicted)

from sklearn import cross_validation
skf = cross_validation.StratifiedKFold(y=y, n_folds=3)
cross_validation.cross_val_score(clf, X, y,cv=skf, n_jobs=1)


#Assign original to clusters
Xdense_predicted = X_test.todense()
test_predicted = clf2.predict(Xdense_predicted)

dfTweets['clusters']= test_predicted
dfCleanedTweets['clusters'] = test_predicted

dfResults = pd.merge(dfCleanedTweets,dfcluster,how='inner',on='clusters')
origTweets = dfTweets[['match_rowid','twitter.text']]
dfResults = pd.merge(dfResults,origTweets,how='inner',on='match_rowid')

dfResults.to_csv('Results/trained_clusters_cleaned.csv', index=False)

#dfResults.to_excel(excel_writer, sheet_name













