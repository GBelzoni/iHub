'''
Created on Jun 27, 2013

@author: phcostello
'''
#
#if __name__ == '__main__':
#    pass



import numpy as np
import pylab as pl
import pandas as pd

from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn import datasets, svm, metrics
from sklearn import tree


# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

import pandas.io.sql as psql
import PSQLUtils
reload(PSQLUtils)
import sqlite3

import pickle



pd.set_printoptions(max_colwidth = 400)
pd.set_option(max_colwidth = 400)
path = "/home/phcostello/Documents/Data/iHub/S3_RawData/"
dbfile = "CrowdSourcingData.sqlite"
con = sqlite3.connect(path + dbfile)
PSQLUtils.showTables(con, display=True)
df = PSQLUtils.readDB(con,'FeaturesReduced3000T10000FwithWords', 0,0)


#Splitting data to features and target
len(df.columns)
target = df.pop('Newsworthy')
len(df.columns)
features = df
features.pop('rowid')
features.pop('match_rowid')



#Find char features
typeFirstCol = [type(it) for it in features.values[0]]
#Select just the unicodes
indofunicode = typeFirstCol.index(unicode)
indofunicode
##ConvertThese to nominal
##if we want to use dummy variables for categorical variables
#from sklearn.feature_extraction import DictVectorizer
#dictNominals = pd.DataFrame( features.iloc[:,indofunicode])
#dictN2 = [dict(r[1]) for r in dictNominals.iterrows()]
#vec = DictVectorizer()
#vectorised_sparse = vec.fit_transform(dictN2)
#vectorised_array = vectorised_sparse.toarray()
#vec.get_feature_names()
#
##if we want to make integers
#nomVals = features.iloc[:,indofunicode]
#levels = set(nomVals)
#lvlmap = dict( [(val,i) for i, val in enumerate(levels)])
#lvlmap
#intvals = [lvlmap[val] for val in nomVals ]
#intvals

#or using pandas
nomVals = features.iloc[:,indofunicode]
factor = pd.Categorical.from_array(nomVals)
factor.levels
factor.labels

#Reset so now we have integers, have to keep track of labelling somehow
features.iloc[:,indofunicode] = factor.labels

#Initialise tree
classifier = svm.SVC(gamma=0.001)
classifiertree = tree.DecisionTreeClassifier(criterion='entropy', 
                                             max_depth=5, 
                                             #min_samples_split, 
                                             #min_samples_leaf, 
                                             #min_density, 
                                             #max_features, 
                                             compute_importances= True)#, 
                                             #random_state)


#Fit tree
features #= features.values
target #= target.values
clf= classifier.fit(features,target)

clf = classifiertree.fit(features, target)

predicted = clf.predict(features)

#Apply metrics
metrics.confusion_matrix(target, predicted)


#visualise
import StringIO, pydot
dot_data = StringIO.StringIO()

feature_names = features.columns

tree.export_graphviz(clf, out_file= dot_data, feature_names=feature_names)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
path = '/home/phcostello/Documents/workspace/iHubCrowdSourcing/'
graph.write_pdf(path + 'outfile.pdf')





def NomToCategorical(nomVals):
    
    '''Just wrapper to convert categorical variable into int levels with level
    label mapping
    '''
    #using pandas
    factor = pd.Categorical.from_array(nomVals)
    return factor.labels, factor.levels

#Apply to bigger set
dfFull = PSQLUtils.readDB(con,'FeatureswithWords', 0,0)
fullTarget = dfFull.pop('Newsworthy')
dfFull.pop('rowid')
dfFull.pop('rowid_x')
dfFull.pop('rowid_y')
dfFull.pop('match_rowid')
fullFeatures = dfFull

#Find char features
typeFirstCol = [type(it) for it in fullFeatures.values[0]]
#Select just the unicodes
indofunicode = typeFirstCol.index(unicode)
indofunicode
##ConvertThese to nominal

#Reset so now we have integers, have to keep track of labelling somehow
nomVals = fullFeatures.iloc[:,indofunicode]
labels, levels = NomToCategorical(nomVals)
labels
levels

#Check if and null features
nans = pd.isnull(fullFeatures).any(0).nonzero()
nanrows = pd.isnull(fullFeatures).any(1).nonzero()
nanrows 
fullFeatures.iloc[70:80,16]
nans[0][0]

#Check if model features match
colfeat = set(features.columns)
colfullfeat = set(fullFeatures.columns)
colfullfeat.difference(colfeat)


fullFeatures.iloc[:,indofunicode] = labels
predictedFull = clf.predict(fullFeatures)

metrics.confusion_matrix(fullTarget, predictedFull)


#Something weird going on with the index
path2 = "/home/phcostello/Documents/workspace/iHubCrowdSourcing/"
predictedFull.index
len(predictedFull)
dfFullwittext = PSQLUtils.readDB(con,'FeaturesPlaces', 0,0, fields=['match_rowid','twitter.text'])
predictedFull.index = dfFullwittext.index
len(dfFullwittext)
dfFullwittext.columns
dfFullwittext.index
dfFullwittext['predicted'] = predictedFull
textandpredictions = dfFullwittext[['match_rowid','twitter.text','predicted']]

tptf = textandpredictions[textandpredictions['predicted']=='t']
len(tptf)
tptf.to_csv(path+'Placestextandprediction.csv')
fullrowid = set(tptf['match_rowid'])


dfreducedwittext = PSQLUtils.readDB(con,'FeaturesReduced', 0,0, fields=['match_rowid','twitter.text'])
len(dfreducedwittext)
dfreducedwittext['predicted'] = target
tpredtf = dfreducedwittext[dfreducedwittext['predicted']== 't']

redrowid = set(tpredtf['match_rowid'])
len(fullrowid.difference(redrowid))


predicted


