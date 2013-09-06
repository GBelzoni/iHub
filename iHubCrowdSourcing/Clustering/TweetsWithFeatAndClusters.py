from BoilerPlate import *
con = sqlite3.connect(dbpath + dbfile, detect_types=sqlite3.PARSE_DECLTYPES)

cur = con.cursor()

sql = 'SELECT * FROM MasterData NATURAL JOIN (SELECT match_rowid , Newsworthy FROM prediction_batch4)'

data = psql.read_frame(sql, con)
data.info()

from NewsworthyTraining.FeatureExtraction import create_user_based_features
user_features = create_user_based_features(data)
user_features.info()

# Use index set operation to get columns in user_features not in data
len(user_features.columns)
unique_feature_cols = user_features.columns - data.columns
unique_feature_cols = unique_feature_cols + pd.Index(['match_rowid']) 
len(unique_feature_cols)
user_features = user_features[unique_feature_cols]

all_features = pd.merge(data, user_features, how='inner', on='match_rowid')
all_features.info()

# swap twitter text to end of list
cols = all_features.columns.tolist()
cols[cols.index('twitter.text')], cols[-1] = cols[-1], cols[cols.index('twitter.text')] 

# clusters = pd.read_csv("Results/clustering_pred_3grams.csv",sep=",")
clusters = pd.read_csv("just_cluster.csv")
clusters.info()

clustersonly = clusters[['clusters', 'match_rowid']]

clustersonly.head()
clustersonly.shape
all_features.shape


datawith_clusters = pd.merge(all_features, clustersonly, how='inner', on='match_rowid')
datawith_clusters.info()
datawith_clusters.to_csv('Results/clusters_and_timing_canclus.csv')

# From Classifying with cluster
regex_data = all_features[all_features['match_rowid'].isin(dfFiltered['match_rowid'])]
regex_data.to_csv('Results/kisumu.csv',index=False)
