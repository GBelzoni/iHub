from BoilerPlate import *
con = sqlite3.connect(dbpath+ dbfile, detect_types=sqlite3.PARSE_DECLTYPES)

cur = con.cursor()

sql = 'SELECT * FROM MasterData NATURAL JOIN (SELECT match_rowid , Newsworthy FROM prediction_batch4)'

data = psql.read_frame(sql, con)
data.info()

from NewsworthyTraining.FeatureExtraction import create_user_based_features
user_features = create_user_based_features(data)
user_features.info()

#Use index set operation to get columns in user_features not in data
len(user_features.columns)
unique_feature_cols = user_features.columns - data.columns
unique_feature_cols = unique_feature_cols + pd.Index(['match_rowid']) 
len(unique_feature_cols)
user_features = user_features[unique_feature_cols]

all_features = pd.merge(data,user_features,how='inner', on = 'match_rowid')
all_features.info()

#swap twitter text to end of list
cols = all_features.columns.tolist()
cols[cols.index('twitter.text')], cols[-1] = cols[-1], cols[cols.index('twitter.text')] 

all_features[cols].to_csv( 'CSV/predicted_true_alldata.csv')

all_features['twitter.text'].to_csv('CSV/pridected_true_text_alldata.csv')

