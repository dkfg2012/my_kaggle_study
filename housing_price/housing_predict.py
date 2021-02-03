import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#preprocessing, clean data
NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])
NAs[NAs.sum(axis=1) > 0] #show columns that column sum of NaN data is larger than 0

def get_score(prediction, labels):
    print('R2: {}'.format(r2_score(prediction, labels)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, labels))))

def train_test(estimator, x_train, x_test, y_train, y_test):
    prediction_train = estimator.predict(x_train)
    get_score(prediction_train, y_train)
    prediction_test = estimator.predict(x_test)
    get_score(prediction_test, y_test)

train_labels = train.pop('SalePrice') #remove the label column in train dataset

features = pd.concat([train, test], keys=['train', 'test'])

# get rid of features that have more than half of missing information or do not correlate to SalePrice
features.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
              axis=1, inplace=True) #only when inplacec = True, columns of "features" variable would drop


#filling NA and converting features
features['MSSubClass'] = features['MSSubClass'].astype(str)
features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0]) #mode, return list that value appears most often
features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())
features['Alley'] = features['Alley'].fillna('NOACCESS')
features["OverallCond"] = features["OverallCond"].astype(str)
features["MasVnrType"] = features["MasVnrType"].fillna(features["MasVnrType"].mode()[0])

for col in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"):
    features[col] = features[col].fillna("NoBSMT")

features["TotalBsmtSF"] = features["TotalBsmtSF"].fillna(0)
features["Electrical"] = features["Electrical"].fillna(features["Electrical"].mode()[0])
features["KitchenAbvGr"] = features["KitchenAbvGr"].astype(str)
features["KitchenQual"] = features["KitchenQual"].fillna(features["KitchenQual"].mode()[0])
features["FireplaceQu"] = features["FireplaceQu"].fillna("NoFP")

for col in ("GarageType", "GarageFinish", "GarageQual"):
    features[col] = features[col].fillna("NoGRG")

features["GarageCars"] = features["GarageCars"].fillna(0.0)
features["SaleType"] = features["SaleType"].fillna(features["SaleType"].mode()[0])
features["YrSold"] = features["YrSold"].astype(str)
features["MoSold"] = features["MoSold"].astype(str)

features["TotalSF"] = features["TotalBsmtSF"] + features["1stFlrSF"] + features["2ndFlrSF"]

features.drop(["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"], axis=1, inplace=True)

# ax = sns.distplot(train_labels)
# train_labels = np.log(train_labels)
# ax = sns.distplot(train_labels)

numeric_features = features.loc[:, ["LotFrontage", "LotArea", "GrLivArea", "TotalSF"]]
numeric_features_standardized = (numeric_features - numeric_features.mean()) / numeric_features.std()

# ax = sns.pairplot(numeric_features_standardized)

# Getting Dummies from Condition1 and Condition2
conditions = set([x for x in features['Condition1']] + [x for x in features["Condition2"]])
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(conditions))), index=features.index, columns=conditions)
for i, cond in enumerate(zip(features['Condition1'], features['Condition2'])):
    dummies.loc[i, cond] = 1
features = pd.concat([features, dummies.add_prefix('Condition_')], axis=1)
features.drop(['Condition1', 'Condition2'], axis=1, inplace=True)

# Getting Dummies from Exterior1st and Exterior2nd
exteriors = set([x for x in features['Exterior1st']] + [x for x in features['Exterior2nd']])
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(exteriors))),
                       index=features.index, columns=exteriors)
for i, ext in enumerate(zip(features['Exterior1st'], features['Exterior2nd'])):
    dummies.loc[i, ext] = 1
features = pd.concat([features, dummies.add_prefix('Exterior_')], axis=1)
features.drop(['Exterior1st', 'Exterior2nd', 'Exterior_nan'], axis=1, inplace=True)


# Getting Dummies from all other categorical vars
for col in features.dtypes[features.dtypes == 'object'].index:
    for_dummy = features.pop(col)
    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)

### Copying features
features_standardized = features.copy()

### Replacing numeric features by standardized values
features_standardized.update(numeric_features_standardized) #based on features_standardized dataframe structure, update the dataframe with data in numeric_features_standardized


### Splitting features
train_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

### Splitting standardized features
train_features_st = features_standardized.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features_st = features_standardized.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

### Shuffling train sets
train_features_st, train_features, train_labels = shuffle(train_features_st, train_features, train_labels, random_state = 5) #they are consider as a merged dataset before undergoing shuffle, therefore, the features set and the label set would still in correct indexing after shuffle
# Splitting
x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=200)
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_features_st, train_labels, test_size=0.1, random_state=200)

#elasticnet used to fit standardized data
ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(x_train_st, y_train_st)
train_test(ENSTest, x_train_st, x_test_st, y_train_st, y_test_st)

# Average R2 score and standart deviation of 5-fold cross-validation
scores = cross_val_score(ENSTest, train_features_st, train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Gradient Boosting
GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(x_train, y_train)
train_test(GBest, x_train, x_test, y_train, y_test)

# Retraining models
GB_model = GBest.fit(train_features, train_labels)
ENST_model = ENSTest.fit(train_features_st, train_labels)

#Do the final prediction based on the average value of the predicted result of elastic net and gradient boosting model
# Getting our SalePrice estimation
Final_labels = (np.exp(GB_model.predict(test_features)) + np.exp(ENST_model.predict(test_features_st))) / 2

# Saving to CSV
pd.DataFrame({'Id': test.Id, 'SalePrice': Final_labels}).to_csv('result.csv', index =False)