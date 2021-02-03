import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn import linear_model

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def get_score(prediction, labels):
    print('R2: {}'.format(r2_score(prediction, labels)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, labels))))

def train_test(estimator, x_train, x_test, y_train, y_test):
    prediction_train = estimator.predict(x_train)
    get_score(prediction_train, y_train)
    prediction_test = estimator.predict(x_test)
    get_score(prediction_test, y_test)

train_labels = train.pop('SalePrice')

dataset = pd.concat([train, test], keys=["Train", "Test"])

#simplify dataset
#remain column
remain_col = ["Id","LotFrontage","OverallQual", "GarageArea", "MSSubClass", "MSZoning", "LotArea", "LotShape", "LandSlope", "Neighborhood"]

dataset.drop([col for col in dataset.columns if col not in remain_col], axis=1, inplace=True)

# Missing_col = dataset.loc[:, dataset.isnull().sum() > 0]

dataset['MSSubClass'] = dataset['MSSubClass'].astype(str)
dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].mode()[0])
dataset['LotFrontage'] = dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean())

numeric_features = dataset.loc[:,['LotFrontage', 'LotArea', 'OverallQual', 'GarageArea']]
numeric_features_standardized = (numeric_features - numeric_features.mean()) / numeric_features.std()

dataset_standardized = dataset.copy()
dataset_standardized.update(numeric_features_standardized)

train_dataset = dataset.loc['Train'].drop("Id", axis=1).select_dtypes(include=[np.number]).values
test_dataset = dataset.loc['Test'].drop("Id", axis=1).select_dtypes(include=[np.number]).values

train_dataset_st = dataset_standardized.loc['Train'].drop("Id", axis=1).select_dtypes(include=[np.number]).values
test_dataset_st = dataset_standardized.loc['Test'].drop("Id", axis=1).select_dtypes(include=[np.number]).values

train_features_st, train_features, train_labels = shuffle(train_dataset_st, train_dataset, train_labels, random_state = 5)
x_train, x_test, y_train, y_test = train_test_split(train_dataset, train_labels, test_size=0.1, random_state=200)
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_dataset_st, train_labels, test_size=0.1, random_state=200)

ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(x_train_st, y_train_st)
train_test(ENSTest, x_train_st, x_test_st, y_train_st, y_test_st)

# Average R2 score and standart deviation of 5-fold cross-validation
scores = cross_val_score(ENSTest, train_features_st, train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))