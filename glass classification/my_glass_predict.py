import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import boxcox
from sklearn.decomposition import PCA
from xgboost import (XGBClassifier, plot_importance)
from sklearn.preprocessing import (FunctionTransformer, StandardScaler)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import (train_test_split, KFold , StratifiedKFold,
                                     cross_val_score, GridSearchCV,
                                     learning_curve, validation_curve)

def get_score(prediction, labels):
    print('R2: {}'.format(r2_score(prediction, labels)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, labels))))

def train_test(estimator, x_test, y_test):
    # prediction_train = estimator.predict(x_train)
    # get_score(prediction_train, y_train)
    prediction_test = estimator.predict(x_test)
    get_score(prediction_test, y_test)

df = pd.read_csv('glass.csv')
features = df.columns[:-1].tolist()
# for feat in features:
#     skew = df[feat].skew()
#     sns.distplot(df[feat], kde= False, label='Skew = %.3f' %(skew), bins=30)
#     plt.legend(loc='best')
#     plt.show()

def outlier_hunt(df):
    outlier_indices = []

    for col in df.columns.tolist():
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)

        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > 2)

    return multiple_outliers

# plt.figure(figsize=(8,6))
# sns.boxplot(df[features])
# plt.show()

outlier_indices = outlier_hunt(df[features])
df = df.drop(outlier_indices).reset_index(drop=True)

X = df[features]
y = df['Type']

random_seed = 7
test_size = 0.2

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=random_seed)

features_boxcox = []

# box cox is one type of normalization
for feature in features:
    bc_transformed, _ = boxcox(df[feature]+1)
    features_boxcox.append(bc_transformed)

features_boxcox = np.column_stack(features_boxcox)
df_bc = pd.DataFrame(data=features_boxcox, columns=features)
df_bc['Type'] = df['Type']

#xgb classifier
# xgb = XGBClassifier()
# xgb = xgb.fit(x_train, y_train)
# score = cross_val_score(xgb, x_train, y_train, cv=10)
# print(score)
# plot_importance(xgb)
# plt.show()

# pca
# pca = PCA(random_state = random_seed)
# pca.fit(x_train)

#compare different training method
n_components = 5
pipelines = []
n_estimators = 200

#print(df.shape)
pipelines.append(('SVC',Pipeline([('sc', StandardScaler()),
#('pca', PCA(n_components = n_components, random_state=seed ) ),
('SVC', SVC(random_state=random_seed))])))


pipelines.append(('KNN',Pipeline([('sc', StandardScaler()),
#('pca', PCA(n_components = n_components, random_state=random_seed ) ),
('KNN', KNeighborsClassifier())])))

pipelines.append(('RF',Pipeline([('sc', StandardScaler()),
#('pca', PCA(n_components = n_components, random_state=random_seed ) ),
('RF', RandomForestClassifier(random_state=random_seed, n_estimators=n_estimators))])))


pipelines.append(('Ada', Pipeline([('sc', StandardScaler()),
#('pca', PCA(n_components = n_components, random_state=random_seed ) ),
('Ada', AdaBoostClassifier(random_state=random_seed,  n_estimators=n_estimators))])))

pipelines.append(('ET',Pipeline([('sc', StandardScaler()),
#('pca', PCA(n_components = n_components, random_state=random_seed ) ),
('ET', ExtraTreesClassifier(random_state=random_seed, n_estimators=n_estimators))])))
pipelines.append(('GB',Pipeline([('sc', StandardScaler()),
#('pca', PCA(n_components = n_components, random_state=random_seed ) ),
('GB', GradientBoostingClassifier(random_state=random_seed)) ]) ))

pipelines.append(('LR', Pipeline([('sc', StandardScaler()),
#('pca', PCA(n_components = n_components, random_state=random_seed ) ),
('LR', LogisticRegression(random_state=random_seed)) ]) ))

results, names  = [], []
num_folds = 10
scoring = 'accuracy'

for name, model in pipelines:
    kfold = StratifiedKFold(n_splits=num_folds, random_state=random_seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring = scoring,
                                n_jobs=-1)
    results.append(cv_results)
    names.append(name)


fig = plt.figure(figsize=(12,8))
fig.suptitle("Algorithms comparison")
ax = fig.add_subplot(1,1,1)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Tuning Random Forests
pipe_rfc = Pipeline([('scl', StandardScaler()), ('rfc', RandomForestClassifier(random_state=random_seed, n_jobs=-1) )])

# Set the grid parameters
param_grid_rfc =  [ {
    'rfc__n_estimators': [100, 200,300,400], # number of estimators
    #'rfc__criterion': ['gini', 'entropy'],   # Splitting criterion
    'rfc__max_features':[0.05 , 0.1], # maximum features used at each split
    'rfc__max_depth': [None, 5], # Max depth of the trees
    'rfc__min_samples_split': [0.005, 0.01], # mininal samples in leafs
    }]
kfold = StratifiedKFold(n_splits=num_folds, random_state= random_seed)
grid_rfc = GridSearchCV(pipe_rfc, param_grid= param_grid_rfc, cv=kfold, scoring=scoring, verbose= 1, n_jobs=-1)

#Fit the pipeline
grid_rfc = grid_rfc.fit(x_train, y_train)