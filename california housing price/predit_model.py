import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

# columns of dataset
# longitude
# latitude
# housingmedianage
# total_rooms
# total_bedrooms
# population
# households
# median_income
# medianhousevalue
# ocean_proximity

df = pd.read_csv('housing.csv')
original_df = df.copy()

#handle missing value
total_missing_value = df.isnull().sum()
missing_value_bar = go.Bar(
    x = total_missing_value.index.to_list(),
    y = total_missing_value.values.tolist(),
    name = 'missing value by column', marker=dict(color="#00FFB6")
)
fig = go.Figure()
fig.add_trace(missing_value_bar)
# plot(fig, 'missing_value.html')
#from the plot, we can see only the number of bedroom contain null

#plot the pie chart to see how many data missing
length_missing_data = df['total_bedrooms'].isnull().sum()
total_data = df.shape[0]
missing_data_pie = go.Pie(
    labels=['valid', 'missing'], values=[total_data, length_missing_data]
)
fig = go.Figure()
fig.add_trace(missing_data_pie)
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size = 20,
                  marker = dict(line=dict(color='#FF3C00', width=2)))
# plot(fig, filename='valid_invalid_pie.html')

ffill_count = df['total_bedrooms'].fillna(method='ffill')
mean_count = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())
bedroom_count_bar = go.Scatter(
    x = ffill_count.index.to_list(),
    y = ffill_count.values.tolist(),
    mode = 'lines',
    name = 'ffill distribution', marker=dict(color="#047C7B")
)

mean_count_bar = go.Scatter(
    x = mean_count.index.to_list(),
    y = mean_count.values.tolist(),
    mode = 'lines',
    name = 'count distribution', marker=dict(color="#00FFB6")
)

fig = make_subplots(rows=2, cols=1)
fig.add_trace(bedroom_count_bar, row=1, col=1)
fig.add_trace(mean_count_bar, row=2, col=1)
# plot(fig, filename='distribution of different nan fill.html')

#the distribution look very similar, therefore we may use ffill to fill the nan
df['total_bedrooms'] = df['total_bedrooms'].fillna(method='ffill')

#then lets check the type of columns, we found that only ocean_proximity column is not in numeric form
df['ocean_proximity'].unique() #there are 5 label
#so we can transform the data using label encoder
le = LabelEncoder()
le.fit(df['ocean_proximity'].unique())
df['ocean_proximity'] = le.transform(df['ocean_proximity'])

#check correlation
corr = df.corr()
heatmap = go.Heatmap(
    z = corr.values,
    x = list(corr.index),
    y = list(corr.index)
)
fig = go.Figure()
fig.add_trace(heatmap)
# plot(fig, filename='correlation heatmap.html')
#we found household, population, total_bedroom, and total_room are highly related
#median house value and median income are highly related as well

df["income_cat"] = pd.cut(df["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

df["rooms_per_household"] = df["total_rooms"]/df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"]/df["total_rooms"]
df["population_per_household"]=df["population"]/df["households"]


corr = df.corr()
heatmap = go.Heatmap(
    z = corr.values,
    x = list(corr.index),
    y = list(corr.index)
)
fig = go.Figure()
fig.add_trace(heatmap)
# plot(fig, filename='new correlation heatmap.html')


# preprocessing and model training
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

from sklearn.compose import ColumnTransformer

housing_num = housing.drop("ocean_proximity", axis=1)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]


full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
housing_prepared = full_pipeline.fit_transform(housing)

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=5, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

from sklearn.metrics import mean_squared_error, mean_absolute_error

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print("RMSE ==> ", forest_rmse)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=5)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("RMSE on Test ==> ",final_rmse)

from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))