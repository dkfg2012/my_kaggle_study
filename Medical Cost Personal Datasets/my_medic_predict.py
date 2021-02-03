import pandas as pd
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.preprocessing import LabelEncoder
import plotly.figure_factory as ff
import seaborn as sns
import numpy as np
import matplotlib as plt
import plotly
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('insurance.csv')

# first check does dataframe contain null
# df.isnull().sum()

# preprocess the non integer type data field
le = LabelEncoder()
le.fit(df['sex'].drop_duplicates())
df['sex'] = le.transform(df['sex']) #0 represent female, 1 represent male
le.fit(df['smoker'].drop_duplicates())
df['smoker'] = le.transform(df['smoker']) #0 represent non smoker, 1 represent smoker
le.fit(df['region'].drop_duplicates())
df['region'] = le.transform(df['region'])

# draw a heatmap to check data correlation
# corr = df.corr()
# hm = go.Heatmap(
#     z = corr.values,
#     x = corr.index.values.tolist(),
#     y = corr.index.values.tolist()
# )
#
# data = [hm]
# layout = go.Layout(title='data correlation heat map')
# figure = go.Figure(data, layout)
# plot(figure)

#by the heatmap, we know the charge and smoker is correlated

#we want to have a look on charges distribution
# charge = df['charges'].values
# his = go.Histogram(
#     x = charge,
#     histnorm='probability',
#     name='Charge distribution',
#     marker=dict(color="#F5B041")
# )
# data = [his]
# layout = go.Layout(title='Charge distribution')
# figure = go.Figure(data, layout)
# plot(figure, 'my_charge_distribution')

#it is left skew

#and since we know smoker and charge are correlated, we want to
#see the distribution of smoker and charges.

#distribution of charge of smoker
# charge_smoker = df[(df['smoker'] == 1)]['charges'].values
# smoker_his = go.Histogram(
#     x = charge_smoker,
#     histnorm='probability',
#     name='Charge distribution',
#     marker=dict(color="#1FCABC")
# )
# figure = plotly.subplots.make_subplots(rows=2, cols=1, subplot_titles=("smoker", "non smoker"), print_grid=False)
# figure.append_trace(smoker_his, 1, 1)
#
# charge_nonsmok = df[(df['smoker'] == 0)]['charges'].values
# nonsmoker_his = go.Histogram(
#     x = charge_nonsmok,
#     histnorm='probability',
#     name='Charge distribution',
#     marker=dict(color="#1F89CA")
# )
# figure.append_trace(nonsmoker_his,2,1)
# figure['layout'].update(showlegend=True, title="smoker charge dist", bargap=0.05)
# plot(figure, filename="my_charge_dis_according_smoker")

# from comparing histogram, we know smoker charge more money than non smoker.

# so we now start the regression
# X = df.drop(['charges'], axis=1)
y = df['charges']
#linear regression
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
# lr = LinearRegression().fit(X_train, y_train)
# y_train_predict = lr.predict(X_train)
# y_test_predict = lr.predict(X_test)
# print(lr.score(X_test, y_test))

#linear regression after preprocessing
# X = df.drop(['charges','region'], axis=1) # if we didnt drop region, the accuracy would just a little increase compare with the last linear regression
# quad = PolynomialFeatures(degree=2) #generate polynomial feature, the degree-2 polynomial features are [1, a, b, a^2, ab, b^2], if input is [a,b]
# X_quad = quad.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_quad, y, random_state=2)
# lr = LinearRegression().fit(X_train, y_train)
# y_train_predict = lr.predict(X_train)
# y_test_predict = lr.predict(X_test)
# print(lr.score(X_test, y_test))

X = df.drop(['charges'], axis = 1)
y = df['charges']
forest = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state = 2)
forest.fit(X_train,y_train)
# forest_train_pred = forest.predict(X_train)
forest_test_pred = forest.predict(X_test)

print(r2_score(y_test,forest_test_pred))
