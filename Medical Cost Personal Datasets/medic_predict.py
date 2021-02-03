import numpy as np
import pandas as pd

# Plotly Packages
import plotly
from plotly import tools
import chart_studio.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

# Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns
from string import ascii_letters

# Statistical Libraries
from scipy.stats import norm
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from scipy import stats


# Regression Modeling
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std


# Other Libraries
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('insurance.csv')
original_df = df.copy()


#plot distribution graph
# charge_distribution = df['charges'].values
# logcharge = np.log(df['charges'])
# trace0 = go.Histogram(
#     x=charge_distribution,
#     histnorm="probability",
#     name="Charge dist", marker = dict(color="blue", ))
# trace1 = go.Histogram(x=logcharge,
#                       histnorm="probability",
#                       name="Charge dist log",
#                       marker = dict(color="red", ))
#
# fig = plotly.subplots.make_subplots(rows=2, cols=1, subplot_titles=("charge dis", "log dis"), print_grid=False)
# fig.append_trace(trace0,1,1)
# fig.append_trace(trace1,2,1)
# fig['layout'].update(showlegend=True, title="charge dis", bargap=0.05)
# plot(fig, filename="custom-sized-subplot-with-subplot-titles")

# draw pie chart for age analysis
df['age_cat'] = np.nan

for i, v in enumerate(df['age']):
    if v >= 18 and v <= 35:
        df['age_cat'].iloc[i] = 'Young Adult'
    elif v > 35 and v <= 55:
        df['age_cat'].iloc[i] = 'Senior Adult'
    elif v > 55:
        df['age_cat'].iloc[i] = "Elder"
#
# labels = df['age_cat'].unique().tolist()
# amount = df['age_cat'].value_counts().tolist()
#
# colors = ['red', 'yellow', 'orange']
# trace = go.Pie(labels = labels, values=amount, hoverinfo='label+percent',
#                textinfo='value', textfont=dict(size=20),
#                marker=dict(colors=colors, line=dict(color='black', width=2))
#                )
# data = [trace]
# layout = go.Layout(title='amount by age cat')
# fig = go.Figure(data, layout)
# plot(fig, filename='pie_chart_age_cat')

# bmi dist plot
# bmi = [df['bmi'].values.tolist()]
# group_labels = ['Body mass index distribution']
# color = ['red']
# fig = ff.create_distplot(bmi, group_labels, colors=color)
# plot(fig, filename='bmi Distplot')

# data corr heat map
# corr = df.corr()
# hm = go.Heatmap(
#     z = corr.values,
#     x = corr.index.values.tolist(),
#     y = corr.index.values.tolist(),
#     # colorscale=[[0,'blue'],[1,'red']]
# )
# data = [hm]
# layout = go.Layout(title="corr heatmap")
#
# fig = go.Figure(data, layout)
# plot(fig, filename='corr_heatmap')

# box graph of bmi by cat
# young_adults = df['bmi'].loc[df['age_cat'] == "Young Adult"].values
# senior_adults = df['bmi'].loc[df['age_cat'] == "Senior Adult"].values
# elders = df['bmi'].loc[df['age_cat'] == "Elder"].values
#
# trace0 = go.Box(y=young_adults, name='young adults', boxmean=True, marker=dict(color="blue"))
# trace1 = go.Box(y=senior_adults, name='senior adults', boxmean=True, marker=dict(color="violet"))
# trace2 = go.Box(y=elders, name='elders', boxmean=True, marker=dict(color="red"))
# data = [trace0, trace1, trace2]
# layout = go.Layout(title='bmi by age cat', xaxis=dict(title='Age cat', titlefont=dict(size=16)), yaxis=dict(title='bmi', titlefont=dict(size=16)))
# fig = go.Figure(data, layout)
# plot(fig, filename='bmi_by_cat')

# statistic analysis （can't understand)
# ordinary least sqaure
# from statsmodels.formula.api import ols
# moore_lm = ols("bmi ~ age_cat", data=df).fit()
# print(moore_lm.summary())

# box graph of bmi according to age cat and is smoker
# youngAdult_smoker = df['bmi'].loc[(df['age_cat'] == "Young Adult") & (df['smoker']=='yes')].values
# seniorAdult_smoker = df['bmi'].loc[(df['age_cat'] == "Senior Adult") & (df['smoker']=='yes')].values
# eld_smoker = df['bmi'].loc[(df['age_cat'] == "Elder") & (df['smoker']=='yes')].values
#
# youngAdult_nonsmoker = df['bmi'].loc[(df['age_cat'] == "Young Adult") & (df['smoker']=='no')].values
# seniorAdult_nonsmoker = df['bmi'].loc[(df['age_cat'] == "Senior Adult") & (df['smoker']=='no')].values
# eld_nonsmoker = df['bmi'].loc[(df['age_cat'] == "Elder") & (df['smoker']=='no')].values
#
# x_data = ['youngAdult_smoker', 'seniorAdult_smoker', 'eld_smoker', 'youngAdult_nonsmoker', 'seniorAdult_nonsmoker', 'eld_nonsmoker']
# y0 = youngAdult_smoker
# y1 = youngAdult_nonsmoker
# y2 = seniorAdult_smoker
# y3 = seniorAdult_nonsmoker
# y4 = eld_smoker
# y5 = eld_nonsmoker
#
# y_data = [y0, y1, y2, y3, y4, y5]
#
# colors = ['rgba(251, 43, 43, 0.5)', 'rgba(125, 251, 137, 0.5)',
#           'rgba(251, 43, 43, 0.5)', 'rgba(125, 251, 137, 0.5)',
#           'rgba(251, 43, 43, 0.5)', 'rgba(125, 251, 137, 0.5)']
#
# traces = []
#
# for xd, yd, cls in zip(x_data, y_data, colors):
#     traces.append(go.Box(
#         y = yd,
#         name = xd,
#         boxpoints='all',
#         jitter = 0.5,
#         whiskerwidth = 0.2,
#         fillcolor = cls,
#         marker = dict(
#             size = 2
#         ),
#         line = dict(
#             width = 1
#         )
#     ))
#
# layout = go.Layout(
#     title='Body Mass Index of Smokers Status by Age Category',
#     xaxis=dict(
#     title="Status",
#     titlefont=dict(
#     size=16)),
#     yaxis=dict(
#         title="Body Mass Index",
#         autorange=True,
#         showgrid=True,
#         zeroline=True,
#         dtick=5,
#         gridcolor='rgb(255, 255, 255)',
#         gridwidth=1,
#         zerolinecolor='rgb(255, 255, 255)',
#         zerolinewidth=2,
#         titlefont=dict(
#         size=16)
#     ),
#     margin=dict(
#         l=40,
#         r=30,
#         b=80,
#         t=100,
#     ),
#     paper_bgcolor='rgb(255, 255, 255)',
#     plot_bgcolor='rgb(255, 243, 192)',
#     showlegend=False
# )
#
# fig = go.Figure(data=traces, layout=layout)
# plot(fig, filename='bmi according to age cat and smoker')

# histogram of charge fee of patient by age cat
# avg_youngadult_charge = df['charges'].loc[df['age_cat'] == "Young Adult"].mean()
# avg_senioradult_charge = df['charges'].loc[df['age_cat'] == "Senior Adult"].mean()
# avg_elder_charge = df['charges'].loc[df['age_cat'] == "Elder"].mean()
# med_youngadult_charge = df['charges'].loc[df['age_cat'] == "Young Adult"].median()
# med_senioradult_charge = df['charges'].loc[df['age_cat'] == "Senior Adult"].median()
# med_elder_charge = df['charges'].loc[df['age_cat'] == "Elder"].median()
#
# average_plot = go.Bar(x=['Young Adult','Senior Adult', 'Elder'],
#                       y=[avg_youngadult_charge, avg_senioradult_charge, avg_elder_charge],
#                       name="Mean",
#                       marker = dict(color="#F5B041")
#                       )
#
# median_plot = go.Bar(x=['Young Adult','Senior Adult', 'Elder'],
#                       y=[med_youngadult_charge, med_senioradult_charge, med_elder_charge],
#                       name="Median",
#                       marker = dict(color="#48C9B0")
#                       )
# fig = plotly.subplots.make_subplots(rows=1, cols=2, specs=[[{}, {}]],
#                                     subplot_titles = ('Average charge by age', 'median charge by age'),
#                                     shared_yaxes = True, print_grid=False
#                                     )
# fig.append_trace(average_plot, 1,1)
# fig.append_trace(median_plot, 1,2)
# fig['layout'].update(showlegend=True, title="age charges", xaxis=dict(title='age cat'),
#                      yaxis=dict(title='patient charges'), bargap = 0.15)
#
# plot(fig, filename='patient charge by age cat')

# catogorize bmi
df['weight_condition'] = np.nan
for i, v in enumerate(df['bmi']):
    if v < 18.5:
        df['weight_condition'].loc[i] = "Underweight"
    elif v >= 18.5 and v < 24.986:
        df["weight_condition"].loc[i] = "Normal Weight"
    elif v >= 25 and v < 29.926:
        df['weight_condition'].loc[i] = 'Overweight'
    elif v >= 30:
        df['weight_condition'].loc[i] = "Obese"

# subplot
# f, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,8))
# #plot relationship between charges and age
# sns.stripplot(x="age_cat", y="charges", data=df, ax=ax1, linewidth=1, palette="Reds")
# ax1.set_title("Relationship between Charges and Age")
# #plot relationship between charges, weight and age
# sns.stripplot(x="age_cat", y="charges", hue="weight_condition", data=df, ax=ax2, linewidth=1, palette="Set2")
# ax2.set_title("Relationship of Weight Condition, Age and Charges")
# #plot relationship between charges and smoker
# sns.stripplot(x="smoker", y="charges", hue="weight_condition", data=df, ax=ax3, linewidth=1, palette="Set2")
# ax3.legend_.remove()
# ax3.set_title("Relationship between Smokers and Charges")
# plt.show()

# create weight charge graph
# fig = ff.create_facet_grid(
#     df, x='age', y='charges', color_name='weight_condition',
#     show_boxes=False, marker={'size': 10, 'opacity': 1.0},
#     colormap={'Underweight': 'rgb(208, 246, 130)', 'Normal Weight': 'rgb(166, 246, 130)',
#              'Overweight': 'rgb(251, 232, 238)', 'Obese': 'rgb(253, 45, 28)'}
# )
# fig['layout'].update(title="Weight status vs charges", width=800, height=600,
#                      plot_bgcolor='rgb(251, 251, 251)', paper_bgcolor='rgb(255, 255, 255)'
#                      )
# plot(fig, filename='Weight status vs charges')

obese_avg = df['charges'].loc[df['weight_condition'] == "Obese"].mean()
df['charge_status'] = np.nan
lst = [df]
for col in lst:
    col.loc[col['charges'] > obese_avg, 'charge_status'] = "Above Average"
    col.loc[col['charges'] < obese_avg, 'charge_status'] = "Below Average"

#showing data cat by smoker or not
# sns.set(style='ticks')
# pal = ["#FA5858", "#58D3F7"]
# sns.pairplot(df, hue='smoker', palette=pal)
# plt.title('Smoker')

total_obese = len(df.loc[df['weight_condition'] == "Obese"])
obese_smoker_prop = len(df.loc[(df['weight_condition'] == "Obese") & (df['smoker'] == "yes")]) / total_obese
obese_smoker_prop = round(obese_smoker_prop, 2)

obese_nonsmoker_prop = len(df.loc[(df["weight_condition"] == "Obese") & (df["smoker"] == "no")])/total_obese
obese_nonsmoker_prop = round(obese_nonsmoker_prop, 2)

charge_obese_smoker = df.loc[(df["weight_condition"] == "Obese") & (df["smoker"] == "yes")].mean()
charge_obese_nonsmoker = df.loc[(df["weight_condition"] == "Obese") & (df["smoker"] == "no")].mean()

# kmean
# X = df[["bmi", "charges"]]
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)




from sklearn.model_selection import train_test_split

# Shuffle our dataset before splitting

original_df = original_df.sample(frac=1, random_state=1)

X = original_df.drop("charges", axis=1)
y = original_df["charges"]

# Split into both training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out

from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

# data preprocessing
# Children as categorical (ordinal varibale)
X_train["children"] = X_train["children"].astype("object")

# Separate numerics and categorical values
numerics = X_train.select_dtypes(exclude="object")
categoricals = X_train.select_dtypes(include="object")

# Pipelines
numerical_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(numerics.columns.tolist())),
    ("std_scaler", StandardScaler()),
])

categorical_pipeline =  Pipeline([
    ("select_numeric", DataFrameSelector(categoricals.columns.tolist())),
    ("std_scaler", CategoricalEncoder(encoding="onehot-dense")),
])

main_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', numerical_pipeline),
    ('cat_pipeline', categorical_pipeline)
])

# Scale our features from our training data
scaled_xtrain = main_pipeline.fit_transform(X_train)

# Let's create the training set by combining the previous X_train and y_train.
train = X_train.join(y_train, lsuffix='_X_train', rsuffix='_y_train')
test = X_test.join(y_test, lsuffix='_X_test', rsuffix='_y_test')

# Random seed
np.random.seed(42)

# Shuffle Randomly the training set
train = train.sample(frac=1)
train.head()

X_train = sm.add_constant(scaled_xtrain)
y_train = y_train.values

model = sm.OLS(y_train, X_train)
results = model.fit()
print(results.summary())