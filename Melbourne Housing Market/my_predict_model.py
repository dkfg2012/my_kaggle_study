import pandas as pd
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Melbourne_housing_FULL.csv')
original_df = df.copy()

# df.info()

#plot number of missing value in each column
total_missing = df.isnull().sum()
missing_value_histogram = go.Histogram(
    histfunc="sum",
    x = total_missing.index,
    y =  total_missing.values,
    histnorm="probability",
    name="Charge dist", marker = dict(color="#1CCBB7", )
)
fig = go.Figure()
fig.add_trace(missing_value_histogram)
# plot(fig, filename="missing value count.html")

#missing data preprocessing
#some preprocessing done by manual
df = df.drop([29483,18443,25717,27390,27391,2536,26210,12043,12043,27150,6017,25839,12539,19583,25635])
df.loc[18523,'Regionname']='Western Metropolitan'
df.loc[26888,'Regionname']='Southern Metropolitan'
df.loc[18523,'Propertycount']=7570.0
df.loc[26888,'Propertycount']=8920.0
fre = df['Bedroom2'].mode()[0]
df['Bedroom2'] = df['Bedroom2'].fillna(fre)
fre = df['Bathroom'].mode()[0]
df['Bathroom'] = df['Bathroom'].fillna(fre)

#do label encoding on regionname
le = LabelEncoder()
le.fit(df['Regionname'].unique())
df['Regionname'] = le.transform(df['Regionname'])

labeldecoder = {}
for i in range(len(le.classes_)):
    l = le.inverse_transform([i])
    labeldecoder[i] = l

#drop less important columns
df = df.drop(['SellerG', 'CouncilArea', 'Date', 'BuildingArea', 'YearBuilt', 'Landsize', 'Car'], axis=1)

#plot null after preproecssing
total_missing = df.isnull().sum()
missing_value_histogram = go.Histogram(
    histfunc="sum",
    x = total_missing.index,
    y = total_missing.values,
    histnorm="probability",
    name="Charge dist", marker = dict(color="#1CCBB7", )
)
fig = go.Figure()
fig.add_trace(missing_value_histogram)
# plot(fig, filename="missing value count after preprocessing.html")

#handle the remaining missing value
fre = df['Lattitude'].mode()[0]
df['Lattitude'] = df['Lattitude'].fillna(fre)
fre = df['Longtitude'].mode()[0]
df['Longtitude'] = df['Longtitude'].fillna(fre)

#plot price distribution
price = df['Price'].value_counts()
trace1 = go.Scatter(
    x = price.index,
    y = price.values,
    mode = 'markers',
    marker = dict(color = 'rgba(200, 50, 55, 0.8)')
)
fig = go.Figure()
fig.add_trace(trace1)
# plot(fig, filename='price distribution.html')

#draw piechart on the percentage of valid row and invalid row in terms of price
valid_rows = df[df['Price'] > 0].shape[0]
invalid_rows = df.shape[0] - valid_rows
pie_chart = go.Pie(labels=['valid', 'invalid'], values=[valid_rows, invalid_rows])
fig = go.Figure()
fig.add_trace(pie_chart)
fig.update_traces(hoverinfo='label+percent', textinfo='percent+label', textfont_size=20,
                  marker=dict(line=dict(color='#000000', width=2)))
# plot(fig, filename='valid-invalid row pie chart')

#fill the price col of the missing row
df["Price"] = df['Price'].fillna(method='ffill')
df.loc[0, 'Price'] = 1480000.0
price = df['Price'].value_counts()
trace1 = go.Scatter(
    x = price.index,
    y = price.values,
    mode = 'markers',
    marker = dict(color = 'rgba(200, 50, 55, 0.8)')
)
fig = go.Figure()
fig.add_trace(trace1)
# plot(fig, filename='price distribution after ffill.html')

corr = df[['Price', 'Rooms', 'Bedroom2', 'Bathroom', 'Longtitude', 'Postcode']].corr()
heatmap = go.Heatmap(z = corr.values, x = list(corr.index), y = list(corr.index))
fig = go.Figure()
fig.add_trace(heatmap)
# plot(fig, filename='correlation heatmap.html')

corr = df.corr()
corr['Price'].sort_values(ascending=False)

# plot number of regionname category
regionname_data = df['Regionname']
bar_chart = go.Bar(
    x = list(df['Regionname'].value_counts().index),
    y = list(df['Regionname'].value_counts().values),
    name="Regionname dist", marker = dict(color="#1CCBB7", )
)
fig = go.Figure()
fig.add_trace(bar_chart)
# plot(fig, filename='region name count.html')

#plot the mean price of each region
region_price = df.groupby('Regionname')['Price'].agg('mean')
bar_chart = go.Bar(
    x = list(region_price.index),
    y = region_price.values.tolist(),
    name='mean price of each region',
    marker = dict(color='#E4790F')
)
fig = go.Figure()
fig.add_trace(bar_chart)
# plot(fig, filename='region price.html')

#plot scatter plot according to longtitude and lattitude
ax = go.Scatter(
    x = df['Longtitude'].tolist(),
    y = df['Lattitude'].tolist(),
    mode='markers'
)
fig = go.Figure()
fig.add_trace(ax)
# plot(fig, filename='longtitude & lattitude.html')

#we focus on the data in regionname = 5, plot number of houses according to suburbs
dfs = df[df['Regionname'] == 5]
bar_chart = go.Bar(
    x = dfs['Suburb'].value_counts().index.to_list(),
    y = dfs['Suburb'].value_counts().values.tolist(),
    name='Suburbs of Southern Metropolitan',
    marker = dict(color='#E4790F')
)
fig = go.Figure()
fig.add_trace(bar_chart)
# plot(fig, filename='Suburbs of Southern Metropolitan.html')

#plot the average price in this suburbs
d = dfs.groupby('Suburb')['Price'].agg('mean').sort_values(ascending = False)
bar_chart = go.Bar(
    x = d.index.tolist(),
    y = d.values.tolist(),
    name='average price in Suburbs of Southern Metropolitan',
    marker = dict(color='#E4790F')
)
fig = go.Figure()
fig.add_trace(bar_chart)
# plot(fig, filename='average price of suburbs of southern metropolitan.html')

#Canterbury is the most expensive suburb, list all house that cost over 4m in this suburbs
print(dfs[(dfs['Type'] == 'h') & (dfs['Price'] >= 4000000) & (dfs['Suburb'] == 'Canterbury')])