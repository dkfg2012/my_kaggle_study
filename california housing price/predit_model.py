import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from plotly.subplots import make_subplots

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