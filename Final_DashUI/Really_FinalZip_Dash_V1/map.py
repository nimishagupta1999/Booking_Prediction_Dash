import pandas as pd

data = pd.read_csv('new_really_actual_1.csv',header=0)

# data['Date2'] = pd.date_range('2018-04-01', periods=len(data))

data['Date'] = pd.to_datetime(data['Date'], format = '%d-%m-%Y' )


data = data.sort_values(by=['Date'])
# data['date'] = data['date'].dt.date
plot_data = data
plot_data = plot_data.groupby(['Date','Country'])[['Bookings']].agg('sum').reset_index()
plot_data['Date'] = plot_data['Date'].dt.strftime('%Y-%m-%d')

# plot_data.head(10)
print(plot_data.shape)

# import plotly.express as px
map = pd.read_excel('isomap.xlsx')

rdict = map.set_index('code2').to_dict()['code3']
rdict2 = map.set_index('code3').to_dict()['name']

plot_data['Country'] = plot_data['Country'].replace(rdict)
plot_data['Country_Name'] = plot_data['Country'].replace(rdict2)


# plot_data.to_csv('worlddata.csv')


import plotly.express as px
import numpy
df = plot_data.iloc[:200] 
fig = px.choropleth(df, locations="Country", color=numpy.log(df["Bookings"]), animation_frame='Date', color_continuous_scale=px.colors.sequential.Sunset ,
hover_name=df["Country_Name"], hover_data= df[['Bookings']] # column to add to hover information 
# color_continuous_scale=px.colors.sequential.Plasma
) 
fig.show()