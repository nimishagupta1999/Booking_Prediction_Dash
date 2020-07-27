import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import numpy
import pandas as pd







data = pd.read_csv('final_actual_bookings.csv',header=0)
f_data = pd.read_csv('final_preds_bookings.csv',header=0)




data['Date'] = pd.to_datetime(data['Date'], format = '%d-%m-%Y' )
f_data['Date'] = pd.to_datetime(f_data['Date'], format = '%d-%m-%Y' )


data = data.sort_values(by=['Date'])

plot_data = data
plot_data = plot_data.groupby(['Date','Country'])[['Bookings']].agg('sum').reset_index()
plot_data['Date'] = plot_data['Date'].dt.strftime('%Y-%m-%d')


f_data = f_data.sort_values(by=['Date'])
# f_data['date'] = f_data['date'].dt.date
plot_f_data = f_data
plot_f_data = plot_f_data.groupby(['Date','Country'])[['Forecast']].agg('sum').reset_index()
plot_f_data['Date'] = plot_f_data['Date'].dt.strftime('%Y-%m-%d')



map = pd.read_excel('isomap.xlsx')

rdict = map.set_index('code2').to_dict()['code3']
rdict2 = map.set_index('code3').to_dict()['name']

plot_data['Country'] = plot_data['Country'].replace(rdict)
plot_data['Country_Name'] = plot_data['Country'].replace(rdict2)

plot_f_data['Country'] = plot_f_data['Country'].replace(rdict)
plot_f_data['Country_Name'] = plot_f_data['Country'].replace(rdict2)
plot_f_data['Bookings'] = plot_f_data['Forecast'].astype(int)  






df = pd.concat([plot_data, plot_f_data], sort=False)
df["Bookings"] = df["Bookings"].astype(int)





map1 = px.choropleth(df, locations="Country", color=numpy.log(df["Bookings"]), animation_frame='Date',color_continuous_scale=px.colors.sequential.YlOrRd ,hover_name=df["Country_Name"], hover_data= df[['Bookings']],width = 1200, height=520,template='plotly_dark')
map1.update_layout(title_text = 'Net Bookings across the globe',paper_bgcolor='black', plot_bgcolor='black',font = {"size": 9, "color":"White"},
        titlefont = {"size": 15, "color":"White"},)



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True






#Expedia Group logo
logo="https://www.logo-designer.co/wp-content/uploads/2018/04/2018-expedia-new-logo-design.png"

#NAVBAR------------------------------->
navbar = dbc.NavbarSimple(
    [
        html.A(
            
            dbc.Row(
                [
                    
                    dbc.Col(dbc.NavbarBrand("Expedia Group", className="project"),style={'padding-left':'30px', 'size':'20px'}),
                    dbc.Col(html.Img(src=logo, height="50px"),style={'padding-left':'30px'}),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://expedia.com",
        )
    ],
    brand="BOOKING PREDICTION PROJECT",
    color="dark",
    dark=True,
    style={'height':'80px'}
)



def get_options(list_get):
    dict_list = []
    for i in list_get:
        dict_list.append({'label': i, 'value': i})

    return dict_list





mapp = pd.read_excel('isomap.xlsx')

rdict = mapp.set_index('code2').to_dict()['code3']
rdict2 = mapp.set_index('code3').to_dict()['name']

df = pd.read_csv("final_actual_bookings.csv")
df2 = pd.read_csv("final_preds_bookings.csv")

df['Country_Code'] = df['Country'].replace(rdict)
df['Country_Name'] = df['Country_Code'].replace(rdict2)

df2['Country_Code'] = df2['Country'].replace(rdict)
df2['Country_Name'] = df2['Country_Code'].replace(rdict2)




country_options = df['Country_Name'].unique()

country_options = np.append(country_options, 'Simulation')


lob_options = df['LOB'].unique()







app.layout = html.Div(
    children=[navbar,html.Span(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='dropdown1', options=get_options(sorted(country_options)),
                                                      multi=False, value='Canada',

                                                      style={'backgroundColor': '#1E1E1E'},
                                                      className='dropdown1'

                                                      ),
                                     ],
                                     style={'color': '#1E1E1E','display':'inline-block',}
                                
                             ),
                                 html.Span(
                                     className='div-for-dropdown1',
                                     children=[
                                         dcc.Dropdown(id='dropdown2', options=get_options(lob_options),
                                                      multi=False,value='LODG', 

                                                      style={'backgroundColor': '#1E1E1E'},
                                                      className='dropdown2'

                                                      ),
                                     ],
                                     style={'color': '#1E1E1E','display':'inline-block'}
                                
                             ),
                    
                    html.Div([html.Span([dcc.Graph(id='timeseries', config={'displayModeBar': True}, animate=True)]),
                        html.Span([dcc.Textarea(
                            id='textarea-example',
                            value='Textarea content initialized\nwith multiple lines of text',
                            style={'width': '17%', 'height': 300,'display':'inline-block','margin-bottom':'85px',},
                    )])
                ]),
                    html.Div([
                              dcc.Graph(id='world_map',figure=map1,style={'width':'50%'}, config={'displayModeBar': True})
                          ])
                            
        ]

)


# Callback for timeseries forecasing of bookings
@app.callback(Output('timeseries', 'figure'),
              [Input('dropdown1', 'value'),
              Input('dropdown2', 'value')])

def update_graph(selected_dropdown_value,selected_dropdown_value2):
 

    
    cats = [selected_dropdown_value]
    lobs = [selected_dropdown_value2]

    if cats == ['Simulation']:
      return simulation_plot(cats) 

    return plot_graphs(cats,lobs)

    


def plot_graphs(cats,lobs):

    df = pd.read_csv("final_actual_bookings.csv")
    prediction_data = pd.read_csv("final_preds_bookings.csv")

    c_df = pd.read_csv("all_final_actual.csv")
    c_prediction_data = pd.read_csv("all_final_prediction.csv")



    df['Country_Code'] = df['Country'].replace(rdict)
    df['Country_Name'] = df['Country_Code'].replace(rdict2)

    prediction_data['Country_Code'] = prediction_data['Country'].replace(rdict)
    prediction_data['Country_Name'] = prediction_data['Country_Code'].replace(rdict2)

    c_df['Country_Code'] = c_df['country_code'].replace(rdict)
    c_df['Country_Name'] = c_df['Country_Code'].replace(rdict2)

    c_prediction_data['Country_Code'] = c_prediction_data['country_code'].replace(rdict)
    c_prediction_data['Country_Name'] = c_prediction_data['Country_Code'].replace(rdict2)


   

    actuals=[]
    preds=[]

    df = df.loc[df.Country_Name.isin(cats)]
    prediction_data = prediction_data.loc[prediction_data.Country_Name.isin(cats)]

    c_df = c_df.loc[c_df.Country_Name.isin(cats)]
    c_prediction_data = c_prediction_data.loc[c_prediction_data.Country_Name.isin(cats)]
    



    temp_df = df.loc[df.LOB.isin(lobs)]
    temp_df['Date'] = pd.to_datetime(temp_df['Date'], format = '%d-%m-%Y' )
    actuals.append(temp_df)
    temp_df = prediction_data.loc[prediction_data.LOB.isin(lobs)]
    temp_df['Date2'] = pd.to_datetime(temp_df['Date'], format = '%d-%m-%Y' )
    preds.append(temp_df)

    temp_df = c_df.loc[c_df.lob.isin(lobs)]
    temp_df['Date'] = pd.to_datetime(temp_df['date'], format = '%d-%m-%Y' )
    actuals.append(temp_df)
    temp_df = c_prediction_data.loc[c_prediction_data.lob.isin(lobs)]
    temp_df['Date2'] = pd.to_datetime(temp_df['date'], format = '%d-%m-%Y' )
    preds.append(temp_df)


    


    

    trace1 = go.Scatter(
        x = actuals[0]['Date'],

        y = actuals[0]['Bookings'].astype(int),
        mode = 'lines',
        name =  'Bookings Data'
    )

    

    trace2 = go.Scatter(
        x = actuals[0]['Date'],
        y = actuals[0]['Prediction'].astype(int),
        visible = 'legendonly',
        mode = 'lines',
        name = ' Bookings Predictions'
    )
    trace3 = go.Scatter(
        x = preds[0]['Date2'],
        y = preds[0]['Forecast'].astype(int),
        mode = 'lines',
        name = 'Bookings Forecast'
    )


    trace6 = go.Scatter(
        x = actuals[1]['Date'],
        y = actuals[1]['cancellations'].astype(int),
        visible = 'legendonly',

        mode = 'lines',
        name = 'Cancellation Data'
    )

    

    trace4 = go.Scatter(
        x = actuals[1]['Date'],
        y = actuals[1]['prediction'].astype(int),
        visible = 'legendonly',
        mode = 'lines',
        name = 'Cancellations Predictions',

    )
    trace5 = go.Scatter(
        x = preds[1]['Date2'],
        y = preds[1]['forecast'].astype(int),
        mode = 'lines',
        name = 'Cancellations Forecast',
        visible = 'legendonly'

    )






    print(cats[0])
  


    layout = go.Layout(
        plot_bgcolor='rgba(0,0,0,0)',
        title = {'text': 'Bookings' } ,
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Number of Bookings"},
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font = {"size": 9, "color":"White"},
        titlefont = {"size": 15, "color":"White"}
        )


    fig = go.Figure(data=[trace1,trace2,trace3,trace4,trace5,trace6], layout=layout)

    return fig





def to_supervised(data,dropNa = True,lag = 3):
    df = pd.DataFrame(data)
    column = []
    column.append(df)
    for i in range(1,lag+1):
        column.append(df.shift(-i))
    df = pd.concat(column,axis=1)
    df.dropna(inplace = True)
    features = data.shape[1]
    df = df.values
    supervised_data = df[:,:features*lag]
    supervised_data = np.column_stack( [supervised_data, df[:,features*lag]])
    return supervised_data




def f(x):
    y = 0
    result = []
    for _ in x:
        result.append(y)
        y += np.random.normal(scale=1)
    return np.array(result)


def simulation_plot(cats):



    x = np.linspace(1, 20 , num = 518)

    import pandas as pd
    Date = pd.date_range('2018-04-01', periods=518)


    Bookings = f(x)*100+5000
    # Bookings = f(x)



    df = pd.DataFrame({
                       'Bookings': Bookings,
                       'Amount': (Bookings*100)+np.random.randint(-10000, 10000) })



    from sklearn.preprocessing import MinMaxScaler
    scaler_list=[]
    features = ['Bookings','Amount']

    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    scaled = df




    timeSteps = 90

    supervised = to_supervised(scaled,lag=timeSteps)
    pd.DataFrame(supervised).head()

    # spiltting the data
    # training on only first year data


    features = df.shape[1]
    train_hours = 250
    X = supervised[:,:features*timeSteps]
    y = supervised[:,features*timeSteps]

    x_train = X[:train_hours,:]
    x_test = X[train_hours:,:]
    y_train = y[:train_hours]
    y_test = y[train_hours:]


    train_index = df.index[:train_hours]
    test_index = df.index[train_hours:]

    # print (x_train.shape,x_test.shape,y_train.shape,y_test.shape)


    #convert data to fit for lstm
    #dimensions = (sample, timeSteps here it is 1, features )

    x_train = x_train.reshape(x_train.shape[0], timeSteps, features)
    x_test = x_test.reshape(x_test.shape[0], timeSteps, features)

   

    import keras
    from keras import backend as K


    #After prediction
    K.clear_session()
    model = keras.models.load_model('models/modelqnewqq_BE')
    model_bk = keras.models.load_model('models/model_mewbqqqk_BE')

    
    dates_to_predict = 90
    future_preds = np.array([0])
    for i in range(dates_to_predict):
        new_test = x_test[-1]
        # print(new_test)
        new =  new_test.reshape((1,new_test.shape[0], new_test.shape[1]))
        temp_bk = model_bk.predict(new)
        # temp_amt = model_amt.predict(new)
        # temp_tw = model_tw.predict(new)
        temp = model.predict(new)

        # print(temp)

        new_test = np.delete(new_test, 0 , axis=0)

        # print(new_test)
        new_test = np.append(new_test,[temp,temp_bk])

        # print(new_test)
        new_test =  new_test.reshape((1,x_test.shape[1], x_test.shape[2]))
        # print(new_test)
        x_test = np.vstack([x_test, new_test])
        # print(x_test_or[j].shape)
        future_preds = np.vstack([future_preds, temp])

        # future_preds.append(temp)

    # print(future_preds)




    x_test_f = x_test
    y_pred_s = future_preds[1:]
    y_pred = model.predict(x_test)
    y_pred_2 = model.predict(x_train)
    K.clear_session()

    x_test_f = x_test_f.reshape(x_test_f.shape[0],x_test_f.shape[2]*x_test_f.shape[1])
    x_train_f = x_train.reshape(x_train.shape[0],x_train.shape[2]*x_train.shape[1])
    

    inv_new = np.concatenate( (y_pred_s, x_test_f[-1*dates_to_predict:,-1:] ) , axis =1)
    inv_new = scaler.inverse_transform(inv_new)
    final_future_pred = inv_new[:,0]
    final_future_pred = final_future_pred


 

    inv_new = np.concatenate( (y_pred, x_test_f[:,-1:] ) , axis =1)
    inv_new = scaler.inverse_transform(inv_new)
    final_pred = inv_new[:,0]
    final_pred = final_pred


    inv_new = np.concatenate( (y_pred_2, x_train_f[:,-1:] ) , axis =1)
    inv_new = scaler.inverse_transform(inv_new)
    final_pred_train = inv_new[:,0]
    final_pred_train = final_pred_train



    y_test = y_test.reshape( len(y_test), 1)


    inv_new = np.concatenate( (y_test, x_test_f[:178,-1:] ) ,axis = 1)
    inv_new = scaler.inverse_transform(inv_new)
    actual_pred = inv_new[:,0]


    y_train = y_train.reshape( len(y_train), 1)

    inv_new = np.concatenate( (y_train, x_train_f[:,-1:] ) ,axis = 1)
    inv_new = scaler.inverse_transform(inv_new)
    x_actual_pred = inv_new[:,0]

    # import tensorflow as tf 
    # tf.keras.backend.clear_session()

    import math

    import pandas as pd 
    times = pd.date_range('2020-06-27', periods=dates_to_predict, freq='1D')

    train_index_2 =  pd.date_range('2018-04-01', periods=250)
    test_index_2 = pd.date_range(train_index_2[-1], periods=268)

    np.array(times)
 
    import plotly
    import plotly.graph_objs as go
    # import chart_studio.plotly as py
    # print(x_train[:12])
    x_train = x_train.reshape(-1)
    # print(len(x_train))
    trace1 = go.Scatter(
        x = train_index_2,
        y = x_actual_pred,
        mode = 'lines',
        name = 'Training Data'
    )
    trace5 = go.Scatter(
        x = train_index_2,
        y = final_pred_train,
        mode = 'lines',
        name = 'Pred Training Data'
    )
    trace2 = go.Scatter(
        x = test_index_2,
        y = actual_pred,
        mode = 'lines',
        name = 'Testing Data'
    )
    trace3 = go.Scatter(
        x = times,
        y = abs(final_future_pred),
        mode='lines',
        name = 'Future Forecast',
        visible = 'legendonly'

    )
    trace4 = go.Scatter(
        x = test_index_2,
        y = final_pred,
        mode='lines',
        name = 'Test Prediction',
        visible = 'legendonly'
    )
    layout = go.Layout(
        title =  " Bookings",
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Number of Bookings"},

        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)'
    )
    fig = go.Figure(data=[trace1,trace2,trace4,trace5], layout=layout)
 
    return fig

















@app.callback(Output('textarea-example', 'value'),
               [Input('dropdown1', 'value'),
              Input('dropdown2', 'value')])


#country_name,lob_name
def update_output(country_name,lob_name):

    if country_name =='Simulation':
        return 'Simulation Data'
    df["Date"] = pd.to_datetime(df["Date"],format = "%d-%m-%Y")
    df['month'] = df['Date'].dt.month
    

    val1=df.loc[(df['Date'] >= '2019-10-01') & (df['Date'] <= '2019-12-31') & (df["Country_Name"] == country_name) & (df["LOB"]==lob_name), "Prediction"].sum()
    
    val2=df.loc[(df['Date'] >= '2020-01-01') & (df['Date'] < '2020-03-01') &(df["Country_Name"] == country_name) & (df["LOB"]==lob_name), "Prediction"].sum()
    
    val3=df.loc[(df['Date'] >= '2020-03-01') &(df["Country_Name"] == country_name) & (df["LOB"]==lob_name), "Prediction"].sum()

    val4=df2.loc[(df2["Country_Name"] == country_name) & (df2["LOB"]==lob_name), "Forecast"].sum()


    # print(df2.head())
    print(country_name, lob_name)

    print(df.loc[(df['Date'] >= '2020-03-01') &(df["Country_Name"] == country_name) & (df["LOB"]==lob_name)])
    # print(df.loc[((df['month'] <= 6) & (df['month'] > 3)) & df['Date'].dt.year==2020 &(df["Country_Name"] == country_name) & (df["LOB"]==lob_name)])

    text = 'Country: '+ country_name + '\nLOB: ' + lob_name+ '\n\n\nTotal number of bookings \nQ4(2019): ' + str(int(val1)) +  '\nQ1(2020): ' + str(int(val2))+ '\nQ2(2020): ' + str(int(val3))+ '\nQ3(2020): ' + str(int(val4))

    return text



if __name__ == '__main__':
    app.run_server(debug=True)