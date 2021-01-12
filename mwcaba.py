import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_table

from datetime import date

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

import pickle
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                meta_tags=[{'name': 'viewport','content': 'width=device-width, initial-scale=1.0'}])

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 1.- Predicción de demanda
# Se lee dataframe simplificado y se entrena el modelo con los mejores parametros
df_simp = pd.read_csv('https://raw.githubusercontent.com/cemljxp/eant_tp/main/APP_SIMP_MW_CABA_2017_2020.csv')
X_simp = df_simp.drop(['Date','Year','MW'], axis=1)
y_simp = df_simp['MW']
X_train_simp, X_test_simp, y_train_simp, y_test_simp = train_test_split(X_simp, y_simp, test_size=0.25, random_state=0)
model_etr_simp = ExtraTreesRegressor(criterion='mse', max_depth=20, n_estimators=130, random_state = 0)
model_etr_simp = model_etr_simp.fit(X_train_simp, y_train_simp)

#with open('modelo.pickle', 'rb') as archivo:
    #model_etr_simp = pickle.load(archivo)

# Se crea el vector con las variables predictoras
val_pred = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 2.- Datos
df_demanda = pd.read_csv('https://raw.githubusercontent.com/cemljxp/eant_tp/main/APP_VAR_MW_CABA_2017_2020.csv')
L_Date = list(df_demanda['Date'])
L_MW = list(df_demanda['MW'])
L_Temp_avg = list(df_demanda['Temp_avg'])
L_Temp_min = list(df_demanda['Temp_min'])
L_Temp_max = list(df_demanda['Temp_max'])
L_hPa = list(df_demanda['hPa'])
L_Hum = list(df_demanda['Hum'])
L_Wind_avg = list(df_demanda['Wind_avg'])
L_Wind_max = list(df_demanda['Wind_max'])

fig = make_subplots(rows=5, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)

fig.add_trace(go.Scatter(name='hPa', x=L_Date, y=L_hPa, line=dict(color='gold', width=1)),
              row=1, col=1)

fig.add_trace(go.Line(name='Temp_avg', x=L_Date, y=L_Temp_avg, line=dict(color='lawngreen', width=2)),
              row=2, col=1)

fig.add_trace(go.Scatter(name='Temp_min',x=L_Date, y=L_Temp_min, line=dict(color='deepskyblue', width=1, dash='dashdot')),
              row=2, col=1)

fig.add_trace(go.Scatter(name='Temp_max',x=L_Date, y=L_Temp_max, line=dict(color='red', width=1, dash='dashdot')),
              row=2, col=1)

fig.add_trace(go.Scatter(name='MW',x=L_Date, y=L_MW,line=dict(color='blue', width=1.5)),
              row=3, col=1)

fig.add_trace(go.Scatter(name='Hum',x=L_Date, y=L_Hum,line=dict(color='deeppink', width=1.5)),
              row=4, col=1)

fig.add_trace(go.Scatter(name='Wind_avg',x=L_Date, y=L_Wind_avg, line=dict(color='orange', width=1.5)),
              row=5, col=1)

fig.add_trace(go.Scatter(name='Wind_max',x=L_Date, y=L_Wind_max, line=dict(color='magenta', width=1, dash='dot')),
              row=5, col=1)
fig.update_yaxes(title_text="Pres. Atmosf. (hPa)", range=[970, 1050],row=1, col=1)
fig.update_yaxes(title_text="Temp. Max, Avg, Min (°C)", row=2, col=1)
fig.update_yaxes(title_text="Demanda (MW)", range=[900, 2300], row=3, col=1)
fig.update_yaxes(title_text="Humedad (%)", range=[0, 110],row=4, col=1)
fig.update_yaxes(title_text="Vel. Viento Max, Avg (km/h)", row=5, col=1)

fig.update_layout(height=1000, width=1500, title_text="Visualización de los Datos", margin=dict(l=20, r=20, t=60, b=20))
fig.update_layout({'plot_bgcolor': 'black','paper_bgcolor': '#404040','font_color': 'white'})
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))
#fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 3.- Outliers
df_demanda2 = df_demanda[df_demanda['Year']!=2020]
df_demanda2["Year"] =df_demanda2["Year"].astype(str)
fig1 = px.scatter(df_demanda2,
                 x='Date',
                 y='MW',
                 title="Demanda Eléctrica C.A.B.A. 2017-2019",
                 color="Year",
                 color_discrete_sequence=px.colors.qualitative.Set1,
                 labels = {'Date':'Fecha', 'MW':'Potencia (MW)', 'Year':'Año'})
fig1.update_layout({'plot_bgcolor': 'black','paper_bgcolor': '#404040','font_color': 'white'}, margin=dict(l=20, r=20, t=40, b=20))
fig1.update_xaxes(showgrid=False)
fig1.update_yaxes(range=[800,2400], tick0=200, dtick=200)
fig1.update_layout(height=500, width=1000)

df_demanda3 = df_demanda[(df_demanda['MW']>1200) & (df_demanda['Year']!=2020)]
df_demanda3["Year"] =df_demanda3["Year"].astype(str)
fig2 = px.scatter(df_demanda3,
                 x='Date',
                 y='MW',
                 title="Demanda Eléctrica C.A.B.A. 2017-2019",
                 color="Year",
                 color_discrete_sequence=px.colors.qualitative.Set1,
                 labels = {'Date':'Fecha', 'MW':'Potencia (MW)', 'Year':'Año'})
fig2.update_layout({'plot_bgcolor': 'black','paper_bgcolor': '#404040','font_color': 'white'}, margin=dict(l=20, r=20, t=40, b=20))
fig2.update_xaxes(showgrid=False)
fig2.update_yaxes(range=[1200,2400], tick0=200, dtick=200)
fig2.update_layout(height=500, width=1000)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 4.- Modelos
df_res = pd.read_csv('https://raw.githubusercontent.com/cemljxp/eant_tp/main/APP_RES_MW_CABA_2017_2020.csv')
L_Date = list(df_res['Date'])
L_MW = list(df_res['MW'])
L_MW_pred_knr = list(df_res['MW_pred_knr'])
L_MW_pred_lr = list(df_res['MW_pred_lr'])
L_MW_pred_dtr = list(df_res['MW_pred_dtr'])
L_MW_pred_abr = list(df_res['MW_pred_abr'])
L_MW_pred_rfr = list(df_res['MW_pred_rfr'])
L_MW_pred_etr = list(df_res['MW_pred_etr'])
L_MW_pred_gbr = list(df_res['MW_pred_gbr'])
L_MW_pred_mlpr = list(df_res['MW_pred_mlpr'])
L_MW_pred_vr = list(df_res['MW_pred_vr'])
L_MW_pred_sr = list(df_res['MW_pred_sr'])

fig3 = go.Figure()

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW, name='MW Real',
                         line=dict(color='blue', width=2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_knr, name='KNeighbors',
                         line=dict(color='firebrick', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_lr, name='LinearReg',
                         line=dict(color='deeppink', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_dtr, name='DecisionTree',
                         line=dict(color='lime', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_abr, name='AdaBoost',
                         line=dict(color='burlywood', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_rfr, name='RandomForest',
                         line=dict(color='greenyellow', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_etr, name='ExtraTrees',
                         line=dict(color='yellow', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_gbr, name='GradientBoosting',
                         line=dict(color='crimson', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_mlpr, name='MultiLayerPerceptron',
                         line=dict(color='gold', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_vr, name='VotingReg',
                         line=dict(color='red', width=1.2)))

fig3.add_trace(go.Scatter(x=L_Date, y=L_MW_pred_sr, name='StackingReg',
                         line=dict(color='aqua', width=1.2)))

fig3.update_layout(height=600, width=1650, title_text="Comparación de Modelos", margin=dict(l=20, r=20, t=80, b=20))
fig3.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'black','font_color': 'white'})
#fig3.update_xaxes(showgrid=False)
fig3.update_yaxes(showgrid=False)
fig3.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))

df_tp = pd.read_csv('https://raw.githubusercontent.com/cemljxp/eant_tp/main/APP_TOL_PRE_MW_CABA_2017_2020.csv')

l_tol_knr = list(df_tp['tol_knr'])
l_tol_lr = list(df_tp['tol_lr'])
l_tol_dtr = list(df_tp['tol_dtr'])
l_tol_abr = list(df_tp['tol_abr'])
l_tol_rfr = list(df_tp['tol_rfr'])
l_tol_etr = list(df_tp['tol_etr'])
l_tol_gbr = list(df_tp['tol_gbr'])
l_tol_mlpr = list(df_tp['tol_mlpr'])
l_tol_vr = list(df_tp['tol_vr'])
l_tol_sr = list(df_tp['tol_sr'])

l_pre_knr = list(df_tp['pre_knr'])
l_pre_lr = list(df_tp['pre_lr'])
l_pre_dtr = list(df_tp['pre_dtr'])
l_pre_abr = list(df_tp['pre_abr'])
l_pre_rfr = list(df_tp['pre_rfr'])
l_pre_etr = list(df_tp['pre_etr'])
l_pre_gbr = list(df_tp['pre_gbr'])
l_pre_mlpr = list(df_tp['pre_mlpr'])
l_pre_vr = list(df_tp['pre_vr'])
l_pre_sr = list(df_tp['pre_sr'])

fig4 = go.Figure()

fig4.add_trace(go.Scatter(x=l_tol_knr, y=l_pre_knr, name='KNeighbors',
                         line=dict(color='firebrick', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_lr, y=l_pre_lr, name='LinearReg',
                         line=dict(color='deeppink', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_dtr, y=l_pre_dtr, name='DecisionTree',
                         line=dict(color='lime', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_abr, y=l_pre_abr, name='AdaBoost',
                         line=dict(color='burlywood', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_rfr, y=l_pre_rfr, name='RandomForest',
                         line=dict(color='greenyellow', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_etr, y=l_pre_etr, name='ExtraTrees',
                         line=dict(color='yellow', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_gbr, y=l_pre_gbr, name='GradientBoosting',
                         line=dict(color='crimson', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_mlpr, y=l_pre_mlpr, name='MultiLayerPerceptron',
                         line=dict(color='gold', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_vr, y=l_pre_vr, name='VotingReg',
                         line=dict(color='red', width=2.5)))

fig4.add_trace(go.Scatter(x=l_tol_sr, y=l_pre_sr, name='StackingReg',
                         line=dict(color='aqua', width=2.5)))

fig4.update_layout(height=600, width=1650, title_text="Precisión de los modelos Vs el Toleterancia de Error",
                    xaxis_title='Toleterancia (%)',
                    yaxis_title='Precisión (%)',
                    showlegend=True)
fig4.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'black','font_color': 'white'})
#fig4.update_xaxes(showgrid=False)
fig4.update_yaxes(showgrid=False)

df_rmse = pd.read_csv('https://raw.githubusercontent.com/cemljxp/eant_tp/main/APP_RMSE_MW_CABA_2017_2020.csv')
MW_avg = df_demanda3["MW"].mean()

fig5 = px.bar(df_rmse, x='model', y='rmse', height=500, width=1000, color='rmse', color_continuous_scale='jet_r')
fig5.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'black','font_color': 'white'})
fig5.update_layout(xaxis_tickangle=-90)
fig5.update_layout(title_text='Cross-Validation RMSE (%)- MW avg = {:.2f}'.format(MW_avg))
fig5.update_yaxes(title_text="RMSE (%)", range=[0, 11])
fig5.update_xaxes(title_text="Modelos")

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 5.- COVID-19
df20 = pd.read_csv('https://raw.githubusercontent.com/cemljxp/eant_tp/main/APP_PRED20_MW_CABA_2017_2020.csv')

df20_01= df20[df20['Month']=='January']
df20_02= df20[df20['Month']=='February']
df20_03= df20[df20['Month']=='March']
df20_04= df20[df20['Month']=='April']
df20_05= df20[df20['Month']=='May']
df20_06= df20[df20['Month']=='June']
df20_07= df20[df20['Month']=='July']
df20_08= df20[df20['Month']=='August']
df20_09= df20[df20['Month']=='September']
df20_10= df20[df20['Month']=='October']
df20_11= df20[df20['Month']=='November']


L_Date20_01 = list(df20_01['Date'])
L_Date20_02 = list(df20_02['Date'])
L_Date20_03 = list(df20_03['Date'])
L_Date20_04 = list(df20_04['Date'])
L_Date20_05 = list(df20_05['Date'])
L_Date20_06 = list(df20_06['Date'])
L_Date20_07 = list(df20_07['Date'])
L_Date20_08 = list(df20_08['Date'])
L_Date20_09 = list(df20_09['Date'])
L_Date20_10 = list(df20_10['Date'])
L_Date20_11 = list(df20_11['Date'])

L_MW20_01 = list(df20_01['MW'])
L_MW20_02 = list(df20_02['MW'])
L_MW20_03 = list(df20_03['MW'])
L_MW20_04 = list(df20_04['MW'])
L_MW20_05 = list(df20_05['MW'])
L_MW20_06 = list(df20_06['MW'])
L_MW20_07 = list(df20_07['MW'])
L_MW20_08 = list(df20_08['MW'])
L_MW20_09 = list(df20_09['MW'])
L_MW20_10 = list(df20_10['MW'])
L_MW20_11 = list(df20_11['MW'])

L_MW_pred20_01 = list(df20_01['MW_pred'])
L_MW_pred20_02 = list(df20_02['MW_pred'])
L_MW_pred20_03 = list(df20_03['MW_pred'])
L_MW_pred20_04 = list(df20_04['MW_pred'])
L_MW_pred20_05 = list(df20_05['MW_pred'])
L_MW_pred20_06 = list(df20_06['MW_pred'])
L_MW_pred20_07 = list(df20_07['MW_pred'])
L_MW_pred20_08 = list(df20_08['MW_pred'])
L_MW_pred20_09 = list(df20_09['MW_pred'])
L_MW_pred20_10 = list(df20_10['MW_pred'])
L_MW_pred20_11 = list(df20_11['MW_pred'])

# Variables de Visualización de Barras de Error
ey_vis = True # Activa/Desactiva las barras de error
ey_pval = 9.2 # Porcentaje de Barras de Error



l_mse = []
l_rmse = []
y = [L_MW20_01, L_MW20_02, L_MW20_03, L_MW20_04, L_MW20_05, L_MW20_06, L_MW20_07, L_MW20_08, L_MW20_09, L_MW20_10, L_MW20_11]
y_pred = [L_MW_pred20_01, L_MW_pred20_02, L_MW_pred20_03, L_MW_pred20_04, L_MW_pred20_05, L_MW_pred20_06, L_MW_pred20_07, L_MW_pred20_08, L_MW_pred20_09, L_MW_pred20_10, L_MW_pred20_11]

for i in range (0,11):
    l_mse.append(mean_squared_error(y[i], y_pred[i]))
    l_rmse.append(np.sqrt(l_mse[i]))

mes = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre']

import plotly.graph_objects as go
fig7 = go.Figure([go.Bar(x = mes, y = l_rmse,
                        marker=dict(color=l_rmse, colorbar=dict(title="Potencia (MW)"),colorscale="jet"))])

fig7.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'black','font_color': 'white'})
fig7.update_layout(height=400, width=1000)
fig7.update_xaxes(showgrid=False)
#fig.update_yaxes(showgrid=False)
fig7.update_layout(xaxis_tickangle=0)
fig7.update_layout(title_text='RMSE de la Demanda Eléctrica 2020 (MW)', legend_title="Legend Title")
fig7.update_yaxes(title_text="Variación de Demanda Eléctrica (MW)", range=[0, 300])

l_hab = []
for r in l_rmse:
    l_hab.append(r*1000000/514)

fig8 = go.Figure([go.Bar(x = mes, y = l_hab,
                        marker=dict(color=l_hab, colorbar=dict(title="Habitantes x 1.000"),colorscale="jet"))])

fig8.update_layout({'plot_bgcolor': 'black','paper_bgcolor': 'black','font_color': 'white'})
fig8.update_layout(height=400, width=1000)
fig8.update_xaxes(showgrid=False)
#fig.update_yaxes(showgrid=False)
fig8.update_layout(xaxis_tickangle=0)
fig8.update_layout(title_text='Reducción Equivalente de Demanda Promedio de Habitantes', legend_title="miles de Habitante")
fig8.update_yaxes(title_text="Habitantes")
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
tabs_styles = {
    'height': '10px'
}
tab_style = {
    'borderBottom': '1px solid #ffffff',
    'padding': '4px',
    'fontWeight': 'bold',
    'color': 'black',
}

tab_selected_style = {
    'borderTop': '1px solid #00BFFF',
    'borderBottom': '1px solid #00BFFF',
    'backgroundColor': '#00BFFF',
    'color': 'navy',
    'padding': '4px',
    'fontWeight': 'bold'
}

theme0 =  {
    'dark': True,
    'detail': '#00BFFF',
    'primary': '#00BFFF',
    'secondary': '#00BFFF',
}

theme1 =  {
    'dark': True,
    'detail': 'limegreen',
    'primary': 'limegreen',
    'secondary': 'limegreen',
}

theme2 =  {
    'dark': True,
    'detail': 'red',
    'primary': 'red',
    'secondary': 'red',
}

theme3 =  {
    'dark': True,
    'detail': 'cyan',
    'primary': 'cyan',
    'secondary': 'cyan',
}
#------------------------------------------------------------------------------
app.layout = html.Div([
    dbc.Row([
        dbc.Col(html.H2('Estimación de Demanda Electrica CABA e Impacto del COVID-19',
            className='text-center font-weight-bolder, mb-2 mt-2'), width=12, align='center'),
            ]),
    dbc.Row([
        dbc.Col(
            dbc.Tabs([
                dbc.Tab(label='Estimación de MW', active_tab_style=tab_selected_style,
                        children=[
                        daq.DarkThemeProvider(theme=theme0, children=[
                            dbc.Row([
                                dbc.Col([
                                    dcc.DatePickerSingle(id='date_pick', date=date(2020, 1, 1), style={'margin-left': '209px', 'margin-top': '35px'}),
                                    daq.BooleanSwitch(id='sw_holiday', on=False, label = 'Feriado No Laborable', color = '#00BFFF', className='text-center mb-2 mt-2'),
                                    ]),
                                dbc.Col([
                                    html.H4('Demanda Estimada (MW)', className="text-center mt-3"),
                                    daq.LEDDisplay(id='LED_display', value=8888.88, size=60,
                                                color="#00BFFF", backgroundColor="#000000",
                                                className="text-center")
                                    ]),
                                dbc.Col([
                                    html.H4('Maximo Histórico (%)', className="text-center"),
                                    daq.Tank(id='max_hist_val',
                                    value=110,
                                    min=0,
                                    max=100,
                                    style={'margin-left': '220px'},
                                    className='mt-2',
                                    showCurrentValue=True,
                                    height = 130,
                                    width = 120)]
                                    ),
                                ]),
                            ]),
                        html.Br(),

                        dbc.Row([
                            dbc.Col(html.H5(id='temp_avg_val', style={'textAlign':'center'})),
                            dbc.Col(html.H5(id='temp_max_val', style={'textAlign':'center'})),
                            dbc.Col(html.H5(id='temp_min_val', style={'textAlign':'center'})),
                        ]),

                    daq.DarkThemeProvider(theme=theme0, children=[
                        dbc.Row([
                            dbc.Col(daq.Knob(id='temp_avg',
                                color = 'limegreen',
                                size = 100,
                                #min = round(min(L_Temp_avg))-5,
                                max = round(max(L_Temp_avg))+5,
                                value=round(sum(L_Temp_avg)/len(L_Temp_avg)),
                                className='text-center'
                                )
                            ),
                            dbc.Col(daq.Knob(id='temp_max',
                                color = 'red',
                                size = 100,
                                #min = 10,
                                max = round(max(L_Temp_max))+5,
                                value=round(max(L_Temp_max)),
                                className='text-center')
                                ),
                            dbc.Col(daq.Knob(id='temp_min',
                                color = 'cyan',
                                size = 100,
                                min = round(min(L_Temp_min))-5,
                                max = 35,
                                value=round(min(L_Temp_min)),
                                className='text-center')
                            ),
                            ])
                        ]),

                        dbc.Row([
                            dbc.Col(html.H5(id='wind_avg_val', style={'textAlign':'center'})),
                            dbc.Col(html.H5(id='hum_val', style={'textAlign':'center'})),
                            dbc.Col(html.H5(id='hpa_val', style={'textAlign':'center'})),
                        ]),


                            dbc.Row([
                                dbc.Col(dcc.Slider(id='wind_avg',
                                    min = 0,
                                    max = round(max(L_Wind_avg)),
                                    step=0.1,
                                    value=round(sum(L_Wind_avg)/len(L_Wind_avg)),
                                    updatemode='drag',
                                    className='text-right')
                                    ),
                                dbc.Col(dcc.Slider(id='hum',
                                    min = 0,
                                    max = 100,
                                    step=0.1,
                                    value=round(sum(L_Hum)/len(L_Hum)),
                                    updatemode='drag',)
                                    ),
                                dbc.Col(dcc.Slider(id='hpa',
                                    min = round(min(L_hPa))-5,
                                    max = round(max(L_hPa))+5,
                                    step=0.1,
                                    value=round(sum(L_hPa)/len(L_hPa)),
                                    updatemode='drag',)
                                    ),
                                ]),
                    daq.DarkThemeProvider(theme=theme0, children=[
                            dbc.Row([
                                dbc.Col(daq.Gauge(
                                    id='wind_avg_mea',
                                    value=5,
                                    max=round(max(L_Wind_avg)),
                                    min=0,
                                    size=130,
                                    color = 'magenta',
                                    style={'margin-left': '190px'},)
                                ),
                                dbc.Col(daq.Gauge(
                                    id='hum_mea',
                                    value=5,
                                    max=100,
                                    min=0,
                                    size=130,
                                    color = 'yellow',
                                    style={'margin-left': '190px'},)
                                ),
                                dbc.Col(daq.Gauge(
                                    id='hpa_mea',
                                    value=90,
                                    max=1200,
                                    min=0,
                                    size=130,
                                    color = 'deeppink',
                                    style={'margin-left': '190px'},)
                                ),
                            ]),
                        ]),
                    ]),
                dbc.Tab(label='Resumen',active_tab_style=tab_selected_style,
                        children=[
                            html.Br(),
                        dbc.Row([
                            html.H4('Problema:'),
                            dcc.Markdown
                            ('''
                            La demanda eléctrica de la Ciudad de Buenos Aires está relacionada a factores
                            climáticos tales como la temperatura, humedad y velocidad del viento promedio
                            que se experimentan en la ciudad, durante las estaciones de Primavera y Verano
                            se incrementa la demanda a medida que se incrementa la temperatura promedio
                            y durante Otoño e Invierno disminuye a medida que disminuye la temperatura
                            promedio, este patrón de comportamiento permite planificar mantenimientos
                            preventivos durante las épocas de menor consumo con la finalidad de garantizar
                            una mayor confiabilidad y disponibilidad del suministro de energía eléctrica
                            durante los periodos de mayor consumo.

                            A partir del **20 de Marzo 2020**, se inicia una cuarentena obligatoria en
                            varias regiones de la Argentina, lo que obligó a varios grupos industriales
                            y comerciales de diferentes sectores a paralizar sus labores cotidianas y
                            por ende a reducir su consumo de energía eléctrica, por otro lado, hubo sectores
                            que incrementaron su consumo de energía eléctrica debido al incrememento de
                            sus actividades diarias, como por ejemplo el sector del área de la salud,
                            medicina o asistencia médica, otro grupo que incrementó su consumo promedio
                            fueron los consumidores residenciales, motivado a la necesidad de pasar mayor
                            tiempo en sus hogares debido a la cuarentena obligatoria o la implementación
                            del trabajo a distancia (Home Office), esta dicotomía genera las siguientes
                            interrogantes: **¿Como fué afectada la demanda de energía eléctrica de la
                            Ciudad de Buenos Aires durante el año 2020 debido a la cuarentena obligatoria?**
                            y **¿Cuanto se redujo o incrementó la demanda de energía eléctrica en la Ciudad
                            de Buenos Aires durante la cuarentena obligatoria?**.
                            '''),
                            html.Br(),
                            html.H4('Alcance:'),
                            dcc.Markdown
                            ('''
                            Este trabajo empleará **técnicas de Machine Learning para evaluar distintos
                            modelos de aprendisaje supervisado**, utilizando el lenguaje **Python**,
                            para la **creación de un modelo que permita estimar el comportamiento de
                            la demanda eléctrica de la Ciudad de Buenos Aires** en función de variables
                            metereológicas (Temperatura, Humedad, Presión, Velocidad de Viento) y variables
                            asociadas al calendario (Día de la Semana, Mes, Feriados No Laborables),
                            se utilizará dicho modelo para **cuantificar el impacto promedio mensual**
                            de la cuarentena obligatoria, producto de la pandemia mundial asociada al
                            COVID-19, en el consumo de potencia eléctrica diaria (MW) de la Ciudad Autónoma
                            de Buenos Aires.

                            _______________________________________________________________________________
                            '''),
                        ], className='ml-5 mr-5 mb-2'),

                            dbc.Row(html.H5('Trabajo Final del Programa de Ciencia de Datos con R y Python, Escuela Argentina de Nuevas Tecnologías (EANT)'), className='ml-5 mr-5 mb-2 text-info'),
                            dbc.Row(html.H5('Integrantes: Carlos Martinez (cemljxp@gmail.com - https://github.com/cemljxp), Gustavo Prieto'), className='ml-5 mr-5 mb-2 text-success'),
                            dbc.Row(html.H5('Palabras Claves: Potencia Eléctrica, Estimación de Demanda, Predicción de Consumo, COVID-19, CABA, Argentina, Machine Learning, Python, Dash, Aprendisaje Supervisado'), className='ml-5 mr-5 mb-2'),
                            dbc.Row(html.H5('15/01/2021'), className='ml-5 mr-5 mb-2 text-warning'),

                    ]),

                dbc.Tab(label='Variables',active_tab_style=tab_selected_style,
                        children=[
                            html.Br(),
                            dbc.Row([html.H5('En esta sección se muestran las graficas de todas las variables.'),], className='ml-5 mr-5 mb-2'),
                            dbc.Row([dcc.Graph(figure=fig,style={'width': '100%','padding-left':'3%', 'padding-right':'3%'}),], className='ml-5 mr-5 mb-2'),
                            dbc.Row([html.H5('Observamos que la curva de demanda anual tiene tres (03) maximos:'),], className='ml-5 mr-5 mb-2'),
                            dbc.Row([html.H5('* A principios del mes de Enero y finales del mes de Diciembre debido al del Verano.'),], className='ml-5 mr-5 mb-2'),
                            dbc.Row([html.H5('* A mediados del mes de Julio cuando se alcanzan los mínimos de tempertura durante el Invierno.'),], className='ml-5 mr-5 mb-2'),
                            html.Br(),
                    ]),

                dbc.Tab(label='Outliers',active_tab_style=tab_selected_style,
                        children=
                            [
                            html.Br(),
                            dbc.Row(html.H5('Sí solo vemos los datos del periodo 2017-2019 podremos identificar los Outliers de este conjunto de datos que usaremos para entrenar y probar los modelos predictivios. Recordemos que los valores del año 2020 están afectados por la cuarentenea obligatoria debido a la pandemia mundial de COVID-19.'), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Graph(figure=fig1,style={'width': '100%','padding-left':'20%', 'padding-right':'25%'}), className='ml-5 mr-5 mb-2 '),
                            dbc.Row(html.H5('Se identifican dos (02) outliers:'), className='ml-5 mr-5 mb-2'),
                            dbc.Row([html.H5('* 16 de Junio 2019 (Dia de la falla que afectó a toda la Argentina y paises vecinos)'), dcc.Markdown('''[ Falla Argentina, Uruguay y Paraguay](https://es.wikipedia.org/wiki/Apag%C3%B3n_el%C3%A9ctrico_de_Argentina,_Paraguay_y_Uruguay_de_2019)''')], className='ml-5 mr-5 mb-2 text-danger'),
                            dbc.Row(html.H5('* 25 de Diciembre 2019'), className='ml-5 mr-5 mb-2 text-warning'),
                            dbc.Row(html.H5('Se decidió eliminar todos aquellos puntos con una potencia inferior a 1.200 MW'), className='ml-5 mr-5 mb-2 text-success'),

                            dbc.Row(dcc.Graph(figure=fig2,style={'width': '100%','padding-left':'20%', 'padding-right':'25%'}), className='ml-5 mr-5 mb-2'),
                            html.Br(),
                    ]),

                dbc.Tab(label='Modelos Evaluados',active_tab_style=tab_selected_style,
                        children=
                            [
                            html.Br(),
                            dbc.Row(html.H5('En esta sección se comparan los resultados de los modelos de predicción evaluados.'), className='ml-5 mr-5 mb-2'),

                            dcc.Graph(figure=fig3,style={'width': '100%','padding-left':'3%', 'padding-right':'3%'}),
                            dbc.Row(dcc.Markdown ('''Nota: Se pueden encender/apagar los resultados haciendo click en la leyenda'''), className='ml-5 mr-5 mb-2 text-warning'),

                            dbc.Row(html.H5('De la comparación gráfica de predicciones observamos que los modelos con mejores resultados son: ExtraTreeRegressor, RandomForestRegressor y VotingRegressor (conformado por ExtraTreeRegressor y RandomForestRegressor)'), className='ml-5 mr-5 mb-2 text-info'),

                            dbc.Row(html.H5('Cuantificar la precisión (observaciones predichas correctamente) usando solo la comparación grafica de los modelos, para seleccionar el mejor  modelo no es facil, en tal sentido se propone un métrica que cuantifique la precisión basada en bandas de tolerancia, es decir, cuantificar todas las predicciones que se encuentran dentro del rango del valor predicho +/- una tolerancia y así poder comparar el desempeño de cada modelo.'),
                            className='ml-5 mr-5 mb-2'),

                            dcc.Graph(figure=fig4,style={'width': '100%','padding-left':'3%', 'padding-right':'3%'}),
                            dbc.Row(dcc.Markdown ('''Nota: Se pueden encender/apagar los resultados haciendo click en la leyenda'''), className='ml-5 mr-5 mb-2 text-warning'),
                            dbc.Row(html.H5('Los modelos que alcanzan un 99% de precisión con menos del 10% de tolerancia son: ExtraTreeRegressor (9,2%), RandomForestRegressor (9,6%) y VotingRegressor (9,8%) (conformado por ExtraTreeRegressor y RandomForestRegressor'), className='ml-5 mr-5 mb-2 text-info'),

                            dbc.Row(html.H5('Comparando los valores de RMSE obtenidos por Cross-Validation para todos lo modelos'), className='ml-5 mr-5 mb-2'),
                            dcc.Graph(figure=fig5,style={'width': '100%','padding-left':'20%', 'padding-right':'25%'}),

                            dbc.Row(html.H5('Los modelos que tienen menor RMSE (%) son: VotingRegressor (4,55%), RandomForestRegressor (4,65%) y ExtraTreeRegressor (4,67%)'), className='ml-5 mr-5 mb-2 text-info'),

                            dbc.Row(html.H5('De los resultados presentados, se selecciona el ExtraTreeRegressor como mejor modelo para predicir la demanda en este caso de estudio.'), className='ml-5 mr-5 mb-2 text-success'),
                            html.Br(),
                    ]),

                dbc.Tab(label='COVID-19',active_tab_style=tab_selected_style,
                        children=
                            [
                            html.Br(),
                            dbc.Row(html.H5('En esta sección se muestra el impacto de la cuarentena obligatoria a partir del 20 de Marzo 2020 sobre la demanda de potencia eléctrica en la Ciudad de Buenos Aires'), className='ml-5 mr-5 mb-2'),
                            #dcc.Graph(figure=fig6,style={'width': '100%','padding-left':'1%', 'padding-right':'1%'}),
                            dbc.Row(daq.BooleanSwitch(id='sw_err', on=True, label = 'Barras de Error', color = '#119DFF', style={'padding-left':'1%', 'padding-right':'90%'}), className='ml-5 mr-5 mb-2'),
                            dbc.Row([
                                dbc.Col(
                                    dcc.Slider(id='err_sld', min=0, max=10, step=0.2, value=9.2,
                                        marks={0: {'label': '0%'}, 1: {'label': '1%'}, 2: {'label': '2%'},
                                            3: {'label': '3%'}, 4: {'label': '4%'}, 5: {'label': '5%'},
                                            6: {'label': '6%'}, 7: {'label': '7%'}, 8: {'label': '8%'},
                                            9: {'label': '9%'}, 10: {'label': '10%'}})
                                    ),
                                dbc.Col(),
                                ], className='ml-5 mr-5 mb-2'),
                            dbc.Row(html.Div(id='sld_err_val', style={'width': '100%','padding-left':'1%', 'padding-right':'1%'}),className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Graph(id='graph6', style={'width': '100%','padding-left':'1%', 'padding-right':'1%'}),className='ml-5 mr-5 mb-2'),

                            dbc.Row(html.H5('A partir del 20 de Marzo se empieza a notar el impacto de la cuarentena en la demanda de potencia eléctrica, es el primer punto donde la demanda real es inferior al valor mínimo de la banda inferior de tolerancia de la demanda predicha.'), className='ml-5 mr-5 mb-2'),

                            dbc.Row(dcc.Graph(figure=fig7,style={'width': '100%','padding-left':'18%', 'padding-right':'25%'}), className='ml-5 mr-5 mb-2'),

                            dbc.Row(html.H5('Para tener un orden de magnitud de cuan grande fué el impacto en la demanda se recurrió al informe de resultados del Consumo de Energía en la Ciudad de Buenos Aires 2013, Informe N° 663, publicado en marzo de 2014'), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''[Consumo de Energía en la Ciudad de Buenos Aires 2013, Informe N° 663](https://www.estadisticaciudad.gob.ar/eyc/wp-content/uploads/2015/04/ir_2014_663.pdf)'''), className='ml-5 mr-5 mb-2'),
                            dbc.Row(html.H5('En este informe se indica que el consumo promedio de energía eléctrica por habitante en CABA es de 4.500 kWh lo que equivale a un consumo promedio continuo por habitante de 514 W, en tal sentido graficamos la cantidad de habitantes promedios que representa la variación de la demanda'), className='ml-5 mr-5 mb-2'),

                            dbc.Row(dcc.Graph(figure=fig8,style={'width': '100%','padding-left':'18%', 'padding-right':'25%'}), className='ml-5 mr-5 mb-2'),

                            dbc.Row(html.H5('El impacto respecto a la cantidad de personas es equivalente a una reducción de 552 mil habitantes en promedio de consumo de energía eléctrica, lo que equivaldría al 18% de la población de acuerdo al censo de 2010 (3.075.646 hab).'), className='ml-5 mr-5 mb-2 text-danger'),

                            dbc.Row(html.H5('Se nota que la recuperación de la demanda de consumo de energía eléctrica a partir del mes Mayo 2020 coincide con las primeras medidas de flexibilización de la cuarentenea'), className='ml-5 mr-5 mb-2 text-success'),
                            html.Br(),
                            html.Br(),
                    ]),

                dbc.Tab(label='Datasets',active_tab_style=tab_selected_style,
                        children=
                            [
                            dbc.Row(dash_table.DataTable(
                                id='table',
                                columns=[{"name": i, "id": i} for i in df_demanda.columns],
                                data=df_demanda.to_dict('records'),
                                style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                                style_cell={'backgroundColor': 'rgb(50, 50, 50)',
                                            'color': 'white',
                                            'textAlign': 'center'},
                                            ), className='ml-5 mr-5'),
                            ]),

                dbc.Tab(label='Referencias', active_tab_style=tab_selected_style,
                        children=
                            [
                            dbc.Row(html.H4('Fuentes de Datos:'), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''[Base de Datos Diaria 2017 - 2020 - 12/11/2020](https://portalweb.cammesa.com/Pages/comdemcovid19.aspx)'''), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''[Generación de energía eléctrica (MW) por tipo de central. Ciudad de Buenos Aires](https://www.estadisticaciudad.gob.ar/eyc/?p=113254)'''), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''[METEORED - Datos meterológicos y ambientales históricos Agrentina](https://www.meteored.com.ar/tiempo-en_Buenos+Aires-America+Sur-Argentina-Ciudad+Autonoma+de+Buenos+Aires-SABE-sactual-13584.html)'''), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''[TUTIEMPO.NET - Datos meterológicos y ambientales históricos Agrentina](https://www.tutiempo.net/clima/01-2017/ws-875820.html)'''), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''[Dias Feriados Argentina 2017 - 2020](https://www.infobae.com/feriados-argentina/)'''), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''[16/06/2019 - Falla Sistema Eléctrico Argentina-Uruguay](https://www.infobae.com/sociedad/2019/06/16/una-falla-masiva-en-el-sistema-de-interconexion-electrica-dejo-sin-energia-a-toda-la-argentina-y-uruguay/)'''), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''[Falla Argentina, Uruguay y Paraguay](https://es.wikipedia.org/wiki/Apag%C3%B3n_el%C3%A9ctrico_de_Argentina,_Paraguay_y_Uruguay_de_2019)'''), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''[Repositorio Github del Proyecto](https://github.com/cemljxp/eant_tp)'''), className='ml-5 mr-5 mb-2'),
                            dbc.Row(html.H4('Bibliografía:'), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''Müller Andreas C. y Guido Sarah Introduction to Machine Learning with Python, O’Reilly Media, Inc. 2017.'''), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''Albon Chris Machine Learning with Python Cookbook Practical Solutions from Preprocessing to Deep Learning, O’Reilly Media, Inc. 2018.'''), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''[Plotly Python Open Source Graphing Library](https://plotly.com/python/)'''), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''[scikit-learn Machine Learning in Python](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)'''), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''[Comprehensive Python Beautiful Soup Web Scraping Tutorial](https://www.youtube.com/watch?v=GjKQ6V_ViQE)'''), className='ml-5 mr-5 mb-2'),
                            dbc.Row(dcc.Markdown('''[]()'''), className='ml-5 mr-5 mb-2'),
                            ]),

                ], className='ml-4 mr-4 mb-2'),
            width=12),
        ]),
    ])

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
@app.callback(
    [Output('temp_avg_val', 'children'),
    Output(component_id='temp_max', component_property='min'),
    Output(component_id='temp_min', component_property='max'),
    Output('temp_max_val', 'children'),
    Output('temp_min_val', 'children'),
    Output('wind_avg_val', 'children'),
    Output('hum_val', 'children'),
    Output('hpa_val', 'children'),
    Output(component_id='LED_display', component_property='value'),
    Output('max_hist_val', 'value'),
    Output('wind_avg_mea', 'value'),
    Output('hum_mea', 'value'),
    Output('hpa_mea', 'value')],
    [dash.dependencies.Input('temp_avg', 'value'),
    dash.dependencies.Input('temp_max', 'value'),
    dash.dependencies.Input('temp_min', 'value'),
    dash.dependencies.Input('wind_avg', 'value'),
    dash.dependencies.Input('hum', 'value'),
    dash.dependencies.Input('hpa', 'value'),
    dash.dependencies.Input('sw_holiday', 'on'),
    Input('date_pick', 'date')])
def update_predict(temp_avg, temp_max, temp_min, wind_avg, hum, hpa, sw_holiday, date_pick):
    temp_max_min = round(temp_avg,0) + 1
    temp_min_max = round(temp_avg,0) - 1
    if sw_holiday == True:
        val_pred[0][0] = 1
    else:
        val_pred[0][0] = 0
    val_pred[0][1] = round(temp_avg,2)
    val_pred[0][2] = round(temp_max,2)
    val_pred[0][3] = round(temp_min,2)
    val_pred[0][4] = hpa
    val_pred[0][5] = hum
    val_pred[0][6] = wind_avg
    date_object = date.fromisoformat(date_pick)
    date_p = date.weekday(date_object)
    if date_p == 5:
        val_pred[0][7] = 1
    else:
        val_pred[0][7] = 0
    if date_p == 6:
        val_pred[0][8] = 1
    else:
        val_pred[0][8] = 0
    y_pred = model_etr_simp.predict(val_pred)
    led_val = round(y_pred[0],2)
    MW_max = max(L_MW)
    p_MW_max = round(led_val*100/MW_max,1)
    return ['Temperatura Promedio ({:.2f} °C)'.format(temp_avg), temp_max_min, temp_min_max, 'Temperatura Máxima ({:.2f} °C)'.format(temp_max), 'Temperatura Mínima ({:.2f} °C)'.format(temp_min), 'Velocidad de Viento Promedio ({} km/h)'.format(wind_avg), 'Humedad Relativa ({}%) '.format(hum), 'Presión Atmosférica ({} hPa) '.format(hpa), led_val, p_MW_max, wind_avg, hum, hpa]
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
@app.callback(
    Output('graph6', 'figure'),
    Output('sld_err_val', 'children'),
    [dash.dependencies.Input('sw_err', 'on')],
    [dash.dependencies.Input('err_sld', 'value')])
def update_bars(swerr, err_sld):
    if swerr == True:
        ey_vis = True
    else:
        ey_vis = False
    ey_pval = err_sld
    fig6 = make_subplots(rows=4, cols=3,
                        vertical_spacing=0.03)
    fig6.add_trace(go.Scatter(name='MW_Ene_20', x=L_Date20_01, y=L_MW20_01, line=dict(color='blue', width=1.8)),
                  row=1, col=1)
    fig6.add_trace(go.Scatter(name='MW_pred_Ene_20', x=L_Date20_01, y=L_MW_pred20_01,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y = dict(type='percent', value = ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=1, col=1)
    fig6.add_trace(go.Scatter(name='MW_Feb_20', x=L_Date20_02, y=L_MW20_02, line=dict(color='blue', width=1.8)),
                  row=1, col=2)
    fig6.add_trace(go.Scatter(name='MW_pred_Feb_20', x=L_Date20_02, y=L_MW_pred20_02,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=1, col=2)

    fig6.add_trace(go.Scatter(name='MW_Mar_20', x=L_Date20_03, y=L_MW20_03, line=dict(color='blue', width=1.8)),
                  row=1, col=3)
    fig6.add_trace(go.Scatter(name='MW_pred_Mar_20', x=L_Date20_03, y=L_MW_pred20_03,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=1, col=3)

    fig6.add_trace(go.Scatter(name='MW_Abr_20', x=L_Date20_04, y=L_MW20_04, line=dict(color='blue', width=1.8)),
                  row=2, col=1)
    fig6.add_trace(go.Scatter(name='MW_pred_Abr_20', x=L_Date20_04, y=L_MW_pred20_04,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=2, col=1)

    fig6.add_trace(go.Scatter(name='MW_May_20', x=L_Date20_05, y=L_MW20_05, line=dict(color='blue', width=1.8)),
                  row=2, col=2)
    fig6.add_trace(go.Scatter(name='MW_pred_May_20', x=L_Date20_05, y=L_MW_pred20_05,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=2, col=2)

    fig6.add_trace(go.Scatter(name='MW_Jun_20', x=L_Date20_06, y=L_MW20_06, line=dict(color='blue', width=1.8)),
                  row=2, col=3)
    fig6.add_trace(go.Scatter(name='MW_pred_Jun_20', x=L_Date20_06, y=L_MW_pred20_06,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=2, col=3)

    fig6.add_trace(go.Scatter(name='MW_Jul_20', x=L_Date20_07, y=L_MW20_07, line=dict(color='blue', width=1.8)),
                  row=3, col=1)
    fig6.add_trace(go.Scatter(name='MW_pred_Jul_20', x=L_Date20_07, y=L_MW_pred20_07,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=3, col=1)

    fig6.add_trace(go.Scatter(name='MW_Ago_20', x=L_Date20_08, y=L_MW20_08, line=dict(color='blue', width=1.8)),
                  row=3, col=2)
    fig6.add_trace(go.Scatter(name='MW_pred_Ago_20', x=L_Date20_08, y=L_MW_pred20_08,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=3, col=2)

    fig6.add_trace(go.Scatter(name='MW_Sep_20', x=L_Date20_09, y=L_MW20_09, line=dict(color='blue', width=1.8)),
                  row=3, col=3)
    fig6.add_trace(go.Scatter(name='MW_pred_Sep_20', x=L_Date20_09, y=L_MW_pred20_09,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=3, col=3)

    fig6.add_trace(go.Scatter(name='MW_Oct_20', x=L_Date20_10, y=L_MW20_10, line=dict(color='blue', width=1.8)),
                  row=4, col=1)
    fig6.add_trace(go.Scatter(name='MW_pred_Otc_20', x=L_Date20_10, y=L_MW_pred20_10,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=4, col=1)

    fig6.add_trace(go.Scatter(name='MW_Nov_20', x=L_Date20_11, y=L_MW20_11, line=dict(color='blue', width=1.8)),
                  row=4, col=2)
    fig6.add_trace(go.Scatter(name='MW_pred_Nov_20', x=L_Date20_11, y=L_MW_pred20_11,
                             line=dict(color='red', width=1.5, dash='dot'),
                             error_y=dict(type='percent', value=ey_pval, thickness=1, width=2, visible=ey_vis)),
                  row=4, col=2)

    fig6.update_yaxes(title_text="Demanda (MW)",row=1, col=1)
    fig6.update_yaxes(title_text="Demanda (MW)", row=2, col=1)
    fig6.update_yaxes(title_text="Demanda (MW)", row=3, col=1)
    fig6.update_yaxes(title_text="Demanda (MW)", row=4, col=1)
    fig6.update_yaxes(title_text="Demanda (MW)", row=5, col=1)
    fig6.update_yaxes(title_text="Demanda (MW)", row=6, col=1)

    fig6.update_layout(height=1200, width=1650,
                      title_text="Demanda Real Vs Demanda Predicha 2020")
    fig6.update_layout({'plot_bgcolor': 'black','paper_bgcolor': '#404040','font_color': 'white'}, margin=dict(l=30, r=30, t=100, b=20))
    fig6.update_xaxes(showgrid=False)
    fig6.update_yaxes(showgrid=False)
    fig6.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))
    return fig6, 'Barras de Error: {}% '.format(err_sld)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
