# -*- coding: utf-8 -*-

# ------------------------------------------------- #
# This is a MIT-BIH database ECG signal display
# ------------------------------------------------- #

import dash
from dash.dependencies import Input, Output # P2
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import numpy as np
import wfdb
import os
from collections import Counter
from numpy import nan
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

# Read signal file
record = wfdb.rdsamp('mitdb/100')
# Read annotation (strip the .dat)
ann = wfdb.rdann('mitdb/100', 'atr')

# The signal is in the position 0 of 'record'
data = record[0]

# X vector
X_Vector = list(range(0, 3000))

Y_Vector = []
Y_Vector2 = []
for el in range (0, 3000):
    Y_Vector = Y_Vector + [data[el][0]]
    Y_Vector2 = Y_Vector2 + [data[el][1]]

app = dash.Dash()

# App of graph of ECG record

app.layout = html.Div(children=[
    html.H1(children='Graph ECG MIT-BIH database with Dash'),

    html.Div(children='''
        Plotting a record of MIT-BIH database
    '''),
    # Graph of the ECG record
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': X_Vector, 'y': Y_Vector, 'type': 'line', 'name': "Graph name"},
            ],
            'layout': {
                'title': "ECG record"
            }
        }
    ),

    # Range slider
    dcc.RangeSlider(
    id='my-range-slider',
    min=0,
    max=40000,
    step=1000,
    value=[0, 3000]
    ),
    html.Div(id='output-container-range-slider'),

    # Dropdown (input database record)
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': '100', 'value': '100'},
            {'label': '101', 'value': '101'},
            {'label': '102', 'value': '102'},
            {'label': '103', 'value': '103'},
            {'label': '104', 'value': '104'},
            {'label': '105', 'value': '105'},
            {'label': '106', 'value': '106'},
            {'label': '107', 'value': '107'},
            {'label': '108', 'value': '108'},
            {'label': '109', 'value': '109'},
            {'label': '111', 'value': '111'},
            {'label': '112', 'value': '112'},
            {'label': '113', 'value': '113'},
            {'label': '114', 'value': '114'},
            {'label': '115', 'value': '115'},
            {'label': '116', 'value': '116'},
            {'label': '117', 'value': '117'},
            {'label': '118', 'value': '118'},
            {'label': '119', 'value': '119'},
            {'label': '121', 'value': '121'},
            {'label': '122', 'value': '122'},
            {'label': '123', 'value': '123'},
            {'label': '124', 'value': '124'},
            {'label': '200', 'value': '200'},
            {'label': '201', 'value': '201'},
            {'label': '202', 'value': '202'},
            {'label': '203', 'value': '203'},
            {'label': '205', 'value': '205'},
            {'label': '207', 'value': '207'},
            {'label': '208', 'value': '208'},
            {'label': '209', 'value': '209'},
            {'label': '210', 'value': '210'},
            {'label': '212', 'value': '212'},
            {'label': '213', 'value': '213'},
            {'label': '214', 'value': '214'},
            {'label': '215', 'value': '215'},
            {'label': '217', 'value': '217'},
            {'label': '219', 'value': '219'},
            {'label': '220', 'value': '220'},
            {'label': '221', 'value': '221'},
            {'label': '222', 'value': '222'},
            {'label': '223', 'value': '223'},
            {'label': '228', 'value': '228'},
            {'label': '230', 'value': '230'},
            {'label': '231', 'value': '231'},
            {'label': '232', 'value': '232'},
            {'label': '233', 'value': '233'},
            {'label': '234', 'value': '234'},
        ],
        value='100'
    ),
    html.Div(id='output-container')


])

# Graph callback callback
@app.callback(
    dash.dependencies.Output('example-graph', 'figure'),
    [dash.dependencies.Input('my-range-slider', 'value'),
     dash.dependencies.Input('my-dropdown', 'value'),
    ])
def update_graph(value, value2):
    # Read signal file
    record = wfdb.rdsamp('mitdb/'+value2)
    # Read annotation (strip the .dat)
    ann = wfdb.rdann('mitdb/'+value2, 'atr')

    # The signal is in the position 0 of 'record'
    data = record[0]

    X_Vector = list(range(value[0], value[1]))
    Y_Vector = []
    Y_Vector2 = []
    for el in range (value[0], value[1]):
        Y_Vector = Y_Vector + [data[el][0]]
        Y_Vector2 = Y_Vector2 + [data[el][1]]
    return {
        'data': [
        {'x': X_Vector, 'y': Y_Vector, 'type': 'line', 'name': "Graph name"}],
        'layout': {
            'title': "ECG record"
    
        }
        }


# Range slider callback
@app.callback(
    dash.dependencies.Output('output-container-range-slider', 'children'),
    [dash.dependencies.Input('my-range-slider', 'value')])
def update_output(value):
    return 'Samples selected: "{}"'.format(value)


if __name__ == '__main__':
    app.run_server(debug=True)
