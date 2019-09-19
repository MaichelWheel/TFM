# -*- coding: utf-8 -*-

# ---------------------------------------------------- #
# This is a MIT-BIH database ECG signal PCA (Class N)
# ---------------------------------------------------- #
# Classes of beats: (Typical: N, S, V, F, Q
# N		Normal beat
# L		Left bundle branch block beat
# R		Right bundle branch block beat
# B		Bundle branch block beat (unspecified)
# A		Atrial premature beat
# a		Aberrated atrial premature beat
# J		Nodal (junctional) premature beat
# S		Supraventricular premature or ectopic beat (atrial or nodal)
# V		Premature ventricular contraction
# r		R-on-T premature ventricular contraction
# F		Fusion of ventricular and normal beat
# e		Atrial escape beat
# j		Nodal (junctional) escape beat
# n		Supraventricular escape beat (atrial or nodal)
# E		Ventricular escape beat
# /		Paced beat
# f		Fusion of paced and normal beat
# Q		Unclassifiable beat
# ?		Beat not classified during learning

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

twoch_df = pd.DataFrame()
firstch_df = pd.DataFrame()
#MR20190501
Y_List = []
X_List = list(range(0,240,1))

a = 0

# We exctract 120 samples to the left and right from the beat label 
half_qrs=120

# Search in each file name of the folder 'mitdb'
for filename in os.listdir('mitdb'): # DEBUG: To avoid reading every file
#for i in range (1): # DEBUG: To avoid reading every file
    #filename = '100.dat' # DEBUG: To avoid reading every file
    # If file name is a .dat and is not one of (102, 104, 107 or 217) recordings
    if filename.endswith(".dat") and not filename.startswith(('102','104','107','217')):
        # Read annotation (strip the .dat)
        ann = wfdb.rdann('mitdb/' + filename.strip('.dat'), 'atr')
        # Read signal file
        record = wfdb.rdsamp('mitdb/' + filename.strip('.dat'))
        
        # The signal is in the position 0 of 'record'
        data = record[0]
        
        # Prepare containers
        signals, classes = [], []
        firstch_signals = []
        firstch_class = []
                
        # Beat extraction
        for it, beat in enumerate(ann.symbol):
            if beat == 'N':# in good_beats:

                a = a + 1
                
                start = ann.sample[it] - half_qrs # getting first index of beat window
                end = ann.sample[it] + half_qrs # gettint last index of beat window
                qrs = data[start : end, :] # takes 240 samples, and both channels (:)
                #MR20190501
                qrs_firstch = data[start : end, 0] # takes 240 samples, only first channel (0)
                    
                # This may happen at the edges. len(qrs) must be 240
                if len(qrs) != 2 * half_qrs: continue
                    
                # Here conditional and different Y_List for every kind of beat    
                Y_List.append(qrs_firstch)
                
                # Extracting only first channel signals and classes 
                firstch_beat_class = '{}_{}'.format(record[1]['sig_name'][0], beat)
                firstch_signals.append(qrs[:,0].reshape(240,))
                firstch_class.append(firstch_beat_class)
            
                # Keep the channel type in the class name (both channels)
                for ch in range(1): #2 DEBUG: Only 1 channel.
                    beat_class = '{}_{}'.format(record[1]['sig_name'][ch], beat)
                    signals.append(qrs[:, ch].reshape(240,))
                    classes.append(beat_class)
                
        # Build new frame of full signals (to append afterwards)       
        new_frame = pd.DataFrame({'qrs_data' : signals,
                              'qrs_type' : classes})
        
        # Build new frame of first channel (to append afterwards)
        new_frame_firstch = pd.DataFrame({'qrs_data' : firstch_signals,
                              'qrs_type' : firstch_class})
        
        # Append full channel data frame
        twoch_df = twoch_df.append(new_frame)
        # Append first channel data frame
        firstch_df = firstch_df.append(new_frame_firstch)

print(a)
        
#MR20190505
pca_df = pd.DataFrame(Y_List, columns = X_List) # 240 columns (240 samples)
#pca_df.head() # just to print

# Scale before PCA #MR20190509
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Scaler
scaler.fit(pca_df)
mean_vector = scaler.mean_

# Apply transform to both the training set and the test set.
pca_df_scale = scaler.transform(pca_df)
#pca_df_scale # just to print

# --------------------------------------------------------------------------- # 
# --------------------------------- SCALE ----------------------------------- # 
# --------------------------------------------------------------------------- # 
# Number of components
pca = PCA().fit(pca_df_scale)
"""
# Plot explained variance
plt.figure(figsize=(10,5));
plt.plot(np.linspace(1,240,240),np.cumsum(pca.explained_variance_ratio_))
plt.xlim([1,10])
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance');
plt.grid(True)
plt.yticks(np.arange(0.5, 1.01, 0.05))
plt.show()
"""
# Chosen 5 components (95% of explained variance)
pca = PCA(n_components=5)
pca.fit(pca_df_scale)
print(pca.explained_variance_ratio_)
#print(pca.components_) #5x240
"""
# Plot segment of ECG (and PCA)
pc = pca.components_
mv = mean_vector
p1 = 0
p2 = 0
p3 = 0
p4 = 0
p5 = 0

# Init value of Result_Vector: mv
Result_Vector = mv

# Plot of the ECG segment
list240 = list(range(0, 240))

app = dash.Dash()

# App of graph of ECG segment (240 samples)
app.layout = html.Div(children=[
    html.H1(children='Graph ECG with Dash: Beats N'),

    html.Div(children='''
        Plotting mean vector and principal components
    '''),
    
    # Graph of the ECG segment (240 samples)
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                # Base line
                {'x': list240, 'y': mv, 'type': 'line', 'name': "Mean vector"},
                # Positive line
                {'x': list240, 'y': Result_Vector, 'type': 'line', 'name': "Result vector"},
            ],
            'layout': {
                'title': "Hiho"
            }
        }
    ),

    # Slider 1
    dcc.Slider(
        id='my-slider',
        min=-1,
        max=1,
        step=0.1,
        value=0,
    ),
    html.Div(id='slider-output-container'),

    # Slider 2
    dcc.Slider(
        id='my-slider2',
        min=-1,
        max=1,
        step=0.1,
        value=0,
    ),
    html.Div(id='slider-output-container2'),

    # Slider 3
    dcc.Slider(
    id='my-slider3',
    min=-1,
    max=1,
    step=0.1,
    value=0,
    ),
    html.Div(id='slider-output-container3'),

    # Slider 4
    dcc.Slider(
    id='my-slider4',
    min=-1,
    max=1,
    step=0.1,
    value=0,
    ),
    html.Div(id='slider-output-container4'),

    # Slider 5
    dcc.Slider(
    id='my-slider5',
    min=-1,
    max=1,
    step=0.1,
    value=0,
    ),
    html.Div(id='slider-output-container5')

])

# ------------------------------------------------- #
# --------------------- LABELS -------------------- #
# ------------------------------------------------- #

# Slider 1 label
@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('my-slider', 'value')])
def update_output(value):        
    return 'P1: {}'.format(value)

# Slider 2 label
@app.callback(
    dash.dependencies.Output('slider-output-container2', 'children'),
    [dash.dependencies.Input('my-slider2', 'value')])
def update_output(value):
    return 'P2: {}'.format(value)

# Slider 3 label
@app.callback(
    dash.dependencies.Output('slider-output-container3', 'children'),
    [dash.dependencies.Input('my-slider3', 'value')])
def update_output(value):
    return 'P3: {}'.format(value)

# Slider 4 label
@app.callback(
    dash.dependencies.Output('slider-output-container4', 'children'),
    [dash.dependencies.Input('my-slider4', 'value')])
def update_output(value):
    return 'P4: {}'.format(value)

# Slider 5 label
@app.callback(
    dash.dependencies.Output('slider-output-container5', 'children'),
    [dash.dependencies.Input('my-slider5', 'value')])
def update_output(value):
    return 'P5: {}'.format(value)

# ------------------------------------------------- #
# ------------------- COMPONENTS ------------------ #
# ------------------------------------------------- #


# Sliders callback
@app.callback(
    dash.dependencies.Output('example-graph', 'figure'),
    [dash.dependencies.Input('my-slider', 'value'),
     dash.dependencies.Input('my-slider2', 'value'),
     dash.dependencies.Input('my-slider3', 'value'),
     dash.dependencies.Input('my-slider4', 'value'),
     dash.dependencies.Input('my-slider5', 'value')        
    ])
def update_graph(value, value2, value3, value4, value5):
    p1 = 5*value
    p2 = 5*value2
    p3 = 5*value3
    p4 = 5*value4
    p5 = 5*value5

    
    Result_Vector = mv + p1*pc[0] + p2*pc[1] + p3*pc[2] + p4*pc[3] + p5*pc[4]
    #Result_Vector = mv + Result_Vector_100 / 100
    return {
        'data': [
            {'x': list240, 'y': mv, 'type': 'line', 'name': "Mean vector"},
            {'x': list240, 'y': Result_Vector, 'type': 'line', 'name': "Result vector"}            
        ]}

if __name__ == '__main__':
    app.run_server(debug=True)
"""
