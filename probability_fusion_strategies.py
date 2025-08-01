#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 09:36:43 2025

@authors: adugnamullissa, natetrux
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go

#function for P5 fusion method
def pChange(prior, likelihood):
    return (prior * likelihood)/((prior * likelihood) + ((1 - prior) * (1 - likelihood)))

#define PVV and PVH arrays
PVV = np.linspace(0, 1, 100) #st.slider("PVV", min_value=0.0, max_value=1.0, value=0.85, step=0.01) #np.linspace(0, 1, 100)
PVV = PVV[:,np.newaxis]
PVH = np.linspace(0, 1, 100)
X, Y = np.meshgrid(PVV, PVH)

#current luca model methods
P0 = X*Y
# print('LUCA',P0)
P00 = np.maximum(PVV,PVH)
# print('max',P00)
#soft union
P1 = PVV+PVH-PVV*PVH
# print('soft union',P1)
#bayesian 
P5 = pChange(PVV, PVH)
# print('bayesian',P5)

####################### streamlit app start ##############################

fig0 = go.Figure(data = go.Surface(x=PVH,y=PVH, z=P0,colorscale='Viridis',))

fig0.update_layout(
    title="luca multiplication",
    scene=dict(
        xaxis_title='PVV',
        yaxis_title='PVH',
        zaxis_title='Fused Probability',
        camera=dict(
            eye=dict(x=0, y=0, z=2)  # Look straight down from above
        )
    )
)


#logisstic fusion
w1, w2, b = 1.0, 1.0, -0.5
w1 = st.slider("w1", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
w2 = st.slider("w2", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
b = st.slider("b", min_value=-5.0, max_value=5.0, value=-0.5, step=0.1)


z = w1 * X + w2 * Y + b

sigmoid = 1 / (1 + np.exp(-z))


if np.any(sigmoid < 0) or np.any(sigmoid > 1):
    st.warning('⚠️ Sigmoid values out of bounds — check for overflow')
    P3 = np.zeros_like(sigmoid)
else:
    P3 = sigmoid


fig1 = go.Figure(data = go.Surface(x=PVH,y=PVH, z=P3,colorscale='Viridis',))


fig1.update_layout(
    title="logistic fusion",
    scene=dict(
        xaxis_title='PVV',
        yaxis_title='PVH',
        zaxis_title='Fused Probability',
        camera=dict(
            eye=dict(x=0, y=0, z=2)  # Look straight down from above
        )
    )
)

st.plotly_chart(fig1, use_countainer_width=True)


#weighted average
alpha = st.slider("alpha", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
P4 = alpha * X + (1 - alpha) * Y
# print('weighted average',P4)


fig2 = go.Figure(data = go.Surface(x=PVH,y=PVH, z=P4,colorscale='Viridis',))

fig2.update_layout(
    title="weighted average",
    scene=dict(
        xaxis_title='PVV',
        yaxis_title='PVH',
        zaxis_title='Fused Probability',
        camera=dict(
            eye=dict(x=0, y=0, z=2)  # Look straight down from above
        )
    )
)

st.plotly_chart(fig2, use_countainer_width=True)



upper_threshold = st.slider("upper_threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
lower_threshold = st.slider("lower_threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
#rule based
# if min(PVV, PVH) > 0.7:
#     P6 = PVV*PVH
# elif min(PVV, PVH) > 0.5:
#     P6 = max(PVV,PVH)
# else:
#     P6 = 0.0

#element-wise version
mins = np.minimum(X,Y)
maxes = np.maximum(X, Y)
P6 = np.zeros_like(X)
mask1 = mins > upper_threshold
mask2 = (mins > lower_threshold) & ~mask1

P6[mask1] = X[mask1]*Y[mask1]
P6[mask2] = maxes[mask2]


fig3 = go.Figure(data = go.Surface(x=PVH,y=PVH, z=P6,colorscale='Viridis',))


fig3.update_layout(
    title="rule-based",
    scene=dict(
        xaxis_title='PVV',
        yaxis_title='PVH',
        zaxis_title='Fused Probability',
        camera=dict(
            eye=dict(x=0, y=0, z=2)  # Look straight down from above
        )
    )
)

st.plotly_chart(fig3, use_countainer_width=True)

# fig.add_trace(go.Scatter(x=PVH, y=P0, mode='lines', name='luca multiplication'))
# fig.add_trace(go.Scatter(x=PVH, y=P00, mode='lines', name='luca maximum'))
# fig.add_trace(go.Scatter(x=PVH, y=P1, mode='lines', name='soft union'))
# fig.add_trace(go.Scatter(x=PVH, y=P3, mode='lines', name='logistic fusion'))
# fig.add_trace(go.Scatter(x=PVH, y=P4, mode='lines', name='weighted average'))
# fig.add_trace(go.Scatter(x=PVH, y=P5, mode='lines', name='bayesian'))





# fig1.show()


####################### streamlit app end ##############################