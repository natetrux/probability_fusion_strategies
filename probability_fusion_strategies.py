#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 09:36:43 2025

@author: adugnamullissa
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go

def pChange(prior, likelihood):
    return (prior * likelihood)/((prior * likelihood) + ((1 - prior) * (1 - likelihood)))

PVV = st.slider("PVV", min_value=0.0, max_value=1.0, value=0.85, step=0.01)#np.linspace(0, 1, 100)
PVH = np.linspace(0, 1, 100)

#luca
P0 = PVV*PVH
# print('LUCA',P0)
P00 = np.maximum(PVV,PVH)
# print('max',P00)




#soft union
P1 = PVV+PVH-PVV*PVH
# print('soft union',P1)




#logisstic fusion
w1, w2, b = 1.0, 1.0, -0.5
w1 = st.slider("w1", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
w2 = st.slider("w2", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
b = st.slider("b", min_value=-1.0, max_value=1.0, value=-0.5, step=0.1)


z = w1 * PVV + w2 * PVH + b

sigmoid = 1 / (1 + np.exp(-z))


if np.any(sigmoid < 0) or np.any(sigmoid > 1):
    st.warning('⚠️ Sigmoid values out of bounds — check for overflow')
    P3 = np.zeros_like(sigmoid)
else:
    P3 = sigmoid




#weighted average
alpha = st.slider("alpha", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
P4 = alpha * PVV + (1 - alpha) * PVH
# print('weighted average',P4)





#bayesian 
P5 = pChange(PVV, PVH)
# print('bayesian',P5)







# #rule based
# if min(PVV, PVH) > 0.7:
#     P6 = PVV*PVH
# elif min(PVV, PVH) > 0.5:
#     P6 = max(PVV,PVH)
# else:
#     P6 = 0.0

# print('rule based',P6)



fig = go.Figure()
fig.add_trace(go.Scatter(x=PVH, y=P0, mode='lines', name='luca multiplication'))
fig.add_trace(go.Scatter(x=PVH, y=P00, mode='lines', name='luca maximum'))
fig.add_trace(go.Scatter(x=PVH, y=P1, mode='lines', name='soft union'))
fig.add_trace(go.Scatter(x=PVH, y=P3, mode='lines', name='logistic fusion'))
fig.add_trace(go.Scatter(x=PVH, y=P4, mode='lines', name='weighted average'))
fig.add_trace(go.Scatter(x=PVH, y=P5, mode='lines', name='bayesian'))

fig.update_layout(
    title="probability fusion strategies",
    xaxis_title = "VVH",
    yaxis_title = "y"
)

st.plotly_chart(fig, use_countainer_width=True)