import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd


def plot_res(residuals):
    fig = go.Figure()
    for __model in residuals.columns:
        fig.add_trace(
            go.Histogram(x=residuals[__model], 
                         name = __model,
                         histnorm= 'probability', 
                         nbinsx= 10)
                )
        
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75, nbinsx = 10)
        
    return st.plotly_chart(fig, use_container_width=True)
        

def ts_plotly_chart(target, data):
    fig = go.Figure()
    fig.add_vline(x=target.index[-1], line_width=3, line_dash="dash", line_color="green")
    __colour_pallete = px.colors.qualitative.Alphabet
    #data = pd.concat([data, forecast], ignore_index= False, axis=0)

    for __model in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, 
                            y=data[__model], 
                            name = __model,
                            opacity=0.5,
                        line=dict(width=2,
                                    color=__colour_pallete.pop(),
                                    dash = 'dot')
                        )
                )
        
    # add target
    fig.add_trace(
        go.Scatter(x=target.index, 
                                 y=target, 
                                 name = target.name,
                                line=dict(width=3,
                                          color="white")
                                )
                        )
    
    return st.plotly_chart(fig, use_container_width=True)