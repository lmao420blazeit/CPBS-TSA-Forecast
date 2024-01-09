import streamlit as st
import pandas as pd

def variable_selectbox(data):
    option = st.selectbox(
    'What is the Y variable to be analyzed?',
    ('Quantity of Sales', 'Total Sales ($)', 'Sales Profit'))

    if option == "Quantity of Sales":
        target = data.groupby("MonthYear")["QTD_SALES"].sum()
        target.index = pd.to_datetime(target.index, format='%Y/%m')

    elif option == "Total Sales ($)":
        target = data.groupby("MonthYear")["MZN_SALES"].sum()
        target.index = pd.to_datetime(target.index, format='%Y/%m')

    else:
        target = data.groupby("MonthYear")["PROFIT_SALES"].sum()
        target.index = pd.to_datetime(target.index, format='%Y/%m')        

    return target