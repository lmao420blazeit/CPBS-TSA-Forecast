import streamlit as st
import pandas as pd

def variable_selectbox():
    option = st.selectbox(
    'What is the Y variable to be analyzed?',
    ('Quantity of Sales', 'Total Sales ($)', 'Sales Profit'),
    key = "var_sb")

    if option == "Quantity of Sales":
        #target = data.groupby("MonthYear")["QTD_SALES"].sum()
        #target.index = pd.to_datetime(target.index, format='%Y/%m')
        return "QTD_SALES"

    elif option == "Total Sales ($)":
        #target = data.groupby("MonthYear")["MZN_SALES"].sum()
        #target.index = pd.to_datetime(target.index, format='%Y/%m')
        return "MZN_SALES"

    else:
        #target = data.groupby("MonthYear")["PROFIT_SALES"].sum()
        #target.index = pd.to_datetime(target.index, format='%Y/%m')    
        return "PROFIT_SALES"    
    
def name_subfam_selectbox(data):
    # Create the selectbox
    selected_option = st.selectbox("Filter by product subfamily.", 
                                   options= [None] + [i for i in data["NAME_SUBFAM"].unique()])
    
    return selected_option


def estab_radio(data):
    estab = st.radio(
        "Filter by establishment name. :house:",
        options=[None] + [i for i in data["ESTAB_NAME"].unique()],
        key = "estab_rad"
    )    
    #return data.query("ESTAB_NAME == '{i}'")
    return estab