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
    
def name_fam_selectbox(data):
    # Create the selectbox
    selected_option = st.selectbox("Filter by product family.", 
                                   options= [None] + [i for i in data["NAME_FAM"].unique()])
    
    return selected_option

def client_selectbox(data):
    selected_option = st.selectbox("Filter by client.", 
                                   options= [None] + [i for i in data["CLI"].unique()])
    #return data.query("ESTAB_NAME == '{i}'")
    return selected_option


def estab_radio(data):
    estab = st.radio(
        "Filter by establishment name. :house:",
        options=[None] + [i for i in data["ESTAB_NAME"].unique()],
        key = "estab_rad",
        horizontal= True

    )    
    #return data.query("ESTAB_NAME == '{i}'")
    return estab

def metrics_selector():
    metrics = ["mean_squared_error", 
                "mean_absolute_error", 
                "mean_absolute_percentage_error", 
                "r2_score",
                "explained_variance_score",
                "mean_squared_log_error"]
    
    options = st.multiselect(
        'Select the model metrics.',
        default = metrics[:3],
        options = metrics)
    
    return options