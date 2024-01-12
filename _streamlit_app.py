import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from tsa_estimators._build_model import build_model
import streamlit as st
#import sys, inspect

# Local imports
from st_components.charts import plot_res, ts_plotly_chart
from st_components.selectbox import variable_selectbox, estab_radio, name_fam_selectbox, client_selectbox, metrics_selector
from tsa_estimators.custom_estimators import StackedEstimator
from utils import format_and_capitalize # pretify column headers

#clsmembers = inspect.getmembers(statsmodels.tsa.holtwinters, inspect.isclass)
#print(clsmembers)
#print(inspect.getargspec(SimpleExpSmoothing.__init__))



# Streamlit app
def main():
    data = pd.read_csv("data.csv")
    data = data[data.ESTAB_NAME != 'NAMPULA']
    st.set_option('deprecation.showPyplotGlobalUse', True)

    st.set_page_config(page_title="Time Series Forecasting",
                        layout="wide")
    
    

    col1, col2 = st.columns([0.3, 0.6])
    with col1:
        st.title("TUDOBOM Forecasting App")

    with col2:
        metrics = metrics_selector()

    col1, col2, col3, col4 = st.columns([0.25, 0.4, 0.4, 0.4])

    # Define filtering variables
    # move this into a seperate file and join into a unique func
    with col1:
        # ESTAB NAME
        estab = estab_radio(data)

    with col2:
        # NAME FAM
        prod_fam = name_fam_selectbox(data)

    with col3:
        # CLI
        client = client_selectbox(data)

    with col4:
        # Y variable
        target_var = variable_selectbox()

    if estab == None:
        #target = data.groupby("MonthYear")[target_var].sum()
        #target.index = pd.to_datetime(target.index, format='%Y/%m')
        estab = data["ESTAB_NAME"].unique()

    if prod_fam == None:
        prod_fam = data["NAME_FAM"].unique()

    if client == None:
        client = data["CLI"].unique()

    # filter the data
    target = data.query("ESTAB_NAME in @estab and NAME_FAM in @prod_fam and CLI in @client")

    target = target.groupby("MonthYear")
    target = target[target_var].sum()
    
    target.index = pd.to_datetime(target.index, format='%Y/%m')

    # defining models
    # suports all statsmodels.tsa models
    # __models: list[[model_object: statsmodels.tsa, alias: str]]
    __models = [[SimpleExpSmoothing(endog = target, 
                                   initialization_method="estimated"),
                                   "Exponential Smoothing"],
                [Holt(endog = target,
                      initialization_method = "estimated", 
                      exponential = True),
                      "Holt-model"],
                [ExponentialSmoothing(endog = target, 
                                     seasonal_periods=12, 
                                     trend="add", 
                                     seasonal="add", 
                                     use_boxcox=False, 
                                     initialization_method="estimated"),
                                     "HW-Additive"],
                [ExponentialSmoothing(endog = target, 
                                     seasonal_periods=12, 
                                     trend="add", 
                                     seasonal="mul", 
                                     use_boxcox=False, 
                                     initialization_method="estimated"),
                                     "HW-Multiplicative"],
                [ExponentialSmoothing(endog = target, 
                                     seasonal_periods=12, 
                                     trend="add", 
                                     seasonal="add", 
                                     use_boxcox=False,
                                     damped_trend = True,
                                     initialization_method="estimated"),
                                     "HW-Additive-damped-trend"],
                [ExponentialSmoothing(endog = target, 
                                     seasonal_periods=12, 
                                     trend="add", 
                                     seasonal="mul", 
                                     use_boxcox=False,
                                     damped_trend = True,
                                     initialization_method="estimated"),
                                     "HW-Multiplicative-damped-trend"]]                                     
    

    # build models
    _metrics_df, _params, _pred, _res, _forecast = build_model(models = __models, 
                                                               target = target, 
                                                               metrics = metrics)

    from sklearn.linear_model import LinearRegression

    # stacking estimator
    se = StackedEstimator(meta_classifier = LinearRegression(),
                     stackeddataset = _pred,
                     target = target,
                     alias = "Stacking estimator (LR)"
                     )
    
    # generate stacking estimator prediction and concat to baseestimators
    se_pred = se.predict(pd.concat([_pred, _forecast], 
                                   ignore_index= False, 
                                   axis=0))
    
    # generate stacking metrics and concat to baseestimators 
    se_metrics = se.get_metrics(target = target, 
                                features = _pred,
                                metrics = metrics
                                )

    # reindex for timeseries plot
    se_pred.index = pd.concat([_pred, 
                               _forecast], 
                               ignore_index= False, 
                               axis=0).index
    
    _metrics_df = pd.concat([_metrics_df, se_metrics], ignore_index= False, axis=1).T
    _metrics_df.columns = _metrics_df.columns.to_series().apply(format_and_capitalize)


    # Visualize the regression line
    ts_plotly_chart(target, pd.concat([_pred, _forecast, se_pred], ignore_index= False, axis=0))


    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.subheader("Model Evaluation:")
        st.table(_metrics_df)

    with col2:
        st.subheader("Residuals:")
        _res.columns = _res.columns.to_series().apply(format_and_capitalize)
        plot_res(_res)

    st.subheader("Params:")
    _params = pd.concat(_params, ignore_index= False, axis=1).T.drop(["use_boxcox", "lamda", "remove_bias", "initial_seasons"], axis = 1)
    _params.columns = _params.columns.to_series().apply(format_and_capitalize)
    st.table(_params)

    st.subheader("Forecast:")
    se_pred = pd.concat([se_pred, _forecast], ignore_index= False, axis=1)
    se_pred.columns = se_pred.columns.to_series().apply(format_and_capitalize)
    st.table(_forecast)



if __name__ == "__main__":
    main()