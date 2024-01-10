import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
#from statsmodels.tsa.arima.model import ARIMA
from tsa_estimators._build_model import build_model
import streamlit as st
from st_components.charts import plot_res, ts_plotly_chart
from st_components.selectbox import variable_selectbox, estab_radio, name_subfam_selectbox
from tsa_estimators.custom_estimators import StackedEstimator
#import sys, inspect

#clsmembers = inspect.getmembers(statsmodels.tsa.holtwinters, inspect.isclass)
#print(clsmembers)
#print(inspect.getargspec(SimpleExpSmoothing.__init__))

# Streamlit app
def main():
    data = pd.read_csv("data.csv")
    data = data[data.ESTAB_NAME != 'NAMPULA']
    st.set_option('deprecation.showPyplotGlobalUse', True)

    st. set_page_config(page_title="Time Series Forecasting",
                        layout="wide")
    st.title("Forecasting App")

    col1, col2, col3 = st.columns([0.25, 0.4, 0.4])

    with col1:
        estab = estab_radio(data)

    with col2:
        prod_subfam = name_subfam_selectbox(data)

    with col3:
        target_var = variable_selectbox()

    if estab == None:
        #target = data.groupby("MonthYear")[target_var].sum()
        #target.index = pd.to_datetime(target.index, format='%Y/%m')
        estab = data["ESTAB_NAME"].unique()

    if prod_subfam == None:
        prod_subfam = data["NAME_SUBFAM"].unique()

    target = data.query("ESTAB_NAME in @estab and NAME_SUBFAM in @prod_subfam")
    target = target.groupby("MonthYear")
    target = target[target_var].sum()
    
    target.index = pd.to_datetime(target.index, format='%Y/%m')


    # defining models
    # suports all statsmodels.tsa models
    __models = [[SimpleExpSmoothing(endog = target, 
                                   initialization_method="estimated"),
                                   "SimpleExponentialSmoothing"],
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
    _metrics_df, _params, _pred, _res, _forecast = build_model(__models, target)

    from sklearn.linear_model import LinearRegression

    # stacking estimator
    se = StackedEstimator(stacking_method = LinearRegression(),
                     stackeddataset = _pred,
                     target = target
                     )
    
    # generate stacking estimator prediction and concat to baseestimators
    se_pred = se.predict(pd.concat([_pred, _forecast], 
                                   ignore_index= False, 
                                   axis=0))
    
    # generate stacking metrics and concat to baseestimators 
    se_metrics = se.get_metrics(target = target, 
                                features = _pred)

    # reindex for timeseries plot
    se_pred.index = pd.concat([_pred, 
                               _forecast], 
                               ignore_index= False, 
                               axis=0).index
    
    _metrics_df = pd.concat([_metrics_df, se_metrics], ignore_index= False, axis=1)


    # Visualize the regression line
    ts_plotly_chart(target, pd.concat([_pred, _forecast, se_pred], ignore_index= False, axis=0))


    st.subheader("Model Evaluation:")
    st.table(_metrics_df.T)


    st.subheader("Residuals:")
    plot_res(_res)

    st.subheader("Params:")
    st.table(pd.concat(_params, ignore_index= False, axis=1).T.drop(["use_boxcox", "lamda", "remove_bias", "initial_seasons"], axis = 1))



if __name__ == "__main__":
    main()