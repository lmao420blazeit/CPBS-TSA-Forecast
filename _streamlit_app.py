import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
#from statsmodels.tsa.arima.model import ARIMA
from tsa_estimators._build_model import build_model
import streamlit as st
from st_components.charts import plot_res, ts_plotly_chart
from st_components.selectbox import variable_selectbox
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

    target = variable_selectbox(data)

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
                                     "HW-Mult"],
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
                                     "HW-Additive-damped-trend"]]                                     
    

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


    # Evaluate the model
    #mse = mean_squared_error(y_test, predictions)
    st.subheader("Model Evaluation:")
    st.table(_metrics_df.T)


    st.subheader("Residuals:")
    plot_res(_res)
    #st.table(_res.style.highlight_min(axis=1, color='blue'))



if __name__ == "__main__":
    main()