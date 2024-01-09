from old.baseclass_template import _EstimatorBaseClass
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import sklearn.metrics
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class HWSimpleExpSmoothEstimator(_EstimatorBaseClass):

    def __init__(self, **kwargs):
        self.base_model = SimpleExpSmoothing(**kwargs).fit()
    

    def predict(self, timestep):
        return(self.base_model.predict(timestep))

    def get_metrics(self, 
                target, 
                timestep, 
                metrics = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "r2_score"]
                ):
        model_pred = self.predict(timestep)
        metrics_list = [getattr(sklearn.metrics, x) for x in metrics]
        metrics_ = pd.DataFrame(index = [x.__name__ for x in metrics_list],
                                data = [x(target, model_pred) for x in metrics_list], 
                                columns = [self.__class__.__name__])
        return (metrics_)

    @property
    def get_params(self):
        params = pd.DataFrame(
            index=[__key for __key in self.base_model.params.keys()]
        )
        params[self.__class__.__name__] = self.base_model.params
        return(params)

class HWExponentialSmoothing(_EstimatorBaseClass):
    """
    Apply Exponential Smoothing with Holt-Winters method to the input time series data.

    Parameters:
    - data (array-like): Time series data to be forecasted.
    - trend (str): Type of trend component, either "add" or "multiplicative".
    - seasonal (str): Type of seasonal component, either "add" or "multiplicative".
    - seasonal_periods (int): Number of seasons in a year for seasonal component.
    - use boxcox
    - damping_trend
    - remove bias

    """
    def __init__(self, 
                 **kwargs):
        self.base_model = ExponentialSmoothing(**kwargs).fit()
    

    def predict(self, 
                timestep):
        return(self.base_model.predict(timestep))

    def get_metrics(self, 
                target, 
                timestep, 
                metrics = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "r2_score"]
                ):
        model_pred = self.predict(timestep)
        metrics_list = [getattr(sklearn.metrics, x) for x in metrics]
        metrics_ = pd.DataFrame(index = [x.__name__ for x in metrics_list],
                                data = [x(target, model_pred) for x in metrics_list], 
                                columns = [self.__class__.__name__])
        return (metrics_)

    @property
    def get_params(self):
        params = pd.DataFrame(
            index=[__key for __key in self.base_model.params.keys()]
        )
        params[self.__class__.__name__] = self.base_model.params
        return(params)
    
class ARIMAEstimator(_EstimatorBaseClass):

    def __init__(self, **kwargs):
        self.base_model = ARIMA(**kwargs).fit()
    

    def predict(self, timestep):
        return(self.base_model.predict(timestep))

    def get_metrics(self, 
                target, 
                timestep, 
                metrics = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "r2_score"]
                ):
        model_pred = self.predict(timestep)
        metrics_list = [getattr(sklearn.metrics, x) for x in metrics]
        metrics_ = pd.DataFrame(index = [x.__name__ for x in metrics_list],
                                data = [x(target, model_pred) for x in metrics_list], 
                                columns = [self.__class__.__name__])
        return (metrics_)

    @property
    def get_params(self):
        params = pd.DataFrame(
            index=[__key for __key in self.base_model.params.keys()]
        )
        params[self.__class__.__name__] = self.base_model.params
        return(params)


    
if __name__ == "__main__":
    df1 = pd.read_csv("data.csv")
    df1.columns = df1.columns.str.replace("\n","_")
    df1.columns = df1.columns.str.replace(" ","_")

    data = df1

    X = data.groupby("MonthYear")["QTD_SALES"].sum()
    X.index = pd.to_datetime(X.index, format='%Y/%m')


    hw = HWSimpleExpSmoothEstimator(endog = X, initialization_method="estimated")
    print(hw.get_params)
    print(hw.get_metrics(X, X.index[0]))

    hw = HWExponentialSmoothing(endog = X,     
                                seasonal_periods=12,
                                trend="add",
                                seasonal="add",
                                use_boxcox=True,
                                initialization_method="estimated"
                                )
    print(hw.get_params)
    print(hw.get_metrics(X, X.index[0]))

    hw = ARIMAEstimator(endog = X, order=(3, 1, 1), trend="t")
    print(hw.get_params)
    print(hw.get_metrics(X, X.index[0]))