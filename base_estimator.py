from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import sklearn.metrics
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import abc
#import statistics

class EstimatorTemplate(abc.ABC):
    """
    Estimator Template class for all model methods
    """

    @abc.abstractmethod
    def get_metrics():
        pass

    @abc.abstractmethod
    def get_params():
        pass

    @abc.abstractmethod
    def predict():
        pass

    @abc.abstractclassmethod
    def forecast():
        pass

class BaseEstimator(EstimatorTemplate):
    """
    Interface class for statsmodels.tsa models
    """
    def __init__(self, model_obj):
        self.model_obj = model_obj # keep track of base model
        self.base_model = model_obj.fit()
    

    def predict(self, timestep):
        df = self.base_model.predict(timestep)
        df.name = str(self.model_obj.__class__.__name__)
        return(df)

    def get_metrics(self, 
                target, 
                timestep, 
                metrics = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "r2_score"]
                ):
        model_pred = self.predict(timestep)
        metrics_list = [getattr(sklearn.metrics, x) for x in metrics]
        metrics_ = pd.DataFrame(index = [x.__name__ for x in metrics_list],
                                data = [x(target, model_pred) for x in metrics_list], 
                                columns = [self.model_obj.__class__.__name__])
        return (metrics_)

    @property
    def get_params(self):
        params = pd.DataFrame(
            index=[__key for __key in self.base_model.params.keys()]
        )
        params[self.model_obj.__class__.__name__] = self.base_model.params
        return(params)
    
    @property
    def get_residuals(self):
        params = pd.DataFrame(
            index=[__key for __key in self.base_model.resid.index]
        )
        params[self.model_obj.__class__.__name__] = self.base_model.resid
        return(params)
    
    def forecast(self, steps):
        forecast = self.base_model.forecast(steps = steps)
        forecast.name = self.model_obj.__class__.__name__
        print(forecast)
        return (forecast)
    
    @property
    def get_stderr(self):
        # stderr is the +- stderr(residuals)
        pass
    