from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import sklearn.metrics
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from base_estimator import EstimatorTemplate

class StackedEstimator(EstimatorTemplate):

    def __init__(self, 
                 stacking_method,
                 stackeddataset,
                 target, 
                 method="cross_val",
                 **kwargs):
        
        self.model_obj = stacking_method
        self.stacking_method = self._fit_model(stackeddataset,
                                               target,
                                               method="cross_val",
                                               **kwargs)

    def predict(self, timestep):
        df = pd.DataFrame(self.stacking_method.predict(timestep))
        df.columns = [str(self.model_obj.__class__.__name__)]
        return(df)        

    @property
    def get_params(self):
        params = pd.DataFrame(
            index=[__key for __key in self.stacking_method.feature_names_in_]
        )
        params[self.model_obj.__class__.__name__] = self.stacking_method.coef_
        return(params)

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

    def _fit_model(self, 
                stackeddataset, 
                target,
                method = "cross_val", 
                **kwargs):
        
        if method not in ["cross_val", "test_split"]:
            raise Exception("Validation method isn't implented. ('cross_val', 'test_split')")
        
        if method == "cross_val":
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.model_selection import cross_validate
            #import statistics

            cv_results = cross_validate(
                self.model_obj,
                stackeddataset, 
                target, 
                cv=TimeSeriesSplit(gap = 0, n_splits = 5), 
                scoring= ('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error'),
                return_estimator= True,
                return_train_score= False
                )
            
            return cv_results.get("estimator")[-1]
            
        else:
            from sklearn.model_selection import train_test_split

            x_train, x_test, y_train, y_test = train_test_split(stackeddataset, 
                                                                target, 
                                                                test_size = 0.4, 
                                                                random_state = 42)
            return self.model_obj.fit(x_train, y_train)
            
            #model_pred = self.stacking_method.predict((x_test), y_test)

            #metrics_ = pd.DataFrame(index = [x.__name__ for x in ['r2', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error']],
            #                        data = [x(target, model_pred) for x in ['r2', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error']], 
            #                        columns = [self.stacking_method.__class__.__name__])
        
    def forecast(self, timesteps):
        raise NotImplementedError
            

        