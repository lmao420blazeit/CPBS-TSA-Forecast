import sklearn.metrics
import pandas as pd
from tsa_estimators.base_estimator import EstimatorTemplate

class StackedEstimator(EstimatorTemplate):

    def __init__(self, 
                 stacking_method,
                 stackeddataset,
                 target, 
                 **kwargs):
        
        self.model_obj = stacking_method
        self.stacking_method = self._fit_model(stackeddataset,
                                               target,
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
                features, 
                metrics = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "r2_score"]
                ):
        model_pred = self.predict(features)
        metrics_list = [getattr(sklearn.metrics, x) for x in metrics]
        metrics_ = pd.DataFrame(index = [x.__name__ for x in metrics_list],
                                data = [x(target, model_pred) for x in metrics_list], 
                                columns = [self.model_obj.__class__.__name__])
        return (metrics_)

    def _fit_model(self, 
                stackeddataset, 
                target,
                **kwargs):
        
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(stackeddataset, 
                                                            target, 
                                                            test_size = 0.4, 
                                                            random_state = 42)
        return self.model_obj.fit(x_train, y_train)
        
    def forecast(self, timesteps):
        raise NotImplementedError
            

        