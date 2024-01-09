import pandas as pd
from base_estimator import BaseEstimator
from custom_estimators import StackedEstimator


def build_model(models: list[BaseEstimator], target: pd.DataFrame, timestamps = None):
    if timestamps == None:
        timestamps = target.index[0]

    # add the ability to predict custom timestamps
    _metrics_df = []
    _params = []
    _pred = []
    _res = []
    _forecast = []
    for __model in models:
        hw = BaseEstimator(model_obj= __model)
        _params.append(hw.get_params)
        _metrics_df.append(hw.get_metrics(target, timestamps))
        _pred.append(hw.predict(timestamps))
        _res.append(hw.get_residuals)
        _forecast.append(hw.forecast(5))

    _pred = pd.concat(_pred, ignore_index= False, axis=1)
    _metrics_df = pd.concat(_metrics_df, ignore_index= False, axis=1)
    _res = pd.concat(_res, ignore_index= False, axis=1)
    _forecast = pd.concat(_forecast, ignore_index= False, axis=1)

    return(_metrics_df, _params, _pred, _res, _forecast)


    """
    from sklearn import linear_model

    est = StackedEstimator(linear_model.Ridge(), _pred, X, method="test_split")
    #print(est.predict(_pred.to_numpy()))
    #print(est.get_params)
    print(est.get_metrics(X, _pred.to_numpy()))
    est = StackedEstimator(linear_model.Ridge(), _pred, X)
    #print(est.predict(_pred.to_numpy()))
    #print(est.get_params)
    print(est.get_metrics(X, _pred.to_numpy()))"""