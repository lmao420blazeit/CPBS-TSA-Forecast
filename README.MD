# CPBS Demo Forecasting App using univariate TSA models

https://cpbs-tsa-forecast.streamlit.app/
Sandbox project to test time series forecasting models in a small business dataset using statsmodels, sklearn and streamlit.

## Table of Contents

- [Project Title](#project-title)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)

## Introduction

### Scope of the project

- Compare the results of different TSA models
- Provide a custom sandbox framework
    - Create custom TSA models
    - Add your own model params and aliases
    - Define the metrics you want (full support to sklearn regressor metrics)
    - Define stacking regression models using TSA as base models
- Residuals plot for normality check
- Forecasting and model params

### Not in scope

- Optimization of model params (such as GridSearchCV)
- Normal tests for residuals (such as Shapiro Wilk)
- Extensive user interface to add and customize models outside of python script

## Features

### Custom model constructor

Supporting all statsmodels.tsa models as well as others:
    - Holt
    - Holt Winters
    - SimpleExpSmoothing
    - ARMA
    - ARIMA

### Stacking model constructor with TSA as base models

Supports any sklearn.linear_model class:
    - ElasticNet
    - Ridge
    - Lasso
    - LinearRegression
    - more to confirm (in theory all should be supported)

## Getting Started

#### Passing models constructor object list

Each model should be passed as a list [model_object: statsmodels.tsa, alias]. If no model alias is given, consider an empty string.

``` Python
# models: list[[model_object: statsmodels.tsa, alias: str]]
__models = [[SimpleExpSmoothing(endog = target, 
                                initialization_method="estimated"),
                                ""],
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
```

#### Using stacked models

### Prerequisites

The project was built on Python 3.10.6.

### Installation

```bash
# Example installation commands
git clone https://github.com/yourusername/yourproject.git
cd yourproject
pip install -r requirements.txt
run in console: streamlit run _streamlit_app.py
