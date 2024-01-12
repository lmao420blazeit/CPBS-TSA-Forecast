# CPBS Demo Forecasting App using univariate TSA models

A brief description of your project goes here.

## Table of Contents

- [Project Title](#project-title)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)

## Introduction

Briefly introduce your project, providing context and explaining its purpose.

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

### Ensemble model constructor with TSA as base models

Supports any sklearn.linear_model class:
    - ElasticNet
    - Ridge
    - Lasso
    - LinearRegression
    - more to confirm (in theory should be able to support everyone)

## Getting Started

Provide instructions for setting up and running your project. Include information about prerequisites and installation steps.

### Prerequisites

List any software, libraries, or dependencies that need to be installed before running your project.

### Installation

Provide step-by-step instructions on how to install and set up your project.

```bash
# Example installation commands
git clone https://github.com/yourusername/yourproject.git
cd yourproject
pip install -r requirements.txt
run in console: streamlit run _streamlit_app.py
