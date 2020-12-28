# Import Library
from ipywidgets import interact
import numpy as np
import pandas as pd

from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import Legend, LegendItem


from sktime.datasets import load_airline
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    EnsembleForecaster,
    ReducedRegressionForecaster,
    TransformedTargetForecaster,
)
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import sMAPE, smape_loss
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.utils.plotting import plot_series
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.bats import BATS
from sktime.forecasting.tbats import TBATS

from sklearn.neighbors import KNeighborsRegressor
from joblib import dump, load
from warnings import simplefilter
import warnings


# Visualization Functions
def slice_data(data, start_year, end_year, start_month, end_month):
    data = data.copy(deep=True)
    data = data[data.date > pd.Timestamp(f'{start_year}-{start_month}-01T00:00:00.000Z')][data.date < pd.Timestamp(f'{end_year}-{end_month}-01T00:00:00.000Z')]
    return data


def plot_customer_sale_freq(data):
    x = range(len(data.customer_id.value_counts()))
    y = data.customer_id.value_counts()

    TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"

    # create a new plot with the tools above, and explicit ranges
    p = figure(tools=TOOLS, x_range=(-50, max(x)+100), y_range=(-500, max(y)+500), plot_height=400, plot_width=600)

    # add a circle renderer with vectorized colors and sizes
    p.circle(x, y, radius=7, fill_color='darkslategray', fill_alpha=1.0, line_color=None)

    show(p, notebook_handle=True)


def plot_product_sale_freq(data):
    x = range(len(data.product_id.value_counts()))
    y = data.product_id.value_counts()

    TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"

    # create a new plot with the tools above, and explicit ranges
    p = figure(tools=TOOLS, x_range=(-3, max(x)+3), y_range=(-10000, max(y)+10000), plot_height=400, plot_width=600)

    # add a circle renderer with vectorized colors and sizes
    p.circle(x, y, radius=0.3, fill_color='darkslategray', fill_alpha=1.0, line_color=None)

    show(p, notebook_handle=True)


def plot_product_monthly_sales(year, data, product_id):
    data = data.copy(deep=True)
    data = slice_data(data, str(year), str(year + 1), '01', '01')
    transactions = data[data.product_id == product_id].date
    month_sales = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in transactions:
        month_sales[i.month - 1] += 1

    categories = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    p = figure(title=f'{year}_{product_id}_Sales', x_range=categories, plot_height=400, plot_width=600)
    p.vbar(x=categories, top=month_sales, width=0.9)

    show(p, notebook_handle=True)


# Forecasting Functions
def prepare_data(data, customer_id, start='2017-01', end='2019-04'):
    """
    function for easy exctraction of our model input from original dataset
    Parameters
    ----------
    data: pandas DataFrame
        main dataset with customer_id, product_id and Timestamp

    customer_id: int

    start: string
        start year and month in '2020-01' format

    end: string
        end year and month in '2020-01' format *** this month will not be included ***

    Returns
    -------
    y_series: pandas.Series
        sales data as a pd.Series and pd.period_range index
    """
    data = data.copy(deep=True)
    data = data[data.customer_id == customer_id]
    p_index = pd.period_range(start=start, end=end, periods=None, freq='M', name='Priod')
    freq = []
    for i in range(len(p_index) - 1):
        start_month = str(p_index[i]).split('-')[1]
        end_month = str(p_index[i + 1]).split('-')[1]
        start_year = str(p_index[i]).split('-')[0]
        end_year = str(p_index[i + 1]).split('-')[0]
        freq.append(len(data[data.date >= pd.Timestamp(f'{start_year}-{start_month}-01T00:00:00Z')][
                            data.date <= pd.Timestamp(f'{end_year}-{end_month}-01T00:00:00Z')]))

    y_series = pd.Series(freq, index=p_index[:-1], name='Sales', dtype='float64')
    return y_series


def forecast(data, customer_id, start='2017-01', end='2019-04', model_type='NaiveForecaster', test_size_month=5,
             model_storage_path=''):
    """
    Main function for build forecasting model on selected customer and time interval, save the model and plotting

    Parameters
    ----------
    data: pandas DataFrame
        main dataset with customer_id, product_id and Timestamp

    customer_id: int

    start: string
        start year and month in '2020-01' format

    end: string
        end year and month in '2020-01' format *** this month will not be included ***

    model_type:
        type of model to use in forecasting
        select from : ['NaiveForecaster', 'PolynomialTrendForecaster', 'ThetaForecaster', 'KNeighborsRegressor',
                       'ExponentialSmoothing', 'AutoETS', 'AutoARIMA', 'TBATS', 'BATS', 'EnsembleForecaster']

    test_size_month:
        number of month that will be excluded from end of interval to use as test dataset

    model_storage_path: string
        the folder that you want to store saved models
    Returns
    -------
    sMAPE Loss: print

    plot: matplotlib figure
        plot train, test and predicted values
    """
    y_train, y_test = temporal_train_test_split(prepare_data(data, customer_id, start=start, end=end),
                                                test_size=test_size_month)
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    if model_type == 'NaiveForecaster':
        forecaster = NaiveForecaster(strategy="last", sp=12)
    elif model_type == 'PolynomialTrendForecaster':
        forecaster = PolynomialTrendForecaster(degree=2)
    elif model_type == 'ThetaForecaster':
        forecaster = ThetaForecaster(sp=6)
    elif model_type == 'KNeighborsRegressor':
        regressor = KNeighborsRegressor(n_neighbors=1)
        forecaster = ReducedRegressionForecaster(regressor=regressor, window_length=12, strategy="recursive")
    elif model_type == 'ExponentialSmoothing':
        forecaster = ExponentialSmoothing(trend="add", seasonal="multiplicative", sp=12)
    elif model_type == 'AutoETS':
        forecaster = AutoETS(auto=True, sp=12, n_jobs=-1)
    elif model_type == 'AutoARIMA':
        forecaster = AutoARIMA(sp=12, suppress_warnings=True)
    elif model_type == 'TBATS':
        forecaster = TBATS(sp=12, use_trend=True, use_box_cox=False)
    elif model_type == 'BATS':
        forecaster = BATS(sp=12, use_trend=True, use_box_cox=False)
    elif model_type == 'EnsembleForecaster':
        forecaster = EnsembleForecaster([
            ("ses", ExponentialSmoothing(seasonal="multiplicative", sp=12)),
            ("holt", ExponentialSmoothing(trend="add", damped_trend=False, seasonal="multiplicative", sp=12),),
            ("damped", ExponentialSmoothing(trend="add", damped_trend=True, seasonal="multiplicative", sp=12),), ])

    try:
        forecaster.fit(y_train)
    except:
        forecaster.fit(y_train + 1)

    y_pred = forecaster.predict(fh)
    dump(forecaster, f'{model_storage_path}/{customer_id}_{model_type}_{start}_{end}_{test_size_month}.model')

    print('sMAPE Loss :', smape_loss(y_pred, y_test))
    plot = plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
    return plot


def load_forecast(data, model_path='Models/6689489_NaiveForecaster_2017-01_2019-04_5.model'):
    """
    Load saved forcasting model and plotting

    Parameters
    ----------
    data: pandas DataFrame
        main dataset with customer_id, product_id and Timestamp

    model_path: .model file
        path to previously saved model

    Returns
    -------
    sMAPE Loss: print

    plot: matplotlib figure
        plot train, test and predicted values
    """
    y_train, y_test = temporal_train_test_split(prepare_data(data, int(model_path.split('_')[0].split('/')[-1]),
                                                             start=model_path.split('_')[-3],
                                                             end=model_path.split('_')[-2]),
                                                test_size=int(model_path.split('_')[-1].split('.')[0]))
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    f = load(model_path)

    y_pred = f.predict(fh)

    print('sMAPE Loss :', smape_loss(y_pred, y_test))
    plot = plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
    return plot

