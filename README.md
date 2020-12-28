# Visualiziation and Forecasting of Time Series
with focus on modularity and ease of use

## Usage
1. download utils.py file
2. put that in the same directory as the production.py file
3. ```python
    from utils import *
    ```

## Goals

1. Create an ordered (descending) plot that shows the total number of transactions per customer from the most active customer to the least active one.

Solution: ```plot_customer_sale_freq``` function do this job. you can see this function example usage in the ```Customer purchase freq``` part of the ```Data_Visualization.ipynb``` notebook.


2. Given any product ID, create a plot to show its transaction frequency per month for the year 2018.

Solution: ```plot_product_monthly_sales``` function do this job. you can see this function example usage in the ```Purchase freq through time``` part of the ```Data_Visualization.ipynb``` notebook.

3. Build a model to predict the total number of transactions for the next three months per customer anywhere in 2019. For example, given all data up to the end of January 2019, predict the size of the transactions between Feb 1st and April 30th for each customer. Then, measure the performance of your model with appropriate metrics and visuals to validate the entire process.

Solution: ```forecast``` function does this job. you can see this function example usage in the ```Build and Save Model``` part of the ```Forcasting_SKTime.ipynb``` notebook. this function has a docstring that is available via help(). there is many alternative model algorithm accessible in this function that you select one of them with "model_type" parameter, example usage of each algorithm is located in ```Model_Evaluations.ipynb``` notebook. by default this function use ```NaiveForecaster``` algorithm, one of the leading algorithm in time series forecasting is ```AutoARIMA```, you can see a good comparison [Here](https://medium.com/analytics-vidhya/time-series-forecasting-arima-vs-lstm-vs-prophet-62241c203a3b).
the resulting model will be saved and can be loaded for future use via the ```load_forecast``` function.


4. At any time, what are the top 5 products that drove the highest sales over the last six months? Do you see a seasonality effect in this data set?

Solution: you can see the solution in the ```Check seasonality in data``` part of the ```Data_Visualization.ipynb``` notebook. in short, a decreasing trend in data makes the effect of seasonality negligible.
