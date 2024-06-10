# developer: Foundry Digital
# Copyright © 2023 Foundry Digital

# Import required models
from datetime import datetime, timedelta
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import ta
from typing import Tuple
import yfinance as yf
import pandas as pd
import numpy as np 
from neuralforecast import NeuralForecast
from pytz import timezone

def prep_data_chaotic(drop_na:bool = True) -> DataFrame:
    """
    Prepare data by calling Yahoo Finance SDK & computing technical indicators.

    The function gets the last 60 day data for the S&P 500 index at a 5m granularity
    and then computes the necessary technical indicators, resets index and drops rows
    with NA values if mentioned.

    Input:
        :param drop_na: The drop_na flag is used to tell the function whether to drop rows
                        with nan values or keep them.
        :type drop_na: bool

    Output:
        :returns: A pandas dataframe with the OHLCV data, along with the some technical indicators.
                  The dataframe also has the next close as a column to predict future close price using
                  current data.
        :rtype: pd.DataFrame
    """
    # Fetch S&P 500 data - when capturing data any interval, the max we can go back is 60 days
    # using Yahoo Finance's Python SDK
    data = yf.download('^GSPC', period='60d', interval='5m')

    # Calculate technical indicators - all technical indicators computed here are based on the 5m data
    # For example - SMA_50, is not a 50-day moving average, but is instead a 50 5m moving average
    # since the granularity of the data we procure is at a 5m interval. 

    # Drop NaN values
    if(drop_na):
        data.dropna(inplace=True)

    data.reset_index(inplace=True)

    return data

def round_down_time(dt:datetime, interval_minutes:int = 5) -> datetime:
    """
    Find the time of the last started `interval_minutes`-min interval, given a datetime

    Input:
        :param dt: The datetime value which needs to be rounded down to the last 5m interval
        :type dt: datetime

        :param interval_minutes: interval_minutes gives the interval we want to round down by and
                            the default is set to 5 since the price predictions being done
                            now are on a 5m interval
        :type interval_minutes: int

    Output:
        :returns: A datetime of the last started 5m interval
        :rtype: datetime
    """

    # Round down the time to the nearest interval
    rounded_dt = dt - timedelta(minutes=dt.minute % interval_minutes,
                                seconds=dt.second,
                                microseconds=dt.microsecond)

    return rounded_dt


def extract_data(data: pd.DataFrame, unique_id='BTCUSD', target='Close'):
    data = data.reset_index()
    data['y'] = data[target]
    data['ds'] = data['Datetime']
    X = data[['y', 'ds']]
    X['unique_id'] = unique_id
    X.loc[:, 'ds'] = pd.to_datetime(X['ds'], format='%Y-%m-%d %H:%M:%S%z', utc=True)
    return X




def predict_chaotic(timestamp:datetime, model,input_len = 1000) -> float:
    """
    Predicts the close price of the next 5m interval

    The predict function also ensures that the data is procured - using yahoo finance's python module,
    prepares the data and gets basic technical analysis metrics, and finally predicts the model
    and scales it based on the scaler used previously to create the model.

    Input:
        :param timestamp: The timestamp of the instant at which the request is sent by the validator
        :type timestamp: datetime.datetime

        :param scaler: The scaler used to scale the inputs during model training process
        :type scaler: sklearn.preprocessing.MinMaxScaler

        :param model: The model used to make the predictions - in this case a .h5 file
        :type model: A keras model instance

    Output:
        :returns: The close price of the 5m period that ends at the timestamp passed by the validator
        :rtype: float
    """
    
    # calling this to get the data - the information passed by the validator contains
    # only a timestamp, it is on the miners to get the data and prepare is according to their requirements
    data = prep_data_chaotic(drop_na=False)


    # The timestamp sent by the validator need not be associated with an exact 5m interval
    # It's on the miners to ensure that the time is rounded down to the last completed 5 min candle
    pred_time = round_down_time(datetime.fromisoformat(timestamp))

    matching_row = data[data['Datetime'] <= pred_time].tail(input_len)

    print(pred_time, matching_row.tail(1))

    # Check if matching_row is empty
    if matching_row.empty:
        print("No matching row found for the given timestamp.")
        return 0.0

    # data.to_csv('mining_models/base_miner_data.csv')
    input = extract_data(matching_row)


    prediction = model.predict(input)
    print(f"Pred len is {prediction.shape[0]}")
    modelname = str(model.models[0])



    return np.array(prediction[modelname].values).reshape(1,-1)

# Uncomment this section if you wanna do a local test without having to run the miner
# on a subnet. This main block (kinda) mimics the actual validator response being sent
if(__name__=='__main__'):
    #mse = create_and_save_base_model_regression(scaler, X, y)

    #model = joblib.load('mining_models/base_linear_regression.joblib')
    model = NeuralForecast.load('mining_models/chaotic_snp/')
    ny_timezone = timezone('America/New_York')
    current_time_ny = datetime.now(ny_timezone)
    timestamp = current_time_ny.isoformat()
    data = prep_data_chaotic()
    prediction = predict_chaotic(timestamp, model) 
    print(f"current pred is { prediction[0].tolist()}")

 