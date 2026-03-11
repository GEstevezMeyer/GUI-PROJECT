import pandas as pd 
import tensorflow as tf 
import yfinance as yf
from data_windowing import WindowGenerator
import numpy as np



def extract_ticket_data(ticket:str,period = "max") -> pd.DataFrame:
    df = yf.Ticker(ticket).history(period = period)
    if df.empty: 
        return False
    return df

def multivariate_input_width(df: pd.DataFrame, threshold=0.2, max_lag=20):
    n_vars = df.shape[1]
    input_width = 1

    for lag in range(1, max_lag+1):
        for col in df.columns:
            x = df[col].values
            x_mean = np.mean(x)
            cov = np.sum((x[lag:] - x_mean) * (x[:-lag] - x_mean)) / (len(x)-1)
            var = np.var(x, ddof=1)
            acf = cov / var
            if abs(acf) >= threshold:
                input_width = max(input_width, lag)
    shift = max(1, input_width // 2)
    return input_width, shift


def create_window_class(df:pd.DataFrame) -> WindowGenerator:
    input_width, shift = multivariate_input_width(df)
    wg = WindowGenerator(df,input_width,shift)
    return wg





if __name__ == "__main__": 
    print(extract_ticket_data("nenenenenen")) 

    data = pd.DataFrame({
        "x": np.arange(1000)
    })

    wg = create_window_class(data)
    print(wg.training_input)
