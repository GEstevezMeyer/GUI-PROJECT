
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data_windowing import WindowGenerator
import tensorflow as tf 
import tomllib
import plotext as plt
from textual.widgets import Static


with open("parameters.toml","rb") as f:
    toml_data:dict = tomllib.load(f)
    FIGSIZE = tuple(toml_data["FIGSIZE"])


import plotext as plt
from textual.widgets import Static

class PlotterTF(Static):

    def history_plot_text(self, history: dict) -> None:
        train_mae = history.history['mae']
        val_mae = history.history['val_mae']
        epochs = list(range(1, len(train_mae) + 1))

        plt.clt()
        plt.clear_data()
        plt.plot(epochs, train_mae, label="train MAE")
        plt.plot(epochs, val_mae, label="val MAE")
        plt.title("Training History")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")

        graph_str = plt.build()
        self.update(graph_str)  

    def plot_prediction_test_general_text(self, wg, model, mean: float, std: float, label_encoder: int = None) -> None:
        if label_encoder is not None:
            x, y = wg.split(wg.raw_df.to_numpy(), wg.input_width, wg.shift, wg.label_width, 2)
        else:
            x, y = wg.split(wg.raw_df.to_numpy(), wg.input_width, wg.shift, wg.label_width)

        y_pred = model.predict(x).flatten()
        y_pred = (y_pred + mean) * std

        y_real = y.flatten()
        y_real = (y_real + mean) * std

        indices = list(range(len(y_real)))

        plt.clt()
        plt.clear_data()
        plt.plot(indices, y_pred, label="Predicted")
        plt.plot(indices, y_real, label="Real")
        plt.title("Prediction vs Real")
        plt.xlabel("Index")
        plt.ylabel("Value")

        graph_str = plt.build()
        self.update(graph_str)