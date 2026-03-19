import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt 
from data_windowing import WindowGenerator
import tensorflow as tf 
import tomllib


with open("parameters.toml","rb") as f:
    toml_data:dict = tomllib.load(f)
    FIGSIZE = tuple(toml_data["FIGSIZE"])



def history_plot(history:dict) ->None: 
    plt.plot(history.history['mae'], label='train MAE')
    plt.plot(history.history['val_mae'], label='val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()


    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_prediction_test_general(wg: WindowGenerator, model: tf.keras.Sequential,
                    mean:float, std: float,label_encoder:int = None,):
    
    if label_encoder is not None:
        x,y = wg.split(wg.raw_df.to_numpy(),wg.input_width,wg.shift,wg.label_width,2)
    else:
        x,y = wg.split(wg.raw_df.to_numpy(),wg.input_width,wg.shift,wg.label_width)

    plt.figure(figsize=FIGSIZE)

    y_pred = model.predict(x).flatten()
    y_pred = (y_pred + mean) * std  

    y_real = y.flatten()
    y_real = (y_real + mean) * std  

    plt.plot(y_pred, label="Predicted")
    plt.plot(y_real, label="Real")
    plt.legend()
    plt.show()
