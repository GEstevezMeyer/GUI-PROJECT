import pandas as pd 
import tensorflow as tf 
import numpy as np 


class WindowGenerator():
    def __init__(self,df:pd.DataFrame,input_width:int, shift:int,p_val_df:float = 0.2, p_test_df: float = 0.1,label_width:int = 1):

        if isinstance(df, pd.Series):
            df = df.to_frame()
        self.raw_df = df 
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        if (p_test_df+p_val_df) < 1:
            training_end = int((len(df)*(1-p_test_df-p_val_df)))
            validation_end = int((len(df)*(1-p_val_df)))
            self.training_df = self.raw_df.iloc[0:training_end:1].to_numpy().flatten()
            self.test_df = self.raw_df.iloc[training_end:validation_end:1].to_numpy().flatten()
            self.val_df = self.raw_df.iloc[validation_end::1].to_numpy().flatten()

         
        
if __name__ == "__main__": 

    n = 100

    df = pd.DataFrame({
        "price": np.arange(n)
    })

    wg = WindowGenerator(df["price"],input_width=10,shift=1,p_val_df=0.2,p_test_df=0.1)

    print(wg.training_df)



