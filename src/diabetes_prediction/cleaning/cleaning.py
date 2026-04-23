import numpy as np
import pandas as pd


class DataCleaning:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def remove_invalid_gender(self): 
        self.df = self.df[~(self.df['gender'] == 'Other')]    
        return self

    def remove_inaccurate_infants(self):
        self.df = self.df[~((self.df['age'] < 10) & (self.df['smoking_history'].isin(['not current', 'current', 'ever', 'former'])))]
        return self
    
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates(keep='first')
        return self
    
    def run_cleaning(self):
        return (
            self
            .remove_invalid_gender()
            .remove_inaccurate_infants()
            .remove_duplicates()
            .df
        )
