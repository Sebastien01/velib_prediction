import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

pd.options.mode.chained_assignment = None

def build_features_pipeline():
    
    def create_time_feature(df):
        df['time'] = pd.to_datetime(df['time'])
        df['month'] = df['time'].dt.month
        df['hour'] = df['time'].dt.hour
        df['day'] = df['time'].dt.dayofweek
        df['minute'] = df['time'].dt.minute
        
        return df.drop(columns=['time'])

    create_time_ft = FunctionTransformer(lambda df: create_time_feature(df))
    
    return Pipeline([('time_feature',create_time_ft)])
