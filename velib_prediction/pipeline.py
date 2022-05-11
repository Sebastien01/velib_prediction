import pandas as pd
import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

pd.options.mode.chained_assignment = None

def load_clean_hist(path='/Users/sebastienvallin/code/Sebastien01/velib_prediction/raw_data/historique_velib_v2.csv',
                    y='docks_available'):
    '''
    Loads historical data and cleans closed stations.
    Adds stations cluster info.
    Delete useless-for-training columns.
    '''
    df = pd.read_csv(path, index_col=0)
    df.time = pd.to_datetime(df.time)
    
    #Deleting closed stations
    for col in ['is_installed', 'is_returning','is_renting']:
        df[col] = df[col].map({True:1, False:0})

    closed_stations = [st for st in df.station_id.unique() 
                    if (df[(df.station_id==st)].is_installed.sum() / df[(df.station_id==st)].shape[0]) < 0.7]

    df = df[~df.station_id.isin(closed_stations)]
    
    #Deleting misleading rows
    df = df[~(df.is_installed == 0)]
    df = df[~(df.is_returning == 0)]
    df = df[~(df.is_renting == 0)]
    df = df.drop_duplicates()
    
    #Adding cluster info
    station_df = pd.read_csv('/Users/sebastienvallin/code/Sebastien01/velib_prediction/velib_prediction/data/stations_info.csv')
    df['cluster'] = df['station_id'].map(dict(zip(station_df.station_id,station_df.cluster)))
    df = df.dropna().dropna()
    
    return df[['station_id', 'time', 'capacity', 'temp2m', 'probarain', 'weather', 'wind10m','cluster', y]]


    
    
ohe_pipe = ColumnTransformer([
                    ('ohe',OneHotEncoder(sparse=False, handle_unknown='ignore'),['station_id'])
                    ],remainder='passthrough'
                                )
    
def build_pipe(n_estimators = 3_000):
    ''' Returns a pipeline ready to be trained '''

    pipe = Pipeline([('ohe_',ohe_pipe),
                    ('model',xgb.XGBRegressor(n_estimators=n_estimators,
                                            learning_rate=0.05,
                                            objective='reg:squarederror',
                                            colsample_bylevel= 0.6 , #to prevent overfitting
                                            max_depth = 9,
                                            gamma = 4,
                                            seed=20))
                    ])

    return pipe



def create_time_feature(df): #To use when creating X_train
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.dayofweek
    df['minute'] = df['time'].dt.minute

    return df.drop(columns=['time'])