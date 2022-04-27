import pandas as pd

pd.options.mode.chained_assignment = None

def build_dataset(station_id, target):
    """ 
    :station_id: Station to build a model for
    :target: Item to predict -> 'docks_available', 'mechanical_available' or 'ebike_available'
    """
    USEFUL_COLS = ['time','temp2m', 'probarain', 'weather', 'wind10m', target]
    
    velib_df = pd.read_csv('../data/velib.csv',index_col=0)
    velib_df = velib_df.reset_index().query(f'station_id == {station_id}')[USEFUL_COLS]
    
    return velib_df.drop(columns=target), velib_df[target]

