import pandas as pd
import os
import requests
import os
import pickle

from pipeline import create_time_feature


model_dic = {8: 'docks_cl_8.pkl',
             88: 'docks_cl_88.pkl',
             11: 'docks_cl_11.pkl',
             888: 'docks_cl_888.pkl',
             33: 'docks_cl_33.pkl',
             2: 'docks_cl_2.pkl',
             3: 'docks_cl_3.pkl',
             1: 'docks_cl_1.pkl',
             0: 'docks_cl_0.pkl',
             4: 'docks_cl_4.pkl',
             7: 'docks_cl_7.pkl',
             555: 'docks_cl_555.pkl',
             55: 'docks_cl_55.pkl',
             6: 'docks_cl_6.pkl'}
    


class velibPredictor():
    '''This class can collect meteo datapoints and make velib/docks predictions for a specified date and hour'''
    
    def __init__(self):
        
        self.stations = None
        self.date = None
        self.hour = None

        
        self.station_df = pd.read_csv('data/stations_info.csv').set_index('name')
        self.station_cluster_dic = dict(zip(self.station_df.index, self.station_df.cluster))

        self.station_id_dic = None
        
        self.models = {cluster : pickle.load(open(f"models/{mod}","rb")) for cluster,mod in model_dic.items()}
        
    
    def add_time(self, date, hour):
        
        self.date = date
        self.hour = hour
        
        self.target_date = pd.Timestamp(f"{date} {hour}", tz="Europe/Brussels")
        
        
    def add_stations(self, stations):
        
        self.stations = stations
        self.station_id_dic = {int(self.station_df.at[st,'station_id']) : st for st in self.stations}
        
    
    def get_API_meteo(self, day_shift_nb):
        """Retrieve meteo forecast info via meteo-concept' API, return a pd.DataFrame for a given day"""

       # TOKEN = os.environ["METEO_TOKEN"]
        
        TOKEN = 'b2e61b6debca16466d2155717dfbd66e4174baaf17c9be89c8daae6addcb553f'

        url = f'https://api.meteo-concept.com/api/forecast/daily/{day_shift_nb}/periods?token={TOKEN}&insee=75056'
        rep = requests.get(url)

        assert rep.status_code == 200, f"API call failed with code error {rep.status_code}"
        
        df = pd.DataFrame(rep.json()['forecast']).loc[:,['temp2m','probarain','weather','wind10m','datetime']]
        df['datetime'] = pd.to_datetime(df.datetime)
        return df.set_index('datetime')

    
    def retrieve_meteo_forecast(self):
        
        now = pd.Timestamp.now(tz="Europe/Brussels")
        
        day_shift = (self.target_date - now.floor('d')).days
        assert day_shift <= 13, "No meteo data after 13 days"
        
        if self.target_date.hour < 2:
            df = self.get_API_meteo(day_shift - 1)
            self.meteo_info = df.tail(1)

        else:
            df = self.get_API_meteo(day_shift)
            self.meteo_info = df.loc[(self.target_date - pd.Timedelta(hours=5, minutes=59)) : self.target_date]
            
            
            
    def predict(self):
        
        self.X = self.meteo_info.reset_index().rename({'datetime':'time'}, axis='columns')
        self.X = create_time_feature(self.X)
        
        results = dict()
        
        for st_id, st_name  in self.station_id_dic.items():
            
            X = self.X.assign(station_id = st_id)
            
            #I should have dropped "capacity" during training as it doesn't bring any more info to our model
            X['capacity'] = int(self.station_df.at[st_name, 'capacity']) 
            
            X.columns = ['station_id', 'capacity', 'temp2m', 'probarain', 'weather', 'wind10m',
                 'month', 'hour', 'day', 'minute']
            
            cluster = self.station_cluster_dic.get(st_name)
            model = self.models.get(cluster)
            
            results[st_name] = model.predict(X).item()
        
        return results
    




















class dfConstructor():
    '''This class is an ensemble of methods to fetch velib and meteo data in order to build our dataset'''
    
    def get_station_info(self):
        
        url = "https://velib-metropole-opendata.smoove.pro/opendata/Velib_Metropole/station_information.json"
        rep_info = requests.get(url).json()
        
        df = pd.DataFrame(rep_info['data']['stations'])
        df = df.astype({'station_id':str})
        
        return df[['station_id','name','lat','lon','capacity']]  
    

    def get_stations_status(self):
        
        url = "https://velib-metropole-opendata.smoove.pro/opendata/Velib_Metropole/station_status.json"
        rep = requests.get(url).json()
        
        df = pd.DataFrame(rep['data']['stations'])
        df = df.astype({'is_installed':bool, 'is_returning': bool, 'is_renting':bool, 'station_id':str})
        df = df.join(df.num_bikes_available_types.apply(lambda x: pd.Series({**x[0], **x[1]})))
        df['time'] = pd.Timestamp('now').round(freq='S')
        df.rename(columns={'numDocksAvailable':'docks_available',
                       'mechanical': 'mechanical_available',
                       'ebike': 'ebike_available'} ,inplace=True)
        
        return df[['station_id','docks_available','is_installed',
                   'is_returning','is_renting','mechanical_available','ebike_available','time']]
        
    
    def get_meteo_info(self):
        
        TOKEN = os.environ.get("METEO_TOKEN")
        
        url = f'http://api.meteo-concept.com/api/forecast/nextHours?token={TOKEN}&insee=75056&hourly=true'
        rep = requests.get(url).json()
        
        today_meteo = pd.DataFrame(rep['forecast']).loc[0,['temp2m','probarain','weather','wind10m']]
        
        return today_meteo.rename({'temp2m' : 'temperature',
                                   'probarain' : 'rain_proba',
                                   'wind10m' : 'wind_speed'})
        

    def export_df(self, file_name):
        
        info_df = self.get_station_info()
        status_df = self.get_stations_status()
        
        df = status_df.merge(on='station_id',right = info_df)
        
        meteo_df = self.get_meteo_info()
        
        for col in meteo_df.keys():
            df[col] = meteo_df[col]
        
        df.to_csv(f'{file_name}', header=False, mode='a')   
        
    