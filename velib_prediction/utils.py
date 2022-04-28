import pandas as pd
import os
import requests

from joblib import load
from pathlib import Path


class velibPredictor():
    '''This class can collect meteo datapoints and make velib/docks predictions for a specified date and hour'''
    
    def __init__(self, date, hour):
        
        
        self.target_date = pd.Timestamp(f"{date} {hour}", tz="Europe/Brussels")
        
        self.ATF_DOCKS = load('velib_prediction/modeling/models/artefact_docks.joblib')
        self.ATF_MECA = load('velib_prediction/modeling/models/artefact_meca.joblib')
        self.MAIRIE_DOCKS = load('velib_prediction/modeling/models/mairie_neuf_docks.joblib')
        self.MAIRIE_MECA = load('velib_prediction/modeling/models/mairie_neuf_meca.joblib')

        
    def get_API_meteo(self, day_shift_nb):
        """Retrieve meteo forecast info via meteo-concept' API, return a pd.DataFrame for a given day"""

        TOKEN = os.environ["METEO_TOKEN"]
    
        url = f'https://api.meteo-concept.com/api/forecast/daily/{day_shift_nb}/periods?token={TOKEN}&insee=75056'
        rep = requests.get(url)

        assert rep.status_code == 200, f"API call failed with code error {rep.status_code} : \nTOKEN = {TOKEN}"
        
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
        
        self.X = self.meteo_info
        self.X['month']  = self.target_date.month
        self.X['hour']   = self.target_date.hour
        self.X['day']    = self.target_date.day_of_week
        self.X['minute'] = self.target_date.minute
        
        return {
            'ATF_docks_available'    : self.ATF_DOCKS.predict(self.X).item(),
            'ATF_meca_available'     : self.ATF_MECA.predict(self.X).item(),
            'MAIRIE_docks_available' : self.MAIRIE_DOCKS.predict(self.X).item(),
            'MAIRIE_meca_available'  : self.MAIRIE_MECA.predict(self.X).item()
            }
        




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
        
    