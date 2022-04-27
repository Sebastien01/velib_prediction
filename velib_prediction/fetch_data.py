import requests
import pandas as pd

from utils import dfConstructor
from time import sleep

             
constructor = dfConstructor()

count = 0
while True:
    
    try:
        constructor.export_df(file_name='data/historique_velib_v1.csv')
        
        print(f'Iteration nb {count} done')
        count+=1
        
    except:
        print("error "+str(IOError))

    sleep(60*15) #collect datapoints every 15mins