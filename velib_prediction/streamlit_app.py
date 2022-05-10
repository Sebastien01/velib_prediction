import streamlit as st
import datetime

from utils import velibPredictor

def display_preds(dic):    
    for station in stations:
        scores = dic.get(station)
        
        st.header(station)
        st.write(f"🅿️ : {scores:.0f}")


vp = velibPredictor()

st.title("Combien-de-velib.com")
st.markdown("""Ce site vous permet de prédire le nombre de places libres et de vélos mécaniques disponibles 
            pour les stations et la date de votre choix.""")



stations = st.multiselect("Séléctionnez une ou plusieurs stations :", vp.station_df.index)
vp.add_stations(stations = stations)



today = datetime.datetime.now()
date = st.date_input("Séléctionnez une date :", 
                     min_value = today,
                     max_value = (today + datetime.timedelta(days=13))
                     )

hour = st.time_input('Séléctionnez une heure :',
                     datetime.time(12,30)
                     )

vp.add_time(date = date, hour = hour)


if st.button('Exécuter les prédictions 🤖🧠'):

    vp.retrieve_meteo_forecast()
    display_preds(vp.predict())

    
else:
     st.write('')
