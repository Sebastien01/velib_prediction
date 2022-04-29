import streamlit as st
import datetime

from utils import velibPredictor

def display_preds(dic):    
    for station in stations:
        scores = dic.get(station)
        
        st.header(station)
        st.write(f"🅿️ : {scores[0]:.0f}")
        st.write(f"🚲 : {scores[1]:.0f}")


st.title("Combien-de-velib.com")
st.markdown("""Ce site vous permet de prédire le nombre de places libres et de vélos mécaniques disponibles 
            pour les stations et la date de votre choix.""")

today = datetime.datetime.now()

stations = st.multiselect("Séléctionnez une ou plusieurs stations :", ['Mairie du 9ème', 'Geoffroy - Mairie'])


date = st.date_input("Séléctionnez une date :", 
                     min_value = today,
                     max_value = (today + datetime.timedelta(days=13))
                     )

hour = st.time_input('Séléctionnez une heure :',
                     datetime.time(12,30)
                     )



if st.button('Compute predictions 🤖🧠'):
    vb = velibPredictor(date, hour, stations)
    vb.retrieve_meteo_forecast()
    display_preds(vb.predict())

    
else:
     st.write('')
