import streamlit as st
import datetime

from utils import velibPredictor

def display_preds(dic):
    st.header("La Grange Batelière")
    st.write(f"🅿️ : {dic.get('MAIRIE_docks_available'):.0f}")
    st.write(f"🚲 : {dic.get('MAIRIE_meca_available'):.0f}")

    st.header("Artefact")
    st.write(f"🅿️ : {dic.get('ATF_docks_available'):.0f}")
    st.write(f"🚲 : {dic.get('ATF_meca_available'):.0f}")

st.title("Combien y a t-il de places et de velibs ?")

today = datetime.datetime.now()

date = st.date_input("Pour quel jour voulez-vous connaitre les informations de votre station ?", 
                     min_value = today,
                     max_value = (today + datetime.timedelta(days=13))
                     )

hour = st.time_input('A quelle heure ?',
                     datetime.time(12,30)
                     )



if st.button('Compute predictions 🤖🧠'):
    vb = velibPredictor(date, hour)
    vb.retrieve_meteo_forecast()
    display_preds(vb.predict())
    
else:
     st.write('No date input yet...')
