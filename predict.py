import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('model_rand_pca.pkl') 
preprocessor = joblib.load('preprocessor_pipeline.pkl')
pca = joblib.load('pca.pkl')

df = pd.read_csv("./data/raw_database.csv", dtype={'Flight Phase':'str', 'Visibility':'str',
                                                       'Precipitation':'str','Species ID':'str',
                                                       'Species Name':'str', 'Species Quantity':'str',
                                                       'Flight Impact': 'str', 'Height':'Int64', 
                                                       'Speed':'float64','Distance':'float64',  
                                                       'Fatalities':'Int64', 'Injuries': 'Int64'}, low_memory=False) 
features_numeric = [
    
    'Injuries',
    'Record ID', 'Incident Month', 'Incident Day',
    'Aircraft Mass', 'Engine2 Position', 'Height',
    'Aircraft Damage', 'Radome Strike', 'Radome Damage', 'Windshield Strike',
    'Nose Strike', 'Nose Damage',
    'Engine1 Strike', 'Engine2 Strike', 'Engine3 Strike', 'Engine4 Strike',
    'Engine4 Damage', 'Propeller Strike', 'Propeller Damage',
    'Wing or Rotor Strike', 'Fuselage Strike', 'Fuselage Damage',
    'Landing Gear Strike', 'Landing Gear Damage',
    'Tail Strike', 'Lights Strike', 'Other Strike', 'Other Damage',
    'Engine Make'  
]

feature_string = [
    'Operator ID', 'Aircraft',
    'Species Quantity',
    'Warning Issued', 'Precipitation',
    'Flight Impact',
    'Flight Phase',      
    'Visibility'    
]


st.title('Species Prediction App')

st.markdown("Please enter the following information:")
input_values = {}
for feature in features_numeric:
    input_values[feature] = st.number_input(f"{feature}:", value=1)

for feature in feature_string:
    unique_values = df[feature].dropna().unique() 
    input_values[feature] = st.selectbox(f"Select {feature}:", unique_values)

if st.button('Predict'):

    df_input = pd.DataFrame([input_values])
    expected_columns = features_numeric + feature_string
    df_input = df_input[expected_columns]
    X_preprocessed = preprocessor.transform(df_input)
    X_pca = pca.transform(X_preprocessed)
    prediction = model.predict(X_pca)
    st.success(f"Predicted Species: {prediction[0]}")

