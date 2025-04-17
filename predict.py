import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

df = pd.read_csv("./data/raw_database.csv", dtype={'Flight Phase':'str', 'Visibility':'str',
                                                       'Precipitation':'str','Species ID':'str',
                                                       'Species Name':'str', 'Species Quantity':'str',
                                                       'Flight Impact': 'str', 'Height':'Int64', 
                                                       'Speed':'float64','Distance':'float64',  
                                                       'Fatalities':'Int64', 'Injuries': 'Int64'}, low_memory=False) 

features_numeric = [
    'Aircraft Mass', 'Engines', 'Engine2 Position',  'Engine4 Position',
    'Height', 'Speed', 'Distance',
    'Aircraft Damage', 'Radome Strike', 'Radome Damage', 'Windshield Strike',
    'Windshield Damage', 'Nose Strike', 'Nose Damage', 'Engine1 Strike', 'Engine1 Damage',
    'Engine2 Strike', 'Engine2 Damage', 'Engine3 Strike', 'Engine3 Damage', 'Engine4 Strike',
    'Engine4 Damage', 'Engine Ingested', 'Propeller Strike', 'Propeller Damage',
    'Wing or Rotor Strike', 'Wing or Rotor Damage', 'Fuselage Strike', 'Fuselage Damage',
    'Landing Gear Strike', 'Landing Gear Damage', 'Tail Strike', 'Tail Damage',
    'Lights Strike', 'Lights Damage', 'Other Strike', 'Other Damage','Engine Make'
]
feature_string = [
    'Operator ID', 'Aircraft', 'Aircraft Type', 'Aircraft Make',
    'Aircraft Model',  'Species Quantity', 'Engine Model', 'Engine Type',
    'Engine1 Position','Engine3 Position', 'Airport ID', 'State', 'FAA Region', 'Warning Issued',
    'Flight Phase', 'Visibility', 'Precipitation', 'Flight Impact'
]

st.title('Species Prediction App')

st.markdown("Please enter the following information:")


input_values = {}
for feature in features_numeric:
    input_values[feature] = st.number_input(f"{feature}:", value=1)

for feature in feature_string:
    unique_values = df[feature].dropna().unique()  # Get unique values for dropdown
    input_values[feature] = st.selectbox(f"Select {feature}:", unique_values)

if st.button('Predict'):
    df_input = pd.DataFrame([input_values])
    X_num = df_input[features_numeric]
    X_cat = df_input[feature_string]

    st.write("Shape of categorical features:", X_cat.shape)
    X_cat_encoded = encoder.transform(X_cat)
    X_combined = np.hstack([X_num.values, X_cat_encoded])

    # Scale + PCA
    X_combined = scaler.transform(X_combined)
    X_pca = pca.transform(X_combined)
    prediction = model.predict(X_pca)

    st.success(f"Predicted Species: {prediction[0]}")
