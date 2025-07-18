import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd  
from babel.numbers import format_currency
#st.title("🏠 House Price Prediction App")
col1, col2 = st.columns([4, 6])
with col1:
    st.image("/workspaces/Codes-Please-/House_Price_Prediction_ML_Project/images/house.webp", width=400)
with col2:
    st.markdown("<h1 style='padding-top: 35px;'>House Price Prediction App</h1>", unsafe_allow_html=True)

st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")
st.divider()
st.write("Welcome to the House Price Prediction App! This tool uses machine learning to estimate the selling price of a house based on key features provided by the user. By analyzing parameters such as the square footage area, number of bathrooms and bedrooms, total number of stories, presence of air conditioning, and available parking space, the model predicts a fair market price for the property. The application is powered by a regression algorithm trained on real-world housing data, making it a valuable tool for home buyers, sellers, and real estate professionals alike")
st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("This application is designed to provide quick and intelligent estimates of house prices based on key property features. By analyzing factors like square footage, number of bedrooms and bathrooms, floors, air conditioning availability, and parking space, the model delivers a data-driven prediction to help users understand potential market value.")

st.markdown("<h2 style='color:teal;'>Enter House Details Below</h2>", unsafe_allow_html=True)
st.divider()

pipeline=joblib.load('/workspaces/Codes-Please-/House_Price_Prediction_ML_Project/model/model_pipeline.pkl')
area=st.number_input("Enter The Square feet area of house",min_value=1200,max_value=17000,step=5)
bedroom=st.number_input("Enter number of bedrooms needed",min_value=1,max_value=6,step=1)
bathroom=st.number_input("Enter number of bathroom needed",min_value=1,max_value=4,step=1)
stories=st.number_input("Number of stories",min_value=1,max_value=4,step=1)
aircondition_input = st.selectbox("Do you want Air Conditioning?", ["Yes", "No"])

aircondition = 1 if aircondition_input == "Yes" else 0

parking=st.number_input("Parking Spaces Required",min_value=0,max_value=3)
input_df = pd.DataFrame([[area, bedroom,bathroom, stories, aircondition,parking]],
                        columns=['area','bedrooms','bathrooms','stories','airconditioning','parking'])

if st.button("Predict House Price"):
    st.balloons()
    prediction = pipeline.predict(input_df)[0]
    formatted_price = format_currency(prediction, 'INR', locale='en_IN')
    st.success(f"Predicted Price: Rs{formatted_price}")


st.markdown("---")
st.markdown("Created with ❤️ by Subhankit (https://github.com/SubhankitBaner)")

