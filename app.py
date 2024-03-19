import streamlit as st
import pandas as pd
import pickle 
import numpy as np


df = pd.read_csv("final_cardataset.csv")

with open('car_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

def filter_cars_by_company(selected_company, df):
    sorted_df = df.sort_values(['company', 'name'], ascending=True)
    filtered_cars = sorted_df[sorted_df['company'] == selected_company]['name'].unique()
    return filtered_cars

def main():
    st.title("CAR PRICE PREDICTION")
    st.write("Enter the details of the car to predict its price.")
    st.markdown(
        """
       <style>
       .stApp {
          background-color: lightblue;
      }
      </style>
      """,
     unsafe_allow_html=True
  )
    
    company = st.selectbox("company name", sorted(df['company'].unique()))
    name = st.selectbox('Car Name',filter_cars_by_company(company,df))
    year = st.number_input('Year', min_value=1900, max_value=2023, step=1)
    km_driven = st.number_input('Kilometers Driven', step=1000)
    fuel = st.selectbox('Fuel Type', df['fuel'].unique())
    seller_type = st.selectbox('Seller Type', df['seller_type'].unique())
    transmission = st.selectbox('Transmission', df['transmission'].unique())
    owner = st.selectbox('Owner', df['owner'].unique())

    data = pd.DataFrame({'company':[company],
                         'name': [name],
                         'year': [year],
                         'km_driven': [km_driven],
                         'fuel': [fuel],
                         'seller_type': [seller_type],
                         'transmission': [transmission],
                         'owner': [owner]})
    
    if st.button('Predict Price'):
        transformer, regressor = pipeline
        data_encoded = transformer.transform(data).toarray()
        predictions = regressor.predict(data_encoded)

        success_message = f'Price of the car is {predictions[0]} INR'
        colored_message =  f'<span style="color: blue; font-size: 24px;font-weight: bold;">{success_message}</span>'
        st.markdown(colored_message, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
       



