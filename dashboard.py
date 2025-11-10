import streamlit as st
import requests
import json
from datetime import datetime, date, time # Import time

# API Endpoint
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(layout="wide")
st.title("Credit Card Fraud Detection System 💳")

st.header("Real-Time Transaction Monitor")
st.write("Enter transaction details below to get a fraud prediction.")

# Common categories from the dataset (you might want to get this dynamically)
categories = ['misc_net', 'grocery_pos', 'entertainment', 'gas_transport', 'misc_pos',
             'grocery_net', 'shopping_net', 'shopping_pos', 'food_dining',
             'personal_care', 'health_fitness', 'travel', 'kids_pets', 'home']

col1, col2 = st.columns(2)

with st.form("prediction_form"):
    with col1:
        st.subheader("Transaction Details")
        category = st.selectbox("Category", categories)
        amount = st.number_input("Amount ($)", value=100.0, min_value=0.0, format="%.2f")
        # Use separate date and time inputs, then combine
        trans_date = st.date_input("Transaction Date", datetime.now().date())
        trans_time = st.time_input("Transaction Time", datetime.now().time())
        # Combine date and time for the API
        trans_datetime = datetime.combine(trans_date, trans_time)


        st.subheader("Customer Details")
        gender = st.selectbox("Gender", ["M", "F"])
        job = st.text_input("Job Title", "Software Engineer") # Example job
        dob = st.date_input("Date of Birth", date(1990, 1, 1))
        city_pop = st.number_input("City Population", value=50000, min_value=0)

    with col2:
        st.subheader("Location Details")
        st.write("Customer Location")
        customer_latitude = st.number_input("Customer Latitude", value=34.05, format="%.6f")
        customer_longitude = st.number_input("Customer Longitude", value=-118.24, format="%.6f")

        st.write("Merchant Location")
        merchant_latitude = st.number_input("Merchant Latitude", value=34.00, format="%.6f")
        merchant_longitude = st.number_input("Merchant Longitude", value=-118.30, format="%.6f")

    # Submit button
    submit_button = st.form_submit_button(label='Detect Fraud')

if submit_button:
    # 1. Create the payload
    # Convert datetime/date objects to strings in ISO format for JSON
    payload = {
        "transaction_timestamp": trans_datetime.isoformat(),
        "date_of_birth": dob.isoformat(),
        "category": category,
        "amount": amount,
        "gender": gender,
        "job": job,
        "city_population": city_pop,
        "customer_latitude": customer_latitude,
        "customer_longitude": customer_longitude,
        "merchant_latitude": merchant_latitude,
        "merchant_longitude": merchant_longitude
    }

    try:
        # 2. Send POST request to the API
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        result = response.json()

        # 3. Display the result
        st.subheader("Prediction Result")
        if "error" in result:
             st.error(f"API Error: {result.get('error')} - {result.get('detail', '')}")
             st.json(result.get("traceback", "No traceback provided.")) # Show traceback if available
        elif result.get("is_fraud") == 1:
            st.error(f"**Prediction: FRAUD DETECTED** (Status: {result.get('prediction')})")
            st.write(f"Fraud Probability: {result.get('fraud_probability', 'N/A'):.4f}")
        else:
            st.success(f"**Prediction: NOT FRAUD** (Status: {result.get('prediction')})")
            st.write(f"Fraud Probability: {result.get('fraud_probability', 'N/A'):.4f}")

    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the API. Is `api.py` running?")
    except requests.exceptions.HTTPError as e:
         st.error(f"API Request Error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
         st.error(f"API Request Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred in Streamlit: {e}")
        import traceback
        st.text(traceback.format_exc())