import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
st.markdown("""
    <style>
        /* Global Styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Arial', sans-serif;
            color: white;
            padding: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }

        /* Hero Section */
        .hero {
            background-color: #1E2A47;
            color: white;
            padding: 80px 20px;
            text-align: center;
            margin-bottom: 40px;
            border-radius: 8px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15);
        }

        .hero h1 {
            font-size: 3.5em;
            font-weight: bold;
            margin-bottom: 10px;
            color:White;
        }

        .hero p {
            font-size: 1.2em;
            margin-bottom: 20px;
            text-align: center;
        }

        /* Sidebar Styling */
        .sidebar {
            background-color: #283B5E;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
        }

        .sidebar .sidebar-header {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #A0C4FF;
        }

        .sidebar select, .sidebar input, .sidebar .slider {
            width: 100%;
            padding: 12px;
            border-radius: 5px;
            border: 1px solid #ddd;
            margin-bottom: 15px;
            font-size: 1em;
            transition: border-color 0.3s;
        }

        .sidebar select:focus, .sidebar input:focus, .sidebar .slider:focus {
            border-color: #A0C4FF;
            outline: none;
        }

        /* Button Styles */
        .stButton > button {
            background-color: #1E2A47;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            font-size: 1.1em;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.3s;
        }

        .stButton > button:hover {
            background-color: #A0C4FF;
            transform: translateY(-2px);
        }

        /* Results Box */
        .results {
            background-color: #283B5E;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            margin-top: 30px;
        }

        .results h3 {
            font-size: 2em;
            margin-bottom: 20px;
        }

        .results p {
            font-size: 1.2em;
            margin-bottom: 10px;
        }

        .results .prediction {
            font-size: 1.5em;
            font-weight: bold;
            color: #A0C4FF;
        }

        /* Plot Styles */
        .plot {
            margin-top: 30px;
            background-color: #283B5E;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }

        .plot:hover {
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.15);
        }

        /* Responsive Layout */
        @media (max-width: 768px) {
            .hero h1 {
                font-size: 2.5em;
            }

            .hero p {
                font-size: 1em;
            }

            .sidebar select, .sidebar input, .sidebar .slider {
                font-size: 0.9em;
                padding: 10px;
            }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero">
        <h1>Currency Exchange Rate Predictor</h1>
        <p>Predict the exchange rate to USD based on macroeconomic indicators</p>
    </div>
""", unsafe_allow_html=True)

model = joblib.load("C:\\Users\\santosh nandam\\Downloads\\best_model_RandomForest.pkl")

st.sidebar.header('User Input Parameters')
currency_encoder = LabelEncoder()
currencies = ['EUR', 'USD', 'GBP', 'INR', 'CNY', 'JPY']
currency_encoder.fit(currencies)

def user_input_features():
    currency = st.sidebar.selectbox('Select Currency:', currencies)
    interest_rate = st.sidebar.slider('Interest Rate (%)', 0.0, 20.0, 1.5, 0.1)
    inflation = st.sidebar.slider('Inflation Rate (%)', 0.0, 20.0, 2.0, 0.1)
    gdp_growth = st.sidebar.slider('GDP Growth Rate (%)', -10.0, 10.0, 2.5, 0.1)
    unemployment = st.sidebar.slider('Unemployment Rate (%)', 0.0, 20.0, 5.0, 0.1)
    
    encoded_currency = currency_encoder.transform([currency])[0]

    data = {
        'Currency': [encoded_currency],
        'Interest_Rate': [interest_rate],
        'Inflation_Rate': [inflation],
        'GDP_Growth_Rate': [gdp_growth],
        'Unemployment_Rate': [unemployment]
    }
    
    features = pd.DataFrame(data)
    return features, currency

df, currency = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

if st.button('Get Prediction'):
    with st.spinner('Predicting exchange rate...'):
        try:
            prediction = model.predict(df)[0]
            preds = [model.predict(df) for _ in range(100)]
            noise_level = 0.1
            preds = [model.predict(df * (1 + np.random.uniform(-noise_level, noise_level, df.shape))) for _ in range(100)]
            preds = np.array(preds).flatten()
            lower_bound = np.percentile(preds, 2.5)
            upper_bound = np.percentile(preds, 97.5)

            st.success(f'**Predicted Exchange Rate for {currency} to USD:** {prediction:.4f}')
            st.info(f'95% Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]')
            
            st.subheader('Prediction Visualization')
            fig, ax = plt.subplots()
            ax.plot(preds.flatten(), color='lightblue', alpha=0.6, label='Simulated Predictions')
            ax.axhline(y=prediction, color='red', linestyle='--', label='Mean Prediction')
            ax.fill_between(range(100), lower_bound, upper_bound, color='orange', alpha=0.3, label='Confidence Interval')
            ax.legend()
            ax.set_title(f"Prediction Distribution for {currency}")
            st.pyplot(fig)

        except Exception as e:
            st.error(f'Error during prediction: {e}')

