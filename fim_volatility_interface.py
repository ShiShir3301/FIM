# -*- coding: utf-8 -*-
# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model

# Function to load and validate data
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        required_columns = ['Date', 'DSEX Index']
        if not all(col in df.columns for col in required_columns):
            st.error("Uploaded file is missing required columns: 'Date' and 'DSEX Index'.")
            return None
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'DSEX Index']).sort_values(by='Date')
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Function to calculate returns
def calculate_returns(df):
    df['Days_Diff'] = df['Date'].diff().dt.days
    df = df[df['Days_Diff'] > 0]
    df['Return'] = (df['DSEX Index'] / df['DSEX Index'].shift(1)) ** (365 / df['Days_Diff']) - 1
    return df.dropna(subset=['Return'])

# Function to calculate rolling statistics
def calculate_rolling_stats(df, rolling_window):
    df['Rolling_Return'] = df['Return'].rolling(window=rolling_window).mean()
    df['Rolling_Variance'] = df['Return'].rolling(window=rolling_window).var()
    df['Rolling_Deviation'] = df['Return'].rolling(window=rolling_window).std()
    return df

# Function to calculate GARCH volatility
def calculate_garch_volatility(df, p=1, q=1, dist='normal'):
    returns = df['Return'].dropna()
    if not returns.empty:
        garch_model = arch_model(returns, vol='Garch', p=p, q=q, dist=dist)
        garch_fit = garch_model.fit(disp="off")
        garch_volatility = garch_fit.conditional_volatility
        df['GARCH_Volatility'] = np.nan
        df.loc[returns.index, 'GARCH_Volatility'] = garch_volatility
    else:
        df['GARCH_Volatility'] = np.nan
    return df

# Streamlit app
st.title("DSEX Index Analysis with GARCH Volatility")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file with 'Date' and 'DSEX Index' columns.", type=['csv'])
if uploaded_file:
    data = load_data(uploaded_file)
    if data is not None:
        # Calculate returns
        data = calculate_returns(data)

        # Display variance and deviation
        variance = data['Return'].var()
        deviation = data['Return'].std()
        st.write(f"Variance of returns: {variance:.6f}")
        st.write(f"Standard deviation of returns: {deviation:.6f}")

        # Rolling statistics
        rolling_window = st.slider("Select Rolling Window Size (Days):", min_value=5, max_value=50, value=20)
        data = calculate_rolling_stats(data, rolling_window)

        # GARCH volatility
        st.write("Calculating GARCH Volatility...")
        p = st.slider("GARCH(p):", min_value=1, max_value=5, value=1)
        q = st.slider("GARCH(q):", min_value=1, max_value=5, value=1)
        data = calculate_garch_volatility(data, p=p, q=q)

        # Display line charts
        st.subheader("Line Chart of Returns")
        st.line_chart(data.set_index('Date')['Return'])

        st.subheader("Rolling Statistics")
        st.line_chart(data.set_index('Date')[['Rolling_Return', 'Rolling_Variance', 'Rolling_Deviation']])

        st.subheader("GARCH Volatility")
        st.line_chart(data.set_index('Date')['GARCH_Volatility'])

        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(data)
    else:
        st.error("Unable to process the uploaded file. Please check the file format.")



