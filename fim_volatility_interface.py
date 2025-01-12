# -*- coding: utf-8 -*-
# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from arch import arch_model  # For GARCH modeling

# Function to calculate return, variance, and deviation
def calculate_statistics(df):
    df['Return'] = df['DSEX Index'].pct_change()  # Daily return
    df['Variance'] = df['Return'].var()  # Overall variance
    df['Deviation'] = df['Return'].std()  # Overall standard deviation
    return df

# Function to calculate rolling return, variance, and deviation
def calculate_rolling_statistics(df, window=20):
    df['Rolling_Return'] = df['Return'].rolling(window=window).mean()  # Rolling return
    df['Rolling_Variance'] = df['Return'].rolling(window=window).var()  # Rolling variance
    df['Rolling_Deviation'] = df['Return'].rolling(window=window).std()  # Rolling standard deviation
    return df

# Function to calculate GARCH volatility
def calculate_garch_volatility(df, p=1, q=1, dist="normal"):
    # Ensure no NaN values in the return column for GARCH modeling
    returns = df['Return'].dropna()
    if returns.empty:
        df['GARCH_Volatility'] = np.nan
        return df
    
    # Fit a GARCH(p, q) model
    garch_model = arch_model(returns, vol='Garch', p=p, q=q, dist=dist)
    garch_fit = garch_model.fit(disp="off")
    
    # Predict conditional volatility
    df['GARCH_Volatility'] = np.nan
    df.loc[returns.index, 'GARCH_Volatility'] = garch_fit.conditional_volatility
    return df

# Function to load and process data
@st.cache_data  # Cache the data processing function
def load_data(file_path, rolling_window):
    # Read Excel file and rename columns
    df = pd.read_excel(file_path)
    df.columns = [
        'Date', 
        'DSEX Index', 
        'DSEX Index Change', 
        'Total Trade', 
        'Total Value Taka(mn)', 
        'Total Volume', 
        'Total Market Cap. Taka(mn)'
    ]
    
    # Convert Date column to datetime and sort the data
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    # Calculate return, variance, and deviation
    df = calculate_statistics(df)

    # Calculate rolling statistics with the chosen window size
    df = calculate_rolling_statistics(df, window=rolling_window)

    # Calculate GARCH volatility with default parameters (p=1, q=1, normal distribution)
    df = calculate_garch_volatility(df)

    return df

# Streamlit App
st.title("DSEX Volatility Analysis")

# File uploader for Excel data
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    # Add a slider to control the rolling window size
    rolling_window = st.slider("Select Rolling Window Size (Days)", min_value=4, max_value=20, value=20, step=1)

    # Load and process the uploaded data
    data = load_data(uploaded_file, rolling_window)

    # Display the processed data
    st.subheader("Processed Data")
    st.dataframe(data)

    # Plot volatility metrics
    st.subheader("Volatility Metrics")
    st.line_chart(data[['Date', 'Return']].set_index('Date'))
    st.line_chart(data[['Date', 'Rolling_Variance']].set_index('Date'))
    st.line_chart(data[['Date', 'Rolling_Deviation']].set_index('Date'))
    st.line_chart(data[['Date', 'GARCH_Volatility']].set_index('Date'))

    # Download button for the processed data
    st.download_button(
        label="Download Processed Data as CSV",
        data=data.to_csv(index=False),
        file_name="processed_volatility_data.csv",
        mime="text/csv",
    )



