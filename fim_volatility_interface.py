# -*- coding: utf-8 -*-
# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from arch import arch_model  # For GARCH modeling

# Function to calculate effective and nominal returns
def calculate_returns(df, return_type="nominal", m=12, n=1):
    # Ensure the Date column is sorted and calculate the time difference
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df['Days_Diff'] = df['Date'].diff().dt.days

    if return_type == "nominal":
        # Nominal return: Return relative to the previous value
        df['Return'] = df['DSEX Index'].pct_change()
    elif return_type == "effective":
        # Effective return: Accounts for time between observations
        df['Return'] = (df['DSEX Index'] / df['DSEX Index'].shift(1)) ** (365 / df['Days_Diff']) - 1

    # Drop rows with NaN values created due to shifts
    df = df.dropna(subset=['Return'])
    return df

# Function to calculate variance and standard deviation
def calculate_statistics(df):
    df['Variance'] = df['Return'].var()  # Overall variance
    df['Deviation'] = df['Return'].std()  # Overall standard deviation
    return df

# Function to calculate rolling return, variance, and deviation
def calculate_rolling_statistics(df, rolling_window_return=20, rolling_window_variance=20, rolling_window_deviation=20):
    df['Rolling_Return'] = df['Return'].rolling(window=rolling_window_return).mean()  # Rolling return
    df['Rolling_Variance'] = df['Return'].rolling(window=rolling_window_variance).var()  # Rolling variance
    df['Rolling_Deviation'] = df['Return'].rolling(window=rolling_window_deviation).std()  # Rolling standard deviation
    return df

# Function to calculate GARCH volatility
def calculate_garch_volatility(df, p=1, q=1, dist="normal", rolling_window_garch=None):
    returns = df['Return'].dropna()
    if returns.empty:
        df['GARCH_Volatility'] = np.nan
        return df

    # Fit a GARCH(p, q) model
    garch_model = arch_model(returns, vol='Garch', p=p, q=q, dist=dist)
    garch_fit = garch_model.fit(disp="off")

    # Predict conditional volatility
    garch_volatility = garch_fit.conditional_volatility
    if rolling_window_garch:
        garch_volatility = garch_volatility.rolling(window=rolling_window_garch).mean()

    df['GARCH_Volatility'] = np.nan
    df.loc[returns.index, 'GARCH_Volatility'] = garch_volatility
    return df

# Function to load and process data
@st.cache_data
def load_data(file_path, return_type, m, n, rolling_window_return, rolling_window_variance, rolling_window_deviation, rolling_window_garch, garch_p, garch_q, garch_dist):
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

    # Calculate customizable returns
    df = calculate_returns(df, return_type=return_type, m=m, n=n)

    # Calculate overall variance and deviation
    df = calculate_statistics(df)

    # Calculate rolling statistics with separate window sizes
    df = calculate_rolling_statistics(
        df, 
        rolling_window_return=rolling_window_return, 
        rolling_window_variance=rolling_window_variance, 
        rolling_window_deviation=rolling_window_deviation
    )

    # Calculate GARCH volatility with separate rolling window
    df = calculate_garch_volatility(df, p=garch_p, q=garch_q, dist=garch_dist, rolling_window_garch=rolling_window_garch)

    return df

# Streamlit App
st.title("DSEX Volatility Analysis")

# File uploader for Excel data
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    # Dropdown for return type
    return_type = st.selectbox("Select Return Type", options=["nominal", "effective"], index=0)

    # Parameters for return calculation
    m = st.slider("Compounding Frequency (m)", min_value=1, max_value=365, value=365)
    n = st.slider("Period Selection (n)", min_value=1, max_value=2, value=2)

    # Rolling window sliders for different analyses
    rolling_window_return = st.slider("Rolling Window for Return (Days)", min_value=4, max_value=50, value=20)
    rolling_window_variance = st.slider("Rolling Window for Variance (Days)", min_value=4, max_value=50, value=20)
    rolling_window_deviation = st.slider("Rolling Window for Deviation (Days)", min_value=4, max_value=50, value=20)
    rolling_window_garch = st.slider("Rolling Window for GARCH (Days)", min_value=4, max_value=50, value=20)

    # GARCH parameters
    garch_p = st.slider("GARCH p-parameter", min_value=1, max_value=5, value=1)
    garch_q = st.slider("GARCH q-parameter", min_value=1, max_value=5, value=1)
    garch_dist = st.selectbox("GARCH Distribution", options=["normal", "t", "skewt"], index=0)

    # Load and process the uploaded data
    data = load_data(uploaded_file, return_type, m, n, rolling_window_return, rolling_window_variance, rolling_window_deviation, rolling_window_garch, garch_p, garch_q, garch_dist)

    # Display the processed data
    st.subheader("Processed Data")
    st.dataframe(data)

    # Statistical Summary
    st.subheader("Statistical Summary")
    st.write("Summary of key metrics:")
    st.write(data.describe())

    # Plot volatility metrics
    st.subheader("Volatility Metrics")

    st.write("### Daily Returns")
    st.line_chart(data[['Date', 'Return']].set_index('Date'))

    st.write(f"### Rolling Variance (Window: {rolling_window_variance} Days)")
    st.line_chart(data[['Date', 'Rolling_Variance']].set_index('Date'))

    st.write(f"### Rolling Standard Deviation (Window: {rolling_window_deviation} Days)")
    st.line_chart(data[['Date', 'Rolling_Deviation']].set_index('Date'))

    st.write(f"### GARCH Volatility (Window: {rolling_window_garch} Days, p={garch_p}, q={garch_q}, Distribution: {garch_dist})")
    st.line_chart(data[['Date', 'GARCH_Volatility']].set_index('Date'))

    # Download button for the processed data
    st.download_button(
        label="Download Processed Data as CSV",
        data=data.to_csv(index=False),
        file_name="processed_volatility_data.csv",
        mime="text/csv",
    )



