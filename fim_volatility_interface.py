# -*- coding: utf-8 -*-
# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from arch import arch_model  # For GARCH modeling

# Function to calculate effective and nominal returns
def calculate_returns(df, return_type="nominal", m=12):
    try:
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
        else:
            raise ValueError("Invalid return type selected. Choose either 'nominal' or 'effective'.")
        
        # Drop rows with NaN values created due to shifts
        df = df.dropna(subset=['Return'])
        return df
    except Exception as e:
        st.error(f"Error calculating returns: {e}")
        return pd.DataFrame()

# Function to calculate variance and standard deviation based on return type
def calculate_statistics(df):
    try:
        # Variance and standard deviation are computed for the 'Return' column
        df['Variance'] = df['Return'].expanding().var()
        df['Deviation'] = df['Return'].expanding().std()
        return df
    except Exception as e:
        st.error(f"Error calculating statistics: {e}")
        return pd.DataFrame()

# Function to calculate GARCH volatility
def calculate_garch_volatility(df, p=1, q=1, dist="normal"):
    try:
        returns = df['Return'].dropna()
        if returns.empty:
            st.warning("No valid returns to calculate GARCH volatility.")
            df['GARCH_Volatility'] = np.nan
            return df

        # Fit a GARCH(p, q) model
        garch_model = arch_model(returns, vol='Garch', p=p, q=q, dist=dist)
        garch_fit = garch_model.fit(disp="off")

        # Predict conditional volatility
        df['GARCH_Volatility'] = np.nan
        df.loc[returns.index, 'GARCH_Volatility'] = garch_fit.conditional_volatility
        return df
    except Exception as e:
        st.error(f"Error calculating GARCH volatility: {e}")
        return pd.DataFrame()

# Function to load and process data
@st.cache_data
def load_data(file_path, return_type, m, garch_p, garch_q, garch_dist):
    try:
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

        # Calculate returns
        df = calculate_returns(df, return_type=return_type, m=m)

        # Calculate overall variance and deviation based on returns
        df = calculate_statistics(df)

        # Calculate GARCH volatility
        df = calculate_garch_volatility(df, p=garch_p, q=garch_q, dist=garch_dist)

        return df
    except Exception as e:
        st.error(f"Error loading and processing data: {e}")
        return pd.DataFrame()

# Streamlit App
st.title("DSEX Volatility Analysis")

# File uploader for Excel data
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    # Dropdown for return type
    return_type = st.selectbox("Select Return Type", options=["nominal", "effective"], index=0)

    # Parameters for return calculation
    m = st.slider("Compounding Frequency (m)", min_value=1, max_value=365, value=365)

    # GARCH parameters
    garch_p = st.slider("GARCH p-parameter", min_value=1, max_value=5, value=1)
    garch_q = st.slider("GARCH q-parameter", min_value=1, max_value=5, value=1)
    garch_dist = st.selectbox("GARCH Distribution", options=["normal", "t", "skewt"], index=0)

    # Load and process the uploaded data
    data = load_data(uploaded_file, return_type, m, garch_p, garch_q, garch_dist)

    if not data.empty:
        # Display the processed data
        st.subheader("Processed Data")
        st.dataframe(data)

        # Statistical Summary
        st.subheader("Statistical Summary")
        st.write(data.describe())

        # Plot volatility metrics
        st.subheader("Volatility Metrics")
        st.write("### Daily Returns")
        st.line_chart(data[['Date', 'Return']].set_index('Date'))

        st.write("### Variance")
        st.line_chart(data[['Date', 'Variance']].set_index('Date'))

        st.write("### Standard Deviation")
        st.line_chart(data[['Date', 'Deviation']].set_index('Date'))

        st.write("### GARCH Volatility")
        st.line_chart(data[['Date', 'GARCH_Volatility']].set_index('Date'))

        # Download button for the processed data
        st.download_button(
            label="Download Processed Data as CSV",
            data=data.to_csv(index=False),
            file_name="processed_volatility_data.csv",
            mime="text/csv",
        )
