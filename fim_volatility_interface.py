# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
from arch import arch_model  # For GARCH modeling

# Function to calculate returns
def calculate_returns(df, index_column="DSEX Index"):
    try:
        # Ensure the Date column is sorted
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')

        # Calculate nominal returns
        df['Return'] = df[index_column].pct_change()

        # Drop rows with NaN values created due to shifts
        df = df.dropna(subset=['Return'])
        return df
    except Exception as e:
        st.error(f"Error calculating returns: {e}")
        return pd.DataFrame()

# Function to calculate variance and standard deviation
def calculate_statistics(df):
    try:
        # Variance and standard deviation for 'Return' column
        df['Variance'] = df['Return'].expanding().var()
        df['Standard_Deviation'] = df['Return'].expanding().std()
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

# Function to calculate liquidity proxy, turnover ratio, volume spike, and market cap adjusted return
def calculate_additional_metrics(df):
    try:
        # Liquidity proxy: Total Value / Total Trade
        df['Liquidity_Proxy'] = df['Total Value Taka(mn)'] / df['Total Trade']

        # Turnover ratio: Total Value / Total Market Cap
        df['Turnover_Ratio'] = df['Total Value Taka(mn)'] / df['Total Market Cap. Taka(mn)']

        # Volume spike: Change in Total Volume
        df['Volume_Spike'] = df['Total Volume'].pct_change()

        # Market cap adjusted return
        df['Market_Cap_Adj_Return'] = df['Return'] * df['Total Market Cap. Taka(mn)']

        # Drop NaN rows created by percentage change
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error calculating additional metrics: {e}")
        return pd.DataFrame()

# Function to load and process data for a given sheet
@st.cache_data
def load_data(file_path, sheet_name, calculate_garch, garch_p=1, garch_q=1, garch_dist="normal"):
    try:
        # Load data from the specific sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # Standardize column names for processing
        if sheet_name == "Sheet1":
            df.columns = [
                'Date', 'DSEX Index', 'Total Trade', 'Total Value Taka(mn)', 
                'Total Volume', 'Total Market Cap. Taka(mn)'
            ]
            df = calculate_returns(df)
            df = calculate_statistics(df)
            df = calculate_additional_metrics(df)
        elif sheet_name == "Sheet2":
            df.columns = ['Date', 'DSEX Index']
            df = calculate_returns(df)
            df = calculate_statistics(df)
            if calculate_garch:
                df = calculate_garch_volatility(df, p=garch_p, q=garch_q, dist=garch_dist)

        return df
    except Exception as e:
        st.error(f"Error loading and processing data: {e}")
        return pd.DataFrame()

# Streamlit App
st.title("DSEX Volatility and Metrics Analysis")

# File uploader for Excel data
uploaded_file = st.file_uploader("Upload Excel File with Two Sheets", type=["xlsx"])

if uploaded_file is not None:
    # GARCH calculation toggle for Sheet2
    calculate_garch = st.checkbox("Calculate GARCH Volatility for Sheet2", value=True)
    garch_p = st.slider("GARCH p-parameter", min_value=1, max_value=5, value=1)
    garch_q = st.slider("GARCH q-parameter", min_value=1, max_value=5, value=1)
    garch_dist = st.selectbox("GARCH Distribution", options=["normal", "t", "skewt"], index=0)

    # Load and process Sheet1 data
    st.subheader("Metrics from Sheet1 (73 rows)")
    sheet1_data = load_data(uploaded_file, "Sheet1", calculate_garch=False)
    if not sheet1_data.empty:
        st.dataframe(sheet1_data)
        st.download_button(
            label="Download Sheet1 Processed Data as CSV",
            data=sheet1_data.to_csv(index=False),
            file_name="sheet1_processed_data.csv",
            mime="text/csv",
        )

    # Load and process Sheet2 data
    st.subheader("Metrics from Sheet2 (242 rows)")
    sheet2_data = load_data(uploaded_file, "Sheet2", calculate_garch, garch_p, garch_q, garch_dist)
    if not sheet2_data.empty:
        st.dataframe(sheet2_data)
        st.download_button(
            label="Download Sheet2 Processed Data as CSV",
            data=sheet2_data.to_csv(index=False),
            file_name="sheet2_processed_data.csv",
            mime="text/csv",
        )


