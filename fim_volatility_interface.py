# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from arch import arch_model  # For GARCH modeling

# Function to calculate returns
def calculate_returns(df, index_column="DSEX Index"):
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
        df['Return'] = df[index_column].pct_change()
        df = df.dropna(subset=['Return'])
        return df
    except Exception as e:
        st.error(f"Error calculating returns: {e}")
        return pd.DataFrame()

# Function to calculate variance and standard deviation
def calculate_statistics(df):
    try:
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

        garch_model = arch_model(returns, vol='Garch', p=p, q=q, dist=dist)
        garch_fit = garch_model.fit(disp="off")
        df['GARCH_Volatility'] = np.nan
        df.loc[returns.index, 'GARCH_Volatility'] = garch_fit.conditional_volatility
        return df
    except Exception as e:
        st.error(f"Error calculating GARCH volatility: {e}")
        return pd.DataFrame()

# Function to calculate additional metrics
def calculate_additional_metrics(df):
    try:
        df['Liquidity_Proxy'] = df['Total Value Taka(mn)'] / df['Total Trade']
        df['Turnover_Ratio'] = df['Total Value Taka(mn)'] / df['Total Market Cap. Taka(mn)']
        df['Volume_Spike'] = df['Total Volume'].pct_change()
        df['Market_Cap_Adj_Return'] = df['Return'] * df['Total Market Cap. Taka(mn)']
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error calculating additional metrics: {e}")
        return pd.DataFrame()

# Function to load and process data for a given sheet
@st.cache_data
def load_data(file_path, sheet_name, calculate_garch, garch_p=1, garch_q=1, garch_dist="normal"):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)

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

# Function to plot figures
def plot_figures(df, sheet_name):
    try:
        st.write(f"### Figures for {sheet_name}")

        # Time-series plot of returns
        if 'Return' in df.columns:
            st.write("#### Time-Series of Daily Returns")
            st.line_chart(df.set_index('Date')['Return'], height=400, use_container_width=True)

        # Time-series plot of GARCH Volatility
        if 'GARCH_Volatility' in df.columns:
            st.write("#### Time-Series of GARCH Volatility")
            st.line_chart(df.set_index('Date')['GARCH_Volatility'], height=400, use_container_width=True)

        # Time-series plots for Variance and Standard Deviation
        if 'Variance' in df.columns and 'Standard_Deviation' in df.columns:
            st.write("#### Time-Series of Variance and Standard Deviation")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['Date'], df['Variance'], label='Variance', color='blue')
            ax.plot(df['Date'], df['Standard_Deviation'], label='Standard Deviation', color='orange')
            ax.set_title("Variance and Standard Deviation Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid()
            st.pyplot(fig)

        # Combined figure: GARCH Volatility, Variance, and Standard Deviation
        if 'GARCH_Volatility' in df.columns and 'Variance' in df.columns and 'Standard_Deviation' in df.columns:
            st.write("#### Combined Figure: GARCH Volatility, Variance, and Standard Deviation")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['Date'], df['GARCH_Volatility'], label='GARCH Volatility', color='green')
            ax.plot(df['Date'], df['Variance'], label='Variance', color='blue')
            ax.plot(df['Date'], df['Standard_Deviation'], label='Standard Deviation', color='orange')
            ax.set_title("GARCH Volatility, Variance, and Standard Deviation Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid()
            st.pyplot(fig)

        # Histogram of Returns
        if 'Return' in df.columns:
            st.write("#### Histogram of Daily Returns")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['Return'], bins=50, color='skyblue', edgecolor='black')
            ax.set_title("Histogram of Daily Returns")
            ax.set_xlabel("Return")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error generating plots: {e}")

# Streamlit App
st.title("DSEX Volatility and Metrics Analysis")

uploaded_file = st.file_uploader("Upload Excel File with Two Sheets", type=["xlsx"])

if uploaded_file is not None:
    calculate_garch = st.checkbox("Calculate GARCH Volatility for Sheet2", value=True)
    garch_p = st.slider("GARCH p-parameter", min_value=1, max_value=5, value=1)
    garch_q = st.slider("GARCH q-parameter", min_value=1, max_value=5, value=1)
    garch_dist = st.selectbox("GARCH Distribution", options=["normal", "t", "skewt"], index=0)

    st.subheader("Metrics from Sheet1 (73 rows)")
    sheet1_data = load_data(uploaded_file, "Sheet1", calculate_garch=False)
    if not sheet1_data.empty:
        st.dataframe(sheet1_data)
        plot_figures(sheet1_data, "Sheet1")
        st.download_button(
            label="Download Sheet1 Processed Data as CSV",
            data=sheet1_data.to_csv(index=False),
            file_name="sheet1_processed_data.csv",
            mime="text/csv",
        )

    st.subheader("Metrics from Sheet2 (242 rows)")
    sheet2_data = load_data(uploaded_file, "Sheet2", calculate_garch, garch_p, garch_q, garch_dist)
    if not sheet2_data.empty:
        st.dataframe(sheet2_data)
        plot_figures(sheet2_data, "Sheet2")
        st.download_button(
            label="Download Sheet2 Processed Data as CSV",
            data=sheet2_data.to_csv(index=False),
            file_name="sheet2_processed_data.csv",
            mime="text/csv",
        )

