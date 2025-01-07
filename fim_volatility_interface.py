# -*- coding: utf-8 -*-
"""FIM_volatility_interface.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kguUTHh3HJ9RLrRt_S9luHAt7ZKPil_X
"""
import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model
import plotly.express as px

# Volatility Calculation Functions
def calculate_historical_volatility(data, window=12):
    data['Log_Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data['Historical_Volatility'] = data['Log_Returns'].rolling(window=window).std() * np.sqrt(12)
    return data['Historical_Volatility']

def calculate_garch_volatility(data):
    data['Log_Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna()
    model = arch_model(data['Log_Returns'], vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')
    data['GARCH_Volatility'] = np.sqrt(model_fit.conditional_volatility) * np.sqrt(12)
    return data['GARCH_Volatility']

def calculate_volume_weighted_volatility(data, window=12):
    data['Log_Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data['Weighted_Variance'] = (data['Volume'] * (data['Log_Returns'] - data['Log_Returns'].mean())**2)
    data['Volume_Weighted_Volatility'] = (
        data['Weighted_Variance'].rolling(window=window).sum() / data['Volume'].rolling(window=window).sum()
    ) ** 0.5 * np.sqrt(12)
    return data['Volume_Weighted_Volatility']

def calculate_volatility(data, method="historical", **kwargs):
    if method == "historical":
        return calculate_historical_volatility(data, **kwargs)
    elif method == "garch":
        return calculate_garch_volatility(data)
    elif method == "volume_weighted":
        return calculate_volume_weighted_volatility(data, **kwargs)
    else:
        raise ValueError("Invalid method. Choose from 'historical', 'garch', 'volume_weighted'.")

# Data Loading Function
@st.cache
def load_data(file_path):
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Company Name', 'Date'])

    # Filter for the last 10 years of data
    start_date = pd.Timestamp.now() - pd.DateOffset(years=10)
    df = df[df['Date'] >= start_date]

    # Calculate volatility measures
    # Fix: Applying volatility functions without returning a DataFrame
    df['Historical_Volatility'] = df.groupby('Company Name').apply(
        lambda group: calculate_volatility(group, method="historical", window=12)
    ).reset_index(level=0, drop=True)  # Remove multi-index

    df['GARCH_Volatility'] = df.groupby('Company Name').apply(
        lambda group: calculate_volatility(group, method="garch")
    ).reset_index(level=0, drop=True)  # Remove multi-index

    df['Volume_Weighted_Volatility'] = df.groupby('Company Name').apply(
        lambda group: calculate_volatility(group, method="volume_weighted", window=12)
    ).reset_index(level=0, drop=True)  # Remove multi-index

    return df

# Streamlit App
st.title("Volatility Analysis for Companies and Industries")

# File Upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Sidebar Filters
    st.sidebar.header("Filter Options")
    companies = data['Company Name'].unique()
    selected_companies = st.sidebar.multiselect("Select Companies", companies, default=companies)

    sectors = data['Sector'].unique()
    selected_sectors = st.sidebar.multiselect("Select Sectors", sectors, default=sectors)

    # Filter Data
    filtered_data = data[ 
        (data['Company Name'].isin(selected_companies)) & 
        (data['Sector'].isin(selected_sectors))
    ]

    # Select Volatility Type
    volatility_type = st.selectbox(
        "Select Volatility Type:",
        ["Historical_Volatility", "GARCH_Volatility", "Volume_Weighted_Volatility"]
    )

    # Show Volatility for Companies
    if st.checkbox("Show Volatility for Companies"):
        for company in selected_companies:
            company_data = filtered_data[filtered_data['Company Name'] == company]
            fig = px.line(
                company_data, x='Date', y=volatility_type,
                title=f"{company} - {volatility_type.replace('_', ' ')} Over Time",
                labels={'Date': 'Date', volatility_type: volatility_type.replace('_', ' ')}
            )
            st.plotly_chart(fig)

    # Compare Volatility Across Sectors
    if st.checkbox("Compare Industry Volatilities"):
        industry_volatility = filtered_data.groupby(['Sector', 'Date'])[volatility_type].mean().reset_index()
        fig = px.line(
            industry_volatility, x='Date', y=volatility_type, color='Sector',
            title=f"Sector-Wise {volatility_type.replace('_', ' ')} Comparison",
            labels={'Date': 'Date', volatility_type: volatility_type.replace('_', ' '), 'Sector': 'Sector'}
        )
        st.plotly_chart(fig)

    # Display Raw Data
    if st.checkbox("Show Raw Data"):
        st.write(filtered_data)
else:
    st.info("Please upload an Excel file to begin.")


