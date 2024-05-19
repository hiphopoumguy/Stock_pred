# Stock Price Prediction using LSTM

This project involves developing a stock price prediction model using Long Short-Term Memory (LSTM) networks. The goal is to predict future stock prices based on historical data.

## Project Background

The motivation behind this project is to apply machine learning techniques to financial market data and gain insights into stock price movements. The model aims to help in understanding market trends and making informed investment decisions.

## Features

- Data preprocessing, including handling missing values and outliers
- LSTM model for time series prediction
- Visualization of actual and predicted stock prices

## Data Source

The stock price data is fetched using the `yfinance` library. The example in this project uses Apple Inc. (AAPL) stock data from January 1, 2010, to January 1, 2023.

## Prerequisites

Make sure you have the following packages installed:

```bash
pip install pandas numpy matplotlib tensorflow yfinance scikit-learn
