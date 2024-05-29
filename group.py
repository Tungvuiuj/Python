# Import necessary libraries
import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator, WMAIndicator
from ta.momentum import RSIIndicator
from ta.volume import ChaikinMoneyFlowIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from pmdarima import auto_arima
from prophet import Prophet
import plotly.graph_objects as go

# Set up the titles in Streamlit Application
st.title('Welcome to the Predicting Future Stock Value Application')
st.sidebar.info("This website was created and designed by [IUJ Group]")
st.sidebar.info('Please fill the cells below:')

# Create main function in main interface with 3 categories "Visualize", "Recent Data" and "Predict"
def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

# Set up data downloading function from Yahoo Finance
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

# Set up input information from users
option = st.sidebar.text_input('Enter a Stock Symbol', value='AAPL')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')

data = download_data(option, start_date, end_date)
scaler = StandardScaler()

# Showing technical indicators (Close price, Volume, BB, MACD, RSI, SMA, EMA, WMA, MA)
def tech_indicators():
    st.header('Technical Indicators')
    indicators = st.multiselect('Choose Technical Indicators to Visualize', ['Close Price', 'Volume','Bollinger Bands', 'Moving Average Convergence Divergence', 'Relative Strength Indicator', 'Simple Moving Average (SMA)', 'Exponential Moving Average (EMA)', 'Weighted Moving Average (WMA)', 'Moving Average (MA)'])

    fig = go.Figure()

    # Plot Close Price
    if 'Close Price' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

    # Plot Volume
    if 'Volume' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['Volume'], mode='lines', name='Volume', yaxis='y2'))

    # Plot Bollinger Bands
    if 'Bollinger Bands' in indicators:
        bb_indicator = BollingerBands(data.Close)
        data['bb_h'] = bb_indicator.bollinger_hband()
        data['bb_l'] = bb_indicator.bollinger_lband()
        fig.add_trace(go.Scatter(x=data.index, y=data['bb_h'], mode='lines', name='BB High'))
        fig.add_trace(go.Scatter(x=data.index, y=data['bb_l'], mode='lines', name='BB Low'))

    # Plot MACD
    if 'Moving Average Convergence Divergence' in indicators:
        data['macd'] = MACD(data.Close).macd()
        fig.add_trace(go.Scatter(x=data.index, y=data['macd'], mode='lines', name='MACD'))

    # Plot RSI
    if 'Relative Strength Indicator' in indicators:
        data['rsi'] = RSIIndicator(data.Close).rsi()
        fig.add_trace(go.Scatter(x=data.index, y=data['rsi'], mode='lines', name='RSI'))

    # Plot SMA
    if 'Simple Moving Average (SMA)' in indicators:
        sma_window = st.number_input('Enter SMA window:', min_value=1, value=50)
        data['sma'] = SMAIndicator(data.Close, window=sma_window).sma_indicator()
        fig.add_trace(go.Scatter(x=data.index, y=data['sma'], mode='lines', name=f'SMA {sma_window}'))

    # Plot EMA
    if 'Exponential Moving Average (EMA)' in indicators:
        ema_window = st.number_input('Enter EMA window:', min_value=1, value=50)
        data['ema'] = EMAIndicator(data.Close, window=ema_window).ema_indicator()
        fig.add_trace(go.Scatter(x=data.index, y=data['ema'], mode='lines', name=f'EMA {ema_window}'))

    # Plot WMA
    if 'Weighted Moving Average (WMA)' in indicators:
        wma_window = st.number_input('Enter WMA window:', min_value=1, value=50)
        data['wma'] = WMAIndicator(data.Close, window=wma_window).wma()
        fig.add_trace(go.Scatter(x=data.index, y=data['wma'], mode='lines', name=f'WMA {wma_window}'))

    # Plot MA
    if 'Moving Average (MA)' in indicators:
        ma_window = st.number_input('Enter MA window:', min_value=1, value=50)
        data['ma'] = data['Close'].rolling(window=ma_window).mean()
        fig.add_trace(go.Scatter(x=data.index, y=data['ma'], mode='lines', name=f'MA {ma_window}'))

    # Update layout for dual y-axis
    fig.update_layout(
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False
        )
    )

    st.plotly_chart(fig)

# Showing recent data:
def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(20))

# Showing future value estimation:
def model_engine(model, num):
    df = data[['Close']]
    df['preds'] = data.Close.shift(-num)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df.preds.values
    y = y[:-num]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=5)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \nMAE: {mean_absolute_error(y_test, preds)}')

    forecast_pred = model.predict(x_forecast)
    day = 1
    predictions = []
    for i in forecast_pred:
        predictions.append(i)
        day += 1

    forecast_dates = pd.date_range(end=end_date, periods=num + 1)[1:]
    predicted_data = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': predictions})

    return predicted_data 

# Creating interface for choosing learning model, prediction days, etc.
def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = st.number_input('How many days do you want to forecast?', value=10)
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            predicted_data = model_engine(engine, num)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            predicted_data = model_engine(engine, num)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            predicted_data = model_engine(engine, num)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            predicted_data = model_engine(engine, num)
        elif model == 'XGBoostRegressor':
            engine = XGBRegressor()
            predicted_data = model_engine(engine, num)
        
        st.header('Predicted Stock Prices')
        st.line_chart(predicted_data.set_index('Date'))

if __name__ == '__main__':
    main()
