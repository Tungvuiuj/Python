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

# Set up the titles in Streamlit Application
st.title('Predicting Future Stock Value')
st.sidebar.info('Welcome to the Predicting Future Stock Value Application')
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
option = st.sidebar.text_input('Enter a Stock Symbol', value='TSM')
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

# Showing technical indicators (Close price, BB, MACD, RSI, SMA, EMA, WMA, CMF)
def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close Price', 'Bollinger Bands', 'Moving Average Convergence Divergence', 'Relative Strength Indicator', 'Simple Moving Average (SMA)', 'Exponential Moving Average (EMA)', 'Weighted Moving Average (WMA)', 'Moving Average (MA)',])

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Close', 'upper', 'lower']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    if option == 'Simple Moving Average (SMA)':
        sma_window = st.number_input('Enter SMA window:', min_value=1, value=50)
        sma = SMAIndicator(data.Close, window=sma_window).sma_indicator()
    # EMA
    if option == 'Exponential Moving Average (EMA)':
        ema_window = st.number_input('Enter EMA window:', min_value=1, value=50)
        ema = EMAIndicator(data.Close, window=ema_window).ema_indicator()
    # WMA
    if option == 'Weighted Moving Average (WMA)':
        wma_window = st.number_input('Enter WMA window:', min_value=1, value=50)
        wma = WMAIndicator(data.Close, window=wma_window).wma()
    # MA
    if option == 'Moving Average (MA)':
        ma_window = st.number_input('Enter MA window:', min_value=1, value=50)
        ma = data['Close'].rolling(window=ma_window).mean()

    if option == 'Close Price':
        st.write('Close Price')
        st.line_chart(data.Close)
    elif option == 'Bollinger Bands':
        st.write('Bollinger Bands')
        st.line_chart(bb)
    elif option == 'Moving Average Convergence Divergence':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'Relative Strength Indicator':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option == 'Simple Moving Average (SMA)':
        st.write(f'Simple Moving Average ({sma_window})')
        st.line_chart(sma)
    elif option == 'Exponential Moving Average (EMA)':
        st.write(f'Exponential Moving Average ({ema_window})')
        st.line_chart(ema)
    elif option == 'Weighted Moving Average (WMA)':
        st.write(f'Weighted Moving Average ({wma_window})')
        st.line_chart(wma)
    elif option == 'Moving Average (MA)':
        st.write(f'Moving Average ({ma_window})')
        st.line_chart(ma)

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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=8)
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
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor', 'ARIMA', 'PROPHET'])
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
        elif model == 'ARIMA':
            predicted_data = arima_model(num)
        else:
            predicted_data = prophet_model(num)
        
        st.header('Predicted Stock Prices')
        st.line_chart(predicted_data.set_index('Date'))

# Run application
if __name__ == '__main__':
    main()
