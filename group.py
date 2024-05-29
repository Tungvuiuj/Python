#Import necesarry libraries
import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

#Set up the tittles in Streamlit Application
st.title('Prediciting future stock value')
st.sidebar.info('Welcome to the Predicting Future Stock Value Application. Choose your interest below')
st.sidebar.info("Created and designed by [IUJ Group]")

#Create main function in main interface with 3 categories "Visualize", "Recent Data" and "Predict"
def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()
        
#Set up data downloading function from Yahoo Finance
@st.cache
def download_data(op, start_date, end_date):
    try:
        df = yf.download(op, start=start_date, end=end_date, progress=False)
        if df.empty:
            st.error(f"No data found for {op}. Please check the stock symbol and try again.")
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data for {op}: {e}")
        return None
        
#Set up input information from users
option = st.sidebar.text_input('Enter a Stock Symbol', value='TSM')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration (days)', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End Date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success(f'Start date: `{start_date}`\n\nEnd date: `{end_date}`')
        data = download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must be after start date')

#Downloading data
data = download_data(option, start_date, end_date)
scaler = StandardScaler()

#Showing technical indicators (Close price,BB,MACD,RSI,SMA,EMA)
def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()


    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data.Close)
    elif option == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Exponential Moving Average')
        st.line_chart(ema)

#Showing recent data:
def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(20))

#Showing future value estimation:
def model_engine(model, num):
    df = data[['Close']]
    df['preds'] = data.Close.shift(-num)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df.preds.values
    y = y[:-num]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=10)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \nMAE: {mean_absolute_error(y_test, preds)}')

    forecast_pred = model.predict(x_forecast)
    day = 1
    predictions = []
    for i in forecast_pred:
        predictions.append(i)
        day += 1

    forecast_dates = pd.date_range(end=end_date, periods=num+1)[1:]
    predicted_data = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': predictions})

    return predicted_data

#Creating interface for choosing learning model, prediction days,...
def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = st.number_input('How many days forecast?', value=10)
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
        else:
            engine = XGBRegressor()
            predicted_data = model_engine(engine, num)

        st.header('Predicted Stock Prices')
        st.line_chart(predicted_data.set_index('Date'))

#Run application
if __name__ == '__main__':
    main()
