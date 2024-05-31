import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator, WMAIndicator
from ta.momentum import RSIIndicator
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error
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
        data = download_data(option, start_date, end_date)
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

    st.plotly_chart(fig)

    # Plot Volume in a separate chart
    if 'Volume' in indicators:
        st.write('Volume')
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=data.index, y=data['Volume'], mode='lines', name='Volume'))
        st.plotly_chart(fig_vol)

    # Plot Bollinger Bands in a separate chart
    if 'Bollinger Bands' in indicators:
        st.write('Bollinger Bands')
        bb_indicator = BollingerBands(data.Close)
        data['bb_h'] = bb_indicator.bollinger_hband()
        data['bb_l'] = bb_indicator.bollinger_lband()
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        fig_bb.add_trace(go.Scatter(x=data.index, y=data['bb_h'], mode='lines', name='BB High'))
        fig_bb.add_trace(go.Scatter(x=data.index, y=data['bb_l'], mode='lines', name='BB Low'))
        st.plotly_chart(fig_bb)

    # Plot MACD in a separate chart
    if 'Moving Average Convergence Divergence' in indicators:
        st.write('Moving Average Convergence Divergence')
        data['macd'] = MACD(data.Close).macd()
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['macd'], mode='lines', name='MACD'))
        st.plotly_chart(fig_macd)

    # Plot RSI in a separate chart
    if 'Relative Strength Indicator' in indicators:
        st.write('Relative Strength Indicator')
        data['rsi'] = RSIIndicator(data.Close).rsi()
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['rsi'], mode='lines', name='RSI'))
        st.plotly_chart(fig_rsi)

# Showing recent data:
def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(20))

# Function to train and evaluate models
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

    forecast_dates = pd.date_range(start=data.index[-1], periods=num + 1, freq='B')[1:]
    predicted_data = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': predictions})

    return predicted_data 

# Creating interface for choosing learning model, prediction days, etc.
def predict():
    model_name = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor', 'SVR', 'DecisionTreeRegressor', 'GradientBoostingRegressor', 'LightGBM', 'CatBoost'])
    num = st.number_input('How many days do you want to forecast?', value=10)
    num = int(num)
    if st.button('Predict'):
        if model_name == 'LinearRegression':
            engine = LinearRegression()
        elif model_name == 'RandomForestRegressor':
            engine = RandomForestRegressor()
        elif model_name == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
        elif model_name == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
        elif model_name == 'XGBoostRegressor':
            engine = XGBRegressor()
        elif model_name == 'SVR':
            engine = SVR()
        elif model_name == 'DecisionTreeRegressor':
            engine = DecisionTreeRegressor()
        elif model_name == 'GradientBoostingRegressor':
            engine = GradientBoostingRegressor()
        elif model_name == 'LightGBM':
            engine = lgb.LGBMRegressor()
        elif model_name == 'CatBoost':
            engine = CatBoostRegressor(verbose=0)

        predicted_data = model_engine(engine, num)
        
        st.header('Predicted Stock Prices')
        st.line_chart(predicted_data.set_index('Date'))

if __name__ == '__main__':
    main()
