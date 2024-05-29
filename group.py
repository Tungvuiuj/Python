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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error

st.title('Predicting Future Stock Value')
st.sidebar.info('Welcome to the Predicting Future Stock Value Application')
st.sidebar.info("This website was created and designed by [IUJ Group]")
st.sidebar.info('Please fill the cells below:')

def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

@st.cache
def download_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

option = st.sidebar.text_input('Enter a Stock Symbol', value='AAPL')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End Date', value=today)

if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
        data = download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')

data = download_data(option, start_date, end_date)
scaler = StandardScaler()

def tech_indicators():
    st.header('Technical Indicators')
    indicators = st.multiselect('Select Indicators to Plot', ['Close', 'Bollinger Bands', 'MACD', 'RSI', 'SMA', 'EMA', 'Volume'])

    if 'Close' in indicators:
        plt.plot(data['Close'], label='Close')
    if 'Bollinger Bands' in indicators:
        bb_indicator = BollingerBands(data['Close'])
        data['bb_h'] = bb_indicator.bollinger_hband()
        data['bb_l'] = bb_indicator.bollinger_lband()
        plt.plot(data.index, data['bb_h'], label='Bollinger Bands High')
        plt.plot(data.index, data['bb_l'], label='Bollinger Bands Low')
    if 'MACD' in indicators:
        macd_indicator = MACD(data['Close'])
        data['macd'] = macd_indicator.macd()
        plt.plot(data.index, data['macd'], label='MACD')
    if 'RSI' in indicators:
        rsi_indicator = RSIIndicator(data['Close'])
        data['rsi'] = rsi_indicator.rsi()
        plt.plot(data.index, data['rsi'], label='RSI')
    if 'SMA' in indicators:
        sma_indicator = SMAIndicator(data['Close'])
        data['sma'] = sma_indicator.sma_indicator()
        plt.plot(data.index, data['sma'], label='SMA')
    if 'EMA' in indicators:
        ema_indicator = EMAIndicator(data['Close'])
        data['ema'] = ema_indicator.ema_indicator()
        plt.plot(data.index, data['ema'], label='EMA')
    if 'Volume' in indicators:
        plt.plot(data['Volume'], label='Volume')

    plt.legend()
    st.pyplot()

def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(20))

def model_engine(model, num):
    df = data[['Close']]
    df['preds'] = data['Close'].shift(-num)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df['preds'].values
    y = y[:-num]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=5)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \nMAE: {mean_absolute_error(y_test, preds)}')

    forecast_pred = model.predict(x_forecast)
    predictions = forecast_pred.tolist()
    forecast_dates = pd.date_range(end=end_date, periods=num + 1)[1:]
    predicted_data = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': predictions})

    return predicted_data

def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor', 'SVM', 'Neural Network', 'Gradient Boosting', 'LightGBM', 'CatBoost', 'Naive Bayes', 'Decision Tree','Logistic Regression', 'K-Means Clustering', 'Nearest Neighbors'])
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
        elif model == 'SVM':
            engine = SVR()
            predicted_data = model_engine(engine, num)
        elif model == 'Neural Network':
            from sklearn.neural_network import MLPRegressor
            engine = MLPRegressor(max_iter=500)
            predicted_data = model_engine(engine, num)
        elif model == 'Gradient Boosting':
            engine = GradientBoostingRegressor()
            predicted_data = model_engine(engine, num)
        elif model == 'LightGBM':
            engine = LGBMRegressor()
            predicted_data = model_engine(engine, num)
        elif model == 'CatBoost':
            engine = CatBoostRegressor(silent=True)
            predicted_data = model_engine(engine, num)
        elif model == 'Naive Bayes':
            engine = GaussianNB()
            predicted_data = model_engine(engine, num)
        elif model == 'Decision Tree':
            engine = DecisionTreeRegressor()
            predicted_data = model_engine(engine, num)
        elif model == 'Logistic Regression':
            engine = LogisticRegression()
            predicted_data = model_engine(engine, num)
        elif model == 'K-Means Clustering':
            engine = KMeans(n_clusters=num)
            x = data[['Close']].values
            x = scaler.fit_transform(x)
            kmeans_pred = engine.fit_predict(x)
            st.text(f'Cluster Centers: {engine.cluster_centers_}')
            data['Cluster'] = kmeans_pred
            st.dataframe(data)
            return
        elif model == 'Nearest Neighbors':
            engine = KNeighborsClassifier()
            x = data[['Close']].values
            x = scaler.fit_transform(x)
            x_train, x_test, y_train, y_test = train_test_split(x, kmeans_pred, test_size=0.2, random_state=5)
            engine.fit(x_train, y_train)
            preds = engine.predict(x_test)
            st.text(f'Accuracy: {engine.score(x_test, y_test)}')
            return
        
        st.header('Predicted Stock Prices')
        st.line_chart(predicted_data.set_index('Date'))

if __name__ == '__main__':
    main()
