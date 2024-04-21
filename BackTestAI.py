import pyupbit
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import sys

# 파라미터 설정
access = "iW9QLoF5WRp0po0sRVcWV0ImAGllCar5crRpSvOK"
secret = "zrRS9Ps7UOipuOdt7fQWxMmAhGeJW0xFlsJ27zmx"
upbit = pyupbit.Upbit(access, secret)
initial_balance = 2500000  # 초기 자본금: 250,000 KRW

# 데이터 불러오기
def load_data(ticker, interval, count):
    return pyupbit.get_ohlcv(ticker, interval=interval, count=count)

# LSTM 모델 생성 함수
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 백테스트 실행
def backtest(ticker, interval, count, initial_balance):
    data = load_data(ticker, interval, count)
    close_prices = data['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    model = create_lstm_model((time_step, 1))
    model.fit(X, y, batch_size=1, epochs=1, verbose=1)
    
    test_data = scaled_data[-time_step:].reshape(1, time_step, 1)
    predicted_price_scaled = model.predict(test_data)
    predicted_close_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

    balance = initial_balance
    coins = 0
    transaction_fee = 0.0005
    
    for index, row in data.iterrows():
        current_price = row['close']
        if coins == 0:  # 매수 조건
            if current_price < predicted_close_price:
                coins = balance / current_price * (1 - transaction_fee)
                balance = 0
                print(f"Bought at {current_price}, coins: {coins}")
        elif coins > 0:  # 매도 조건
            if current_price > predicted_close_price:
                balance = coins * current_price * (1 - transaction_fee)
                coins = 0
                print(f"Sold at {current_price}, balance: {balance}")

    final_value = balance + (coins * data.iloc[-1]['close'])
    print(f"Final balance: {final_value}")

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# 백테스트 설정
ticker = "KRW-SHIB"
interval = "minute1"
count = 10080  # 데이터 개수

backtest(ticker, interval, count, initial_balance)
