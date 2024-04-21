import time
import pyupbit
import datetime
import requests
import schedule
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import sys
import winsound as sd

TF_ENABLE_ONEDNN_OPTS=0

access = "iW9QLoF5WRp0po0sRVcWV0ImAGllCar5crRpSvOK"
secret = "zrRS9Ps7UOipuOdt7fQWxMmAhGeJW0xFlsJ27zmx"
myToken = "xoxb-6977539637330-6977568576114-6ICO862atLDqmvtmwBumXCpm"
df = 0
k_flag = 0

def reset_k_flag():
    global k_flag
    k_flag = 0

reset_k_flag()
# 매시간 k_flag를 0으로 리셋
schedule.every().hour.do(reset_k_flag)

def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 100     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

def tellTime():
    print("현재 시간 : ", time.strftime("%Y-%m-%d %H : %M : %S"), "실행 계속 중")

def post_message(token, channel, text):
    """슬랙 메시지 전송"""
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel,"text": text}
    )

def get_ror(k=0.5):
    """K값 갱신함수 1"""
    global k_flag  # 전역 변수 사용 선언
    if k_flag == 0:
        df = pyupbit.get_ohlcv(ticker="KRW-SHIB", count=288, interval="minute60",)
        if df is None:
            print("데이터 불러오기 실패: 데이터가 없습니다.")
            sys.exit()
            return None  # 데이터가 없을 때 None 반환
            
        k_flag = 1
    else:
        df = pyupbit.get_ohlcv(ticker="KRW-SHIB")
        if df is None:
            print("데이터 불러오기 실패: 데이터가 없습니다.")
            sys.exit()
            return None
            

    df['range'] = (df['high'] - df['low']) * k
    df['target'] = df['open'] + df['range'].shift(1)
    df['ror'] = np.where(df['high'] > df['target'],
                        df['close'] / df['target'],
                        1)

    ror = df['ror'].cumprod().iloc[-2]
    return ror

def get_k():
    """K값 갱신함수 2"""
    global k_flag
    k_flag = 0  # 매번 k 값을 계산할 때마다 df를 새로 불러오도록 초기화
    max_ror = 0
    result_k = 0.1
    for k in np.arange(0.1, 1.0, 0.05):
        ror = get_ror(k)
        if ror is None:  # get_ror에서 None 반환 시 루프 중단
            print("ror 계산 실패, k 값 계산을 중단합니다.")
            return None
        if ror > max_ror:
            max_ror = ror
            result_k = k
    return result_k

def get_target_price(ticker, k):
    """변동성 돌파 전략으로 매수 목표가 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
    target_price = df.iloc[0]['close'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * k
    return target_price

def get_start_time(ticker):
    """시작 시간 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
    start_time = df.index[0]
    return start_time

def get_balance(ticker):
    """잔고 조회"""
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            if b['balance'] is not None:
                return float(b['balance'])
            else:
                return 0
    return 0

def get_current_price(ticker):
    """현재가 조회"""
    return pyupbit.get_orderbook(ticker=ticker)["orderbook_units"][0]["ask_price"]

predicted_close_price = 0

def predict_price(ticker):
    """TensorFlow2와 LSTM을 사용하여 다음 지표 가격 예측"""
    global predicted_close_price

    # 데이터 불러오기
    df = pyupbit.get_ohlcv(ticker, interval="minute60", count=240)
    
    # 종가 데이터만 사용
    close_prices = df['close'].values.reshape(-1, 1)
    
    # 데이터 스케일링
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # 데이터셋 생성
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    
    # 시퀀스 길이
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # LSTM 모델 구성
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    # 모델 컴파일
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 모델 학습
    model.fit(X, y, batch_size=1, epochs=2, verbose=1)
    
    # 미래 데이터를 위한 입력
    test_data = scaled_data[-time_step:].reshape(1, time_step, 1)
    
    # 가격 예측
    predicted_price_scaled = model.predict(test_data)
    predicted_close_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
    print("예상 가격은 :", predicted_price_scaled)

# 모델 사용 예시
predict_price("KRW-SHIB")
schedule.every(6).hours.do(lambda: predict_price("KRW-SHIB"))
schedule.every(20).minutes.do(lambda: tellTime())

# 로그인
upbit = pyupbit.Upbit(access, secret)
print("autotrade start")
bestk = get_k()
post_message(myToken, "#general", "Best K 는 " + str(bestk) + " 입니다. 트레이딩을 시작합니다.")
flag = 0

# 자동매매 시작
while True:
    try:
        now = datetime.datetime.now()
        start_time = get_start_time("KRW-SHIB")
        end_time = start_time + datetime.timedelta(days=1)
        if flag == 1:
            formerBestK = bestk
            bestk = (get_k())
            flag = 0
            if bestk is not None:
                post_message(myToken, "#general", "Best K 는 " + str(bestk))
            else:
                post_message(myToken, "#general", "Best K 값을 계산하는 데 실패했습니다.")
            if formerBestK != bestk:
                post_message(myToken, "#general", "BestK가 변동 되었습니다: " + str(bestk) + "\n 현재 KRW 잔고는: " + upbit.get_balance("KRW") + " 원이며 SHIB 잔고는 " + upbit.get_balance("KRW-SHIB") + "입니다.")
        schedule.run_pending()


        if start_time < now < end_time - datetime.timedelta(seconds=10):
            target_price = get_target_price("KRW-SHIB", bestk)
            current_price = get_current_price("KRW-SHIB")
            if target_price < current_price and current_price < predicted_close_price:
                krw = get_balance("KRW")
                if krw > 5000:
                    buy_result = upbit.buy_market_order("KRW-SHIB", krw*0.9995)
                    post_message(myToken,"#general", "SHIB 매수 완료 : " +str(buy_result))
                    beepsound()
                    flag = 1
            else:
                SHIB = get_balance("SHIB")
                if SHIB > 5000 / current_price and current_price >= predicted_close_price:
                    sell_result = upbit.sell_market_order("KRW-SHIB", SHIB*1)
                    post_message(myToken,"#general", "SHIB 매도 완료 : " +str(sell_result))
                    beepsound()
                    flag = 1
        time.sleep(1)
    except Exception as e:
        print(e)
        post_message(myToken,"#general", e)
        time.sleep(1)