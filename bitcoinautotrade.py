import time
import pyupbit
import datetime
import requests
import numpy as np

access = "iW9QLoF5WRp0po0sRVcWV0ImAGllCar5crRpSvOK"
secret = "zrRS9Ps7UOipuOdt7fQWxMmAhGeJW0xFlsJ27zmx"
myToken = "xoxb-6977539637330-6977568576114-6ICO862atLDqmvtmwBumXCpm"

def post_message(token, channel, text):
    """슬랙 메시지 전송"""
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel,"text": text}
    )

def get_ror(k=0.5):
    """K값 갱신함수 1"""
    df = pyupbit.get_ohlcv("KRW-BTC")
    df['range'] = (df['high'] - df['low']) * k
    df['target'] = df['open'] + df['range'].shift(1)

    df['ror'] = np.where(df['high'] > df['target'],
                        df['close'] / df['target'],
                        1)

    ror = df['ror'].cumprod()[-2]
    return ror

def get_k():
    """K값 갱신함수 2"""
    for k in np.arange(0.1, 1.0, 0.05):
        ror1 = get_ror(k)
        ror2 = get_ror(k-0.05)
        resultror = get_ror(0.05)
        if ror1 > ror2:
            resultror = ror1
    return resultror

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

def get_ma15(ticker):
    """15일 이동 평균선 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="day", count=15)
    ma15 = df['close'].rolling(15).mean().iloc[-1]
    return ma15

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

# 로그인
upbit = pyupbit.Upbit(access, secret)
print("autotrade start")
# 시작 메세지 슬랙 전송
post_message(myToken,"#general", "autotrade start")
bestk = get_k()
post_message(myToken, "#general", "Today's Best K is " + bestk)

while True:
    try:
        now = datetime.datetime.now()
        start_time = get_start_time("KRW-BTC")
        end_time = start_time + datetime.timedelta(days=1)
        if flag == 1:
            formerBestK = bestk
            bestk = (get_k)
            flag = 0
            if formerBestK != bestk:
                post_message(myToken, "#general", "BestK has been changed into " + bestk + "\n Your Money is Now " + upbit.get_balance("KRW") + " Won and " + upbit.get_balance("KRW-BTC") + "as BTC")

        if start_time < now < end_time - datetime.timedelta(seconds=10):
            target_price = get_target_price("KRW-BTC", bestk)
            ma15 = get_ma15("KRW-BTC")
            current_price = get_current_price("KRW-BTC")
            if target_price < current_price and ma15 < current_price:
                krw = get_balance("KRW")
                if krw > 5000:
                    buy_result = upbit.buy_market_order("KRW-BTC", krw*0.9995)
                    post_message(myToken,"#crypto", "BTC buy : " +str(buy_result))
                    flag = 1
        else:
            btc = get_balance("BTC")
            if btc > 0.00008:
                sell_result = upbit.sell_market_order("KRW-BTC", btc*0.9995)
                post_message(myToken,"#crypto", "BTC buy : " +str(sell_result))
        time.sleep(1)
    except Exception as e:
        print(e)
        post_message(myToken,"#crypto", e)
        time.sleep(1)