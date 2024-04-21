import pyupbit
import winsound as sd
import os

access = os.getenv("UPBIT_ACCESS_KEY")
secret = os.getenv("UPBIT_SECRET_KEY")
myToken = os.getenv("SLACK_TOKEN")
upbit = pyupbit.Upbit(access, secret)

sd.Beep(2000, 1000)

print(upbit.get_balance("KRW-XRP"))     # KRW-XRP 조회
print(upbit.get_balance("KRW"))         # 보유 현금 조회

print(pyupbit.get_ohlcv("KRW-BTC"))

def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 1000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

beepsound()