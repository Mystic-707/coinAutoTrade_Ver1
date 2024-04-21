import pyupbit
import winsound as sd

access = "iW9QLoF5WRp0po0sRVcWV0ImAGllCar5crRpSvOK"          # 본인 값으로 변경
secret = "zrRS9Ps7UOipuOdt7fQWxMmAhGeJW0xFlsJ27zmx"          # 본인 값으로 변경
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