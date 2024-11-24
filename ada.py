import time
import pyupbit
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium import Env  # gymnasium import

# API 키 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"

# 업비트 객체 생성
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

# 지표 계산 함수들
def get_macd(ticker, short_window=12, long_window=26, signal_window=9):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    short_ema = df['close'].ewm(span=short_window).mean()
    long_ema = df['close'].ewm(span=long_window).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window).mean()
    return macd.iloc[-1], signal.iloc[-1]

def get_rsi(ticker, period=14):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def get_bollinger_bands(ticker, window=20, num_std=2):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    sma = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band.iloc[-1], lower_band.iloc[-1], sma.iloc[-1]

def get_stochastic_oscillator(ticker, k_window=14, d_window=3):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    if df is None or df.empty:  # 데이터가 없을 경우 처리
        return 0.0, 0.0  # 기본값 반환
    low_min = df['low'].rolling(window=k_window).min()
    high_max = df['high'].rolling(window=k_window).max()
    stochastic_k = ((df['close'] - low_min) / (high_max - low_min)) * 100
    stochastic_d = stochastic_k.rolling(window=d_window).mean()
    return stochastic_k.iloc[-1], stochastic_d.iloc[-1]

def get_price_change(ticker):
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=30)
    price_change = df['close'].pct_change().mean()
    return price_change

# AI 환경 설정
class CryptoTradingEnv(gym.Env):  # gym.Env로 수정
    def __init__(self, ticker, initial_balance=100000):
        super(CryptoTradingEnv, self).__init__()
        self.ticker = ticker
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.stock_owned = 0
        self.current_step = 0
        self.done = False
        self.truncated = False

        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float64)  # float64로 수정

        # np_random 추가
        self.np_random = np.random.RandomState()

    def reset(self, seed=None, **kwargs):
        # 시드를 설정 (Stable-Baselines3는 seed를 전달함)
        if seed is not None:
            self.np_random.seed(seed)
        self.balance = self.initial_balance
        self.stock_owned = 0
        self.current_step = 0
        self.done = False
        self.truncated = False
        return self._next_observation(), {}

    def _next_observation(self):
        macd, signal = get_macd(self.ticker)
        rsi = get_rsi(self.ticker)
        upper_band, lower_band, sma = get_bollinger_bands(self.ticker)
        stochastic_k, stochastic_d = get_stochastic_oscillator(self.ticker)
        price_change = get_price_change(self.ticker)
        return np.array([macd, signal, rsi, upper_band, lower_band, sma, stochastic_k, stochastic_d, price_change], dtype=np.float64)

    def step(self, action):
        current_price = pyupbit.get_current_price(self.ticker)
        if action[0] > 0:  # Buy signal
            if self.balance >= current_price:
                self.balance -= current_price
                self.stock_owned += 1
        elif action[1] > 0:  # Sell signal
            if self.stock_owned > 0:
                self.balance += current_price
                self.stock_owned -= 1
        
        # 상태, 보상 계산
        observation = self._next_observation()
        reward = self.balance - self.initial_balance
        self.done = self.current_step >= 200  # 임의의 종료 조건
        self.truncated = False
        return observation, reward, self.done, self.truncated, {}

# 환경 설정
tickers = ["KRW-BTC"]
env = CryptoTradingEnv(tickers[0])

# RL 에이전트 학습
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)
