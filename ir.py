import time
import pyupbit
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

# API 키 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"

# 업비트 객체 생성
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

# 지표 계산 함수
def get_macd(ticker, short_window=12, long_window=26, signal_window=9):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    if df is None or df.empty:
        return 0, 0  # 데이터가 없으면 0 반환
    short_ema = df['close'].ewm(span=short_window).mean()
    long_ema = df['close'].ewm(span=long_window).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window).mean()
    return macd.iloc[-1], signal.iloc[-1]

def get_rsi(ticker, period=14):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    if df is None or df.empty:
        return 50  # 데이터가 없으면 50 반환 (중립 상태)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def get_bollinger_bands(ticker, window=20, num_std=2):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    if df is None or df.empty:
        return 0, 0, 0  # 데이터가 없으면 0 반환
    sma = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band.iloc[-1], lower_band.iloc[-1], sma.iloc[-1]

def get_stochastic_oscillator(ticker, k_window=14, d_window=3):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    if df is None or df.empty:
        return 0, 0  # 데이터가 없으면 0 반환
    low_min = df['low'].rolling(window=k_window).min()
    high_max = df['high'].rolling(window=k_window).max()
    stochastic_k = ((df['close'] - low_min) / (high_max - low_min)) * 100
    stochastic_d = stochastic_k.rolling(window=d_window).mean()
    return stochastic_k.iloc[-1], stochastic_d.iloc[-1]

def get_price_change(ticker):
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=30)
    if df is None or df.empty:
        return 0  # 가격 변화가 없으면 0으로 처리
    price_change = df['close'].pct_change().mean()
    return price_change

# AI 환경 설정
class CryptoTradingEnv(gym.Env):
    def __init__(self, ticker, initial_balance=100000):
        super(CryptoTradingEnv, self).__init__()
        self.ticker = ticker
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.stock_owned = 0
        self.current_step = 0
        self.done = False

        # dtype을 float32로 명시적으로 설정하여 경고를 방지
        self.action_space = spaces.Box(low=np.array([0, 0], dtype=np.float32), 
                                        high=np.array([1, 1], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)  # 가격 변화 추가

    def reset(self):
        self.balance = self.initial_balance
        self.stock_owned = 0
        self.current_step = 0
        self.done = False
        obs = self._next_observation()
        if obs is None:
            print(f"Reset failed for {self.ticker}, returning zero observation.")
            return np.zeros(self.observation_space.shape)  # 비어있는 경우 기본값 반환
        return obs

    def _next_observation(self):
        try:
            macd, signal = get_macd(self.ticker)
            rsi = get_rsi(self.ticker)
            upper_band, lower_band, sma = get_bollinger_bands(self.ticker)
            stochastic_k, stochastic_d = get_stochastic_oscillator(self.ticker)
            price_change = get_price_change(self.ticker)

            # 유효하지 않은 값은 기본값을 사용
            if any(val is None for val in [macd, signal, rsi, upper_band, lower_band, sma, stochastic_k, stochastic_d, price_change]):
                print(f"Invalid data for {self.ticker}, returning zero observation.")
                return np.zeros(self.observation_space.shape)

            return np.array([macd, signal, rsi, upper_band, lower_band, sma, stochastic_k, stochastic_d, price_change], dtype=np.float32)

        except Exception as e:
            print(f"Error in _next_observation: {e}")
            return np.zeros(self.observation_space.shape)  # 예외 발생 시 기본값 반환

    def step(self, action):
        current_price = pyupbit.get_current_price(self.ticker)
        buy_action = action[0]
        sell_action = action[1]
        reward = 0

        # 매수
        if buy_action > 0.5 and self.balance > current_price:
            self.stock_owned += 1
            self.balance -= current_price
            reward += 5  # 매수 보상

        # 매도
        if sell_action > 0.5 and self.stock_owned > 0:
            self.stock_owned -= 1
            self.balance += current_price
            reward += 10  # 매도 보상

        self.current_step += 1
        self.done = self.current_step >= 100
        return self._next_observation(), reward, self.done, {}

# 모델 훈련
def train_sac(tickers):
    def make_env(ticker):
        def _env():
            return CryptoTradingEnv(ticker)
        return _env

    envs = [make_env(ticker) for ticker in tickers]
    env = SubprocVecEnv(envs)

    model = SAC("MlpPolicy", env, verbose=1, buffer_size=200000, batch_size=256, train_freq=4, learning_starts=10000)
    model.learn(total_timesteps=500000)
    return model

# 코인 데이터
def get_top_10_tickers():
    tickers = pyupbit.get_tickers(fiat="KRW")
    volumes = []
    for ticker in tickers:
        df = pyupbit.get_ohlcv(ticker, interval="minute5", count=30)
        volume = df['volume'].sum() if df is not None and not df.empty else 0
        volumes.append((ticker, volume))
    volumes.sort(key=lambda x: x[1], reverse=True)
    return [ticker for ticker, _ in volumes[:10]]

# 메인
if __name__ == "__main__":
    print("AI 자동매매 시작")
    tickers = get_top_10_tickers()
    model = train_sac(tickers)

    while True:
        for ticker in tickers:
            env = CryptoTradingEnv(ticker)
            observation = env.reset()
            action, _ = model.predict(observation)
            buy_action, sell_action = action

            if buy_action > 0.5:
                print(f"Buying {ticker}...")
                # buy_crypto_currency(ticker, env.balance * 0.1)
            if sell_action > 0.5:
                print(f"Selling {ticker}...")
                # sell_crypto_currency(ticker, env.stock_owned)
        time.sleep(60)
