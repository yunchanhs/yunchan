import time
import pyupbit
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# API 키 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"

# 설정값
STOP_LOSS_THRESHOLD = -0.03  # -3% 손절
TAKE_PROFIT_THRESHOLD = 0.05  # +5% 익절
COOLDOWN_TIME = timedelta(minutes=5)
ML_THRESHOLD = 0.02  # 머신러닝 모델의 매수 신호 임계값

recent_trades = {}
entry_prices = {}

# 머신러닝 모델 생성
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Transformer 모델 정의
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model, num_heads, num_layers, num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)로 변환
        x = self.transformer(x, x)
        x = x[-1, :, :]  # 마지막 시퀀스의 출력
        return self.fc_out(x)

# 데이터 수집 및 지표 계산 함수
def get_features(ticker):
    """코인의 과거 데이터와 지표를 가져와 머신러닝에 적합한 피처 생성"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=200)

    if df is None or df.empty:
        raise ValueError(f"{ticker} 데이터가 없습니다.")

    # 지표 계산
    df['macd'], df['signal'] = get_macd(ticker)
    df['rsi'] = get_rsi(ticker)
    df['adx'] = get_adx(ticker)
    df['atr'] = get_atr(ticker)
    df['return'] = df['close'].pct_change()

    # 미래 수익률 계산
    df['future_return'] = (df['close'].shift(-1) - df['close']) / df['close']

    # NaN 제거 전 디버깅
    print(df[['macd', 'signal', 'rsi', 'adx', 'atr', 'return', 'future_return']].head())

    # NaN 제거
    df = df.dropna()  # NaN 제거

    return df

def get_macd(ticker):
    """MACD 계산 함수 (매매에 사용할 지표)"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=200)
    short_window = 12
    long_window = 26
    signal_window = 9
    df['short_ema'] = df['close'].ewm(span=short_window).mean()
    df['long_ema'] = df['close'].ewm(span=long_window).mean()
    df['macd'] = df['short_ema'] - df['long_ema']
    df['signal'] = df['macd'].ewm(span=signal_window).mean()
    return df['macd'].iloc[-1], df['signal'].iloc[-1]

def get_rsi(ticker):
    """RSI 계산 함수 (매매에 사용할 지표)"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=200)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def get_adx(ticker):
    """ADX 계산 함수 (매매에 사용할 지표)"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=200)
    # ADX 계산 로직을 여기에 구현
    # (여기서는 기본적으로 14일로 가정하여 계산)
    return 30  # 예시로 고정값 반환

def get_atr(ticker):
    """ATR 계산 함수 (매매에 사용할 지표)"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=200)
    df['tr'] = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
    df['atr'] = df['tr'].rolling(window=14).mean()
    return df['atr'].iloc[-1]

# Transformer 모델 학습용 데이터셋 클래스
class CryptoDataset(Dataset):
    def __init__(self, ticker, seq_len=50):
        self.data = get_features(ticker)
        self.seq_len = seq_len
        self.features = self.data[['macd', 'signal', 'rsi', 'adx', 'atr', 'return']].values
        self.labels = self.data['future_return'].values

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 모델 훈련 함수
def train_transformer_model(ticker):
    """Transformer 모델 학습"""
    dataset = CryptoDataset(ticker)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    input_dim = dataset.features.shape[1]
    d_model = 64
    num_heads = 4
    num_layers = 2
    output_dim = 1

    model = TransformerModel(input_dim, d_model, num_heads, num_layers, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(10):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch.view(-1, 1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model

# 모델 예측 함수
def predict_with_transformer(ticker, model):
    """Transformer 모델을 사용해 예측"""
    dataset = CryptoDataset(ticker)
    features = dataset.features[-1].reshape(1, -1)
    features = torch.tensor(features, dtype=torch.float32)
    prediction = model(features)
    return prediction.item()

# 매매 함수들 (기존 로직 유지)

# 메인 로직
if __name__ == "__main__":
    upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
    print("자동매매 시작!")

    # 머신러닝 모델 초기 학습
    tickers = pyupbit.get_tickers(fiat="KRW")
    models = {}

    # 각 코인에 대해 Transformer 모델 훈련
    for ticker in tickers:
        try:
            models[ticker] = train_transformer_model(ticker)
        except Exception as e:
            print(f"[{ticker}] 모델 훈련 중 에러 발생: {e}")

    try:
        while True:
            tickers = pyupbit.get_tickers(fiat="KRW")
            krw_balance = get_balance("KRW")

            for ticker in tickers:
                try:
                    now = datetime.now()

                    # 쿨다운 타임 체크
                    if ticker in recent_trades and now - recent_trades[ticker] < COOLDOWN_TIME:
                        continue

                    # Transformer 모델 예측
                    model = models.get(ticker)
                    if model:
                        ml_signal = predict_with_transformer(ticker, model)

                        # 기존 지표 계산
                        macd, signal = get_macd(ticker)
                        rsi = get_rsi(ticker)
                        adx = get_adx(ticker)
                        current_price = pyupbit.get_current_price(ticker)

                        # 매수 조건
                        if ml_signal > ML_THRESHOLD and macd > signal and rsi < 30 and adx > 25 and krw_balance > 5000:
                            buy_amount = krw_balance * 0.1
                            buy_result = buy_crypto_currency(ticker, buy_amount)
                            entry_prices[ticker] = current_price
                            recent_trades[ticker] = now

                    time.sleep(1)

                except Exception as e:
                    print(f"[{ticker}] 매매 중 에러 발생: {e}")

    except KeyboardInterrupt:
        print("자동매매 종료!")
