import time
import pyupbit
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader

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
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        # batch_first=True로 설정하여 성능을 개선
        self.transformer = nn.Transformer(d_model, num_heads, num_layers, num_layers, batch_first=True)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        # batch_first=True이므로 (batch, seq_len, d_model) 형식으로 변환
        x = self.transformer(x, x)
        x = x[:, -1, :]  # 마지막 시퀀스의 출력
        return self.fc_out(x)

# 데이터 수집 및 지표 계산 함수
def get_features(ticker):
    """코인의 과거 데이터와 지표를 가져와 머신러닝에 적합한 피처 생성"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=200)
    df['macd'], df['signal'] = get_macd(ticker)
    df['rsi'] = get_rsi(ticker)
    df['adx'] = get_adx(ticker)
    df['atr'] = get_atr(ticker)
    df['return'] = df['close'].pct_change()
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1  # 다음 5분봉 수익률

    # NaN 값 제거
    df.dropna(inplace=True)
    return df

# 모델 학습을 위한 데이터셋 준비
class TradingDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx+self.seq_len][['macd', 'signal', 'rsi', 'adx', 'atr', 'return']].values
        y = self.data.iloc[idx + self.seq_len]['future_return']
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_transformer_model(ticker):
    """트랜스포머 모델 학습"""
    data = get_features(ticker)
    seq_len = 30  # 시퀀스 길이
    dataset = TradingDataset(data, seq_len)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    input_dim = 6  # 특성 수
    d_model = 64
    num_heads = 8
    num_layers = 2
    output_dim = 1

    model = TransformerModel(input_dim, d_model, num_heads, num_layers, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):  # 학습 epoch 수
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

    return model

def get_ml_signal(ticker, model):
    """머신러닝 모델을 사용해 매수 신호 예측"""
    try:
        features = get_features(ticker)
        latest_data = features[['macd', 'signal', 'rsi', 'adx', 'atr', 'return']].tail(30)  # 최근 30개 데이터
        X_latest = torch.tensor(latest_data.values, dtype=torch.float32).unsqueeze(0)  # (batch, seq_len, features)
        model.eval()
        with torch.no_grad():
            prediction = model(X_latest).item()  # 예측 결과
        return prediction
    except Exception as e:
        print(f"[{ticker}] 머신러닝 신호 계산 중 에러 발생: {e}")
        return 0

# 매매 함수들 (기존 로직 유지)

# 메인 로직
if __name__ == "__main__":
    upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
    print("자동매매 시작!")

    tickers = pyupbit.get_tickers(fiat="KRW")
    models = {}

    # 각 코인에 대해 트랜스포머 모델 학습
    for ticker in tickers:
        print(f"학습 시작: {ticker}")
        model = train_transformer_model(ticker)
        models[ticker] = model

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

                    # 머신러닝 신호 계산
                    ml_signal = get_ml_signal(ticker, models[ticker])

                    # 기존 지표 계산
                    macd, signal = get_macd(ticker)
                    rsi = get_rsi(ticker)
                    adx = get_adx(ticker)
                    current_price = pyupbit.get_current_price(ticker)

                    # 매수 조건
                    if ml_signal > ML_THRESHOLD and macd > signal and rsi < 30 and adx > 25 and krw_balance > 5000:
                        buy_amount = krw_balance * 0.1
                        buy_result = buy_crypto_currency(ticker, buy_amount)
                        if buy_result:
                            entry_prices[ticker] = current_price
                            recent_trades[ticker] = now
                            print(f"[{ticker}] 매수 완료. 금액: {buy_amount:.2f}, 가격: {current_price:.2f}")

                    # 매도 조건 (손절/익절)
                    elif ticker in entry_prices:
                        entry_price = entry_prices[ticker]
                        change_ratio = (current_price - entry_price) / entry_price

                        if change_ratio <= STOP_LOSS_THRESHOLD or change_ratio >= TAKE_PROFIT_THRESHOLD:
                            coin_balance = get_balance(ticker.split('-')[1])
                            sell_result = sell_crypto_currency(ticker, coin_balance)
                            if sell_result:
                                recent_trades[ticker] = now
                                print(f"[{ticker}] 매도 완료. 잔고: {coin_balance:.4f}, 가격: {current_price:.2f}")

                except Exception as e:
                    print(f"[{ticker}] 처리 중 에러 발생: {e}")

            # 1분 대기
            time.sleep(60)

    except Exception as e:
        print(f"시스템 에러: {e}")
