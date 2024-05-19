import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# データのダウンロード
ticker = 'AAPL'  # Appleの株価データを例に使用
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
data = data[['Close']]  # 終値のみを使用

# 異常値の検出と処理（例: 3標準偏差を超える値を異常値とする）
mean = data['Close'].mean()
std = data['Close'].std()
threshold = 3 * std
outliers = (data['Close'] - mean).abs() > threshold
data['Close'][outliers] = np.nan  # 異常値をNaNに置き換え
data['Close'] = data['Close'].interpolate(method='linear')  # 欠損値を線形補間で補完

# データの前処理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 訓練データとテストデータの分割
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# データセットの作成
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

# LSTMの入力形式にデータを変換
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# LSTMモデルの構築
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# モデルのコンパイル
model.compile(optimizer='adam', loss='mean_squared_error')

# モデルの訓練
model.fit(X_train, Y_train, batch_size=1, epochs=1)

# 予測の作成
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 元のスケールに戻す
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# グラフのプロット
plt.figure(figsize=(14, 8))
plt.plot(data.index, scaler.inverse_transform(scaled_data), label='Actual Stock Price')

# 予測データのインデックスを設定
train_index = data.index[time_step:len(train_predict) + time_step]
test_index = data.index[len(train_predict) + (time_step * 2) + 1:len(data) - 1]

plt.plot(train_index, train_predict, label='Train Predict')
plt.plot(test_index, test_predict, label='Test Predict')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
