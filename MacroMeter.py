import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import yfinance as yf
import requests


# 예시: VIX 데이터 수집
vix_data = yf.download('^VIX', start='2020-01-01', end='2025-01-01')
vix_data = vix_data['Close'].rename('VIX')

# 예시: S&P 500 데이터 수집
sp500_data = yf.download('^GSPC', start='2020-01-01', end='2025-01-01')
sp500_data = sp500_data['Close'].rename('SP500')

# 데이터 병합
data = pd.concat([vix_data, sp500_data], axis=1).dropna()

# 수익률 계산
data['SP500_Return'] = data['SP500'].pct_change().shift(-1)
data['Target'] = np.where(data['SP500_Return'] > 0, 1, 0)

# 결측치 제거
data = data.dropna()

# 특징 변수 및 타겟 변수 설정
X = data[['VIX']]
y = data['Target']

# 학습용 및 테스트용 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 로지스틱 회귀 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
