import pandas as pd
import numpy as np
import yfinance as yf

from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. 데이터 다운로드 & 컬럼명 지정
close = yf.download(
    ['^VIX','^GSPC','^IRX','^TNX'],
    start='2015-01-01', end='2024-12-31'
)['Close']
close.columns = ['VIX','SP500','US2Y','US10Y']

# 2. 기본 DataFrame & 파생변수
data = close.dropna().copy()
data['SP500_Return']    = data['SP500'].pct_change().shift(-1)
data['Target']          = (data['SP500_Return']>0).astype(int)
data['Yield_Spread']    = data['US10Y'] - data['US2Y']
data['Fed_Expectation'] = data['Yield_Spread']/10
data['Surprise_Score']  = np.random.normal(0,0.1,len(data))  # 나중에 실제 서프라이즈로 교체

# 3. 기술적 지표
data['MA20']       = SMAIndicator(data['SP500'], window=20).sma_indicator()
data['RSI14']      = RSIIndicator(close=data['SP500'], window=14).rsi()
bb               = BollingerBands(close=data['SP500'], window=20, window_dev=2)
data['BBW']        = bb.bollinger_wband()
macd             = MACD(close=data['SP500'], window_slow=26, window_fast=12, window_sign=9)
data['MACD_diff'] = macd.macd_diff()

# 4. 최종 데이터 준비
data.dropna(inplace=True)
features = ['VIX','Yield_Spread','Fed_Expectation','Surprise_Score',
            'MA20','RSI14','BBW','MACD_diff']
X = data[features]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2
)

# 5. 두 개의 로지스틱 회귀 모델 정의
model_base     = LogisticRegression(max_iter=500)                         # 베이스
model_weighted = LogisticRegression(max_iter=500, class_weight='balanced', C=0.1)

# 6. 각 모델 학습
model_base.fit(X_train, y_train)
model_weighted.fit(X_train, y_train)

# 7. 확률 예측(soft vote)
proba_base     = model_base.predict_proba(X_test)
proba_weighted = model_weighted.predict_proba(X_test)
proba_ensemble = (proba_base + proba_weighted) / 2

# 8. 최종 예측 (threshold 0.5)
y_pred_ensemble = (proba_ensemble[:,1] > 0.5).astype(int)

# 9. 결과 평가
results = X_test.copy()
results['Prob_Ensemble_Down'] = proba_ensemble[:,0]
results['Prob_Ensemble_Up']   = proba_ensemble[:,1]
results['Pred_Ensemble']      = y_pred_ensemble
results['Actual']             = y_test.values

print(results.tail(10))
print()
print(classification_report(y_test, y_pred_ensemble))
