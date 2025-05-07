# -*- coding: utf-8 -*-
"""
FOMC 기대 심리 수치화 및 시장 방향성 확률 예측 모델
필요 패키지:
  pip install yfinance pandas numpy scikit-learn ta fredapi pyfedwatch newsapi-python tweepy vaderSentiment
"""
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf

from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# FedWatch 및 FOMC 일정 관리
from pyfedwatch.fedwatch import FedWatch
from pyfedwatch.fomc import FOMC
# FRED API
from fredapi import Fred
# News API
from newsapi import NewsApiClient
# Twitter sentiment
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- API Key 설정 (환경변수 또는 직접 입력) ---
FRED_API_KEY     = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY")
NEWS_API_KEY     = os.getenv("NEWS_API_KEY", "YOUR_NEWSAPI_KEY")
TWITTER_API_KEY        = os.getenv("TWITTER_API_KEY", "YOUR_TWITTER_API_KEY")
TWITTER_API_SECRET     = os.getenv("TWITTER_API_SECRET", "YOUR_TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN   = os.getenv("TWITTER_ACCESS_TOKEN", "YOUR_TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET  = os.getenv("TWITTER_ACCESS_SECRET", "YOUR_TWITTER_ACCESS_SECRET")

# FRED API 키 검증
if FRED_API_KEY == "YOUR_FRED_API_KEY":
    sys.exit("ERROR: Please set the FRED_API_KEY environment variable. (Get a key at https://fred.stlouisfed.org/)")

# FOMC 회의 일정 (예시)
fomc_dates = [
    '2025-06-15',
    '2025-07-27',
    # ... 추가 회의일
]

# 사용자 정의 Fed Funds Futures 가격 로드 함수
def fetch_ff_futures_prices():
    # CSV 파일에 'date','price' 컬럼이 있다고 가정
    path = './data/contracts/FEDFUTURES.csv'
    df = pd.read_csv(path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df['price']

# 1. 금융 데이터 다운로드 (Yahoo Finance)
symbols = ['^VIX','^GSPC','^IRX','^TNX']  # 'PCR' 제거
close = yf.download(symbols, start='2015-01-01', end='2024-12-31')['Close']
close.columns = ['VIX','SP500','US2Y','US10Y']

# 2. FRED 데이터 (실효연준금리, BEI, MOVE)
fred = Fred(api_key=FRED_API_KEY)
dff = fred.get_series('DFF', start='2015-01-01', end='2024-12-31')   # 일별 실효연준금리
bei = fred.get_series('T10YIE', start='2015-01-01', end='2024-12-31') # 10년 손익분기 인플레이션
move = fred.get_series('MOVE', start='2015-01-01', end='2024-12-31')  # 채권 변동성 지수
fred_df = pd.concat([
    dff.rename('DFF'),
    bei.rename('BEI'),
    move.rename('MOVE')
], axis=1)

# 3. FOMC 객체 및 FedWatch 확률 계산
fomc_cal = FOMC(
    watch_date=pd.Timestamp.today(),
    fomc_dates=fomc_dates,
    num_upcoming=len(fomc_dates)
)
fw = FedWatch(
    watch_date=pd.Timestamp.today(),
    fomc_dates=fomc_dates,
    num_upcoming=len(fomc_dates),
    user_func=fetch_ff_futures_prices,
    path='./data/contracts'
)
df_fw = fw.generate_hike_info(rate_cols=True)
fw_df = df_fw.set_index('meeting_date')
# 기대 금리 계산
prev_rate = dff.shift(1).reindex(fw_df.index).fillna(method='ffill')
fw_df['expected_rate'] = (
    fw_df['no_change_prob'] * prev_rate +
    fw_df['up_25_prob']     * (prev_rate + 0.25) +
    fw_df['down_25_prob']   * (prev_rate - 0.25)
)
# 실제 금리
fw_df['actual_rate'] = dff.reindex(fw_df.index).fillna(method='ffill')
# 이벤트 서프라이즈 스코어
fw_df['Surprise_Score_Event'] = fw_df['actual_rate'] - fw_df['expected_rate']

# 4. 데이터 병합 및 전처리
data = pd.concat([close, fred_df], axis=1).dropna()
data['Surprise_Score'] = 0.0
for date, row in fw_df.iterrows():
    if date in data.index:
        data.at[date, 'Surprise_Score'] = row['Surprise_Score_Event']

# 5. 목표 변수 생성
# 다음날 S&P 500 등락
data['SP500_Return'] = data['SP500'].pct_change().shift(-1)
data['Target'] = (data['SP500_Return'] > 0).astype(int)

# 6. Fed 기대 지표 (실제 FedWatch 기반)
data['Fed_Expectation'] = (
    fw_df['expected_rate'].reindex(data.index).fillna(method='ffill') -
    dff.shift(1).reindex(data.index).fillna(method='ffill')
)
# 수익률곡선 스프레드
data['Yield_Spread'] = data['US10Y'] - data['US2Y']

# 7. 기술적 지표 추가
data['MA20']       = SMAIndicator(data['SP500'], window=20).sma_indicator()
data['RSI14']      = RSIIndicator(close=data['SP500'], window=14).rsi()
bb               = BollingerBands(close=data['SP500'], window=20, window_dev=2)
data['BBW']        = bb.bollinger_wband()
macd             = MACD(close=data['SP500'], window_slow=26, window_fast=12, window_sign=9)
data['MACD_diff'] = macd.macd_diff()

# 8. 뉴스 감성 지표 (NewsAPI + VADER)
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
sid     = SentimentIntensityAnalyzer()
news_sent = []
for date in data.index:
    resp = newsapi.get_everything(
        q='FOMC OR \"Federal Reserve\"',
        from_param=date.strftime('%Y-%m-%d'),
        to=date.strftime('%Y-%m-%d'),
        language='en', page_size=5
    )
    scores = [sid.polarity_scores(a['title'] + ' ' + (a.get('description') or ''))['compound']
              for a in resp.get('articles', [])]
    news_sent.append(np.mean(scores) if scores else 0)
data['News_Sentiment'] = news_sent

# 9. 트위터 감성 지표 (Tweepy + VADER)
auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
api  = tweepy.API(auth)
sid  = SentimentIntensityAnalyzer()
tw_sent = []
for date in data.index:
    try:
        tweets = api.search(q='FOMC', lang='en', count=10,
                            since=date.strftime('%Y-%m-%d'),
                            until=(date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        scores = [sid.polarity_scores(t.text)['compound'] for t in tweets]
        tw_sent.append(np.mean(scores) if scores else 0)
    except:
        tw_sent.append(0)

data['Twitter_Sentiment'] = tw_sent

# 10. 모델 학습용 특징 선정
features = [
    'VIX', 'Yield_Spread', 'Fed_Expectation', 'Surprise_Score',
    'BEI', 'MOVE',
    'MA20', 'RSI14', 'BBW', 'MACD_diff',
    'News_Sentiment', 'Twitter_Sentiment'
]
# 최종 데이터 준비
df = data.dropna()
X, y = df[features], df['Target']

# 11. 학습/테스트 분리 및 모델 학습
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2
)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# 12. 예측 & 평가
proba     = model.predict_proba(X_test)
y_pred    = (proba[:,1] > 0.5).astype(int)
results   = X_test.copy()
results['Prob_Down'] = proba[:,0]
results['Prob_Up']   = proba[:,1]
results['Pred']      = y_pred
results['Actual']    = y_test.values

print(results.tail(10))
print(classification_report(y_test, y_pred))
