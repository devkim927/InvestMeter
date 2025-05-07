# -*- coding: utf-8 -*-
"""
FOMC 기대 심리 수치화 및 시장 방향성 확률 예측 모델
(FRED API 의존 데이터 제외)
필요 패키지:
  pip install yfinance pandas numpy scikit-learn ta pyfedwatch newsapi-python tweepy vaderSentiment
"""
import os
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

# News API
from newsapi import NewsApiClient
# Twitter sentiment
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# News/Twitter API Key 설정 (환경변수 또는 직접 입력)
NEWS_API_KEY     = os.getenv("NEWS_API_KEY", "YOUR_NEWSAPI_KEY")
TWITTER_API_KEY        = os.getenv("TWITTER_API_KEY", "YOUR_TWITTER_API_KEY")
TWITTER_API_SECRET     = os.getenv("TWITTER_API_SECRET", "YOUR_TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN   = os.getenv("TWITTER_ACCESS_TOKEN", "YOUR_TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET  = os.getenv("TWITTER_ACCESS_SECRET", "YOUR_TWITTER_ACCESS_SECRET")

# FOMC 회의 일정 (예시 리스트)
fomc_dates = [
    '2025-06-15',
    '2025-07-27',
    # 추가 회의일
]

# 사용자 정의 Fed Funds Futures 가격 로드 함수
def fetch_ff_futures_prices():
    # CSV 파일에 'date','price' 컬럼이 있다고 가정
    path = './data/contracts/FEDFUTURES.csv'
    df = pd.read_csv(path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    return df['price']

# 1. 금융 데이터 다운로드 (Yahoo Finance)
symbols = ['^VIX','^GSPC','^IRX','^TNX']
close = yf.download(symbols, start='2015-01-01', end='2024-12-31')['Close']
close.columns = ['VIX','SP500','US2Y','US10Y']

# 2. FedWatch 확률 계산 (FOMC 일정 기반)
fw = FedWatch(
    watch_date=pd.Timestamp.today(),
    fomc_dates=fomc_dates,
    num_upcoming=len(fomc_dates),
    user_func=fetch_ff_futures_prices,
    path='./data/contracts'
)(
    watch_date=pd.Timestamp.today(),
    fomc_dates=fomc_dates,
    num_upcoming=len(fomc_dates),
    user_func=fetch_ff_futures_prices,
    path='./data/contracts'
)
df_fw = fw.generate_hike_info(rate_cols=True)
fw_df = df_fw.set_index('meeting_date')[['no_change_prob','up_25_prob','down_25_prob']]

# 3. 데이터 병합 및 전처리
data = close.join(fw_df, how='left').sort_index()
# 확률 전일치 채우기
for col in ['no_change_prob','up_25_prob','down_25_prob']:
    data[col].fillna(method='ffill', inplace=True)
# 결측 제거
data.dropna(inplace=True)

# 4. 목표 변수 생성
# 다음날 S&P500 등락
data['SP500_Return'] = data['SP500'].pct_change().shift(-1)
data['Target'] = (data['SP500_Return'] > 0).astype(int)

# 5. 파생 변수: 수익률 곡선 스프레드
data['Yield_Spread'] = data['US10Y'] - data['US2Y']

# 6. 기술적 지표 추가
data['MA20']       = SMAIndicator(data['SP500'], window=20).sma_indicator()
data['RSI14']      = RSIIndicator(close=data['SP500'], window=14).rsi()
bb               = BollingerBands(close=data['SP500'], window=20, window_dev=2)
data['BBW']        = bb.bollinger_wband()
macd             = MACD(close=data['SP500'], window_slow=26, window_fast=12, window_sign=9)
data['MACD_diff'] = macd.macd_diff()

# 7. 뉴스 감성 지표 (NewsAPI + VADER)
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
sid     = SentimentIntensityAnalyzer()
news_sent = []
for date in data.index:
    resp = newsapi.get_everything(
        q='FOMC OR "Federal Reserve"',
        from_param=date.strftime('%Y-%m-%d'),
        to=date.strftime('%Y-%m-%d'),
        language='en', page_size=5
    )
    scores = [sid.polarity_scores(a['title'] + ' ' + (a.get('description') or ''))['compound']
              for a in resp.get('articles', [])]
    news_sent.append(np.mean(scores) if scores else 0)
data['News_Sentiment'] = news_sent

# 8. 트위터 감성 지표 (Tweepy + VADER)
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

# 9. 모델 학습용 특징 선정
features = [
    'VIX','Yield_Spread',
    'no_change_prob','up_25_prob','down_25_prob',
    'MA20','RSI14','BBW','MACD_diff',
    'News_Sentiment','Twitter_Sentiment'
]
# 데이터 정리
df = data.dropna()
X, y = df[features], df['Target']

# 10. 학습/테스트 분리 및 모델 학습
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2
)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# 11. 예측 & 평가
proba   = model.predict_proba(X_test)
y_pred  = (proba[:,1] > 0.5).astype(int)
results = X_test.copy()
results['Prob_Down']       = proba[:,0]
results['Prob_Up']         = proba[:,1]
results['Predicted_Label'] = y_pred
results['Actual_Label']    = y_test.values

print(results.tail(10))
print(classification_report(y_test, y_pred))
