"""
Hybrid News Sentiment Analyzer (Finnhub + NewsAPI + FinBERT)
Author: Valkyrie Systems

Requirements:
    pip install yfinance requests transformers torch pandas matplotlib
"""

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   # disable TensorFlow
os.environ["TRANSFORMERS_NO_FLAX"] = "1" # disable JAX/Flax

import torch
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# --------------------------
# CONFIG
# --------------------------
NEWSAPI_KEY = "0869389d62ae468d8f199910cff283f2"       # https://newsapi.org
FINNHUB_KEY = "d3dgkc9r01qg5k5rhbbgd3dgkc9r01qg5k5rhbc0"       # https://finnhub.io
SENTIMENT_MODEL = "ProsusAI/finbert"   # Finance-tuned NLP model
DAYS_BACK = 30


# --------------------------
# NewsAPI Fetch
# --------------------------
def fetch_newsapi(ticker: str, days_back: int = DAYS_BACK) -> pd.DataFrame:
    """Fetch recent news from NewsAPI"""
    url = "https://newsapi.org/v2/everything"
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q": ticker,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWSAPI_KEY,
        "pageSize": 100,
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "ok":
        return pd.DataFrame()

    articles = [
        {
            "date": pd.to_datetime(a["publishedAt"]).date(),
            "title": a["title"],
            "description": a.get("description", ""),
            "source": "newsapi",
        }
        for a in data["articles"]
    ]
    return pd.DataFrame(articles)


# --------------------------
# Finnhub Fetch
# --------------------------
def fetch_finnhub(ticker: str, days_back: int = DAYS_BACK) -> pd.DataFrame:
    """Fetch recent news sentiment from Finnhub"""
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={FINNHUB_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    if "sentiment" not in data:
        return pd.DataFrame()

    # Finnhub gives aggregated sentiment already
    df = pd.DataFrame([{
        "date": datetime.utcnow().date(),
        "sentiment": data["sentiment"].get("sentimentScore", 0),
        "source": "finnhub"
    }])
    return df


# --------------------------
# Sentiment Analysis (FinBERT for NewsAPI)
# --------------------------
def analyze_sentiment_newsapi(df: pd.DataFrame) -> pd.DataFrame:
    """Run FinBERT sentiment analysis on NewsAPI headlines"""
    if df.empty:
        return df

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL, dtype=torch.float32)

    # Create pipeline (force device and dtype)
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        framework="pt",        # Force PyTorch
        device=0 if torch.cuda.is_available() else -1
    )

    texts = (df["title"].fillna("") + ". " + df["description"].fillna("")).tolist()
    results = sentiment_pipe(texts, batch_size=16, truncation=True)

    sentiments = []
    for res in results:
        label, score = res["label"].lower(), res["score"]
        if label == "positive":
            val = score
        elif label == "negative":
            val = -score
        else:
            val = 0
        sentiments.append(val)

    df["sentiment"] = sentiments
    return df.groupby("date")["sentiment"].mean().reset_index()


# --------------------------
# Price Data
# --------------------------
def fetch_price_data(ticker: str, days_back: int = DAYS_BACK) -> pd.DataFrame:
    """Fetch OHLCV daily candles from yfinance"""
    start = datetime.utcnow() - timedelta(days=days_back + 2)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"))
    df.reset_index(inplace=True)
    df["date"] = df["Date"].dt.date
    return df


# --------------------------
# Merge & Fusion
# --------------------------
def merge_sentiments(newsapi_df: pd.DataFrame) -> pd.DataFrame:
    """Merge NewsAPI + Finnhub sentiment into a hybrid score"""
    if newsapi_df.empty and finnhub_df.empty:
        return pd.DataFrame()

    # Weighted average: NewsAPI (FinBERT) = 0.6, Finnhub = 0.4
    if not newsapi_df.empty:
        newsapi_df["source"] = "newsapi"

    merged = pd.concat([newsapi_df], ignore_index=True, sort=False)

    # Daily hybrid score
    daily = (
        merged.groupby("date")["sentiment"]
        .apply(lambda x: 1 * x[x.index.isin(newsapi_df.index)].mean()
              if not x.empty else 0)
        .reset_index()
    )
    return daily


def merge_with_prices(sent_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Join sentiment with price and returns"""
    df = price_df.merge(sent_df, on="date", how="left")
    df["sentiment"].fillna(0, inplace=True)
    df["return"] = df["Close"].pct_change()
    return df


# --------------------------
# Analysis & Plot
# --------------------------
def analyze_effect(merged: pd.DataFrame, ticker: str):
    corr = merged[["sentiment", "return"]].corr().iloc[0, 1]
    print(f"\n📊 Hybrid Sentiment vs Return Correlation for {ticker}: {corr:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(merged["date"], merged["Close"], label="Price", color="blue")
    plt.bar(merged["date"], merged["sentiment"] * 50, alpha=0.4, label="Hybrid Sentiment (scaled)")
    plt.legend()
    plt.title(f"{ticker} Price & Hybrid News Sentiment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# --------------------------
# Main Runner
# --------------------------
def run_hybrid_analysis(ticker: str):
    print(f"\n🚀 Running Hybrid Sentiment Analysis for {ticker}...")

    # Fetch news
    newsapi_raw = fetch_newsapi(ticker)
    newsapi_raw.to_csv('news.csv')
    # finnhub_raw = fetch_finnhub(ticker)

    # Process NewsAPI (FinBERT sentiment)
    newsapi_sent = analyze_sentiment_newsapi(newsapi_raw)

    # Merge hybrid
    sent_df = merge_sentiments(newsapi_sent)

    if sent_df.empty:
        print("⚠️ No sentiment data found.")
        return

    # Fetch prices
    price_df = fetch_price_data(ticker)

    # Merge & analyze
    merged = merge_with_prices(sent_df, price_df)
    analyze_effect(merged, ticker)


if __name__ == "__main__":
    run_hybrid_analysis("AAPL")      # Stock
    # run_hybrid_analysis("BTC-USD")   # Crypto
