# sentiment_analysis_emotions.py

import tweepy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from transformers import pipeline

# ==========================
# 1. SETUP TWITTER ACCESS
# ==========================
# Replace with your own keys or use mock data
USE_TWITTER = False  # Set to True if you want real Twitter data

# Twitter API credentials (only needed if USE_TWITTER = True)
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
ACCESS_TOKEN = 'your_access_token'
ACCESS_SECRET = 'your_access_secret'

# ==========================
# 2. FETCH TWEETS
# ==========================
def fetch_tweets(query, count=100):
    auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth)
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode='extended').items(count)
    return [tweet.full_text for tweet in tweets]

# ==========================
# 3. MOCK DATA (if not using Twitter)
# ==========================
def get_mock_tweets():
    return [
        "I'm so happy with the service today!",
        "This is the worst experience I've ever had.",
        "I'm feeling very anxious and stressed lately.",
        "That game was amazing and exciting!",
        "I'm confused about the new policies.",
        "Everything feels dull and meaningless.",
        "Such a joyful moment!",
        "That news was shocking and sad.",
    ]

# ==========================
# 4. EMOTION ANALYSIS
# ==========================
def analyze_emotions(tweets):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
    results = []
    for tweet in tweets:
        emotion = classifier(tweet)[0]['label']
        sentiment = TextBlob(tweet).sentiment.polarity
        results.append({
            "tweet": tweet,
            "emotion": emotion,
            "sentiment_score": sentiment,
            "sentiment": "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
        })
    return pd.DataFrame(results)

# ==========================
# 5. VISUALIZATION
# ==========================
def plot_emotions(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.countplot(y="emotion", data=df, order=df['emotion'].value_counts().index)
    plt.title("Distribution of Emotions")
    plt.xlabel("Count")
    plt.ylabel("Emotion")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(x="sentiment", data=df, order=["positive", "neutral", "negative"])
    plt.title("Overall Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

# ==========================
# 6. MAIN FUNCTION
# ==========================
def main():
    if USE_TWITTER:
        tweets = fetch_tweets("mental health", count=50)
    else:
        tweets = get_mock_tweets()

    print("Analyzing emotions...")
    df = analyze_emotions(tweets)
    print(df[["tweet", "emotion", "sentiment", "sentiment_score"]])
    plot_emotions(df)

if __name__ == "__main__":
    main()
