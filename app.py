import pandas as pd
import numpy as np
from ReviewLoader import ReviewLoader
from openai import OpenAI
import json
import re
import plotly.express as px
import plotly.graph_objects as go

def analyze_review(review_text, client):
    """Use Deepseek API to extract sentiment score and topics from single review"""

    prompt = f"""
        Extract structured sentiment information from this retail review.

        Return JSON with:
        - overall_sentiment (-1 to 1)
        - staff_sentiment (-1 to 1 or null)
        - pricing_sentiment (-1 to 1 or null)
        - product_sentiment (-1 to 1 or null)
        - topics (list)

        Review:
        {review_text}
    """

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": "You are a sentiment analysis expert. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        stream=False,
        reasoning_effort="high"
    )

    result = json.loads(response.choices[0].message.content)
    return result
def safe_load_llm_json(response_text):
    # Strip markdown code blocks
    cleaned = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE)
    return json.loads(cleaned)

def batch_analyze_reviews(df, client, batch_size=10):
    """
    Analyze multiple reviews in batches to reduce API calls.
    Returns DataFrame with sentiment and topics columns.    
    """

    df_copy = df.copy()
    
    overall_sentiments = []
    staff_sentiments = []
    pricing_sentiments = []
    product_sentiments = []
    all_topics = []

    for i in range(0, len(df_copy), batch_size):
        print(f"Analyzing reviews {i} to {i+10}")
        batch = df_copy["comment"].iloc[i:i+batch_size].tolist()
        review_batch = ""

        for idx, review in enumerate(batch):
            review_batch += f"Review {idx+1}: {review}\n"
        
        # Create batch prompt
        prompt = "Analyze the following reviews. Return ONLY valid JSON array. DO NOT wrap in ```json or ```. DO NOT add any explanatory text. Reviews: "
        prompt += review_batch
        prompt += 'Expected format: [{"overall_sentiment": 0.5, "staff_sentiment": 0.5, "pricing_sentiment": 0.5, "product_sentiment": 0.5, "topics": ["service", "price"]}, ...]'
        
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "You are a JSON generator. Return only valid JSON. No markdown, no explanations"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            reasoning_effort="high"
        )
        
        try:
            results = safe_load_llm_json(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw response: {response.choices[0].message.content[:200]}")
            # Fallback to defaults
            results = [{"overall_sentiment": 0, "topics": []} for _ in range(batch_size)]
        
        for result in results:
            overall_sentiments.append(result["overall_sentiment"])
            staff_sentiments.append(result["staff_sentiment"])
            pricing_sentiments.append(result["pricing_sentiment"])
            product_sentiments.append(result["product_sentiment"])
            all_topics.append(result["topics"])
        
    df_copy["overall_sentiment"] = overall_sentiments
    df_copy["staff_sentiment"] = staff_sentiments
    df_copy["pricing_sentiment"] = pricing_sentiments
    df_copy["product_sentiment"] = product_sentiments
    df_copy["topics"] = all_topics
    
    return df_copy


loader = ReviewLoader()
all_reviews = loader.get_reviews()
reviews_with_sentiment = pd.read_csv("all_reviews_with_sentiment.csv", encoding="utf-8")

def stars_to_sentiment(star_rating: int) -> float:
    return (star_rating - 3) / 2

no_comment_reviews = all_reviews[all_reviews["comment"]==""]
no_comment_reviews["overall_sentiment"] = no_comment_reviews["rating"].apply(stars_to_sentiment)
no_comment_reviews["staff_sentiment"] = pd.NA
no_comment_reviews["pricing_sentiment"] = pd.NA
no_comment_reviews["product_sentiment"] = pd.NA
no_comment_reviews["topics"] = [[] for _ in range(len(no_comment_reviews))]

all_reviews = pd.concat([no_comment_reviews, reviews_with_sentiment])
all_reviews["create_time"] = pd.to_datetime(all_reviews["create_time"], format="%Y-%m-%dT%H:%M:%S.%fZ")

def variable_window_by_volume(df, target_count=15, date_col='create_time', topic_col="overall_sentiment"):
    df = df.sort_values(date_col).copy()
    df["sma"] = np.nan
    df["window_size"] = np.nan

    for i in range(len(df)):
        reviews_collected = 0
        window_size = 0

        for j in range(i, -1, -1):
            if df[topic_col].iloc[j] is not None and not np.isnan(df[topic_col].iloc[j]):
                reviews_collected += 1
                window_size += 1
                if reviews_collected >= target_count:
                    break

        window_data = df[topic_col].iloc[i-window_size+1:i+1]
        df.loc[df.index[i], "sma"] = window_data.mean()
        df.loc[df.index[i], "window_size"] = window_size
    
    return df



def plot_store_sentiment(store_name: str, store_df: pd.DataFrame) -> None:
    sma_df = variable_window_by_volume(store_df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sma_df["create_time"], y=sma_df['overall_sentiment'],
                            mode='markers', name='Raw', marker=dict(size=3, opacity=0.5)))
    fig.add_trace(go.Scatter(x=sma_df["create_time"], y=sma_df['sma'],
                            mode='lines', name='7-day SMA', line=dict(width=3, color='red')))
    fig.update_layout(title=f'{store_name} Sentiment 7-day Rolling Average')
    fig.show()
    # fig = px.scatter(
    #     x=monthly_store.index, y=monthly_store["mean_sentiment"],
    #     title=f"{store_name} Sentiment Over Time",
    #     trendline="ols",
    #     labels={"x": "Date", "y": "Mean Sentiment by Month"}
    # )

    # fig.show()


all_reviews["period"] = all_reviews["create_time"].dt.to_period("M").dt.start_time
all_reviews = all_reviews[all_reviews["period"]>pd.to_datetime("2016-01-01")]
store_names = list(all_reviews["store"].unique())
for store in store_names:
    store_df = all_reviews[all_reviews["store"] == store]
    plot_store_sentiment(store, store_df)


