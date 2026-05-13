import pandas as pd
import numpy as np
from ReviewLoader import ReviewLoader
from openai import OpenAI
import json
import re
import plotly.express as px
import plotly.graph_objects as go
import ast

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

all_reviews = pd.read_csv("all_reviews_with_sentiment.csv", encoding="utf-8")
all_reviews["create_time"] = pd.to_datetime(all_reviews["create_time"], format="%Y-%m-%dT%H:%M:%S.%fZ")

def convert_to_list_safe(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            result = ast.literal_eval(value)
            if isinstance(result, list):
                return result
        except:
            pass

    return [str(value)]


all_reviews["topics_list"] = all_reviews["topics"].apply(convert_to_list_safe)

print(type(all_reviews["topics"].iloc[0]))
print(type(all_reviews["topics_list"].iloc[0]))