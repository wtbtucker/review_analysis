import pandas as pd

review_df = pd.read_csv("all_store_reviews.csv", names=["reviewer_name", "rating", "create_time", "reply", "store"], dtype={"rating": int})
print("Total store reviews: ", len(review_df))

print(f"Average store review: {review_df['rating'].mean()}")

average_review_score_by_store = (
    review_df
    .groupby("store")
    .agg(average_rating=("rating", "mean"))
)
print(average_review_score_by_store)