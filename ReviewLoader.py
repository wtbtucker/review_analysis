from typing import List
import pandas as pd
import csv
import os
import json
from dataclasses import dataclass
from datetime import datetime

def main():
    source_data_path = "google-data/"
    store_names = [
        dir_obj.name for dir_obj in os.scandir(source_data_path) if dir_obj.is_dir()
    ]

    all_reviews = []
    old_review_count = 0
    for store_name in store_names:
        store_review_file_paths = get_review_file_names(store_name)
        for file_path in store_review_file_paths:
            review_objs = get_store_reviews(file_path, store_name)
            all_reviews.extend(review_objs)
        
        cur_review_count = len(all_reviews)
        reviews_added = cur_review_count- old_review_count
        print(f"Reviews added for {store_name}: {reviews_added}")
        old_review_count = cur_review_count

    review_df = pd.DataFrame([vars(review) for review in all_reviews])
    review_df.to_csv("all_store_reviews.csv", index=False)



class Review:
    def __init__(self, reviewer_name: str, rating: str, create_time: str, comment: str, reply: bool):
        self.reviewer_name = reviewer_name
        rating_dict = {
            "ONE": 1,
            "TWO": 2,
            "THREE": 3,
            "FOUR": 4,
            "FIVE": 5
        }
        self.rating = rating_dict[rating]
        self.create_time = create_time
        self.comment = comment
        self.reply = bool(reply)
        self.store = None

    
    def __str__(self):
        return (
            f"Review(reviewer='{self.reviewer_name}', "
            f"rating={self.rating}, "
            f"date='{self.create_time}', "
            f"store='{self.store}', "
            f"replied={self.reply}, "
            f"comment='{self.comment}')"
        )

    def set_store(self, store: str):
        self.store = store

def get_store_reviews(file_path: str, store_name) -> List[Review]:
    with open(file_path, 'r', encoding='utf-8') as f:

        data = json.load(f)
        review_dicts = data["reviews"]
        obj_reviews = []

        for review in review_dicts:
            reviewer_name = review["reviewer"]["displayName"]
            rating = review["starRating"]
            comment = review.get("comment", "")
            create_time = review["createTime"]
            reviewReply = "reviewReply" in review
            new_review = Review(reviewer_name, rating, create_time, comment, reviewReply)
            new_review.set_store(store_name)
            obj_reviews.append(new_review)
        return obj_reviews
        
        
def get_review_file_names(store_name: str) -> List[str]:
    """
    Each store directory can have a variable number of review json files
    with name structure "reviews-{random_hash}"
    Extract a list of all review filenames
    Args:
        store_name: string name of store eg. andover
    """

    store_directory_path = f"google-data/{store_name.strip().lower()}"
    
    all_files = [
        dir_obj for dir_obj in os.scandir(store_directory_path) if dir_obj.is_file
    ]

    review_file_paths = [
        file.path for file in all_files if file.name.startswith("reviews")
    ]

    return review_file_paths

main()