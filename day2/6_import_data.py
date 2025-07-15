import os
from google import genai
from google.genai import types
from sklearn.datasets import fetch_20newsgroups

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

"""
The 20 Newsgroups Text Dataset contains 18,000 newsgroups posts on 20 topics divided into training and test sets. 
The split between the training and test datasets are based on messages posted before and after a specific date. 
For this tutorial, you will use sampled subsets of the training and test sets, and perform some processing using Pandas.
"""

newsgroups_train = fetch_20newsgroups(subset="train")
newsgroups_test = fetch_20newsgroups(subset="test")

# View list of class names for dataset
print(newsgroups_train.target_names)
print(len(newsgroups_train.data))
print(newsgroups_train.data[0])

print(newsgroups_test.target_names)
print(len(newsgroups_test.data))
print(newsgroups_test.data[0])