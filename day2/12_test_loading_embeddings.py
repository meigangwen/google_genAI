import os
from google import genai
from google.genai import types
import pandas as pd
import tqdm
from tqdm.rich import tqdm as tqdmr
import warnings
import numpy as np

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

df_train = pd.read_parquet("day2/store/df_train.parquet")
df_test = pd.read_parquet("day2/store/df_test.parquet")

embeddings_test_path = 'day2/store/embeddings_test.npz'
if os.path.exists(embeddings_test_path):
    print("Loading embeddings test ...")
    data = np.load(embeddings_test_path)
    x_test = data["x"]
    y_test = data["y"]

print(x_test)
print(y_test)