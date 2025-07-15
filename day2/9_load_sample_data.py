import os
from google import genai
from google.genai import types
import pandas as pd

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

df_train = pd.read_parquet("day2/store/df_train.parquet")
df_test = pd.read_parquet("day2/store/df_test.parquet")

print(df_train.value_counts("Class Name"))
print(df_test.value_counts("Class Name"))

