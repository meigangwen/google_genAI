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

'''
In this section, you will generate embeddings for each piece of text using the Gemini API embeddings endpoint. 

Task types
The text-embedding-004 model supports a task type parameter that generates embeddings tailored for the specific task.

RETRIEVAL_QUERY	- Specifies the given text is a query in a search/retrieval setting.
RETRIEVAL_DOCUMENT - Specifies the given text is a document in a search/retrieval setting.
SEMANTIC_SIMILARITY - Specifies the given text will be used for Semantic Textual Similarity (STS).
CLASSIFICATION - Specifies that the embeddings will be used for classification.
CLUSTERING - Specifies that the embeddings will be used for clustering.
FACT_VERIFICATION - Specifies that the given text will be used for fact verification.
'''

df_train = pd.read_parquet("day2/store/df_train.parquet")
df_test = pd.read_parquet("day2/store/df_test.parquet")

# Add tqdm to Pandas...
tqdmr.pandas()

# ...But suppress the experimental warning.
warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)

def embed_fn(text: str) -> list[float]:
    # You will be performing classification, so set task_type accordingly.
    response = client.models.embed_content(
        model="models/text-embedding-004",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="classification",
        ),
    )

    return response.embeddings[0].values


def create_embeddings(df):
    df["Embeddings"] = df["Text"].progress_apply(embed_fn)
    return df

df_train = create_embeddings(df_train)
#df_test = create_embeddings(df_test)

x_train = np.stack(df_train["Embeddings"])
y_train = df_train["Encoded Label"]
np.savez("day2/store/embeddings_train.npz", x=x_train, y=y_train)


#x_test = np.stack(df_test["Embeddings"])
#y_test = df_test["Encoded Label"]
#np.savez("day2/store/embeddings_test.npz", x=x_test, y=y_test)