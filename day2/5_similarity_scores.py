import os
from google import genai
from google.genai import types
#from chromadb import Documents, EmbeddingFunction, Embeddings
#import chromadb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

texts = [
    'The quick brown fox jumps over the lazy dog.',
    'The quick rbown fox jumps over the lazy dog.',
    'teh fast fox jumps over the slow woofer.',
    'a quick brown fox jmps over lazy dog.',
    'brown fox jumping over dog',
    'fox > dog',
    # Alternative pangram for comparison:
    'The five boxing wizards jump quickly.',
    # Unrelated text, also for comparison:
    'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus et hendrerit massa. Sed pulvinar, nisi a lobortis sagittis, neque risus gravida dolor, in porta dui odio vel purus.',
]

response = client.models.embed_content(
    model='models/text-embedding-004',
    contents=texts,
    config=types.EmbedContentConfig(task_type='semantic_similarity'))

# Define a short helper function that will make it easier to display longer embedding texts in our visualisation.
def truncate(t: str, limit: int = 50) -> str:
    """Truncate labels to fit on the chart."""
    if len(t) > limit:
        return t[:limit-3] + '...'
    else:
        return t

truncated_texts = [truncate(t) for t in texts]

# Set up the embeddings in a dataframe.
df = pd.DataFrame([e.values for e in response.embeddings], index=truncated_texts)
# Perform the similarity calculation
sim = df @ df.T

# Draw!
sns.heatmap(sim, vmin=0, vmax=1, cmap="Greens")
plt.savefig("day2/store/heatmap.png")

print(sim['The quick brown fox jumps over the lazy dog.'].sort_values(ascending=False))
