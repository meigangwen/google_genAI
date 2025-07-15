import os
from google import genai

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

result = client.models.embed_content(
        model = "gemini-embedding-001",
        contents = "What is the meaning of life?")

"""
The result is a high dimensional vector that captures the meaning of the whole sentence, that is why there are so many values, hundreds or thousands depending on the model

High-dimensional embeddings:
    Allow for more nuanced semantic representation.
    Help the model capture complex relationships (e.g., similarity between "purpose of existence" and "meaning of life").
    Enable things like semantic search, clustering, and recommendation systems.
"""

print(result.embeddings)