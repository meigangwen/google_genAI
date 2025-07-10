import os
from google import genai
#from google.genai import types

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

for model in client.models.list():
    print(model.name)