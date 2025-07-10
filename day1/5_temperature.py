import os
from google import genai
from google.genai import types

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

high_temp_config = types.GenerateContentConfig(temperature=2.0)
print("High temp results:")
for _ in range(5):
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        config=high_temp_config,
        contents='Pick a random colour... (respond in a single word)'
    )
    if response.text:
        print(response.text, '-' * 25)

low_temp_config = types.GenerateContentConfig(temperature=0.0)
print("Low temp results:")
for _ in range(5):
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        config=low_temp_config,
        contents='Pick a random colour... (respond in a single word)'
    )
    if response.text:
        print(response.text, '-' * 25)