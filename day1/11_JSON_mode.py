import os
from google import genai
from google.genai import types
import typing_extensions as typing

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

class PizzaOrder(typing.TypedDict):
    size: str
    ingredients: list[str]
    type: str

response = client.models.generate_content(
    model='gemini-2.0-flash',
    config=types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=PizzaOrder,
    ),
    contents="Can I have a large dessert pizza with apple and chocolate")

print(response.text)
print('-' * 25)
class Event(typing.TypedDict):
    date: str
    participants: list[str]
    name: str

response = client.models.generate_content(
    model='gemini-2.0-flash',
    config=types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=Event,
    ),
    contents="Bob, Adam and Lee are going to the Kangaroo Math Competition on the Sep 5th 2025")

print(response.text)