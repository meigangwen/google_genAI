import os
from google import genai
#from google.genai import types

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

chat = client.chats.create(model='gemini-2.0-flash', history=[])
response = chat.send_message('Hello! My name is Gangwen.')
print(response.text)

response = chat.send_message('Do you remember what my name is?')
print(response.text)