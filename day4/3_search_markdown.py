import os
from google import genai
from google.genai import types
from pprint import pprint
import io

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

config_with_search = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
    temperature=0.0,
)

chat = client.chats.create(model='gemini-2.0-flash')

response = chat.send_message(
    message="What were the medal tallies, by top-10 countries, for the 2024 olympics?",
    config=config_with_search,
)

print(response.text)

'''
ontinuing the chat, now ask the model to convert the data into a chart. The code_execution tool is able to generate code to draw charts, execute that code and return the image. You can see the executed code in the executable_code part of the response.
Combining results from Google Search with tools like live plotting can enable very powerful use cases that require very little code to run.
'''

config_with_code = types.GenerateContentConfig(
    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
    temperature=0.0,
)

response = chat.send_message(
    message="Now plot this as a seaborn chart. Break out the medals too.",
    config=config_with_code,
)

print(response.text)