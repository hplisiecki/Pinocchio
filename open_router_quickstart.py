# pip install openai

import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4.5",   # swap to any OpenRouter model
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Write one short sentence about Warsaw."},
    ],
    extra_headers={
        # Optional according to OpenRouter docs
        "HTTP-Referer": "http://localhost:3000",
        "X-OpenRouter-Title": "local-test-script",
    },
)

print(response.choices[0].message.content)