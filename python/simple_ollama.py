import ollama

# Constants
MODEL = "llama3.2"

# Create a messages list using the same format that we used for OpenAI

messages = [
    {"role": "user", "content": "Is there a buy call option?"}
]

response = ollama.chat(model=MODEL, messages=messages)
print(response['message']['content'])
