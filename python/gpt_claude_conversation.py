import anthropic
import os
import requests
import time
from dotenv import load_dotenv
from openai import OpenAI


class Conversation:
    """
    The Conversation class calls gpt and claude alternately.
    We pass system, user, and assistant contents to the model for history keeping.
    For GPT, gpt_messages are assistant content while claude_messages are user contents.
    For Claude, it's the other way.
    """

    def __init__(self):
        self.openai = OpenAI()
        self.claude = anthropic.Anthropic()
        self.gpt_model = "gpt-4o-mini"
        self.claude_model = "claude-3-5-sonnet-20240620"
        self.gpt_system = "You are a chatbot who talks in a sarcastic tone. \
        You have a know it all personality."
        self.claude_system = "You are a very nice chatbot. \
        But you always like to talk in a obscure tone."
        self.gpt_messages = ["Hey Bro!"]
        self.claude_messages = ["Hey Yo!"]

    def _call_gpt(self):
        messages = [{"role": "system", "content": self.gpt_system}]
        for gpt, claude in zip(self.gpt_messages, self.claude_messages):
            messages.append({"role": "assistant", "content": gpt})
            messages.append({"role": "user", "content": claude})

        stream = self.openai.chat.completions.create(
            model=self.gpt_model,
            messages=messages,
            temperature=0.8,
            stream=True)
        print("gpt: ", end="", flush=True)
        # Handle stream response
        response_list = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ''
            if delta:
                # print without a newline.
                print(delta, end="", flush=True)
                response_list.append(delta)
        final_response = ''.join(response_list)
        # Append the response for history keeping
        self.gpt_messages.append(final_response)
        # print a newline to separate claude's response
        print("\n")

    def _call_claude(self):
        messages = []
        for gpt, claude in zip(self.gpt_messages, self.claude_messages):
            messages.append({"role": "user", "content": gpt})
            messages.append({"role": "assistant", "content": claude})
        # For claude, the last gpt_message is the latest user content it needs to respond.
        messages.append({"role": "user", "content": self.gpt_messages[-1]})
        result = self.claude.messages.stream(
            model=self.claude_model,
            system=self.claude_system,
            messages=messages,
            max_tokens=800,
            temperature=0.6)
        print("claude: ", end="", flush=True)
        # Handle stream response
        response_list = []
        # client.messages.stream() returns a MessageStreamManager
        # That's why we have to use context manager to access the text_stream field
        with result as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                response_list.append(text)
        final_response = ''.join(response_list)
        # Append the response for history keeping
        self.claude_messages.append(final_response)
        # print a newline to separate gpt's response
        print("\n")

    def start_conversation(self):
        self._call_gpt()
        time.sleep(3)
        self._call_claude()
        time.sleep(3)


if __name__ == "__main__":
    # Load environment variables in a file called .env
    load_dotenv(override=True)
    openai_api_key = os.getenv('OPENAI_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

    # Check the openai key

    if not openai_api_key:
        print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
    elif not openai_api_key.startswith("sk-proj-"):
        print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
    elif openai_api_key.strip() != openai_api_key:
        print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
    else:
        print("API key found and looks good so far!")

    # check the anthropic key
    if anthropic_api_key:
        print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
    else:
        print("Anthropic API Key not set")

    conversation = Conversation()
    # print the first two messages.
    print(f"gpt: {conversation.gpt_messages[0]}")
    print(f"claude: {conversation.claude_messages[0]}")

    try:
        while True:
            conversation.start_conversation()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting the program.")

