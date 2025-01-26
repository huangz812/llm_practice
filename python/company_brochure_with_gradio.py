import anthropic
import gradio as gr
import os
import requests
import signal
import sys
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

class CompanyBrochure:

    def __init__(self, url):
        """
        Do some initializing
        Fetch the company url, title and landing page contents from the given url using the BeautifulSoup library
        """
        self.openai = OpenAI()
        self.claude = anthropic.Anthropic()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        }
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)


    @classmethod
    def stream_brochure(cls, url, model):
        """
        Class method that instantiates a CompanyBrochure object.
        Then call the generate_brochure(model)

        Arguments:
            cls: refers to the class itself
            url: the landing page url of the company
            model: the model to be used for answers

        Yields:
            Chunks of stream responses accumulatively
        """
        company_brochure = cls(url)
        yield from company_brochure.__generate_brochure(model)


    def __generate_brochure(self, model):
        system_prompt = "You are an assistant that analyzes the contents of a company website landing page \
        and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown."

        user_prompts = f"Here are the company's title: {self.title}. The web contents are {self.text}"


        if model == 'GPT':
            yield from self.__stream_gpt(system_prompt, user_prompts)
        elif model == 'Claude':
            yield from self.__stream_claude(system_prompt, user_prompts)
        else :
            raise ValueError("Unknown model")

    def __stream_gpt(self, system_prompt, user_prompts):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompts}]
        stream = self.openai.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0.8,
            stream=True)
        responses = ""
        for chunk in stream:
            # We need to send accumulative responses back
            # Because most clients like Gradio don't keep track of history
            responses += chunk.choices[0].delta.content or ''
            yield responses

    def __stream_claude(self, system_prompt, user_prompts):
        result = self.claude.messages.stream(
            model='claude-3-5-sonnet-20240620',
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompts}],
            max_tokens=800,
            temperature=0.6)
        responses = ""
        with result as stream:
            for text in stream.text_stream:
                responses += text
                yield responses


# Function to handle closing the interface
# The 2nd parameter is frame which is required by the signal handler signature
# Since it's not used, use _ instead
def handle_exit_signal(signal, _):
    print("Exiting gradio!")
    gr.close_all()
    sys.exit(0)  # Exit the program

if __name__ == "__main__":
    # Load environment variables in a file called .env
    load_dotenv(override=True)
    openai_api_key = os.getenv('OPENAI_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print("OpenAI API Key not set")

    if anthropic_api_key:
        print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
    else:
        print("Anthropic API Key not set")

    # Register the signal handler for KeyboardInterrupt (Ctrl+C)
    signal.signal(signal.SIGINT, handle_exit_signal)

    view = gr.Interface(
        fn=CompanyBrochure.stream_brochure,
        inputs=[
            gr.Textbox(label="Landing page URL including http:// or https://"),
            gr.Dropdown(["GPT", "Claude"], label="Select model")],
        outputs=[gr.Markdown(label="Brochure:")],
        flagging_mode="never"
    )

    print("launching gradio!")
    view.launch()
