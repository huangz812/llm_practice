# imports

import requests
import ollama
from bs4 import BeautifulSoup

# A class to represent a Webpage
class Website:

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
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


    def summarize(self):
        system_prompt = "You are an assistant that analyzes the contents of a website \
            and provides a short summary, ignoring text that might be navigation related. \
            Respond in markdown."
        user_prompt = f"You are looking at a website titled {self.title}"
        user_prompt += "\nThe contents of this website is as follows; \
            please provide a short summary of this website in markdown. \
            If it includes news or announcements, then summarize these too.\n\n"
        user_prompt += self.text
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
            ]
        response = ollama.chat(model="llama3.2", messages=messages)
        print(response['message']['content'])

if __name__ == "__main__":
    print("Running ollama_web__summarizer module directly")
    website = Website("https://anthropic.com")
    website.summarize()
