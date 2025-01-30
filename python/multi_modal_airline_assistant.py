import base64
import json
import os
import signal
import sys
import gradio as gr
from dotenv import load_dotenv
from io import BytesIO
from openai import OpenAI
from PIL import Image
from pydub import AudioSegment
from pydub.playback import play

class AirlineAssistant:

    TICKET_INFO = {("san francisco", "shanghai"): ("$588", "13 hours 18 minutes"),
                   ("shanghai", "san francisco"): ("$618", "11 hours"),
                   ("los angeles", "shanghai"): ("$600", "13 hours 8 minutes"),
                   ("shanghai", "los angeles"): ("$638", "11 hours 29 minutes")}
    MODEL = "gpt-4o-mini"
    TICKET_FUNCTION = {
        "name": "get_ticket_info",
        "description": "Get the price and distance of a flight ticket from a source city to a destination city. \
        Call this whenever you are asked about a flight ticket information, for example when a customer asks 'How much is a ticket from city A to city B'",
        "parameters": {
            "type": "object",
            "properties": {
                "src": {
                    "type": "string",
                    "description": "The city that the customer wants to travel from",
                },
                "dest": {
                    "type": "string",
                    "description": "The city that the customer wants to travel to",
                },
            },
            "required": ["src", "dest"],
            "additionalProperties": False
        }
    }
    TOOLS = [{"type": "function", "function": TICKET_FUNCTION}]
    
    def __init__(self):
        """
        The assistant will help answer airline ticket price and time to travel
        """
        self.openai = OpenAI()
        self.system_message = "You are a helpful assistant for an Airline called FlightAI. \
        Give short, courteous answers, no more than 1 sentence. \
        Always be accurate. If you don't know the answer, say so."

    def run(self):
        gr.ChatInterface(fn=self.__chat, type="messages").launch()

    def run_multi_modal(self):
        with gr.Blocks() as ui:
            with gr.Row():
                chatbot_output = gr.Chatbot(height=500, type="messages")
                image_output = gr.Image(height=500)
            with gr.Row():
                message_box = gr.Textbox(label="Chat with our AI Assistant:")
            with gr.Row():
                clear_button = gr.Button("Clear")

            def __message_submitted(input_message, history):
                history += [{"role": "user", "content": input_message}]
                # We return "" so we can clear the message_box
                # We append user message to  history and return the new history so we can update the chatbot_output
                return "", history

            message_box.submit(__message_submitted, inputs=[message_box, chatbot_output], outputs=[message_box, chatbot_output]).then(
                self.__multi_modal_chat, inputs=chatbot_output, outputs=[chatbot_output, image_output])

            clear_button.click(lambda: [None, "", None], inputs=None, outputs=[chatbot_output, message_box, image_output], queue=False)

        ui.launch(inbrowser=True)
                

    def __chat(self, message, history):
        """
        Gradio chatInterface requires this function. Gradio will populate both message and history

        Arguments:
            message: the prompt to use, sent back by Gradio
            history: the past conversation, in OpenAI format, sent back by Gradio
        """
        messages = [{"role": "system", "content": self.system_message}] + history + [{"role": "user", "content": message}]
        response = self.openai.chat.completions.create(model=AirlineAssistant.MODEL, messages=messages, tools=AirlineAssistant.TOOLS)
        if response.choices[0].finish_reason=="tool_calls":
            message = response.choices[0].message
            _, _, _, response = self.__handle_tool_call(message)
            messages.append(message)
            messages.append(response)
            response = self.openai.chat.completions.create(model=AirlineAssistant.MODEL, messages=messages)

        return response.choices[0].message.content

    def __multi_modal_chat(self, history):
        messages = [{"role": "system", "content": self.system_message}] + history
        response = self.openai.chat.completions.create(model=AirlineAssistant.MODEL, messages=messages, tools=AirlineAssistant.TOOLS)
        image = None
        if response.choices[0].finish_reason=="tool_calls":
            message = response.choices[0].message
            tool_call_success, src, dest, response = self.__handle_tool_call(message)
            messages.append(message)
            messages.append(response)
            response = self.openai.chat.completions.create(model=AirlineAssistant.MODEL, messages=messages)
            if tool_call_success:
                # Only generate image when there is a legit ticket information
                image = self.__generate_image(src, dest)

        reply = response.choices[0].message.content
        history += [{"role": "assistant", "content": reply}]
        self.__talk(reply)
        return history, image
        

    def __handle_tool_call(self, message):
        tool_call = message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        src = arguments.get('src')
        dest = arguments.get('dest')
        # To handle None, we first assign the return value to result
        result = self.__get_ticket_info([src, dest])
        if result is not None:
            # Now it unpacks tuple into price and distance
            price, distance = result
            print(f"price is {price} and distance is {distance}")
        else:
            price, distance = None, None

        # Even if price and distance are None, we still need to send the response to gpt because it is waiting for our reply
        response = {
            "role": "tool",
            "content": json.dumps({"source_city": src, "destination_city": dest, "price": price, "distance": distance}),
            "tool_call_id": tool_call.id
        }
        
        return result is not None, src, dest, response

    def __get_ticket_info(self, ticket_info_list):
        lowercase_tuple = tuple(item.lower() for item in ticket_info_list)
        return AirlineAssistant.TICKET_INFO.get(lowercase_tuple)

    def __generate_image(self, source, dest):
        image_response = self.openai.images.generate(
                model="dall-e-3",
                prompt=f"An image representing a trip from {source} to {dest}, \
                Please generate an image involving {source} and {dest} in a vibrant pop-art style",
                size="1024x1024",
                n=1,
                response_format="b64_json",
            )
        image_base64 = image_response.data[0].b64_json
        image_data = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_data))

    def __talk(self, message):
        response = self.openai.audio.speech.create(
          model="tts-1",
          voice="alloy",    # Also, try replacing onyx with alloy
          input=message
        )
        
        audio_stream = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        play(audio)
    

# Function to handle closing the interface
# The 2nd parameter is frame which is required by the signal handler signature
# Since it's not used, use _ instead
def handle_exit_signal(signal, _):
    print("Exiting gradio!")
    gr.close_all()
    sys.exit(0)  # Exit the program

if __name__ == '__main__':
    # Load environment variables in a file called .env
    load_dotenv(override=True)
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print("OpenAI API Key not set")

    # Register the signal handler for KeyboardInterrupt (Ctrl+C)
    signal.signal(signal.SIGINT, handle_exit_signal)

    airline_assistant = AirlineAssistant()

    # sys.argv[0] is the script name
    if len(sys.argv) > 1 and sys.argv[1] == 'multi-modal':
        airline_assistant.run_multi_modal()
    else:
        airline_assistant.run()
