import os
import io
import signal
import sys
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import gradio as gr
import subprocess
from huggingface_hub import login, InferenceClient
from transformers import AutoTokenizer


# This class is intended to generate cpp codes given python codes
# It is also able to run the given python codes and generated cpp codes
class CodeGenerator:

    def __init__(self):
        self.system_prompt = "You are an assistant that reimplements Python code in high performance C++ for an M1 Mac. \
            Respond only with C++ code; use comments sparingly and do not provide any explanation other than occasional comments. \
            The C++ response needs to produce an identical output in the fastest possible time."
        self.openai = OpenAI()
        self.claude = anthropic.Anthropic()
        # Login to HF. Somehow it is required to pass in the hf_token explicitly.
        login(os.getenv("HF_TOKEN"), add_to_git_credential=True)
        self.openai_model = "gpt-4o"
        self.claude_model = "claude-3-5-sonnet-20240620"
        self.qwen_model = "Qwen/CodeQwen1.5-7B-Chat"
        self.qwen_model_endpoint_url = "https://fj8m0q4lpx5s7udv.us-east-1.aws.endpoints.huggingface.cloud"

    def run(self):
        css = """
            .python {background-color: #a2c9e8;}
            .cpp {background-color: #c0f0c0;}
            """
        with gr.Blocks(css=css) as ui:
            with gr.Row():
                python = gr.TextArea(label="Enter your python code:")
                cpp = gr.TextArea(label="The converted cpp code:")
            with gr.Row():
                model = gr.Dropdown(["GPT", "Claude", "Qwen"], label="Select model", value="GPT")
                convert = gr.Button("Convert Python to C++")
            with gr.Row():
                run_python = gr.Button("Run Python")
                run_cpp = gr.Button("Run C++")
            with gr.Row():
                python_outputs = gr.TextArea(label="Python outputs:", elem_classes=["python"])
                cpp_outputs = gr.TextArea(label="C++ outputs:", elem_classes=["cpp"])

            convert.click(self.__convert, inputs=[python, model], outputs=[cpp])
            run_python.click(self.__execute_python, inputs=[python], outputs=[python_outputs])
            run_cpp.click(self.__execute_cpp, inputs=[cpp, model], outputs=[cpp_outputs])


        ui.launch(inbrowser=True)

    def __convert(self, python, model):
        if model=="GPT":
            result = self.__gpt_generator(python)
        elif model=="Claude":
            result = self.__claude_generator(python)
        elif model == "Qwen":
            result = self.__qwen_generator(python)
        else:
            raise ValueError("Unknown model")
        for stream_so_far in result:
            yield stream_so_far
    
    def __user_prompt(self, python):
        user_prompt = "Rewrite this Python code in C++ with the fastest possible implementation that produces identical output in the least time. "
        user_prompt += "Respond only with C++ code; do not explain your work other than a few comments. "
        user_prompt += "Pay attention to number types to ensure no int overflows. Remember to #include all necessary C++ packages such as iomanip.\n\n"
        user_prompt += python
        return user_prompt

    # write the converted cpp to a file model_optimized.cpp
    def __write_output_file(self, cpp, model):
        # The returned cpp from LLM are wrapped inside ```cpp and ```, so we need to remove them.
        code = cpp.replace("```cpp","").replace("```","")
        cpp_file_name = model + "_optimized.cpp"
        with open(cpp_file_name, "w") as f:
            f.write(code)
        return cpp_file_name

    def __gpt_generator(self, python):
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.__user_prompt(python)}]
        stream = self.openai.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            stream=True)
        responses = ""
        for chunk in stream:
            # We need to send accumulative responses back
            # Because most clients like Gradio don't keep track of history
            responses += chunk.choices[0].delta.content or ''
            yield responses

    def __qwen_generator(self, python):
        tokenizer = AutoTokenizer.from_pretrained(self.qwen_model)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.__user_prompt(python)}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Somehow need to pass in the hf_token explicitly
        client = InferenceClient(self.qwen_model_endpoint_url, token=os.getenv("HF_TOKEN"))
        stream = client.text_generation(text, stream=True, details=True, max_new_tokens=3000)
        result = ""
        for r in stream:
            result += r.token.text
            yield result

    def __claude_generator(self, python):
        result = self.claude.messages.stream(
            model=self.claude_model,
            system=self.system_prompt,
            messages=[{"role": "user", "content": self.__user_prompt(python)}],
            max_tokens=800)
        responses = ""
        with result as stream:
            for text in stream.text_stream:
                responses += text
                yield responses

    def __execute_python(self, code):
        try:
            # We want to capture the output to the stdout, so we need to redirect sys.stdout to a StringIO buffer
            output = io.StringIO()
            sys.stdout = output
            # By passing a dictionary (global_scope) to exec(), it ensures that all function definitions persist
            # Otherwise, compiler will complain some functions from code such as lcg is not defined.
            global_scope = {}
            exec(code, global_scope)
        finally:
            # Restoring stdout to console
            sys.stdout = sys.__stdout__
        return output.getvalue()

    # Use subprocess to compile and execute C++ code for M1 Mac version
    def __execute_cpp(self, code, model):
        cpp_file_name = self.__write_output_file(code, model)
        cpp_binary_name = os.path.splitext(cpp_file_name)[0]
        try:
            compile_cmd = ["clang++", "-Ofast", "-std=c++17", "-march=armv8.5-a", "-mtune=apple-m1", "-mcpu=apple-m1", "-o", cpp_binary_name, cpp_file_name]
            compile_result = subprocess.run(compile_cmd, check=True, text=True, capture_output=True)
            run_cmd = ["./" + cpp_binary_name]
            run_result = subprocess.run(run_cmd, check=True, text=True, capture_output=True)
            return run_result.stdout
        except subprocess.CalledProcessError as e:
            return f"An error occurred:\n{e.stderr}"
    


# Function to handle closing the interface
# The 2nd parameter is frame which is required by the signal handler signature
# Since it's not used, use _ instead
def handle_exit_signal(signal, _):
    print("Exiting gradio!")
    gr.close_all()
    sys.exit(0)  # Exit the program


if __name__ == "__main__":
    # Load environment variables in a file called .env into os.environ
    load_dotenv(override=True)
    openai_api_key = os.getenv('OPENAI_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    hf_token = os.getenv('HF_TOKEN')

    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print("OpenAI API Key not set")

    if anthropic_api_key:
        print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
    else:
        print("Anthropic API Key not set")

    if hf_token:
        print(f"Hugging face token is found")
    else:
        print("Hugging face token not set")

    # Register the signal handler for KeyboardInterrupt (Ctrl+C)
    signal.signal(signal.SIGINT, handle_exit_signal)

    code_generator = CodeGenerator()

    code_generator.run()
    
