import os
from typing import List

import google.generativeai as genai
from google import genai as gog_genai
from openai import OpenAI
from together import Together


class GenLLM(object):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_model_name(self):
        if '/' in self.model_name:
            return self.model_name.split('/')[-1]
        return self.model_name

    def run_model_chat(self, prompt: str):
        raise TypeError("Not supported")

    def clear(self):
        raise TypeError("Not supported")

    def run_model(self, prompt: str):
        raise TypeError("Not supported")


class GeminiModel(GenLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        print("Using Gemini model-", model_name)
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)
        self.model_info = genai.get_model(model_name)
        print(f"{self.model_info.input_token_limit=}")
        print(f"{self.model_info.output_token_limit=}")

    def run_model(self, prompt: str):
        # print('prompt text:\n', _prompt)
        # print("prompt tok count: ", self.model.count_tokens(_prompt))
        response = self.model.generate_content(prompt)
        # print('response text', response.text)
        # print("response tok count: ", self.model.count_tokens(response.text))
        return response.text

    def run_model_chat(self, prompt: str):
        raise TypeError("Gemini model does not support chat mode, use GeminiChatModel instead")


class GeminiChatModel(GenLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        print("Using Gemini model-", model_name)
        self.client = gog_genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = self.client.chats.create(model="gemini-2.0-flash")

    def run_model_chat(self, prompt: str):
        response = self.model.send_message(prompt)
        return response.text

    def clear(self):
        self.model = self.client.chats.create(model="gemini-2.0-flash")


class TogetherModel(GenLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        print("Using Llama model-", model_name)
        self.model = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.messages = []

    def run_model_chat(self, prompt: str):
        self.messages.append({"role": "user", "content": prompt})
        response = self.run_together(self.model, self.model_name, self.messages)
        self.messages.append({"role": "assistant", "content": response})
        return response

    def run_model(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        return self.run_together(self.model, self.model_name, messages)

    @staticmethod
    def run_together(model, model_name, messages: List):
        # print(f"Prompt: {_prompt}")

        stream = model.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=16000,
            stream=True,
        )

        response = []
        for chunk in stream:
            try:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content or "", end="", flush=True)
                    response.append(chunk.choices[0].delta.content)
            except Exception as e:
                print("Error in chunk:", repr(e))
                continue

        return "".join(response)

    def clear(self):
        self.messages = []


class GPTModel(GenLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        print("Using GPT model-", model_name)
        self.model = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.messages = []

    def run_model(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        return self.run_gpt(self.model, self.model_name, messages)

    def run_model_chat(self, prompt: str):
        self.messages.append({"role": "user", "content": prompt})
        response = self.run_gpt(self.model, self.model_name, self.messages)
        self.messages.append({"role": "assistant", "content": response})
        return response

    @staticmethod
    def run_gpt(model, mname, messages):
        response = model.chat.completions.create(
            model=mname,
            messages=messages,
        )

        response_content = response.choices[0].message.content
        return response_content


    def clear(self):
        self.messages = []