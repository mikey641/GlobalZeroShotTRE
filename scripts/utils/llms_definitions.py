import os
import google.generativeai as genai
from openai import OpenAI
from together import Together


class LLM:
    def __init__(self, api, model_name: str):
        self.model_name = model_name
        self.api = api.lower()
        if self.api == 'gemini':
            print("Using Gemini model-", model_name)
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self.model = genai.GenerativeModel(model_name)
            self.model_info = genai.get_model(model_name)
            print(f"{self.model_info.input_token_limit=}")
            print(f"{self.model_info.output_token_limit=}")
        elif self.api == 'gpt':
            print("Using GPT model-", model_name)
            self.model = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif self.api == 'together':
            print("Using Llama model-", model_name)
            self.model = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        else:
            raise ValueError(f"Model {model_name} not supported")

    def get_model_name(self):
        if '/' in self.model_name:
            return self.model_name.split('/')[-1]
        return self.model_name

    def run_model(self, prompt: str, messages=None):
        if self.api == 'together':
            return self.run_llama(prompt)
        elif self.api == 'gemini':
            return self.run_gemini(prompt)
        elif self.api == 'gpt':
            return self.run_gpt(self.model_name, prompt, messages)
        else:
            raise ValueError(f"Model {self.model_name} not supported")

    def run_gemini(self, _prompt):
        # print('prompt text:\n', _prompt)
        # print("prompt tok count: ", self.model.count_tokens(_prompt))
        response = self.model.generate_content(_prompt)
        # print('response text', response.text)
        # print("response tok count: ", self.model.count_tokens(response.text))
        return response.text

    def run_llama(self, _prompt):
        # print(f"Prompt: {_prompt}")

        stream = self.model.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": _prompt}],
            max_tokens=4096,
            stream=True,
        )

        response = []
        for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
            if chunk.choices[0].delta.content:
                response.append(chunk.choices[0].delta.content)

        return "".join(response)


    def run_gpt(self, model, _prompt, messages):
        if messages is None:
            messages = [
                {
                    "role": "user",
                    "content": _prompt
                },
            ]

        response = self.model.chat.completions.create(
            model=model,
            messages=messages,
        )

        response_content = response.choices[0].message.content
        return response_content
