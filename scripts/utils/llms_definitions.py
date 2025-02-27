import os
import google.generativeai as genai
from openai import OpenAI
from together import Together


def llama_8b(_prompt):
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    print(f"Prompt: {_prompt}")

    stream = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
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


def get_gpt(model, _prompt, messages):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if messages is None:
        messages = [
            {
                "role": "user",
                "content": _prompt
            },
        ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    response_content = response.choices[0].message.content
    return response_content


def gpt4_turbo(_prompt, messages=None):
    return get_gpt("gpt-4-turbo", _prompt, messages)


def gpt4o_mini(_prompt, messages=None):
    return get_gpt("gpt-4o-mini", _prompt, messages)


def gpt4o(_prompt, messages=None):
    return get_gpt("gpt-4o", _prompt, messages)


def gpt4(_prompt, messages=None):
    return get_gpt("gpt-4", _prompt, messages)


def gpt3_5(_prompt, messages=None):
    return get_gpt("gpt-3.5-turbo", _prompt, messages)


def run_gemini_pro(_prompt):
    global gemini_pro_model
    if gemini_pro_model is None:
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        gemini_pro_model = genai.GenerativeModel('gemini-1.5-pro')
        model_info = genai.get_model("models/gemini-1.5-pro")
        print(f"{model_info.input_token_limit=}")
        print(f"{model_info.output_token_limit=}")

    # for m in genai.list_models():
    #     if 'generateContent' in m.supported_generation_methods:
    #         print(m.name)
    print('prompt text:\n', _prompt)
    print("prompt: ", gemini_pro_model.count_tokens(_prompt))

    response = gemini_pro_model.generate_content(_prompt)
    print("response: ", gemini_pro_model.count_tokens(response.text))
    return response.text


def run_gemini_flash(_prompt):
    global gemini_flash_model
    if gemini_flash_model is None:
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        gemini_flash_model = genai.GenerativeModel('gemini-1.5-flash')
        model_info = genai.get_model("models/gemini-1.5-flash")
        print(f"{model_info.input_token_limit=}")
        print(f"{model_info.output_token_limit=}")

    print('prompt text:\n', _prompt)
    print("prompt: ", gemini_flash_model.count_tokens(_prompt))
    response = gemini_flash_model.generate_content(_prompt)
    print("response: ", gemini_flash_model.count_tokens(response.text))
    return response.text
