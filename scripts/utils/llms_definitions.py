import os
from typing import List

try:
    from google import genai as google_genai
    from google.genai import types as google_genai_types
except ImportError:
    google_genai = None
    google_genai_types = None
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
    """Gemini via the google-genai SDK, backed by Vertex AI with ADC.

    Auth: Application Default Credentials (~/.config/gcloud/application_default_credentials.json).
    API keys are intentionally NOT supported (blocked by org security policy).

    Env vars (required):
      GOOGLE_CLOUD_PROJECT   — GCP project with Vertex AI API + billing enabled
      GOOGLE_CLOUD_LOCATION  — region, e.g. "us-central1" (default: us-central1)

    Thinking: `include_thoughts=True` + `thinking_budget=-1` (dynamic) exposes
    reasoning parts. `run_model` prepends any thought parts as <think>…</think>
    so downstream parse_yes_no + trace code sees the same shape as the Together
    DeepSeek-R1 path.
    """

    DEFAULT_LOCATION = "us-central1"

    def __init__(self, model_name: str):
        super().__init__(model_name)
        if google_genai is None:
            raise ImportError(
                "google-genai is not installed. `pip install google-genai`."
            )
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT not set. GeminiModel requires Vertex AI + ADC "
                "(API keys are disallowed by org policy). Set GOOGLE_CLOUD_PROJECT "
                "and optionally GOOGLE_CLOUD_LOCATION, then run "
                "`gcloud auth application-default login`."
            )
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", self.DEFAULT_LOCATION)
        print(f"Using Gemini model- {model_name} (vertexai project={project} location={location})")
        self.client = google_genai.Client(vertexai=True, project=project, location=location)
        self._chat = None

    def _thinking_config(self):
        return google_genai_types.GenerateContentConfig(
            thinking_config=google_genai_types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=-1,
            ),
        )

    @staticmethod
    def _extract(response):
        """Collapse a Gemini response's parts into (thoughts, answer)."""
        thoughts, answer = [], []
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            parts = getattr(candidates[0].content, "parts", None) or []
            for p in parts:
                text = getattr(p, "text", None) or ""
                if getattr(p, "thought", False):
                    thoughts.append(text)
                else:
                    answer.append(text)
        if not answer and not thoughts:
            # Fallback — some responses expose a flat .text
            answer.append(getattr(response, "text", "") or "")
        return "".join(thoughts), "".join(answer)

    def _wrap(self, thoughts, answer):
        if thoughts:
            return f"<think>{thoughts}</think>\n{answer}"
        return answer

    def run_model(self, prompt: str):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._thinking_config(),
        )
        thoughts, answer = self._extract(response)
        return self._wrap(thoughts, answer)

    def run_model_chat(self, prompt: str):
        if self._chat is None:
            self._chat = self.client.chats.create(
                model=self.model_name,
                config=self._thinking_config(),
            )
        response = self._chat.send_message(prompt)
        thoughts, answer = self._extract(response)
        return self._wrap(thoughts, answer)

    def clear(self):
        self._chat = None


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
            max_tokens=12000,
            stream=True,
        )

        thinking = []
        response = []
        for chunk in stream:
            try:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if getattr(delta, 'reasoning', None):
                        thinking.append(delta.reasoning)
                    if delta.content:
                        print(delta.content or "", end="", flush=True)
                        response.append(delta.content)
            except Exception as e:
                print("Error in chunk:", repr(e))
                continue

        result = "".join(response)
        if thinking:
            result = "<think>" + "".join(thinking) + "</think>\n" + result
        return result

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