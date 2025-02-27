import json


class Agent(object):
    def __init__(self):
        pass


class GPTAgent(Agent):
    def __init__(self, llm_client):
        super().__init__()
        self._llm_client = llm_client
        self._messages = list()

    def get_messages(self):
        return self._messages

    def add_message_from_instruct(self, instruct):
        self._messages.append({
            "role": "user",
            "content": instruct
        })

    def call_llm(self):
        try:
            response = self._llm_client(None, self._messages)
            self._messages.append({"role": "assistant", "content": response})
        except Exception as e:
            response = {"Generation Failed"}

        return response


class GPTAgentSimulator(GPTAgent):
    def __init__(self):
        super().__init__(None)
        self._message_idx = 1
        self._messages = json.load(open("data/my_data/expr/sim/155d4_sim.json"))

    def call_llm(self):
        return self.get_response_from_sim()

    def get_response_from_sim(self):
        resp = self._messages[self._message_idx]['content']
        self._message_idx += 2
        return resp

    def add_message_from_instruct(self, instruct):
        pass
