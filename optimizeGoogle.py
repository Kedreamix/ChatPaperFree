"""
A simple wrapper for the official Google GeminiPro API
"""
import json
import os
import threading
import time
import requests
import tiktoken
from typing import Generator
from queue import PriorityQueue as PQ
import json
import os
import time
import google.generativeai as genai
ENCODER = tiktoken.get_encoding("gpt2")
class chatPaper:
    """
    Official Google API
    """
    def __init__(
        self,
        api_keys: list,
        proxy = None,
        api_proxy = None,
        max_tokens: int = 4000,
        temperature: float = 0.5,
        top_p: float = 1.0,
        model_name: str = "gemini-Pro",
        reply_count: int = 1,
        system_prompt = "You are ChatPaper, A paper reading bot",
        lastAPICallTime = time.time()-100,
        apiTimeInterval = 20,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.apiTimeInterval = apiTimeInterval
        self.session = requests.Session()
        self.api_keys = PQ()
        for key in api_keys:
            self.api_keys.put((lastAPICallTime,key))
        self.proxy = proxy
        if self.proxy:
            proxies = {
                "http": self.proxy['http'],
                "https": self.proxy['https'],
            }
            os.environ["http_proxy"] = self.proxy['http']
            os.environ["https_proxy"] = self.proxy['https']
            self.session.proxies = proxies
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reply_count = reply_count
        self.decrease_step = 250
        self.conversation = {}
        if self.token_str(self.system_prompt) > self.max_tokens:
            raise Exception("System prompt is too long")
        self.lock = threading.Lock()
        
    def get_api_key(self):
        with self.lock:
            apiKey = self.api_keys.get()
            delay = self._calculate_delay(apiKey)
            time.sleep(delay)
            self.api_keys.put((time.time(), apiKey[1]))
            return apiKey[1]

    def _calculate_delay(self, apiKey):
        elapsed_time = time.time() - apiKey[0]
        if elapsed_time < self.apiTimeInterval:
            return self.apiTimeInterval - elapsed_time
        else:
            return 0

    def add_to_conversation(self, message: str, role: str, convo_id: str = "default"):
        if(convo_id not in self.conversation):
            self.reset(convo_id)
        self.conversation[convo_id].append({"role": role, "parts": [message]})

    def add_to_conversation_last(self, message: str, role: str, convo_id: str = "default"):
        if(convo_id not in self.conversation):
            self.reset(convo_id)
        self.conversation[convo_id][-1]["parts"] += message

    def __truncate_conversation(self, convo_id: str = "default"):
        """
        Truncate the conversation
        """
        last_dialog = self.conversation[convo_id][-1]
        query = str(last_dialog['parts'][0])
        if(len(ENCODER.encode(str(query)))>self.max_tokens):
            query = query[:int(1.5*self.max_tokens)]
        while(len(ENCODER.encode(str(query)))>self.max_tokens):
            query = query[:self.decrease_step]
        self.conversation[convo_id] = self.conversation[convo_id][:-1]
        # print(self.conversation)
        full_conversation = "\n".join([str(x["parts"][0]) for x in self.conversation[convo_id]],)
        if len(ENCODER.encode(full_conversation)) > self.max_tokens:
            self.conversation_summary(convo_id=convo_id)
        full_conversation = ""
        for x in self.conversation[convo_id]:
            full_conversation = str(x["parts"][0]) + "\n" + full_conversation
        while True:
            if (len(ENCODER.encode(full_conversation+query)) > self.max_tokens):
                query = query[:self.decrease_step]
            else:
                break
        # print(last_dialog)
        last_dialog['parts'][0] = str(query)
        self.conversation[convo_id].append(last_dialog)

    def ask_stream(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        **kwargs,
    ) -> Generator:
        if convo_id not in self.conversation:
            self.reset(convo_id=convo_id)
        self.add_to_conversation_last(prompt, "user", convo_id=convo_id)
        self.__truncate_conversation(convo_id=convo_id)
        
        genai.configure(api_key=self.get_api_key())

        model = genai.GenerativeModel('gemini-pro')

        response = model.generate_content(self.conversation[convo_id],
                                          generation_config=genai.types.GenerationConfig(
                                            # Only one candidate for now.
                                            candidate_count=kwargs.get("n", self.reply_count),
                                            # stop_sequences=['x'],
                                            max_output_tokens=self.max_tokens,
                                            temperature=kwargs.get("temperature", self.temperature)),
                                          stream = True)
        # try:
        #     response.text
        # except Exception as e:
        #     print(f"Exception: {e}")
        #     raise Exception(
        #         f"Gemini Error",
        #     )
        
        for line in response:    
            yield line.text

    def ask(self, prompt: str, role: str = "user", convo_id: str = "default", **kwargs):
        """
        Non-streaming ask
        """
        response = self.ask_stream(
            prompt=prompt,
            role=role,
            convo_id=convo_id,
            **kwargs,
        )
        full_response: str = "".join(response)
        self.add_to_conversation(full_response, role, convo_id=convo_id)
        usage_token = self.token_str(prompt)
        com_token = self.token_str(full_response)
        total_token = self.token_cost(convo_id=convo_id)
        return full_response, usage_token, com_token, total_token

    def check_api_available(self):
        url = f"https://generativelanguage.googleapis.com/v1beta3/models/text-bison-001:generateText?key={self.get_api_key()}"

        headers = {
            'Content-Type': 'application/json',
        }

        data = {
            'prompt': {
                'text': "hello",
            },
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            print(response)
        else:
            print(False)
        if response.status_code == 200:
            return True
        else:
            return False
       
    def reset(self, convo_id: str = "default", system_prompt = None):
        """
        Reset the conversation
        """
        self.conversation[convo_id] = [
            {"role": "user", "parts": [str(system_prompt or self.system_prompt)]},
        ]
        
    def conversation_summary(self, convo_id: str = "default"):
        input = ""
        role = ""
        for conv in self.conversation[convo_id]:
            if (conv["role"]=='user'):
                role = 'User'
            else:
                role = 'Gemini'
            input+=role+' : '+conv['content']+'\n'
        prompt = "Your goal is to summarize the provided conversation in English. Your summary should be concise and focus on the key information to facilitate better dialogue for the large language model.Ensure that you include all necessary details and relevant information while still reducing the length of the conversation as much as possible. Your summary should be clear and easily understandable for the ChatGpt model providing a comprehensive and concise summary of the conversation."
        if(self.token_str(str(input)+prompt)>self.max_tokens):
            input = input[self.token_str(str(input))-self.max_tokens:]
        while self.token_str(str(input)+prompt)>self.max_tokens:
            input = input[self.decrease_step:]
        prompt = prompt.replace("{conversation}", input)
        self.reset(convo_id='conversationSummary')
        response = self.ask(prompt,convo_id='conversationSummary')
        while self.token_str(str(response))>self.max_tokens:
            response = response[:-self.decrease_step]
        self.reset(convo_id='conversationSummary',system_prompt='Summariaze our diaglog')
        self.conversation[convo_id] = [
            {"role": "user", "parts": [self.system_prompt]},
            {"role": "user", "parts": ["Summariaze our diaglog"]},
            {"role": 'model', "parts": [response]},
        ]
        return self.conversation[convo_id]
    def token_cost(self,convo_id: str = "default"):
        return len(ENCODER.encode("\n".join([x["parts"][0] for x in self.conversation[convo_id]])))
    def token_str(self,content:str):
        return len(ENCODER.encode(content))
def main():
    return
