import time
from typing import Any

import openai
from openai import OpenAI

from ..types import MessageList, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)

import random

class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        openai_parameters: dict | None = None,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        if openai_parameters is not None:
            if openai_parameters.get("multi_ollma", False):
                api_key = openai_parameters.get("api_key", None)
                base_url_list = openai_parameters.get("base_url", None)
                self.client = [OpenAI(base_url=base_url, api_key=api_key) for base_url in base_url_list]
            else:
                self.client = OpenAI(**openai_parameters)
        else:
            self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                if isinstance(self.client, list):
                    client = random.choice(self.client)
                    # print(f"Using client {client.base_url} for sampling")
                else:
                    client = self.client
                response = client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception

from transformers import pipeline
import torch


class ChatCompletionSamplerLocal(SamplerBase):
    """
    Sample locally by transformers model
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        system_message: str | None = None,
        # temperature: float = 0.5,
        max_tokens: int = 1024,
        device_map: dict = {"": 3}
    ):
        # self.api_key_name = "OPENAI_API_KEY"
        # self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.pipe = pipeline(
            "text-generation",
            model=model,
            torch_dtype="auto",
            device_map=device_map,
        )
        # self.temperature = temperature
        self.max_tokens = max_tokens
    #     self.image_format = "url"

    # def _handle_image(
    #     self,
    #     image: str,
    #     encoding: str = "base64",
    #     format: str = "png",
    #     fovea: int = 768,
    # ):
    #     new_image = {
    #         "type": "image_url",
    #         "image_url": {
    #             "url": f"data:image/{format};{encoding},{image}",
    #         },
    #     }
    #     return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list

            response = self.pipe(
                message_list,
                max_new_tokens=1024,
            )
            content = response[0]["generated_text"][-1]
            print(f"local content:{content}")
            return SamplerResponse(
                response_text=content,
                response_metadata={},
                actual_queried_message_list=message_list,
            )




        # trial = 0
        # while True:
        #     try:
        #         response = self.client.chat.completions.create(
        #             model=self.model,
        #             messages=message_list,
        #             temperature=self.temperature,
        #             max_tokens=self.max_tokens,
        #         )
        #         content = response.choices[0].message.content
        #         if content is None:
        #             raise ValueError("OpenAI API returned empty response; retrying")
        #         return SamplerResponse(
        #             response_text=content,
        #             response_metadata={"usage": response.usage},
        #             actual_queried_message_list=message_list,
        #         )
        #     # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
        #     except openai.BadRequestError as e:
        #         print("Bad Request Error", e)
        #         return SamplerResponse(
        #             response_text="No response (bad request).",
        #             response_metadata={"usage": None},
        #             actual_queried_message_list=message_list,
        #         )
        #     except Exception as e:
        #         exception_backoff = 2**trial  # expontial back off
        #         print(
        #             f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
        #             e,
        #         )
        #         time.sleep(exception_backoff)
        #         trial += 1
            # unknown error shall throw exception