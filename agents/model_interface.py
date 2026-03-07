import os
import re
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv

load_dotenv()


class BaseModel(ABC):
    @abstractmethod
    def generate(self, system: str, user: str, temperature: float = 0.7) -> str:
        raise NotImplementedError


class StubModel(BaseModel):
    _counter = 0

    @classmethod
    def reset_counter(cls):
        cls._counter = 0

    def generate(self, system: str, user: str, temperature: float = 0.7) -> str:
        StubModel._counter += 1

        text = user.replace(",", " ")
        m = re.search(r"what is\s+(\d+)\s*([+\-*/])\s*(\d+)", text, re.I)
        if m:
            a = int(m.group(1))
            op = m.group(2)
            b = int(m.group(3))

            if op == "+":
                ans = a + b
            elif op == "-":
                ans = a - b
            elif op == "*":
                ans = a * b
            else:
                ans = a / b if b != 0 else "undefined"

            return (
                f"Step 1: Identify the operation.\n"
                f"Step 2: Compute {a} {op} {b} = {ans}.\n"
                f"The answer is {ans}."
            )

        return "Step 1: Analyze the problem.\nStep 2: Solve carefully.\nThe answer is A."


class DartmouthChatModel(BaseModel):
    """
    Dartmouth Chat API wrapper using OpenAI-style chat completions over HTTP.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_url = os.getenv(
            "DARTMOUTH_CHAT_API_URL",
            "https://chat.dartmouth.edu/api/chat/completions",
        )
        self.api_key = os.getenv("DARTMOUTH_CHAT_API_KEY")

        if not self.api_key:
            raise ValueError(
                "DARTMOUTH_CHAT_API_KEY is missing. Add it to your .env file."
            )

        bad = [(i, ch, hex(ord(ch))) for i, ch in enumerate(self.api_key) if ord(ch) > 127]
        if bad:
            raise ValueError(
                f"DARTMOUTH_CHAT_API_KEY contains non-ASCII characters: {bad[:5]}"
            )

    def generate(self, system: str, user: str, temperature: float = 0.7) -> str:
        resp = requests.post(
            self.api_url,
            headers={
                "Authorization": f"bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "stream": False,
            },
            timeout=120,
        )

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(
                f"Dartmouth Chat API request failed: {resp.status_code} {resp.text}"
            ) from e

        data = resp.json()

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected Dartmouth API response format: {data}") from e


def load_model(model_name: str) -> BaseModel:
    if model_name == "stub":
        return StubModel()

    return DartmouthChatModel(model_name=model_name)