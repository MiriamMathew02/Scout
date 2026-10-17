import json
from types import SimpleNamespace

import pytest


class FakeLLM:
    """OpenAI-shaped client; `reply(prompt)` returns the JSON object to send back."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        self.prompts.append(prompt)
        content = json.dumps(self.reply(prompt))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def fake_llm():
    return FakeLLM
