# %%
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Callable, Iterable, Iterator, Protocol

import tiktoken


# %%
class MessageTemplate(Protocol):
    content: str

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        ...

    def get_role(self) -> str:
        ...

class FunctionTemplate(Protocol):
    name: str
    description: str
    params: dict[str, Any] = {}

    @property
    def full_params(self) -> dict[str, Any]:
        ...

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        ...

class ChatModels(Enum):
    GPT4 = "gpt-4"
    GPT4_0613 = "gpt-4-0613"
    GPT4_32K = "gpt-4-32k"
    GPT4_32K0613 = "gpt-4-32k-0613"
    GPT35 = "gpt-3.5-turbo"
    GPT35_0613 = "gpt-3.5-turbo-0613"
    GPT35_16K = "gpt-3.5-turbo-16k"
    GPT35_16K0613 = "gpt-3.5-turbo-16k-0613"

@dataclass
class ChatPrompt:
    model: ChatModels
    messages: list[MessageTemplate] = field(default_factory=list)
    functions: list[FunctionTemplate] = field(default_factory=list)

    @property
    def encoding(self) -> tiktoken.Encoding:
        return tiktoken.encoding_for_model(self.model.value)
    
    @property
    def token_count(self) -> int:
        num_tokens = 0
        for message in self.messages:
            num_tokens += len(self.encoding.encode(message.get_role()))
            num_tokens += len(self.encoding.encode(message.content))
            num_tokens += 4 if message.get_role() else 3 # every content (3 tokens) and name (1 token)
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def register_example(self, fewshot: Iterable[MessageTemplate]) -> None:
        self.messages.extend(fewshot)

    def register_message(self, message: MessageTemplate) -> None:
        self.messages.append(message)

    def register_function(self, func: FunctionTemplate) -> None:
        self.functions.append(func)

    def populate(self, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if self.functions:
            return partial(fn, model=self.model.value, messages=list(map(dict, self.messages)), functions=list(map(dict, self.functions)))
        return partial(fn, model=self.model.value, messages=list(map(dict, self.messages)))


