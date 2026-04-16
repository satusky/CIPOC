import json

from openai import OpenAI, RateLimitError
from pydantic import BaseModel
from typing import Literal

from limiter import RateLimiter


class ChatClient:
    DEFAULT_LIMITER_SETTINGS = {
        "max_retries": 10,
        "initial_delay": 1.0,
        "exponential_base": 2.0,
        "max_delay": 60.0,
        "jitter": True,
        "retry_on": (RateLimitError,),
        "logger_name": None,
    }

    def __init__(
        self,
        model_name: str,
        api_key: str,
        endpoint_url: str,
        tools: list[dict] | None = None,
        tool_choice_mode: Literal["auto", "required", "none"] | dict = "auto",
        limiter_kwargs: dict | Literal["default"] | None = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.endpoint_url, max_retries=0)
        self._tools = tools or []
        self.tool_choice_mode = tool_choice_mode

        self._limiter_settings: dict | None = None
        self._limiter_wrapped_chat = None
        if limiter_kwargs is not None:
            if limiter_kwargs == "default":
                limiter_kwargs = {}
            self.set_limiter(limiter_kwargs)

    def __repr__(self) -> str:
        rep = f"""
Model name: {self.model_name}
Endpoint URL: {self.endpoint_url}
Tool choice mode: {self.tool_choice_mode}
Tools: {json.dumps(self.get_tools(), indent=2)}
Limiter: {json.dumps(self._limiter_settings, indent=2)}
"""
        return rep

    def set_limiter(self, limiter_kwargs: dict | None = None):
        """Enable and configure the rate limiter. Pass None or {} to use defaults."""
        if self._limiter_settings is None:
            self._limiter_settings = self.DEFAULT_LIMITER_SETTINGS

        if limiter_kwargs:
            self._limiter_settings.update(limiter_kwargs)

        limiter = RateLimiter(**self._limiter_settings)
        original_chat = self.__class__._raw_chat
        self._limiter_wrapped_chat = limiter(lambda self, **kw: original_chat(self, **kw))

    def remove_limiter(self):
        """Disable the rate limiter."""
        self._limiter_settings = None
        self._limiter_wrapped_chat = None

    @staticmethod
    def tool_from_pydantic(name: str, description: str, data_model: type[BaseModel]) -> dict:
        schema = data_model.model_json_schema()

        # Strip fields that strict OpenAI-compatible endpoints reject:
        # - "title" at the top level and on each property
        # - missing "additionalProperties": false (required for strict mode)
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema["additionalProperties"] = False

        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema,
                "strict": True,
            }
        }

        return tool

    def add_tool_from_pydantic(self, name: str, description: str, data_model: type[BaseModel]):
        self._tools.append(ChatClient.tool_from_pydantic(name=name, description=description, data_model=data_model))

    def get_tools(self):
        return self._tools

    def _raw_chat(self, **kwargs):
        """Core chat method — always unwrapped."""
        if self._tools:
            kwargs.update({"tools": self._tools, "tool_choice": self.tool_choice_mode})
        response = self.client.chat.completions.create(model=self.model_name, **kwargs)
        return response

    def _chat(self, **kwargs):
        """Dispatches to rate-limited or raw chat depending on config."""
        if self._limiter_wrapped_chat is not None:
            return self._limiter_wrapped_chat(**kwargs)
        return self._raw_chat(**kwargs)

    def chat(self, messages, **kwargs):
        return self._chat(messages=messages, **kwargs)


class OpenAIChatClient(ChatClient):
    def __init__(self, reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = "medium", **kwargs):
        super().__init__(**kwargs)
        self._reasoning_effort = reasoning_effort

    def chat(self, messages: list[dict], reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None, **kwargs):
        reasoning_effort = reasoning_effort or self._reasoning_effort  # type: ignore
        if reasoning_effort is not None:
            kwargs.update({"reasoning_effort": reasoning_effort})

        return self._chat(messages=messages, **kwargs)


class AnthropicChatClient(ChatClient):
    def __init__(self, thinking_enabled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._thinking_enabled = thinking_enabled

    def chat(self, messages: list[dict], thinking_enabled: bool | None = None, thinking_tokens: int = 1024, **kwargs):
        if thinking_enabled or (thinking_enabled is None and self._thinking_enabled):
            if not thinking_tokens > 0:
                raise ValueError("Thinking token budget must be an integer > 0")

            thinking = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": thinking_tokens
                }
            }

            extra_body = kwargs.get("extra_body", {})
            extra_body.update(thinking)
            kwargs.update({"extra_body": extra_body})

        return self._chat(messages=messages, **kwargs)