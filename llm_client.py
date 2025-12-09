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
        limiter_kwargs: dict | None = None
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.endpoint_url, max_retries=0)
        self._tools = tools or []
        self.tool_choice_mode = tool_choice_mode
        
        self._limiter_settings = self.DEFAULT_LIMITER_SETTINGS
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
    
    def _set_limiter(self):
        limiter = RateLimiter(**self._limiter_settings)
        # Get undecorated _chat function and wrap it
        original_chat = self.__class__._chat
        self._chat = limiter(lambda **kw: original_chat(self, **kw))

    def set_limiter(self, limiter_kwargs: dict | None = None):
        self._limiter_settings.update(limiter_kwargs or self.DEFAULT_LIMITER_SETTINGS)
        self._set_limiter()

    def get_limiter_settings(self):
        return self._limiter_settings
    
    @staticmethod
    def tool_from_pydantic(name: str, description: str, data_model: type[BaseModel]) -> dict:
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": data_model.model_json_schema()
            }
        }

        return tool

    def add_tool_from_pydantic(self, name: str, description: str, data_model: type[BaseModel]):
        self._tools.append(ChatClient.tool_from_pydantic(name=name, description=description, data_model=data_model))

    def get_tools(self):
        return self._tools
    
    def _chat(self, **kwargs):
        if self._tools:
            kwargs.update({"tools": self._tools, "tool_choice": self.tool_choice_mode})

        response = self.client.chat.completions.create(model=self.model_name, **kwargs)
        return response
    
    def chat(self, messages, **kwargs):
        return self._chat(messages=messages, **kwargs)
    

class OpenAIChatClient(ChatClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def chat(self, messages: list[dict], reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = "medium", **kwargs):
        if reasoning_effort is not None:
            kwargs.update({"reasoning_effort": reasoning_effort})

        return self._chat(messages=messages, **kwargs)


class AnthropicChatClient(ChatClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def chat(self, messages: list[dict], thinking_enabled: bool = True, thinking_tokens: int = 1024, **kwargs):
        if thinking_enabled:
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

        if self._tools:
            for tool in self._tools:
                tool["function"]["parameters"].pop("title", None)

        return self._chat(messages=messages, **kwargs)