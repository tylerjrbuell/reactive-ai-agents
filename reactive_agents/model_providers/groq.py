import os
from groq import Groq, Stream
from groq.types.chat import ChatCompletion, ChatCompletionChunk
from .base import BaseModelProvider


class GroqModelProvider(BaseModelProvider):
    id = "groq"

    def __init__(self, model="llama3-groq-70b-8192-tool-use-preview"):
        self.name = __class__.id
        self.model = model
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.validate_model()
        super().__init__(model=model, name=self.name)

    def validate_model(self, **kwargs):
        supported_models = self.client.models.list().model_dump().get("data", [])
        if self.model not in [m.get("id") for m in supported_models]:
            raise Exception(f"Model {self.model} is not currently supported by Groq.")

    async def get_chat_completion(
        self,
        messages: list,
        stream: bool = False,
        tools: list | None = None,
        tool_use: bool = True,
        options: dict | None = None,
        format: str = "",
    ) -> dict | Stream[ChatCompletionChunk] | None:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                response_format=(
                    {"type": "json_object"} if format == "json" else {"type": "text"}
                ),
                tool_choice="required" if tool_use else "none",
                stream=stream,
                **(options or {}),
            )
            if type(completion) is ChatCompletion:
                result = completion.choices[0].model_dump()
                return result
            elif type(completion) is Stream[ChatCompletionChunk]:
                return completion
        except Exception as e:
            raise Exception(f"Groq Chat Completion Error: {e}")

    async def get_completion(
        self, **kwargs
    ) -> dict | Stream[ChatCompletionChunk] | None:
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        completion = client.chat.completions.create(
            model=self.model,
            tools=kwargs["tools"] if kwargs.get("tools") else None,
            messages=[
                {"role": "system", "content": kwargs["system"]},
                {"role": "user", "content": kwargs["prompt"]},
            ],
            response_format=(
                {"type": "json_object"}
                if kwargs.get("format") == "json"
                else {"type": "text"}
            ),
        )
        if type(completion) is ChatCompletion:
            result = {}
            result["response"] = (
                completion.choices[0].model_dump().get("message", {}).get("content", "")
            )
            return result
        elif type(completion) is Stream[ChatCompletionChunk]:
            return completion
