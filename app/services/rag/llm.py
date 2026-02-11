from typing import List
from mistralai import Mistral, Messages, MessagesTypedDict
from mistralai.models import UserMessage, SystemMessage
from app.services.rag.prompts import INITIAL_PROMPT, REFINE_PROMPT_TEMPLATE

class LLMService:
    def __init__(self, client: Mistral, model: str, max_tokens: int):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def _call_llm(self, messages: List[Messages] | List[MessagesTypedDict]) -> str | None:
        chat_response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens
        )
        if chat_response.choices and isinstance(chat_response.choices[0].message.content, str):
            return chat_response.choices[0].message.content
        return None

    def get_initial_answer(self, question: str, context: str) -> str | None:
        messages = [
            SystemMessage(content=INITIAL_PROMPT),
            UserMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        ]
        return self._call_llm(messages)

    def get_refined_answer(self, question: str, existing_answer: str, new_context: str) -> str | None:
        refine_prompt = REFINE_PROMPT_TEMPLATE.format(question=question, existing_answer=existing_answer)
        messages = [
            SystemMessage(content=refine_prompt),
            UserMessage(content=f"New Context:\n{new_context}")
        ]
        return self._call_llm(messages)
