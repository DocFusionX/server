from typing import List
from mistralai import Mistral, Messages, MessagesTypedDict
from mistralai.models import UserMessage, SystemMessage
from app.services.rag.prompts import MAP_PROMPT, REDUCE_PROMPT, HYDE_PROMPT

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

    def generate_hypothetical_answer(self, question: str) -> str | None:
        messages = [
            SystemMessage(content=HYDE_PROMPT),
            UserMessage(content=question)
        ]
        return self._call_llm(messages)

    def generate_answer_map_reduce(self, question: str, contexts: List[str]) -> str | None:
        individual_answers = []
        for context in contexts:
            map_prompt_formatted = MAP_PROMPT.format(question=question, context=context)
            messages = [
                SystemMessage(content="You are an expert Q&A system."),
                UserMessage(content=map_prompt_formatted)
            ]
            answer = self._call_llm(messages)
            if answer:
                individual_answers.append(answer)

        if not individual_answers:
            return "Could not generate any answers from the provided documents."

        answers_str = "\n---\n".join(individual_answers)
        reduce_prompt_formatted = REDUCE_PROMPT.format(question=question, answers=answers_str)
        messages = [
            SystemMessage(content="You are an expert Q&a system."),
            UserMessage(content=reduce_prompt_formatted)
        ]

        return self._call_llm(messages)
