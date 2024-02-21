"""
This module contains the ChatPersona class which defines a customizable Chat AI experience
based on the "CustomChatPersonas" section defined in the ChatPersoanaConfig.ini file.
"""

from typing import Any, Dict

from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAI

from chat_config import ChatPersonaConfig


class ChatPersona:
    """
    Class defining a customized Chat Persona which interacts with OpenAI LLMs via LangChain
    """

    def __init__(self, config: ChatPersonaConfig, persona: str) -> None:
        self._config = config
        self._persona = persona
        self._prompt_text = config.persona_prompt_text(persona)
        self._description = config.persona_description(persona)
        self._verbose = config.persona_verbose(persona)
        temp = config.persona_temperature(persona)
        self._llm = OpenAI(
            model=config.llm_model_name,
            temperature=temp if 0.0 <= temp <= 1.0 else config.llm_temperature,
        )
        self._chat = ChatOpenAI()

    def chat_loop(self) -> None:
        """
        Invokes an interactive chat loop for this persona for all interactions
        """
        print(f"**** Starting up the '{self._persona}' Chat Persona ****")
        print(f"**** {self._description} ****\n")
        if self._verbose is True:
            print(f"**** Configured instruction prompt is:\n{self._prompt_text}\n")
        print("Type 'q' to quit\n")

        while True:
            msg = input("--> ")
            if msg == "q":
                break
            text_resp = self.ask_llm_with_prompt(msg)
            print(f"\nString response from OpenAI LLM prompt (no memory): {text_resp}")
            dict_resp = self.ask_llm_using_chain(msg)
            print(
                f"\nDictionary response from OpenAI LLM chain (with memory):\n{dict_resp}"
            )
            msg_resp = self.ask_with_chat(msg)
            print(f"\nBaseMessage response from ChatOpenAI:\n{msg_resp}")

    def ask_llm_with_prompt(self, inquiry: str) -> str:
        """
        Prompts the LLM to respond to specific a persona based inquiry.
        Does NOT use a chain or maintain conversation history
        """
        # Remove chat_history from our prompt since this invokation has no memory
        template = PromptTemplate(
            template=self._prompt_text.replace('Chat History: {chat_history}',''),
            input_variables=["question"],
        )
        response = self._llm.invoke(template.format(question=inquiry))
        return response

    def ask_llm_using_chain(self, inquiry: str) -> Dict[str, Any]:
        """
        Uses a chain to prompt the LLM to respond to a persona specific inquiry.

        """
        template = PromptTemplate(
            template=self._prompt_text, input_variables=["chat_history", "question"]
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="question", return_messages=True
        )
        chain = LLMChain(
            llm=self._llm, prompt=template, memory=memory, verbose=self._verbose
        )
        response = chain.invoke({"question": inquiry})
        return response

    def ask_with_chat(self, question: str) -> BaseMessage:
        """
        Uses the ChatOpenAI object to send an inquiry and return a message
        """
        instructions = SystemMessage(content=self._prompt_text)
        response = self._chat.invoke([instructions, HumanMessage(content=question)])
        return response
