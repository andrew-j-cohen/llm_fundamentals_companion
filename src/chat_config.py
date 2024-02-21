"""
This module defines the global configuration for executing Chat Personas.
"""

import os
from configparser import ConfigParser


class ChatPersonaConfig:
    """
    A wrapper class for the global configuration which handles lookups and type conversions.
    """

    def __init__(self) -> None:
        self._config = ConfigParser()
        # Ensure that case is preserved for parameters in the ini file
        self._config.optionxform = str
        self._config.read(os.getenv("CONFIG_INI", "chat_config.ini"))
        # Ensure the API key environment variable is set for ChatOpenAPI!
        os.environ["OPENAI_API_KEY"] = self._config["LLM"]["OpenAPIKey"]

    @property
    def openai_api_key(self) -> str:
        """Returns the OpenAI API key to use for all LLM invokations"""
        return self._config["LLM"]["OpenAPIKey"]

    @property
    def llm_temperature(self) -> float:
        """Returns the temperature setting between 0.0 and 1.0 to use with the LLM.
        The default is 0.0 (i.e., well grounded)."""
        return self._config["LLM"].getfloat("Temperature", 0.0)

    @property
    def llm_model_name(self) -> str:
        """Returns the name of the model to use with the LLM.
        The default is gpt-3.5-turbo-instruct."""
        return self._config["LLM"]["Model"]

    @property
    def movie_enabled(self) -> bool:
        """Returns True if the Movie persona is enabled, otherwise False."""
        return self._config["MovieChatPersona"].getboolean("enabled")

    @property
    def movie_verbose(self) -> bool:
        """Returns True if verbose logging is enabled for the Movie persona,
        otherwise False."""
        return self._config["MovieChatPersona"].getboolean("verbose")

    @property
    def movie_agent_prompt(self) -> str:
        """Returns the name of the Agent Prompt to use for the Movie Chat Persona.
        The default is hwchase17/react-chat."""
        return self._config["MovieChatPersona"].get(
            "agent_prompt", "hwchase17/react-chat"
        )

    @property
    def movie_max_iterations(self) -> int:
        """Returns the max_iterations for the LLM to use for the Movie Chat Persona."""
        return self._config["MovieChatPersona"].getint("max_iterations")

    @property
    def movie_prompt_template(self) -> str:
        """Returns the prompt template for the LLM to use for the Movie Chat Persona."""
        return str(self._config["MovieChatPersona"]["prompt_template"])

    @property
    def movie_vector_search_enabled(self) -> bool:
        """Returns True if Neo4j vector search is enabled, otherwise False."""
        return self._config["MovieChatPersona"].getboolean("vector_search_enabled")

    @property
    def custom_personas(self) -> list[str]:
        """Returns the list of configured and enabled custom Chat Personas."""
        personas = list()
        config = self._config["CustomChatPersonas"]
        for persona in config.keys():
            name, setting = persona.split(".")
            if setting.lower() == "enabled" and config.getboolean(persona):
                personas.append(name)
        return personas

    @property
    def neo4j_config(self) -> dict[str, str]:
        """Returns the Neo4j DB configuration settings."""
        config = self._config["Neo4j"]
        return {
            "url": config.get("url", "localhost"),
            "username": config.get("username", "neo4j"),
            "password": config.get("password", None),
        }

    def persona_description(self, persona: str) -> str:
        """Returns description for the name of the custom persona."""
        return str(self._config["CustomChatPersonas"][f"{persona}.description"])

    def persona_temperature(self, persona: str) -> float:
        """Returns the LLM temperature to use for the specified custom persona.
        If set this will override the global LLM temperature."""
        return self._config["CustomChatPersonas"].getfloat(f"{persona}.temperature")

    def persona_prompt_text(self, persona: str) -> str:
        """Returns the prompt text to use for the specified custom persona."""
        return str(self._config["CustomChatPersonas"][f"{persona}.prompt_template"])

    def persona_verbose(self, persona: str) -> bool:
        """Returns True if verbose logging is enabled for the specified custom persona,
        otherwise returns False.
        """
        return self._config["CustomChatPersonas"].getboolean(f"{persona}.verbose")
