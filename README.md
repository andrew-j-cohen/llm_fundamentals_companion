# Chat Personas using LLMs and Neo4j

A simple harness for interacting with OpenAI LLMs and Neo4j via LangChain.

This is a companion for the sample code in the free [Neo4j && LLM Fundamentals course](https://graphacademy.neo4j.com/courses/llm-fundamentals/).

The examples here build on the code samples in this course to provide a configuration based OOP implementation with customizable "Persona" classes.

This allows for easy creation, modification and testing of different chat personas via configuration (i.e., INI file) without having to change Python code for every behavioral LLM change (i.e, temperature, prompt template, etc.)

## Executing
Modify the chat_config.ini to:
1. Set your [OpenAI API Key](https://platform.openai.com/api-keys).
2. Review the 3 sample chat personas in the "CustomChatPersonas" section. Modify these settings between runs to test responses with different LLM and prompt settings. Add your own as well! :) 
4. Execute main.py

## Notes
1. The custom personas do not interact with Neo4j (currently)
2. The movie persona requires a pre-populated Neo4j movies database only if "vector_search_enabled" is true in chat_config.ini.
   - A sandbox database can be created when you register and take [Neo4j && LLM Fundamentals course](https://graphacademy.neo4j.com/courses/llm-fundamentals)

## Enhancements
Are most welcome! :)
