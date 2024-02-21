"""
This is a companion program for the exercises in the free Neo4j && LLM Fundamentals course.

The Neo4j LLM Fundamentals course on Neo4j GraphAcademy can be found at:
    https://graphacademy.neo4j.com/courses/llm-fundamentals

The examples here build on the code samples in the course to illustrate a configuration
based OOP implementation with customizable "Persona" classes.
"""

from chat_config import ChatPersonaConfig
from custom_chat_persona import ChatPersona
from movie_chat_persona import MovieChatPersona

# Initialize global configuration for LLM, Neo4j and OpenAI
CONFIG = ChatPersonaConfig()

#
# Execute the Movie chat if configured
#
if CONFIG.movie_enabled is True:
    movie_persona = MovieChatPersona(config=CONFIG)

    if CONFIG.movie_vector_search_enabled is True:
        movie_vector_search = movie_persona.movie_vector_search()
        movie_vector_search.init_neo4j()
        
        print(f"\nMovie DB Count: {movie_vector_search.list_movies()}")

        # Run the same movie plot simliarity search with both methods
        user_query = movie_vector_search.plot_similarity_search(10)
        movie_vector_search.plot_retriever(user_query)

        # Execute the full tool chain for unlimited queries (until the user quits)
        movie_vector_search.plot_retriever_with_tools()

    movie_persona.chat_loop()

#
# Execute all enabled custom Chat Personas
#
for persona in CONFIG.custom_personas:
    custom_persona = ChatPersona(config=CONFIG, persona=persona)
    custom_persona.chat_loop()
