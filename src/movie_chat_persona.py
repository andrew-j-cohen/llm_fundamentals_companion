"""
This module builds on the code samples in:

 https://graphacademy.neo4j.com/courses/llm-fundamentals/3-intro-to-langchain/6-retrievers/

 It defines a Movie Chat persona providing:

 1. Tools for querying movie plots and trailers.
 2. Similarity search on a Neo4j movies database with vectorized plots.

"""

from typing import Callable

import neo4j
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_community.tools import YouTubeSearchTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from chat_config import ChatPersonaConfig


class _MovieTools:
    """
    Internal tools used for Movie inquiries.
    """

    def __init__(self) -> None:
        return

    def chat_tool(self, run_func: Callable) -> Tool:
        """Return tool used for movie chat"""
        return Tool.from_function(
            name="Movie Chat",
            description="For when you need to chat about movies. \
                    The question will be a string. Return a string.",
            func=run_func,
            return_direct=True,
        )

    def trailer_tool(self, run_func: Callable) -> Tool:
        """Return tool used for movie trailer search"""
        return Tool.from_function(
            name="Movie Trailer Search",
            description="Use when needing to find a movie trailer. \
                    The question will include the word 'trailer'. \
                        Return a link to a YouTube video.",
            func=run_func,
            return_direct=True,
        )

    def plot_tool(self, run_func: Callable) -> Tool:
        """Return tool used for movie plot similarity search"""
        return Tool.from_function(
            name="Movie Plot Search",
            description="For when you need to compare a plot to a movie. \
                The question will be a string. Return a string.",
            func=run_func,
            return_direct=True,
        )

    def chat_loop(self, prompt: str, agent_executor: AgentExecutor) -> None:
        """ Invokes a chat loop with an AgentExecutor"""
        while True:
            query = input(prompt)
            if query == "q":
                break
            response = agent_executor.invoke({"input": query})
            print(response["output"])


class MovieVectorSearch:
    """
    This class adds vector search capabilities for the Movie Chat persona.
    Uses OpenAI embeddings.
    """

    def __init__(self, chat: ChatOpenAI, config: ChatPersonaConfig) -> None:
        self._chat = chat
        self._config = config
        self._tools = _MovieTools()

    def init_neo4j(self) -> None:
        """Ensures the Neo4j movies database is prepped for vector searches"""
        self._create_embeddings()
        try:
            self._create_index()
        except neo4j.exceptions.ClientError:
            # Ignore if index already exist
            pass

    def list_movies(self) -> int:
        """
        Outputs the list of movies in the Neo4j database along with the graph schema.
        Does NOT interact with the LLM.
        """
        graph = self._neo4j_graph()
        result = graph.query("""MATCH (m:Movie) RETURN m.title""")
        print(f"\nMovie Graph Schema:\n{graph.schema}")
        return len(result)

    def plot_similarity_search(self, limit: int) -> str:
        """
        Vector similarity search for movie plots in the Neo4j database.
        Does NOT interact with the LLM. Returns the original user query.
        """
        query = input(
            "\nAsk me about a movie that is similar to a named movie and \
            I'll find it with Neo4j vector search! \
            (e.g., Find movies similar to Braveheart)> "
        )
        movie_plot_vector = self._create_movie_plot_vector()
        result = movie_plot_vector.similarity_search(query=query, k=limit)
        results = "\nResults:\n"
        for doc in result:
            results += doc.metadata["title"] + "-" + doc.page_content
        print(results)
        return query

    def plot_retriever(self, query: str) -> None:
        """
        Vector similarity search for movie plots in the Neo4j database
        using a Retriever. Does NOT interact with the LLM.
        """
        movie_plot_vector = self._create_movie_plot_vector()
        plot_retriever = RetrievalQA.from_llm(
            llm=self._chat, retriever=movie_plot_vector.as_retriever()
        )
        result = plot_retriever.invoke({"query": query})
        print(f"\nHere are results for the query using a Retriever\n: {result}\n")

    def plot_retriever_with_tools(self):
        """
        Finds YouTube trailers for all movie plots in the Neo4j database.
        Interacts with the LLM.
        """
        prompt = PromptTemplate(
            template=self._config.movie_prompt_template,
            input_variables=["chat_history", "input"],
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        chat_chain = LLMChain(llm=self._chat, prompt=prompt, memory=memory)

        plot_retriever = RetrievalQA.from_llm(
            llm=self._chat,
            retriever=
                self._create_movie_plot_vector().as_retriever(),
            verbose=self._config.movie_verbose,
            return_source_documents=True,
        )

        def run_retriever(query):
            results = plot_retriever.invoke({"query": query})
            # format the results
            movies = "\n".join(
                [
                    doc.metadata["title"] + " - " + doc.page_content
                    for doc in results["source_documents"]
                ]
            )
            return movies

        tools = [
            self._tools.chat_tool(chat_chain.run),
            self._tools.trailer_tool(YouTubeSearchTool().run),
            self._tools.plot_tool(run_retriever),
        ]

        agent_prompt = hub.pull(self._config.movie_agent_prompt)
        agent = create_react_agent(self._chat, tools, agent_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            max_interations=self._config.movie_max_iterations,
            verbose=True,
            handle_parse_errors=True,
        )

        self._tools.chat_loop(
            prompt="\nAsk me anything about movies and I'll find the answer \
                with Neo4j in the LLM chain! (Type q to quit)> ",
            agent_executor=agent_executor,
        )

    def _create_movie_plot_vector(self) -> Neo4jVector:
        """
        Creates a Neo4j vector from the movie plot embeddings
        """
        neo4j_config = self._config.neo4j_config
        return Neo4jVector.from_existing_index(
            OpenAIEmbeddings(),
            url=neo4j_config["url"],
            username=neo4j_config["username"],
            password=neo4j_config["password"],
            index_name="moviePlots",
            embedding_node_property="embedding",
            text_node_property="plot",
        )

    def _create_embeddings(self) -> int:
        """Loads OpenAI embeddings into the Neo4j movies DB"""
        graph = self._neo4j_graph()
        result = graph.query(
            """LOAD CSV WITH HEADERS
        FROM 'https://data.neo4j.com/llm-fundamentals/openai-embeddings.csv'
        AS row
        MATCH (m:Movie {movieId: row.movieId})
        CALL db.create.setNodeVectorProperty(m, 'embedding', 
            apoc.convert.fromJsonList(row.embedding))
        RETURN count(*)"""
        )
        return int(result[0]["count(*)"])

    def _create_index(self) -> None:
        """Creates the movie plots index from the embedding property"""
        graph = self._neo4j_graph()
        result = graph.query(
            """CALL db.index.vector.createNodeIndex(
                'moviePlots',
                'Movie',
                'embedding',
                1536,
                'cosine'
            )"""
        )
        print(f"index: {result}")

    def _neo4j_graph(self) -> Neo4jGraph:
        neo4j_config = self._config.neo4j_config
        return Neo4jGraph(
            url=neo4j_config["url"],
            username=neo4j_config["username"],
            password=neo4j_config["password"],
        )


class MovieChatPersona:
    """
    This class implements a Movie Persona using an AgentExecutor,
    as illustrated in the Neo4j & LLM Fundamentals course:

        https://graphacademy.neo4j.com/courses/llm-fundamentals/3-intro-to-langchain/4-agents/

    Leverages the YouTubeSearch tool for finding movie trailers.
    Does not interact with Neo4j.
    """

    def __init__(self, config: ChatPersonaConfig) -> None:
        self._chat = ChatOpenAI()
        self._config = config
        self._tools = _MovieTools()

    def chat_loop(self) -> None:
        """
        This function queries movies using an agent with tools which interact
        with the LLM chain and the YouTube search tool.

        Does NOT query a Neo4j database.
        """
        self._tools.chat_loop(
            prompt="\nAsk me anything about movies and I'll find the answer \
                  without Neo4j in the LLM chain! (Type q to quit)> ",
            agent_executor=self._build_agent_executor(),
        )

    def movie_vector_search(self) -> MovieVectorSearch:
        """Returns vector search capabilities for the Neo4j movies database"""
        return MovieVectorSearch(config=self._config, chat=self._chat)

    def _build_agent_executor(self) -> AgentExecutor:
        """
        This internal function builds agent executor for the Movie Chat persona
        """
        prompt = PromptTemplate(
            template=self._config.movie_prompt_template,
            input_variables=["chat_history", "input"],
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        chat_chain = LLMChain(llm=self._chat, prompt=prompt, memory=memory)

        tools = [
            self._tools.chat_tool(chat_chain.run),
            self._tools.trailer_tool(YouTubeSearchTool().run),
        ]

        agent = create_react_agent(
            self._chat, tools, hub.pull(self._config.movie_agent_prompt)
        )

        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=self._config.movie_verbose,
            max_iterations=self._config.movie_max_iterations,
        )
