"""
LangChain Client

High-level client for interacting with LangChain functionality,
including chains, agents, vector stores, and document processing.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from integrations.frameworks.base import BaseClient
from observability.logging import get_logger

from .models import (
    LangChainAgent,
    LangChainAgentType,
    LangChainChain,
    LangChainChainType,
    LangChainConfig,
    LangChainDocument,
    LangChainQuery,
    LangChainResponse,
    LangChainVectorStoreType,
)


class LangChainClient(BaseClient):
    """High-level LangChain client."""

    def __init__(self, config: LangChainConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(__name__)

        # LangChain components (initialized lazily)
        self._llm = None
        self._embeddings = None
        self._vector_store = None
        self._memory = None

        # Storage for chains and agents
        self._chains: Dict[str, Any] = {}
        self._agents: Dict[str, Any] = {}
        self._tools: Dict[str, Any] = {}

        # Initialize LangChain
        self._initialize_langchain()

    def _initialize_langchain(self):
        """Initialize LangChain components."""
        try:
            # Set API keys if available
            if hasattr(self.config, 'openai_api_key') and self.config.openai_api_key:
                os.environ['OPENAI_API_KEY'] = self.config.openai_api_key

            self.logger.info("LangChain client initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain: {e}")
            raise

    def _get_llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            try:
                if self.config.llm_type == "openai":
                    from langchain.chat_models import ChatOpenAI
                    from langchain.llms import OpenAI

                    if "gpt-3.5-turbo" in self.config.llm_model or "gpt-4" in self.config.llm_model:
                        self._llm = ChatOpenAI(
                            model_name=self.config.llm_model,
                            temperature=self.config.llm_temperature,
                            max_tokens=self.config.llm_max_tokens,
                            streaming=self.config.llm_streaming,
                        )
                    else:
                        self._llm = OpenAI(
                            model_name=self.config.llm_model,
                            temperature=self.config.llm_temperature,
                            max_tokens=self.config.llm_max_tokens,
                        )

                elif self.config.llm_type == "anthropic":
                    from langchain.llms import Anthropic

                    self._llm = Anthropic(
                        model=self.config.llm_model,
                        temperature=self.config.llm_temperature,
                        max_tokens_to_sample=self.config.llm_max_tokens,
                    )

                elif self.config.llm_type == "huggingface":
                    from langchain.llms import HuggingFacePipeline

                    self._llm = HuggingFacePipeline.from_model_id(
                        model_id=self.config.llm_model,
                        task="text-generation",
                        model_kwargs={"temperature": self.config.llm_temperature},
                    )

                else:
                    raise ValueError(f"Unsupported LLM type: {self.config.llm_type}")

            except ImportError as e:
                self.logger.error(f"LangChain import error: {e}")
                raise ImportError("LangChain is not installed. Please install with: pip install langchain")
            except Exception as e:
                self.logger.error(f"Failed to create LLM: {e}")
                raise

        return self._llm

    def _get_embeddings(self):
        """Get or create embeddings instance."""
        if self._embeddings is None:
            try:
                if self.config.embedding_type == "openai":
                    from langchain.embeddings import OpenAIEmbeddings

                    self._embeddings = OpenAIEmbeddings(
                        model=self.config.embedding_model, chunk_size=self.config.embedding_chunk_size
                    )

                elif self.config.embedding_type == "huggingface":
                    from langchain.embeddings import HuggingFaceEmbeddings

                    self._embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)

                else:
                    raise ValueError(f"Unsupported embedding type: {self.config.embedding_type}")

            except Exception as e:
                self.logger.error(f"Failed to create embeddings: {e}")
                raise

        return self._embeddings

    def _get_vector_store(self, documents: Optional[List] = None):
        """Get or create vector store instance."""
        try:
            embeddings = self._get_embeddings()

            if self.config.vector_store_type == LangChainVectorStoreType.CHROMA:
                from langchain.vectorstores import Chroma

                if documents:
                    self._vector_store = Chroma.from_documents(
                        documents=documents, embedding=embeddings, **self.config.vector_store_config
                    )
                else:
                    self._vector_store = Chroma(embedding_function=embeddings, **self.config.vector_store_config)

            elif self.config.vector_store_type == LangChainVectorStoreType.FAISS:
                from langchain.vectorstores import FAISS

                if documents:
                    self._vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
                else:
                    # Create empty FAISS index
                    from langchain.schema import Document

                    dummy_doc = Document(page_content="dummy")
                    self._vector_store = FAISS.from_documents([dummy_doc], embeddings)
                    self._vector_store.delete([0])  # Remove dummy document

            elif self.config.vector_store_type == LangChainVectorStoreType.PINECONE:
                import pinecone
                from langchain.vectorstores import Pinecone

                pinecone.init(**self.config.vector_store_config.get('init_kwargs', {}))

                if documents:
                    self._vector_store = Pinecone.from_documents(
                        documents=documents,
                        embedding=embeddings,
                        index_name=self.config.vector_store_config.get('index_name', 'default'),
                    )
                else:
                    self._vector_store = Pinecone(
                        embedding=embeddings, index_name=self.config.vector_store_config.get('index_name', 'default')
                    )

            else:
                raise ValueError(f"Unsupported vector store type: {self.config.vector_store_type}")

            return self._vector_store

        except Exception as e:
            self.logger.error(f"Failed to create vector store: {e}")
            raise

    def _get_memory(self):
        """Get or create memory instance."""
        if self._memory is None:
            try:
                if self.config.memory_type == "buffer":
                    from langchain.memory import ConversationBufferMemory

                    self._memory = ConversationBufferMemory(**self.config.memory_config)

                elif self.config.memory_type == "summary":
                    from langchain.memory import ConversationSummaryMemory

                    self._memory = ConversationSummaryMemory(llm=self._get_llm(), **self.config.memory_config)

                elif self.config.memory_type == "entity":
                    from langchain.memory import ConversationEntityMemory

                    self._memory = ConversationEntityMemory(llm=self._get_llm(), **self.config.memory_config)

                else:
                    from langchain.memory import ConversationBufferMemory

                    self._memory = ConversationBufferMemory()

            except Exception as e:
                self.logger.error(f"Failed to create memory: {e}")
                raise

        return self._memory

    def _load_tools(self, tool_names: List[str]):
        """Load specified tools."""
        loaded_tools = []

        try:
            from langchain.agents import load_tools
            from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
            from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper

            for tool_name in tool_names:
                if tool_name in self._tools:
                    loaded_tools.append(self._tools[tool_name])
                    continue

                if tool_name == "wikipedia":
                    tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
                    self._tools[tool_name] = tool
                    loaded_tools.append(tool)

                elif tool_name == "ddg-search":
                    tool = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())
                    self._tools[tool_name] = tool
                    loaded_tools.append(tool)

                elif tool_name in ["python_repl", "llm-math", "requests"]:
                    # Use LangChain's built-in tool loader
                    tools = load_tools([tool_name], llm=self._get_llm())
                    for tool in tools:
                        self._tools[tool_name] = tool
                        loaded_tools.extend(tools)

                else:
                    self.logger.warning(f"Unknown tool: {tool_name}")

            return loaded_tools

        except Exception as e:
            self.logger.error(f"Failed to load tools: {e}")
            return []

    async def create_chain(
        self, chain_id: str, chain_type: LangChainChainType, config: Optional[Dict[str, Any]] = None
    ) -> LangChainChain:
        """Create a new chain."""
        try:
            self.logger.info(f"Creating LangChain chain: {chain_id} ({chain_type})")

            chain_config = config or {}
            llm = self._get_llm()

            if chain_type == LangChainChainType.LLM:
                from langchain.chains import LLMChain
                from langchain.prompts import PromptTemplate

                prompt_template = chain_config.get('prompt_template', "Question: {question}\nAnswer:")
                prompt = PromptTemplate(
                    input_variables=chain_config.get('input_variables', ['question']), template=prompt_template
                )

                chain = LLMChain(llm=llm, prompt=prompt)

            elif chain_type == LangChainChainType.CONVERSATION:
                from langchain.chains import ConversationChain

                memory = self._get_memory()
                chain = ConversationChain(llm=llm, memory=memory)

            elif chain_type == LangChainChainType.QA:
                from langchain.chains import QAWithSourcesChain

                chain = QAWithSourcesChain.from_llm(llm=llm)

            elif chain_type == LangChainChainType.RETRIEVAL_QA:
                from langchain.chains import RetrievalQA

                vector_store = self._get_vector_store()
                retriever = vector_store.as_retriever(search_kwargs={"k": self.config.retrieval_k})

                chain = RetrievalQA.from_chain_type(
                    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
                )

            elif chain_type == LangChainChainType.SUMMARIZATION:
                from langchain.chains.summarize import load_summarize_chain

                chain = load_summarize_chain(llm=llm, chain_type=chain_config.get('summarize_type', 'stuff'))

            else:
                raise ValueError(f"Unsupported chain type: {chain_type}")

            # Store chain
            self._chains[chain_id] = chain

            # Create chain metadata
            chain_metadata = LangChainChain(chain_id=chain_id, chain_type=chain_type, config=chain_config)

            self.logger.info(f"Created LangChain chain: {chain_id}")
            return chain_metadata

        except Exception as e:
            self.logger.error(f"Failed to create chain {chain_id}: {e}")
            raise

    async def create_agent(
        self, agent_id: str, agent_type: LangChainAgentType, tools: List[str], config: Optional[Dict[str, Any]] = None
    ) -> LangChainAgent:
        """Create a new agent."""
        try:
            self.logger.info(f"Creating LangChain agent: {agent_id} ({agent_type})")

            agent_config = config or {}
            llm = self._get_llm()
            loaded_tools = self._load_tools(tools)

            if not loaded_tools:
                raise ValueError("No valid tools provided for agent")

            if agent_type == LangChainAgentType.ZERO_SHOT_REACT:
                from langchain.agents import AgentType, initialize_agent

                agent = initialize_agent(
                    tools=loaded_tools,
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=agent_config.get('verbose', False),
                )

            elif agent_type == LangChainAgentType.CONVERSATIONAL_REACT:
                from langchain.agents import AgentType, initialize_agent

                memory = self._get_memory()
                agent = initialize_agent(
                    tools=loaded_tools,
                    llm=llm,
                    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                    memory=memory,
                    verbose=agent_config.get('verbose', False),
                )

            elif agent_type == LangChainAgentType.OPENAI_FUNCTIONS:
                from langchain.agents import AgentType, initialize_agent

                agent = initialize_agent(
                    tools=loaded_tools,
                    llm=llm,
                    agent=AgentType.OPENAI_FUNCTIONS,
                    verbose=agent_config.get('verbose', False),
                )

            else:
                # Default to zero-shot react
                from langchain.agents import AgentType, initialize_agent

                agent = initialize_agent(
                    tools=loaded_tools,
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=agent_config.get('verbose', False),
                )

            # Store agent
            self._agents[agent_id] = agent

            # Create agent metadata
            agent_metadata = LangChainAgent(agent_id=agent_id, agent_type=agent_type, tools=tools, config=agent_config)

            self.logger.info(f"Created LangChain agent: {agent_id}")
            return agent_metadata

        except Exception as e:
            self.logger.error(f"Failed to create agent {agent_id}: {e}")
            raise

    async def execute_chain(self, chain_id: str, query: LangChainQuery) -> LangChainResponse:
        """Execute a chain."""
        try:
            self.logger.info(f"Executing chain: {chain_id}")

            if chain_id not in self._chains:
                raise ValueError(f"Chain not found: {chain_id}")

            chain = self._chains[chain_id]

            # Prepare input
            if hasattr(chain, 'input_keys'):
                if len(chain.input_keys) == 1:
                    input_data = {chain.input_keys[0]: query.query}
                else:
                    input_data = query.chain_config
            else:
                input_data = {"question": query.query}

            # Execute chain
            output = chain(input_data)

            response = LangChainResponse.from_langchain_output(output, query, query.chain_type)

            self.logger.info(f"Chain executed successfully: {chain_id}")
            return response

        except Exception as e:
            self.logger.error(f"Chain execution failed: {chain_id}: {e}")
            return LangChainResponse(success=False, error=str(e), framework="langchain")

    async def execute_agent(self, agent_id: str, query: LangChainQuery) -> LangChainResponse:
        """Execute an agent."""
        try:
            self.logger.info(f"Executing agent: {agent_id}")

            if agent_id not in self._agents:
                raise ValueError(f"Agent not found: {agent_id}")

            agent = self._agents[agent_id]

            # Execute agent
            output = agent.run(query.query)

            response = LangChainResponse.from_langchain_output(output, query, None)

            self.logger.info(f"Agent executed successfully: {agent_id}")
            return response

        except Exception as e:
            self.logger.error(f"Agent execution failed: {agent_id}: {e}")
            return LangChainResponse(success=False, error=str(e), framework="langchain")

    async def add_documents(self, documents: List[LangChainDocument], vector_store_id: Optional[str] = None) -> bool:
        """Add documents to vector store."""
        try:
            self.logger.info(f"Adding {len(documents)} documents to vector store")

            # Convert to LangChain documents
            langchain_docs = [doc.to_langchain_document() for doc in documents]

            # Get or create vector store
            vector_store = self._get_vector_store()

            # Add documents
            vector_store.add_documents(langchain_docs)

            self.logger.info(f"Added {len(documents)} documents to vector store")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return False

    async def search_documents(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None
    ) -> List[LangChainDocument]:
        """Search documents in vector store."""
        try:
            vector_store = self._get_vector_store()

            # Perform similarity search
            docs = vector_store.similarity_search(query, k=k)

            # Convert back to our document format
            return [LangChainDocument.from_langchain_document(doc) for doc in docs]

        except Exception as e:
            self.logger.error(f"Document search failed: {e}")
            return []

    async def list_chains(self) -> List[str]:
        """List all created chains."""
        return list(self._chains.keys())

    async def list_agents(self) -> List[str]:
        """List all created agents."""
        return list(self._agents.keys())

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Test LangChain imports
            from langchain.llms import OpenAI

            # Test LLM creation
            llm = self._get_llm()

            return {
                'status': 'healthy',
                'framework': 'langchain',
                'llm_type': self.config.llm_type,
                'llm_model': self.config.llm_model,
                'embedding_type': self.config.embedding_type,
                'vector_store_type': self.config.vector_store_type.value,
                'loaded_chains': len(self._chains),
                'loaded_agents': len(self._agents),
                'available_tools': len(self._tools),
            }

        except Exception as e:
            return {'status': 'unhealthy', 'framework': 'langchain', 'error': str(e)}
