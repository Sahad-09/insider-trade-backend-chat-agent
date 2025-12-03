"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""

import asyncio
from typing import (
    AsyncGenerator,
    Optional,
)
from urllib.parse import quote_plus

from asgiref.sync import sync_to_async
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import (
    END,
    StateGraph,
)
from langgraph.graph.state import (
    Command,
    CompiledStateGraph,
)
from langgraph.types import (
    RunnableConfig,
    StateSnapshot,
)
from mem0 import AsyncMemory
from psycopg_pool import AsyncConnectionPool

from app.core.config import (
    Environment,
    settings,
)
from app.core.langgraph.tools import tools
from app.core.logging import logger
from app.core.metrics import llm_inference_duration_seconds
from app.core.prompts import load_system_prompt
from app.schemas import (
    GraphState,
    Message,
)
from app.services.llm import ReasoningResponse, llm_service
from app.utils import (
    dump_messages,
    prepare_messages,
    process_llm_response,
)


class LangGraphAgent:
    """Manages the LangGraph Agent/workflow and interactions with the LLM.

    This class handles the creation and management of the LangGraph workflow,
    including LLM interactions, database connections, and response processing.
    """

    def __init__(self):
        """Initialize the LangGraph Agent with necessary components."""
        # Use the LLM service with tools bound
        self.llm_service = llm_service
        self.llm_service.bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None
        self.memory: Optional[AsyncMemory] = None
        logger.info(
            "langgraph_agent_initialized",
            model=settings.DEFAULT_LLM_MODEL,
            environment=settings.ENVIRONMENT.value,
        )

    async def _long_term_memory(self) -> AsyncMemory:
        """Initialize the long term memory."""
        if self.memory is None:
            # Configure LLM provider for memory
            llm_config = {"model": settings.LONG_TERM_MEMORY_MODEL}
            # Ollama provider commented out - using OpenAI instead
            # if settings.LONG_TERM_MEMORY_PROVIDER == "ollama":
            #     llm_config["base_url"] = settings.OLLAMA_BASE_URL

            # Configure embedder based on provider
            # Ollama provider commented out - using OpenAI instead
            # if settings.LONG_TERM_MEMORY_PROVIDER == "ollama":
            #     # Use Ollama for embeddings when using Ollama provider
            #     embedder_provider = "ollama"
            #     # Use an Ollama-compatible embedding model (default: nomic-embed-text)
            #     embedder_model = settings.LONG_TERM_MEMORY_EMBEDDER_MODEL
            #     # If using OpenAI model name, default to Ollama embedding model
            #     if "text-embedding" in embedder_model.lower() or "ada" in embedder_model.lower():
            #         embedder_model = "nomic-embed-text"  # Default Ollama embedding model
            #         logger.info(
            #             "using_ollama_embedding_model",
            #             original_model=settings.LONG_TERM_MEMORY_EMBEDDER_MODEL,
            #             using_model=embedder_model,
            #         )
            #     embedder_config = {
            #         "model": embedder_model,
            #         "base_url": settings.OLLAMA_BASE_URL,
            #     }
            # else:
            # Use OpenAI for embeddings (default)
            embedder_provider = "openai"
            embedder_config = {"model": settings.LONG_TERM_MEMORY_EMBEDDER_MODEL}
            # Require API key for OpenAI
            if not settings.OPENAI_API_KEY:
                logger.warning(
                    "openai_api_key_missing_for_embeddings",
                    message="OPENAI_API_KEY not set but required for OpenAI embeddings. Long-term memory may fail.",
                )

            self.memory = await AsyncMemory.from_config(
                config_dict={
                    "vector_store": {
                        "provider": "pgvector",
                        "config": {
                            "collection_name": settings.LONG_TERM_MEMORY_COLLECTION_NAME,
                            "dbname": settings.POSTGRES_DB,
                            "user": settings.POSTGRES_USER,
                            "password": settings.POSTGRES_PASSWORD,
                            "host": settings.POSTGRES_HOST,
                            "port": settings.POSTGRES_PORT,
                        },
                    },
                    "llm": {
                        "provider": settings.LONG_TERM_MEMORY_PROVIDER,
                        "config": llm_config,
                    },
                    "embedder": {"provider": embedder_provider, "config": embedder_config},
                    # "custom_fact_extraction_prompt": load_custom_fact_extraction_prompt(),
                }
            )
            # Original OpenAI-only configuration (commented out):
            # self.memory = await AsyncMemory.from_config(
            #     config_dict={
            #         "vector_store": {
            #             "provider": "pgvector",
            #             "config": {
            #                 "collection_name": settings.LONG_TERM_MEMORY_COLLECTION_NAME,
            #                 "dbname": settings.POSTGRES_DB,
            #                 "user": settings.POSTGRES_USER,
            #                 "password": settings.POSTGRES_PASSWORD,
            #                 "host": settings.POSTGRES_HOST,
            #                 "port": settings.POSTGRES_PORT,
            #             },
            #         },
            #         "llm": {
            #             "provider": "openai",
            #             "config": {"model": settings.LONG_TERM_MEMORY_MODEL},
            #         },
            #         "embedder": {"provider": "openai", "config": {"model": settings.LONG_TERM_MEMORY_EMBEDDER_MODEL}},
            #         # "custom_fact_extraction_prompt": load_custom_fact_extraction_prompt(),
            #     }
            # )
        return self.memory

    async def _get_connection_pool(self) -> AsyncConnectionPool:
        """Get a PostgreSQL connection pool using environment-specific settings.

        Returns:
            AsyncConnectionPool: A connection pool for PostgreSQL database.
        """
        if self._connection_pool is None:
            try:
                # Configure pool size based on environment
                max_size = settings.POSTGRES_POOL_SIZE

                connection_url = (
                    "postgresql://"
                    f"{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
                    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
                )

                self._connection_pool = AsyncConnectionPool(
                    connection_url,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                    },
                )
                await self._connection_pool.open()
                logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
            except Exception as e:
                logger.error("connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we might want to degrade gracefully
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_connection_pool", environment=settings.ENVIRONMENT.value)
                    return None
                raise e
        return self._connection_pool

    async def _get_relevant_memory(self, user_id: str, query: str) -> str:
        """Get the relevant memory for the user and query.

        Args:
            user_id (str): The user ID.
            query (str): The query to search for.

        Returns:
            str: The relevant memory.
        """
        try:
            memory = await self._long_term_memory()
            results = await memory.search(user_id=str(user_id), query=query)
            print(results)
            return "\n".join([f"* {result['memory']}" for result in results["results"]])
        except Exception as e:
            logger.error("failed_to_get_relevant_memory", error=str(e), user_id=user_id, query=query)
            return ""

    async def _update_long_term_memory(self, user_id: str, messages: list[dict], metadata: dict = None) -> None:
        """Update the long term memory.

        Args:
            user_id (str): The user ID.
            messages (list[dict]): The messages to update the long term memory with.
            metadata (dict): Optional metadata to include.
        """
        try:
            memory = await self._long_term_memory()
            await memory.add(messages, user_id=str(user_id), metadata=metadata)
            logger.info("long_term_memory_updated_successfully", user_id=user_id)
        except Exception as e:
            # Log error but don't fail the entire request if memory update fails
            logger.warning(
                "failed_to_update_long_term_memory",
                user_id=user_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )

    async def _chat(self, state: GraphState, config: RunnableConfig) -> Command:
        """Process the chat state and generate a response.

        Args:
            state (GraphState): The current state of the conversation.

        Returns:
            Command: Command object with updated state and next node to execute.
        """
        session_id = config.get("configurable", {}).get("thread_id", "unknown")
        
        # Colored log for CHAT node (Blue background)
        logger.info(
            "\033[44m\033[97m[CHAT NODE]\033[0m chat_node_entered",
            session_id=session_id,
            message_count=len(state.messages),
        )
        
        # Get the current LLM instance for metrics
        current_llm = self.llm_service.get_llm()
        model_name = (
            getattr(current_llm, "model_name", None)
            or getattr(current_llm, "model", None)
            or settings.DEFAULT_LLM_MODEL
        )

        SYSTEM_PROMPT = load_system_prompt(long_term_memory=state.long_term_memory)

        # Prepare messages with system prompt
        messages = prepare_messages(state.messages, current_llm, SYSTEM_PROMPT)

        try:
            # Use LLM service with automatic retries and circular fallback
            with llm_inference_duration_seconds.labels(model=model_name).time():
                response_message = await self.llm_service.call(dump_messages(messages))

            # Process response to handle structured content blocks
            response_message = process_llm_response(response_message)

            logger.info(
                "llm_response_generated",
                session_id=config["configurable"]["thread_id"],
                model=model_name,
                environment=settings.ENVIRONMENT.value,
            )

            # Determine next node based on whether there are tool calls
            if response_message.tool_calls:
                logger.info(
                    "\033[44m\033[97m[CHAT NODE]\033[0m chat_node_routing_to_tool_call",
                    session_id=session_id,
                    tool_calls_count=len(response_message.tool_calls),
                )
                goto = "tool_call"
            else:
                logger.info(
                    "\033[44m\033[97m[CHAT NODE]\033[0m chat_node_routing_to_end",
                    session_id=session_id,
                )
                goto = END

            return Command(update={"messages": [response_message]}, goto=goto)
        except Exception as e:
            logger.error(
                "llm_call_failed_all_models",
                session_id=config["configurable"]["thread_id"],
                error=str(e),
                environment=settings.ENVIRONMENT.value,
            )
            raise Exception(f"failed to get llm response after trying all models: {str(e)}")

    # Define our tool node
    async def _tool_call(self, state: GraphState, config: RunnableConfig) -> Command:
        """Process tool calls from the last message.

        Args:
            state: The current agent state containing messages and tool calls.
            config: The runnable configuration with session metadata.

        Returns:
            Command: Command object with updated messages and routing back to chat.
        """
        session_id = config.get("configurable", {}).get("thread_id", "unknown")
        outputs = []
        
        # Colored log for TOOL_CALL node (Green background)
        logger.info(
            "\033[42m\033[97m[TOOL_CALL NODE]\033[0m tool_call_node_entered",
            session_id=session_id,
            tool_count=len(state.messages[-1].tool_calls),
            tool_names=[tc.get("name") for tc in state.messages[-1].tool_calls],
        )
        
        for tool_call in state.messages[-1].tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            
            # Color mapping for different tools
            tool_colors = {
                "duckduckgo_search": "\033[43m\033[30m",  # Yellow background, black text
                "brave_search": "\033[45m\033[97m",  # Magenta background, white text
            }
            tool_color = tool_colors.get(tool_name, "\033[46m\033[97m")  # Cyan default
            
            logger.info(
                f"{tool_color}[TOOL: {tool_name.upper()}]\033[0m tool_execution_started",
                session_id=session_id,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call.get("id"),
            )
            
            try:
                tool_result = await self.tools_by_name[tool_name].ainvoke(tool_args)
                
                logger.info(
                    f"{tool_color}[TOOL: {tool_name.upper()}]\033[0m tool_execution_completed",
                    session_id=session_id,
                    tool_name=tool_name,
                    result_length=len(str(tool_result)),
                    result_preview=str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else str(tool_result),
                )
                
                outputs.append(
                    ToolMessage(
                        content=tool_result,
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    )
                )
            except Exception as e:
                # Color mapping for different tools (same as above)
                tool_colors = {
                    "duckduckgo_search": "\033[43m\033[30m",  # Yellow background, black text
                    "brave_search": "\033[45m\033[97m",  # Magenta background, white text
                }
                tool_color = tool_colors.get(tool_name, "\033[46m\033[97m")  # Cyan default
                
                logger.error(
                    f"{tool_color}[TOOL: {tool_name.upper()}]\033[0m tool_execution_failed",
                    session_id=session_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    error=str(e),
                    exc_info=True,
                )
                # Create error message for the tool
                outputs.append(
                    ToolMessage(
                        content=f"Error executing tool {tool_name}: {str(e)}",
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    )
                )
        
        logger.info(
            "\033[42m\033[97m[TOOL_CALL NODE]\033[0m tool_call_node_completed",
            session_id=session_id,
            output_count=len(outputs),
        )
        
        return Command(update={"messages": outputs}, goto="chat")

    async def create_graph(self) -> Optional[CompiledStateGraph]:
        """Create and configure the LangGraph workflow.

        Returns:
            Optional[CompiledStateGraph]: The configured LangGraph instance or None if init fails
        """
        if self._graph is None:
            try:
                graph_builder = StateGraph(GraphState)
                graph_builder.add_node("chat", self._chat, ends=["tool_call", END])
                graph_builder.add_node("tool_call", self._tool_call, ends=["chat"])
                graph_builder.set_entry_point("chat")
                graph_builder.set_finish_point("chat")

                # Get connection pool (may be None in production if DB unavailable)
                connection_pool = await self._get_connection_pool()
                if connection_pool:
                    checkpointer = AsyncPostgresSaver(connection_pool)
                    await checkpointer.setup()
                else:
                    # In production, proceed without checkpointer if needed
                    checkpointer = None
                    if settings.ENVIRONMENT != Environment.PRODUCTION:
                        raise Exception("Connection pool initialization failed")

                self._graph = graph_builder.compile(
                    checkpointer=checkpointer, name=f"{settings.PROJECT_NAME} Agent ({settings.ENVIRONMENT.value})"
                )

                logger.info(
                    "graph_created",
                    graph_name=f"{settings.PROJECT_NAME} Agent",
                    environment=settings.ENVIRONMENT.value,
                    has_checkpointer=checkpointer is not None,
                )
            except Exception as e:
                logger.error("graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we don't want to crash the app
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_graph")
                    return None
                raise e

        return self._graph

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Get a response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for Langfuse tracking.
            user_id (Optional[str]): The user ID for Langfuse tracking.

        Returns:
            list[dict]: The response from the LLM.
        """
        if self._graph is None:
            self._graph = await self.create_graph()
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [CallbackHandler()],
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }
        relevant_memory = (
            await self._get_relevant_memory(user_id, messages[-1].content)
        ) or "No relevant memory found."
        try:
            response = await self._graph.ainvoke(
                input={"messages": dump_messages(messages), "long_term_memory": relevant_memory},
                config=config,
            )
            # Run memory update in background without blocking the response
            asyncio.create_task(
                self._update_long_term_memory(
                    user_id, convert_to_openai_messages(response["messages"]), config["metadata"]
                )
            )
            return self.__process_messages(response["messages"])
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")

    async def get_response_with_reasoning(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        reasoning_effort: str = "medium",
    ) -> AsyncGenerator[dict, None]:
        """Get a response using o3-mini with reasoning summary.
        
        This method uses o3-mini model which has internal reasoning capabilities.
        It streams the thinking state and then provides the final response with
        reasoning token usage information.
        
        Args:
            messages: The messages to send to the LLM.
            session_id: The session ID for tracking.
            user_id: The user ID for tracking.
            reasoning_effort: "low", "medium", or "high"
            
        Yields:
            dict: Thinking state updates with format:
                {
                    "thinking_title": str,
                    "response": str,
                    "status": "thinking" | "reasoning" | "done",
                    "reasoning": list[str],
                    "model": str
                }
        """
        try:
            # First emit that we're starting with o3-mini reasoning
            yield {
                "thinking_title": "🧠 o3-mini reasoning",
                "response": "",
                "status": "reasoning",
                "reasoning": [],
                "model": "o3-mini"
            }
            
            # Get memory context
            relevant_memory = (
                await self._get_relevant_memory(user_id, messages[-1].content)
            ) or "No relevant memory found."
            
            # Prepare messages in OpenAI format
            system_prompt = load_system_prompt(long_term_memory=relevant_memory)
            openai_messages = [
                {"role": "system", "content": system_prompt},
            ]
            
            # Add conversation history
            for msg in messages:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            logger.info(
                "calling_o3_mini_reasoning",
                session_id=session_id,
                reasoning_effort=reasoning_effort,
                message_count=len(openai_messages),
            )
            
            # Call o3-mini with reasoning
            reasoning_response: ReasoningResponse = await llm_service.call_o3_with_reasoning(
                messages=openai_messages,
                reasoning_effort=reasoning_effort,
            )
            
            # Emit the final response with reasoning info
            yield {
                "thinking_title": "",
                "response": reasoning_response.content,
                "status": "done",
                "reasoning": reasoning_response.reasoning_steps,
                "model": reasoning_response.model,
                "reasoning_tokens": reasoning_response.reasoning_tokens,
                "output_tokens": reasoning_response.output_tokens,
            }
            
            logger.info(
                "o3_mini_reasoning_completed",
                session_id=session_id,
                response_length=len(reasoning_response.content),
                reasoning_tokens=reasoning_response.reasoning_tokens,
            )
            
        except Exception as e:
            logger.error(
                "o3_mini_reasoning_failed",
                session_id=session_id,
                error=str(e),
                exc_info=True,
            )
            yield {
                "thinking_title": "",
                "response": f"Error with o3-mini: {str(e)}",
                "status": "done",
                "reasoning": [],
                "model": "o3-mini"
            }

    async def get_stream_response(
        self, messages: list[Message], session_id: str, user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.

        Yields:
            str: Tokens of the LLM response.
        """
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [CallbackHandler()],
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }
        if self._graph is None:
            self._graph = await self.create_graph()

        relevant_memory = (
            await self._get_relevant_memory(user_id, messages[-1].content)
        ) or "No relevant memory found."

        try:
            sent_content = ""  # Track what we've already sent to avoid duplicates
            async for token, _ in self._graph.astream(
                {"messages": dump_messages(messages), "long_term_memory": relevant_memory},
                config,
                stream_mode="messages",
            ):
                try:
                    # Get current content from the message token
                    current_content = token.content if hasattr(token, "content") else str(token)
                    
                    # Only yield the new content (delta) to avoid duplication
                    if current_content and isinstance(current_content, str):
                        if current_content.startswith(sent_content):
                            # Extract only the new part
                            new_content = current_content[len(sent_content):]
                            if new_content:
                                sent_content = current_content
                                yield new_content
                        elif current_content != sent_content:
                            # Handle case where content structure changed (shouldn't happen normally)
                            # Yield the difference if possible, otherwise the whole new content
                            if sent_content:
                                # Try to find overlap
                                if current_content.endswith(sent_content):
                                    new_content = current_content[: -len(sent_content)]
                                else:
                                    new_content = current_content
                            else:
                                new_content = current_content
                            
                            if new_content:
                                sent_content = current_content
                                yield new_content
                except Exception as token_error:
                    logger.error("Error processing token", error=str(token_error), session_id=session_id)
                    # Continue with next token even if current one fails
                    continue

            # After streaming completes, get final state and update memory in background
            state: StateSnapshot = await sync_to_async(self._graph.get_state)(config=config)
            if state.values and "messages" in state.values:
                asyncio.create_task(
                    self._update_long_term_memory(
                        user_id, convert_to_openai_messages(state.values["messages"]), config["metadata"]
                    )
                )
        except Exception as stream_error:
            logger.error("Error in stream processing", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_stream_response_with_thinking(
        self, messages: list[Message], session_id: str, user_id: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """Get a stream response with thinking states from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.

        Yields:
            dict: Thinking state updates with format:
                {
                    "thinking_title": str,
                    "response": str,
                    "status": "thinking" | "done"
                }
        """
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [CallbackHandler()],
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }
        if self._graph is None:
            self._graph = await self.create_graph()

        relevant_memory = (
            await self._get_relevant_memory(user_id, messages[-1].content)
        ) or "No relevant memory found."

        try:
            sent_content = ""
            accumulated_response = ""
            previous_node = None
            last_tool_name = None
            previous_state_messages = []
            
            # Stream updates to track node transitions and content
            async for update in self._graph.astream(
                {"messages": dump_messages(messages), "long_term_memory": relevant_memory},
                config,
                stream_mode="updates",
            ):
                # Track node transitions
                for node_name, node_output in update.items():
                    if node_name == "__end__":
                        continue
                    
                    # Get current state messages
                    current_messages = node_output.get("messages", []) if "messages" in node_output else []
                    
                    # Detect node transitions
                    if node_name != previous_node:
                        previous_node = node_name
                        
                        # Emit thinking state based on node
                        if node_name == "chat":
                            # Check previous state to see if tool calls were made
                            if previous_state_messages:
                                last_prev_msg = previous_state_messages[-1]
                                if hasattr(last_prev_msg, "tool_calls") and last_prev_msg.tool_calls:
                                    # LLM just decided to use tools
                                    tool_names = [tc.get("name") for tc in last_prev_msg.tool_calls]
                                    last_tool_name = tool_names[0] if tool_names else None
                                    if "duckduckgo_search" in tool_names:
                                        thinking_state = {
                                            "thinking_title": "searching web",
                                            "response": "",
                                            "status": "thinking"
                                        }
                                        logger.info(
                                            "thinking_state_emitted",
                                            session_id=session_id,
                                            thinking_title=thinking_state["thinking_title"],
                                            status=thinking_state["status"],
                                            node=node_name,
                                            tool_names=tool_names,
                                        )
                                        yield thinking_state
                                    else:
                                        thinking_title = f"using {last_tool_name}" if last_tool_name else "processing"
                                        thinking_state = {
                                            "thinking_title": thinking_title,
                                            "response": "",
                                            "status": "thinking"
                                        }
                                        logger.info(
                                            "thinking_state_emitted",
                                            session_id=session_id,
                                            thinking_title=thinking_state["thinking_title"],
                                            status=thinking_state["status"],
                                            node=node_name,
                                            tool_names=tool_names,
                                        )
                                        yield thinking_state
                                elif current_messages:
                                    # Check if we're generating content
                                    last_msg = current_messages[-1]
                                    if hasattr(last_msg, "content") and last_msg.content and not isinstance(last_msg, ToolMessage):
                                        # LLM is generating response - emit thinking state
                                        thinking_state = {
                                            "thinking_title": "thinking",
                                            "response": "",
                                            "status": "thinking"
                                        }
                                        logger.info(
                                            "thinking_state_emitted",
                                            session_id=session_id,
                                            thinking_title=thinking_state["thinking_title"],
                                            status=thinking_state["status"],
                                            node=node_name,
                                        )
                                        yield thinking_state
                            else:
                                # First chat node entry
                                thinking_state = {
                                    "thinking_title": "thinking",
                                    "response": "",
                                    "status": "thinking"
                                }
                                logger.info(
                                    "thinking_state_emitted",
                                    session_id=session_id,
                                    thinking_title=thinking_state["thinking_title"],
                                    status=thinking_state["status"],
                                    node=node_name,
                                )
                                yield thinking_state
                        
                        elif node_name == "tool_call":
                            # Tool execution started - use last_tool_name from previous state
                            if last_tool_name == "duckduckgo_search":
                                thinking_state = {
                                    "thinking_title": "searching web",
                                    "response": "",
                                    "status": "thinking"
                                }
                                logger.info(
                                    "thinking_state_emitted",
                                    session_id=session_id,
                                    thinking_title=thinking_state["thinking_title"],
                                    status=thinking_state["status"],
                                    node=node_name,
                                    tool_name=last_tool_name,
                                )
                                yield thinking_state
                            elif last_tool_name:
                                thinking_title = f"using {last_tool_name}"
                                thinking_state = {
                                    "thinking_title": thinking_title,
                                    "response": "",
                                    "status": "thinking"
                                }
                                logger.info(
                                    "thinking_state_emitted",
                                    session_id=session_id,
                                    thinking_title=thinking_state["thinking_title"],
                                    status=thinking_state["status"],
                                    node=node_name,
                                    tool_name=last_tool_name,
                                )
                                yield thinking_state
                            else:
                                thinking_state = {
                                    "thinking_title": "executing tool",
                                    "response": "",
                                    "status": "thinking"
                                }
                                logger.info(
                                    "thinking_state_emitted",
                                    session_id=session_id,
                                    thinking_title=thinking_state["thinking_title"],
                                    status=thinking_state["status"],
                                    node=node_name,
                                )
                                yield thinking_state
                    
                    # Extract content from messages if available
                    if current_messages:
                        for msg in current_messages:
                            if hasattr(msg, "content") and msg.content and not isinstance(msg, ToolMessage):
                                current_content = str(msg.content)
                                
                                # Only yield new content (delta)
                                if current_content and isinstance(current_content, str):
                                    if current_content.startswith(sent_content):
                                        new_content = current_content[len(sent_content):]
                                        if new_content:
                                            sent_content = current_content
                                            accumulated_response += new_content
                                            thinking_state = {
                                                "thinking_title": "",
                                                "response": new_content,
                                                "status": "thinking"
                                            }
                                            logger.debug(
                                                "content_chunk_streamed",
                                                session_id=session_id,
                                                chunk_length=len(new_content),
                                                accumulated_length=len(accumulated_response),
                                            )
                                            yield thinking_state
                                    elif current_content != sent_content:
                                        if sent_content:
                                            if current_content.endswith(sent_content):
                                                new_content = current_content[: -len(sent_content)]
                                            else:
                                                new_content = current_content
                                        else:
                                            new_content = current_content
                                        
                                        if new_content:
                                            sent_content = current_content
                                            accumulated_response += new_content
                                            thinking_state = {
                                                "thinking_title": "",
                                                "response": new_content,
                                                "status": "thinking"
                                            }
                                            logger.debug(
                                                "content_chunk_streamed",
                                                session_id=session_id,
                                                chunk_length=len(new_content),
                                                accumulated_length=len(accumulated_response),
                                            )
                                            yield thinking_state
                    
                    # Update previous state for next iteration
                    previous_state_messages = current_messages

            # Get final state and emit done status
            state: StateSnapshot = await sync_to_async(self._graph.get_state)(config=config)
            if state.values and "messages" in state.values:
                # Extract final response
                final_messages = state.values["messages"]
                final_response = ""
                for msg in reversed(final_messages):
                    if hasattr(msg, "content") and msg.content and not isinstance(msg, ToolMessage):
                        final_response = str(msg.content)
                        break
                
                # Emit final response with done status
                final_response_text = final_response if final_response else accumulated_response
                thinking_state = {
                    "thinking_title": "",
                    "response": final_response_text,
                    "status": "done"
                }
                logger.info(
                    "thinking_state_completed",
                    session_id=session_id,
                    status=thinking_state["status"],
                    response_length=len(final_response_text),
                )
                yield thinking_state
                
                # Update memory in background
                asyncio.create_task(
                    self._update_long_term_memory(
                        user_id, convert_to_openai_messages(final_messages), config["metadata"]
                    )
                )
        except Exception as stream_error:
            logger.error("Error in stream processing", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        if self._graph is None:
            self._graph = await self.create_graph()

        state: StateSnapshot = await sync_to_async(self._graph.get_state)(
            config={"configurable": {"thread_id": session_id}}
        )
        return self.__process_messages(state.values["messages"]) if state.values else []

    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        openai_style_messages = convert_to_openai_messages(messages)
        # keep just assistant and user messages
        return [
            Message(role=message["role"], content=str(message["content"]))
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        try:
            # Make sure the pool is initialized in the current event loop
            conn_pool = await self._get_connection_pool()

            # Use a new connection for this specific operation
            async with conn_pool.connection() as conn:
                for table in settings.CHECKPOINT_TABLES:
                    try:
                        await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                        logger.info(f"Cleared {table} for session {session_id}")
                    except Exception as e:
                        logger.error(f"Error clearing {table}", error=str(e))
                        raise

        except Exception as e:
            logger.error("Failed to clear chat history", error=str(e))
            raise
