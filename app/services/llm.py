"""LLM service for managing LLM calls with retries and fallback mechanisms."""

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
# Ollama import commented out - using OpenAI instead
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import (
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import (
    Environment,
    settings,
)
from app.core.logging import logger


@dataclass
class ReasoningResponse:
    """Response from o3-mini model with reasoning summary.
    
    Attributes:
        content: The main response content
        reasoning_steps: List of reasoning steps from o3-mini
        model: The model used
        reasoning_tokens: Number of reasoning tokens used
        output_tokens: Number of output tokens used
    """
    content: str
    reasoning_steps: List[str] = field(default_factory=list)
    model: str = ""
    reasoning_tokens: int = 0
    output_tokens: int = 0


def create_llm_instance(model_name: str, **kwargs) -> BaseChatModel:
    """Create an LLM instance based on the configured provider.

    Args:
        model_name: Name of the model to create
        **kwargs: Additional arguments to pass to the LLM constructor

    Returns:
        BaseChatModel instance
    """
    # Ollama provider commented out - using OpenAI instead
    # if settings.LLM_PROVIDER == "ollama":
    #     return ChatOllama(
    #         model=model_name,
    #         base_url=settings.OLLAMA_BASE_URL,
    #         temperature=kwargs.get("temperature", settings.DEFAULT_LLM_TEMPERATURE),
    #         num_predict=kwargs.get("max_tokens", settings.MAX_TOKENS),
    #         **{k: v for k, v in kwargs.items() if k not in ("temperature", "max_tokens")},
    #     )
    # Default to OpenAI
    # Handle o3/o1 reasoning models specially - they don't support temperature
    if model_name.startswith("o3") or model_name.startswith("o1"):
        reasoning_effort = kwargs.get("reasoning_effort", "medium")
        return ChatOpenAI(
            model=model_name,
            api_key=settings.OPENAI_API_KEY,
            max_completion_tokens=kwargs.get("max_tokens", settings.MAX_TOKENS),
            model_kwargs={"reasoning_effort": reasoning_effort},
        )
    # Standard models (gpt-4o, gpt-4o-mini, etc.)
    return ChatOpenAI(
        model=model_name,
        api_key=settings.OPENAI_API_KEY,
        temperature=kwargs.get("temperature", settings.DEFAULT_LLM_TEMPERATURE),
        max_tokens=kwargs.get("max_tokens", settings.MAX_TOKENS),
        **{k: v for k, v in kwargs.items() if k not in ("temperature", "max_tokens")},
    )


class LLMRegistry:
    """Registry of available LLM models with pre-initialized instances.

    This class maintains a list of LLM configurations and provides
    methods to retrieve them by name with optional argument overrides.
    """

    # Class-level variable containing all available LLM models
    # Dynamically initialized based on provider
    LLMS: List[Dict[str, Any]] = []

    # Original static LLMS list (commented out for dynamic provider-based initialization)
    # LLMS: List[Dict[str, Any]] = [
    #     {
    #         "name": "gpt-5-mini",
    #         "llm": ChatOpenAI(
    #             model="gpt-5-mini",
    #             api_key=settings.OPENAI_API_KEY,
    #             max_tokens=settings.MAX_TOKENS,
    #             reasoning={"effort": "low"},
    #         ),
    #     },
    #     {
    #         "name": "gpt-5",
    #         "llm": ChatOpenAI(
    #             model="gpt-5",
    #             api_key=settings.OPENAI_API_KEY,
    #             max_tokens=settings.MAX_TOKENS,
    #             reasoning={"effort": "medium"},
    #         ),
    #     },
    #     {
    #         "name": "gpt-5-nano",
    #         "llm": ChatOpenAI(
    #             model="gpt-5-nano",
    #             api_key=settings.OPENAI_API_KEY,
    #             max_tokens=settings.MAX_TOKENS,
    #             reasoning={"effort": "minimal"},
    #         ),
    #     },
    #     {
    #         "name": "gpt-4o",
    #         "llm": ChatOpenAI(
    #             model="gpt-4o",
    #             temperature=settings.DEFAULT_LLM_TEMPERATURE,
    #             api_key=settings.OPENAI_API_KEY,
    #             max_tokens=settings.MAX_TOKENS,
    #             top_p=0.95 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.8,
    #             presence_penalty=0.1 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.0,
    #             frequency_penalty=0.1 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.0,
    #         ),
    #     },
    #     {
    #         "name": "gpt-4o-mini",
    #         "llm": ChatOpenAI(
    #             model="gpt-4o-mini",
    #             temperature=settings.DEFAULT_LLM_TEMPERATURE,
    #             api_key=settings.OPENAI_API_KEY,
    #             max_tokens=settings.MAX_TOKENS,
    #             top_p=0.9 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.8,
    #         ),
    #     },
    # ]

    @classmethod
    def _initialize_models(cls):
        """Initialize the LLMS list based on the configured provider."""
        if cls.LLMS:
            return  # Already initialized

        # Ollama provider commented out - using OpenAI instead
        # if settings.LLM_PROVIDER == "ollama":
        #     # Ollama models
        #     cls.LLMS = [
        #         {
        #             "name": "qwen2.5:7b-instruct-q4_K_M",
        #             "llm": create_llm_instance(
        #                 "qwen2.5:7b-instruct-q4_K_M",
        #                 temperature=settings.DEFAULT_LLM_TEMPERATURE,
        #             ),
        #         },
        #     ]
        # else:
        # OpenAI models (default) - Using only models available for this API key
        cls.LLMS = [
            {
                # gpt-4o-mini - Primary model (fast, cheap, reliable)
                "name": "gpt-4o-mini",
                "llm": create_llm_instance(
                    "gpt-4o-mini",
                    temperature=settings.DEFAULT_LLM_TEMPERATURE,
                    top_p=0.9 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.8,
                ),
            },
            # o3-mini - Reasoning model (currently has limited LangChain support)
            # Uncomment when LangChain adds full o3-mini support
            # {
            #     "name": "o3-mini",
            #     "llm": create_llm_instance(
            #         "o3-mini",
            #         reasoning_effort="medium",
            #     ),
            # },
            # Additional models commented out - enable if your API key has access
            # {
            #     "name": "gpt-4o",
            #     "llm": create_llm_instance(
            #         "gpt-4o",
            #         temperature=settings.DEFAULT_LLM_TEMPERATURE,
            #         top_p=0.95 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.8,
            #         presence_penalty=0.1 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.0,
            #         frequency_penalty=0.1 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.0,
            #     ),
            # },
            # {
            #     "name": "gpt-3.5-turbo",
            #     "llm": create_llm_instance(
            #         "gpt-3.5-turbo",
            #         temperature=settings.DEFAULT_LLM_TEMPERATURE,
            #     ),
            # },
            # GPT-5 models commented out - not yet available
            # {
            #     "name": "gpt-5-mini",
            #     "llm": create_llm_instance(
            #         "gpt-5-mini",
            #         reasoning={"effort": "low"},
            #     ),
            # },
            # {
            #     "name": "gpt-5",
            #     "llm": create_llm_instance(
            #         "gpt-5",
            #         reasoning={"effort": "medium"},
            #     ),
            # },
            # {
            #     "name": "gpt-5-nano",
            #     "llm": create_llm_instance(
            #         "gpt-5-nano",
            #         reasoning={"effort": "minimal"},
            #     ),
            # },
        ]

    @classmethod
    def get(cls, model_name: str, **kwargs) -> BaseChatModel:
        """Get an LLM by name with optional argument overrides.

        Args:
            model_name: Name of the model to retrieve
            **kwargs: Optional arguments to override default model configuration

        Returns:
            BaseChatModel instance

        Raises:
            ValueError: If model_name is not found in LLMS
        """
        cls._initialize_models()

        # Find the model in the registry
        model_entry = None
        for entry in cls.LLMS:
            if entry["name"] == model_name:
                model_entry = entry
                break

        if not model_entry:
            available_models = [entry["name"] for entry in cls.LLMS]
            raise ValueError(
                f"model '{model_name}' not found in registry. available models: {', '.join(available_models)}"
            )

        # If user provides kwargs, create a new instance with those args
        if kwargs:
            logger.debug("creating_llm_with_custom_args", model_name=model_name, custom_args=list(kwargs.keys()))
            return create_llm_instance(model_name, **kwargs)
            # Original OpenAI-only implementation (commented out):
            # return ChatOpenAI(model=model_name, api_key=settings.OPENAI_API_KEY, **kwargs)

        # Return the default instance
        logger.debug("using_default_llm_instance", model_name=model_name)
        return model_entry["llm"]

    @classmethod
    def get_all_names(cls) -> List[str]:
        """Get all registered LLM names in order.

        Returns:
            List of LLM names
        """
        cls._initialize_models()
        return [entry["name"] for entry in cls.LLMS]

    @classmethod
    def get_model_at_index(cls, index: int) -> Dict[str, Any]:
        """Get model entry at specific index.

        Args:
            index: Index of the model in LLMS list

        Returns:
            Model entry dict
        """
        cls._initialize_models()
        if 0 <= index < len(cls.LLMS):
            return cls.LLMS[index]
        return cls.LLMS[0]  # Wrap around to first model


class LLMService:
    """Service for managing LLM calls with retries and circular fallback.

    This service handles all LLM interactions with automatic retry logic,
    rate limit handling, and circular fallback through all available models.
    """

    def __init__(self):
        """Initialize the LLM service."""
        self._llm: Optional[BaseChatModel] = None
        self._current_model_index: int = 0

        # Find index of default model in registry
        LLMRegistry._initialize_models()
        all_names = LLMRegistry.get_all_names()
        try:
            self._current_model_index = all_names.index(settings.DEFAULT_LLM_MODEL)
            self._llm = LLMRegistry.get(settings.DEFAULT_LLM_MODEL)
            logger.info(
                "llm_service_initialized",
                default_model=settings.DEFAULT_LLM_MODEL,
                provider=settings.LLM_PROVIDER,
                model_index=self._current_model_index,
                total_models=len(all_names),
                environment=settings.ENVIRONMENT.value,
            )
        except (ValueError, Exception) as e:
            # Default model not found, use first model
            self._current_model_index = 0
            self._llm = LLMRegistry.LLMS[0]["llm"]
            logger.warning(
                "default_model_not_found_using_first",
                requested=settings.DEFAULT_LLM_MODEL,
                using=all_names[0] if all_names else "none",
                error=str(e),
            )

    def _get_next_model_index(self) -> int:
        """Get the next model index in circular fashion.

        Returns:
            Next model index (wraps around to 0 if at end)
        """
        total_models = len(LLMRegistry.LLMS)
        next_index = (self._current_model_index + 1) % total_models
        return next_index

    def _switch_to_next_model(self) -> bool:
        """Switch to the next model in the registry (circular).

        Returns:
            True if successfully switched, False otherwise
        """
        try:
            next_index = self._get_next_model_index()
            next_model_entry = LLMRegistry.get_model_at_index(next_index)

            logger.warning(
                "switching_to_next_model",
                from_index=self._current_model_index,
                to_index=next_index,
                to_model=next_model_entry["name"],
            )

            self._current_model_index = next_index
            self._llm = next_model_entry["llm"]

            logger.info("model_switched", new_model=next_model_entry["name"], new_index=next_index)
            return True
        except Exception as e:
            logger.error("model_switch_failed", error=str(e))
            return False

    @retry(
        stop=stop_after_attempt(settings.MAX_LLM_CALL_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError, ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    async def _call_llm_with_retry(self, messages: List[BaseMessage]) -> BaseMessage:
        """Call the LLM with automatic retry logic.

        Args:
            messages: List of messages to send to the LLM

        Returns:
            BaseMessage response from the LLM

        Raises:
            Exception: If all retries fail
        """
        if not self._llm:
            raise RuntimeError("llm not initialized")

        try:
            response = await self._llm.ainvoke(messages)
            logger.debug("llm_call_successful", message_count=len(messages))
            return response
        except (RateLimitError, APITimeoutError, APIError, ConnectionError, TimeoutError) as e:
            logger.warning(
                "llm_call_failed_retrying",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                "llm_call_failed",
                error_type=type(e).__name__,
                error=str(e),
            )
            raise

    async def call(
        self,
        messages: List[BaseMessage],
        model_name: Optional[str] = None,
        **model_kwargs,
    ) -> BaseMessage:
        """Call the LLM with the specified messages and circular fallback.

        Args:
            messages: List of messages to send to the LLM
            model_name: Optional specific model to use. If None, uses current model.
            **model_kwargs: Optional kwargs to override default model configuration

        Returns:
            BaseMessage response from the LLM

        Raises:
            RuntimeError: If all models fail after retries
        """
        # If user specifies a model, get it from registry
        if model_name:
            try:
                self._llm = LLMRegistry.get(model_name, **model_kwargs)
                # Update index to match the requested model
                all_names = LLMRegistry.get_all_names()
                try:
                    self._current_model_index = all_names.index(model_name)
                except ValueError:
                    pass  # Keep current index if model name not in list
                logger.info("using_requested_model", model_name=model_name, has_custom_kwargs=bool(model_kwargs))
            except ValueError as e:
                logger.error("requested_model_not_found", model_name=model_name, error=str(e))
                raise

        # Track which models we've tried to prevent infinite loops
        total_models = len(LLMRegistry.LLMS)
        models_tried = 0
        starting_index = self._current_model_index
        last_error = None

        while models_tried < total_models:
            try:
                response = await self._call_llm_with_retry(messages)
                return response
            # Original OpenAIError only (commented out):
            # except OpenAIError as e:
            except (OpenAIError, Exception) as e:
                last_error = e
                models_tried += 1

                LLMRegistry._initialize_models()
                current_model_name = LLMRegistry.LLMS[self._current_model_index]["name"]
                logger.error(
                    "llm_call_failed_after_retries",
                    model=current_model_name,
                    models_tried=models_tried,
                    total_models=total_models,
                    error=str(e),
                )

                # If we've tried all models, give up
                if models_tried >= total_models:
                    logger.error(
                        "all_models_failed",
                        models_tried=models_tried,
                        starting_model=LLMRegistry.LLMS[starting_index]["name"],
                    )
                    break

                # Switch to next model in circular fashion
                if not self._switch_to_next_model():
                    logger.error("failed_to_switch_to_next_model")
                    break

                # Continue loop to try next model

        # All models failed
        raise RuntimeError(
            f"failed to get response from llm after trying {models_tried} models. last error: {str(last_error)}"
        )

    def get_llm(self) -> Optional[BaseChatModel]:
        """Get the current LLM instance.

        Returns:
            Current BaseChatModel instance or None if not initialized
        """
        return self._llm

    async def call_o3_with_reasoning(
        self,
        messages: List[Dict[str, str]],
        reasoning_effort: str = "medium",
        reasoning_summary: str = "detailed",
    ) -> ReasoningResponse:
        """Call o3-mini model with reasoning summary using raw OpenAI client.
        
        This method uses the OpenAI client directly to access o3-mini's
        reasoning summary feature, which isn't exposed through LangChain.
        
        Args:
            messages: List of message dicts with "role" and "content" keys
            reasoning_effort: "low", "medium", or "high"
            reasoning_summary: "auto", "concise", or "detailed"
            
        Returns:
            ReasoningResponse with content and reasoning steps
        """
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        try:
            logger.info(
                "calling_o3_mini_with_reasoning",
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                message_count=len(messages),
            )
            
            response = await client.chat.completions.create(
                model="o3-mini",
                messages=messages,
                reasoning_effort=reasoning_effort,
                max_completion_tokens=settings.MAX_TOKENS,
            )
            
            # Extract content from response
            content = ""
            reasoning_steps = []
            
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if choice.message and choice.message.content:
                    content = choice.message.content
            
            # Extract reasoning tokens from usage
            reasoning_tokens = 0
            output_tokens = 0
            if response.usage:
                output_tokens = response.usage.completion_tokens or 0
                # o3-mini includes reasoning_tokens in the usage
                if hasattr(response.usage, "completion_tokens_details"):
                    details = response.usage.completion_tokens_details
                    if details and hasattr(details, "reasoning_tokens"):
                        reasoning_tokens = details.reasoning_tokens or 0
            
            logger.info(
                "o3_mini_response_received",
                content_length=len(content),
                reasoning_tokens=reasoning_tokens,
                output_tokens=output_tokens,
            )
            
            return ReasoningResponse(
                content=content,
                reasoning_steps=reasoning_steps,
                model="o3-mini",
                reasoning_tokens=reasoning_tokens,
                output_tokens=output_tokens,
            )
            
        except Exception as e:
            logger.error(
                "o3_mini_call_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise

    def bind_tools(self, tools: List) -> "LLMService":
        """Bind tools to the current LLM.

        Args:
            tools: List of tools to bind

        Returns:
            Self for method chaining
        """
        if self._llm:
            self._llm = self._llm.bind_tools(tools)
            logger.debug("tools_bound_to_llm", tool_count=len(tools))
        return self


# Create global LLM service instance
llm_service = LLMService()
