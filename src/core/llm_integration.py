import google.generativeai as genai
from openai import OpenAI # Import OpenAI library
import os
from dotenv import load_dotenv
from typing import Optional, Any # Added Any for now
from abc import ABC, abstractmethod # Added ABC and abstractmethod
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Configuration from environment variables ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower() # Default to Gemini
# Gemini Config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-pro")
# DeepSeek Config (using OpenAI compatible API)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE") # e.g., "https://api.deepseek.com/v1"
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")


# --- LLM Client Interface (Abstract Base Class) ---
class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generates text based on the provided prompt.

        Args:
            prompt: The input prompt for the model.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated text as a string, or None if generation failed.
        """
        pass

# --- Concrete Gemini Client Implementation ---
class GeminiClient(BaseLLMClient):
    """LLM Client implementation for Google Gemini."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = GEMINI_MODEL_NAME):
        self.api_key = api_key or GOOGLE_API_KEY
        self.model_name = model_name
        self._configure_gemini()
        self.model = genai.GenerativeModel(self.model_name)
        logger.info(f"GeminiClient initialized with model: {self.model_name}")

    def _configure_gemini(self):
        """Configures the Gemini API."""
        if not self.api_key:
            logger.error("GOOGLE_API_KEY environment variable not set and no api_key provided.")
            raise ValueError("GOOGLE_API_KEY not configured.")
        try:
            genai.configure(api_key=self.api_key)
            logger.debug("Gemini API configured.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}", exc_info=True)
            raise

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generates text using the configured Google Gemini model."""
        try:
            # Ensure API is configured (in case configuration expires or changes)
            # self._configure_gemini() # Configuration is usually done once at init
            logger.debug(f"Generating text with Gemini model: {self.model_name}")
            # Pass kwargs if needed, though generate_content might not use them directly
            response = self.model.generate_content(prompt, **kwargs)
            logger.debug(f"Gemini response finished reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'N/A'}")
            # Handle potential content safety blocks or empty responses
            if not response.parts:
                 logger.warning(f"Gemini response has no parts. Feedback: {response.prompt_feedback}")
                 # Check if block_reason indicates safety issue
                 if response.prompt_feedback and response.prompt_feedback.block_reason:
                     return f"Error: Content generation blocked due to {response.prompt_feedback.block_reason}. Finish reason: {response.prompt_feedback.finish_reason}"
                 return None # No parts and no explicit block reason
            return response.text
        except Exception as e:
            logger.error(f"Error generating text with Gemini ({self.model_name}): {e}", exc_info=True)
            return f"Error: Exception during Gemini generation: {e}" # Return error message instead of None


# --- Concrete DeepSeek Client Implementation (OpenAI compatible) ---
class DeepSeekClient(BaseLLMClient):
    """LLM Client implementation for DeepSeek (using OpenAI compatible API)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model_name: str = DEEPSEEK_MODEL_NAME):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.base_url = base_url or DEEPSEEK_API_BASE
        self.model_name = model_name

        if not self.api_key or not self.base_url:
            logger.error("DEEPSEEK_API_KEY or DEEPSEEK_API_BASE not configured.")
            raise ValueError("DeepSeek API key or base URL not configured.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"DeepSeekClient initialized with model: {self.model_name} via base URL: {self.base_url}")

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generates text using the configured DeepSeek model via OpenAI compatible API."""
        try:
            logger.debug(f"Generating text with DeepSeek model: {self.model_name}")
            # Combine prompt with standard message format
            messages = [{"role": "user", "content": prompt}]
            # Merge any additional kwargs like temperature, max_tokens
            request_params = {"model": self.model_name, "messages": messages, **kwargs}

            response = self.client.chat.completions.create(**request_params)

            # Check if response and choices are valid
            if response and response.choices:
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                logger.debug(f"DeepSeek response finished reason: {finish_reason}")
                # Handle potential safety blocks indicated by finish_reason
                if finish_reason == 'content_filter':
                     logger.warning("DeepSeek generation stopped due to content filter.")
                     return "Error: Content generation blocked by content filter."
                return content.strip() if content else None
            else:
                 logger.warning(f"DeepSeek response structure invalid or empty choices: {response}")
                 # Try to get error details if available
                 error_details = getattr(response, 'error', 'Unknown error')
                 return f"Error: Invalid response structure from DeepSeek. Details: {error_details}"
        except Exception as e:
            logger.error(f"Error generating text with DeepSeek ({self.model_name}): {e}", exc_info=True)
            return f"Error: Exception during DeepSeek generation: {e}"


# --- Factory Function to get the appropriate LLM Client ---
def get_llm_client(provider: Optional[str] = None, **kwargs) -> BaseLLMClient:
    """
    Factory function to instantiate and return the appropriate LLM client
    based on the LLM_PROVIDER environment variable or the provider argument.

    Args:
        provider (Optional[str]): Override the LLM provider (e.g., "gemini", "deepseek").
                                   If None, uses the LLM_PROVIDER environment variable.
        **kwargs: Additional arguments to pass to the specific client's constructor
                  (e.g., api_key, model_name).

    Returns:
        An instance of BaseLLMClient (either GeminiClient or DeepSeekClient).

    Raises:
        ValueError: If the provider is unknown or configuration is missing.
    """
    selected_provider = (provider or LLM_PROVIDER).lower()
    logger.info(f"Attempting to get LLM client for provider: {selected_provider}")

    if selected_provider == "gemini":
        try:
            return GeminiClient(**kwargs)
        except ValueError as e:
            logger.error(f"Failed to initialize GeminiClient: {e}")
            raise
        except Exception as e:
             logger.error(f"Unexpected error initializing GeminiClient: {e}", exc_info=True)
             raise ValueError(f"Could not initialize GeminiClient: {e}") from e
    elif selected_provider == "deepseek":
        try:
            return DeepSeekClient(**kwargs)
        except ValueError as e:
            logger.error(f"Failed to initialize DeepSeekClient: {e}")
            raise
        except Exception as e:
             logger.error(f"Unexpected error initializing DeepSeekClient: {e}", exc_info=True)
             raise ValueError(f"Could not initialize DeepSeekClient: {e}") from e
    else:
        logger.error(f"Unknown LLM provider specified: {selected_provider}")
        raise ValueError(f"Unsupported LLM provider: {selected_provider}. Choose 'gemini' or 'deepseek'.")


# Example usage (updated to use the factory and client instance)
if __name__ == '__main__':
    # Import setup_logging relative to src potentially
    try:
        from src.utils.log_setup import setup_logging # Assuming log setup is in utils
    except ImportError:
        # Fallback basic config if utils not found during simple script run
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        def setup_logging(**kwargs): pass # Dummy function


    # Setup logging to see debug messages during test
    setup_logging(log_level=logging.DEBUG)

    test_prompt = "Explain the concept of 'Zero-Shot Learning' in simple terms."

    try:
        # Get the client using the factory function
        # The factory will read the .env file based on load_dotenv() at the top
        logger.info(f"--- Testing LLM Integration (Provider from env: {LLM_PROVIDER}) ---")
        llm_client = get_llm_client() # Use configured provider

        # Generate text using the client instance's method
        generated_content = llm_client.generate(test_prompt)

        if generated_content:
            # Check if generation resulted in an error message returned by the client
            if generated_content.startswith("Error:"):
                 logger.error(f"\\nGeneration failed: {generated_content}")
            else:
                 logger.info("\\n--- Generation Result ---")
                 logger.info(generated_content)
        else:
            # Handle cases where generate returns None (less likely now with error strings)
            logger.error("\\nFailed to generate content (received None).")

        # Example overriding provider (if credentials for both are set)
        # logger.info("\\n--- Testing DeepSeek Override ---")
        # try:
        #     deepseek_client = get_llm_client(provider="deepseek")
        #     deepseek_content = deepseek_client.generate("What is DeepSeek?")
        #     logger.info(deepseek_content or "Failed")
        # except ValueError as e:
        #     logger.error(f"Could not test DeepSeek override: {e}")


    except ValueError as ve:
        logger.error(f"Configuration or Initialization error: {ve}")
    except Exception as ex:
        logger.exception(f"An unexpected error occurred during test: {ex}", exc_info=True)
