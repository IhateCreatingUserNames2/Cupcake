import logging
import os
import json
from datetime import datetime


class PromptLogger:
    """
    A logging system to capture and analyze OpenAI API prompts.
    """

    def __init__(self, log_dir="prompt_logs", log_file=None):
        """
        Initialize the prompt logger

        Args:
            log_dir: Directory to store log files
            log_file: Optional specific log filename (defaults to timestamp-based name)
        """
        self.log_dir = log_dir

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Set up log file name with timestamp if not provided
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"openai_prompts_{timestamp}.log"

        self.log_file_path = os.path.join(log_dir, log_file)

        # Configure logger
        self.logger = logging.getLogger("prompt_logger")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Counter for logging
        self.prompt_counter = 0

        self.logger.info(f"PromptLogger initialized, logging to {self.log_file_path}")

    def log_prompt(self, module_name, function_name, prompt, model=None, temperature=None, max_tokens=None,
                   other_params=None):
        """
        Log an OpenAI prompt with metadata

        Args:
            module_name: Name of the module making the request
            function_name: Name of the function making the request
            prompt: The actual prompt text or messages array
            model: OpenAI model being used
            temperature: Temperature setting
            max_tokens: Max tokens setting
            other_params: Dictionary of other parameters
        """
        self.prompt_counter += 1

        log_entry = {
            "prompt_id": self.prompt_counter,
            "timestamp": datetime.now().isoformat(),
            "module": module_name,
            "function": function_name,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt": prompt
        }

        # Add other parameters if provided
        if other_params:
            log_entry.update(other_params)

        # Log as formatted JSON
        self.logger.debug(json.dumps(log_entry, indent=2, ensure_ascii=False))

        # Also log a short summary to console
        prompt_summary = str(prompt)
        if isinstance(prompt, list):
            # For message-based prompts, extract content from the last user message
            for msg in reversed(prompt):
                if msg.get('role') == 'user':
                    prompt_summary = msg.get('content', '')[:50]
                    break
        else:
            prompt_summary = prompt[:50]

        self.logger.info(f"Prompt #{self.prompt_counter} from {module_name}.{function_name}: {prompt_summary}...")

        return self.prompt_counter

    def log_response(self, prompt_id, response_text, token_usage=None, response_time=None):
        """
        Log a response from OpenAI

        Args:
            prompt_id: ID of the prompt this is responding to
            response_text: The text of the response
            token_usage: Token usage information
            response_time: Time taken for response
        """
        log_entry = {
            "prompt_id": prompt_id,
            "timestamp": datetime.now().isoformat(),
            "response_text": response_text[:500] + ("..." if len(response_text) > 500 else ""),
            "response_length": len(response_text),
            "token_usage": token_usage,
            "response_time": response_time
        }

        # Log as formatted JSON
        self.logger.debug(json.dumps(log_entry, indent=2, ensure_ascii=False))

        # Summary log
        self.logger.info(
            f"Response to prompt #{prompt_id}: {len(response_text)} chars, {token_usage} tokens, {response_time:.2f}s")


# Create a decorator to automatically log OpenAI calls
def log_openai_prompt(prompt_logger, module_name=None):
    """
    Decorator for logging OpenAI prompts

    Args:
        prompt_logger: PromptLogger instance
        module_name: Name of the module (defaults to function's module)
    """

    def decorator(func):
        import functools
        import time

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get module name if not provided
            nonlocal module_name
            if module_name is None:
                module_name = func.__module__

            function_name = func.__name__

            # Extract prompt, model, and other parameters
            prompt = kwargs.get('messages', kwargs.get('prompt', 'Unknown prompt'))
            model = kwargs.get('model', 'Unknown model')
            temperature = kwargs.get('temperature', None)
            max_tokens = kwargs.get('max_tokens', None)

            # Log the prompt
            prompt_id = prompt_logger.log_prompt(
                module_name,
                function_name,
                prompt,
                model,
                temperature,
                max_tokens,
                {k: v for k, v in kwargs.items() if
                 k not in ['messages', 'prompt', 'model', 'temperature', 'max_tokens']}
            )

            # Time the API call
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            # Extract the response
            response_text = "Unknown response"
            token_usage = None

            if hasattr(result, 'choices') and result.choices:
                if hasattr(result.choices[0], 'message'):
                    response_text = result.choices[0].message.content
                elif hasattr(result.choices[0], 'text'):
                    response_text = result.choices[0].text

            if hasattr(result, 'usage'):
                token_usage = result.usage.total_tokens

            # Log the response
            prompt_logger.log_response(
                prompt_id,
                response_text,
                token_usage,
                end_time - start_time
            )

            return result

        return wrapper

    return decorator


# Patching OpenAI client for automatic logging
def patch_openai_client(client, prompt_logger):
    """
    Patch an OpenAI client to automatically log all requests

    Args:
        client: OpenAI client instance
        prompt_logger: PromptLogger instance

    Returns:
        The patched client
    """
    # Store original methods
    original_completion_create = client.completions.create
    original_chat_completion_create = client.chat.completions.create

    # Patch completions.create
    client.completions.create = log_openai_prompt(prompt_logger, "openai.completions")(original_completion_create)

    # Patch chat.completions.create
    client.chat.completions.create = log_openai_prompt(prompt_logger, "openai.chat.completions")(
        original_chat_completion_create)

    prompt_logger.logger.info("OpenAI client patched for automatic prompt logging")

    return client


