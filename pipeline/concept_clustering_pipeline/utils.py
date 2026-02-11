import yaml
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional
import os

# dotenv for loading .env file
try:
    from dotenv import load_dotenv
    from pathlib import Path
    # Load .env file from current directory
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        import logging
        logging.getLogger(__name__).debug(f"Loaded .env file from: {env_path}")
    else:
        # Try loading from current working directory
        load_dotenv()
except ImportError:
    # Environment variables can still be used without dotenv
    pass
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"Failed to load .env file: {e}")


class Config:
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            logging.warning(f"Configuration file not found: {self.config_path}. Using defaults")
            return {}
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            try:
                config = yaml.safe_load(f)
                return config if config else {}
            except Exception as e:
                logging.error(f"Configuration file load error: {e}")
                return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


def setup_logging(config: Config):
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = config.get('logging.file', None)
    
    # Set logging level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Set default formatter
    formatter = logging.Formatter(log_format)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler (if configured)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress verbose HTTP logging from httpx and urllib3
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def save_intermediate_result(data: Any, file_path: Path, description: str = ""):
    logger = logging.getLogger(__name__)
    
    # Create directory
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        desc = f" ({description})" if description else ""
        logger.info(f"Intermediate result saved: {file_path}{desc}")
    except Exception as e:
        logger.error(f"Failed to save intermediate result: {file_path}, error: {e}")


class LLMClient:
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # API settings
        self.api_type = config.get('llm.api_type', 'openai')  # 'openai'
        
        # API key load order: config file -> .env file -> environment variable
        config_api_key = config.get('llm.api_key')
        
        # Try to load from environment variables (including .env file loaded by dotenv)
        env_openai_key = os.getenv('OPENAI_API_KEY')
        
        # Debug: Check if .env was loaded
        if not env_openai_key:
            self.logger.debug("No API key found in environment variables. Checking .env file...")
            # Try to manually load .env if dotenv didn't work
            try:
                from dotenv import load_dotenv
                from pathlib import Path
                env_file = Path('.env')
                if env_file.exists():
                    load_dotenv(dotenv_path=env_file, override=True)
                    env_openai_key = os.getenv('OPENAI_API_KEY')
                    if env_openai_key:
                        self.logger.info("Successfully loaded API key from .env file")
            except Exception as e:
                self.logger.debug(f"Could not manually load .env: {e}")
        
        self.api_key = (
            config_api_key or 
            env_openai_key
        )
        
        # Log API key loading status (without exposing the key)
        if self.api_key:
            key_source = "config file" if config_api_key else ("environment variable" if env_openai_key else ".env file")
            self.logger.info(f"API key loaded from {key_source} (length: {len(self.api_key)} characters, starts with: {self.api_key[:7]}...)")
        else:
            self.logger.error(
                "API key not found! "
                "Please set OPENAI_API_KEY in .env file or as environment variable."
            )
        
        # Set GPT-5.2 as default model
        self.model = config.get('llm.model', 'gpt-5.2')
        self.max_retries = config.get('llm.max_retries', 3)
        self.retry_delay = config.get('llm.retry_delay', 1.0)
        self.temperature = config.get('llm.temperature', 0.7)
        self.max_tokens = config.get('llm.max_tokens', 2000)
        
        # Initialize API client
        self.client = self._init_client()
    
    def _init_client(self):
        if self.api_type == 'openai':
            try:
                import openai
                # OpenAI v1.0+ API
                if hasattr(openai, 'OpenAI'):
                    if self.api_key:
                        self.logger.info(f"Initializing OpenAI client with API key (key length: {len(self.api_key)})")
                        client = openai.OpenAI(api_key=self.api_key)
                        # Verify API key is set
                        if not client.api_key:
                            self.logger.error("API key was not set in OpenAI client!")
                            # Try again with explicit key
                            client = openai.OpenAI(api_key=self.api_key)
                        return client
                    else:
                        # Try to get from environment
                        env_key = os.getenv('OPENAI_API_KEY')
                        if env_key:
                            self.logger.info("Initializing OpenAI client with API key from environment variable")
                            return openai.OpenAI(api_key=env_key)
                        else:
                            self.logger.warning("No API key provided. Initializing OpenAI client without explicit API key (may fail)")
                            return openai.OpenAI()  # Get API key from environment variable
                # OpenAI v0.x API (legacy)
                else:
                    if self.api_key:
                        openai.api_key = self.api_key
                        self.logger.info("Initializing OpenAI client (legacy) with provided API key")
                    else:
                        env_key = os.getenv('OPENAI_API_KEY')
                        if env_key:
                            openai.api_key = env_key
                            self.logger.info("Initializing OpenAI client (legacy) with API key from environment")
                        else:
                            self.logger.warning("No API key found for legacy OpenAI client")
                    return openai
            except ImportError:
                self.logger.warning("openai package is not installed. Please install with 'pip install openai'.")
                return None
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
                return None
        else:
            self.logger.warning(f"Unsupported API type: {self.api_type}")
            return None
    
    def _call_openai(self, prompt: str, user_input: str) -> Optional[str]:
        if self.client is None:
            self.logger.error("OpenAI client is None. Cannot make API call.")
            return None
        
        # Verify API key is available
        if not self.api_key:
            # Try to get from environment one more time
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                self.logger.error("No API key available for OpenAI API call")
                return None
        
        try:
            # Latest models like GPT-5.2 use max_completion_tokens
            # Legacy models use max_tokens
            use_max_completion_tokens = 'gpt-5' in self.model.lower() or 'o1' in self.model.lower()
            
            # OpenAI v1.0+ API (OpenAI client object)
            if hasattr(self.client, 'chat'):
                kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_input}
                    ],
                    "temperature": self.temperature
                }
                
                if use_max_completion_tokens:
                    kwargs["max_completion_tokens"] = self.max_tokens
                else:
                    kwargs["max_tokens"] = self.max_tokens
                
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content:
                    self.logger.debug(f"API response received: {len(content)} characters")
                else:
                    self.logger.warning("API returned empty content")
                return content
            # OpenAI v0.x API (legacy module)
            elif hasattr(self.client, 'ChatCompletion'):
                kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_input}
                    ],
                    "temperature": self.temperature
                }
                
                if use_max_completion_tokens:
                    kwargs["max_completion_tokens"] = self.max_tokens
                else:
                    kwargs["max_tokens"] = self.max_tokens
                
                response = self.client.ChatCompletion.create(**kwargs)
                content = response.choices[0].message.content
                if content:
                    self.logger.debug(f"API response received: {len(content)} characters")
                else:
                    self.logger.warning("API returned empty content")
                return content
            else:
                self.logger.error("Cannot recognize OpenAI client format.")
                return None
        except Exception as e:
            self.logger.error(f"OpenAI API call error: {e}")
            return None
    
    def get_response(self, prompt: str, user_input: str) -> Optional[str]:
        if self.api_type == 'openai':
            return self._call_openai(prompt, user_input)
        else:
            self.logger.error(f"Unsupported API type: {self.api_type}")
            return None
    
    def get_response_with_retry(self, prompt: str, user_input: str) -> Optional[str]:
        for attempt in range(self.max_retries):
            response = self.get_response(prompt, user_input)
            
            if response is not None:
                return response
            
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                self.logger.warning(f"LLM response failed. Retrying after {wait_time:.1f}s ({attempt + 1}/{self.max_retries})")
                time.sleep(wait_time)
        
        self.logger.error(f"LLM response failed: failed after {self.max_retries} retries")
        return None


class PromptLoader:
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Prompt directory
        prompt_dir = config.get('prompts.directory', 'prompts')
        self.prompt_dir = Path(prompt_dir)
    
    def load_prompt(self, prompt_file: str) -> str:
        # If absolute path
        prompt_path = Path(prompt_file)
        if prompt_path.is_absolute() or prompt_path.exists():
            full_path = prompt_path
        else:
            # If relative path, search in prompt directory
            full_path = self.prompt_dir / prompt_file
        
        if not full_path.exists():
            self.logger.warning(f"Prompt file not found: {full_path}")
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"Prompt file load error: {e}")
            raise

