import ollama
from ollama import Options, Message as OllamaMessage
import time
from typing import cast, Optional, List, Iterator, Mapping, Any
from llms.model import BaseModel

class OllamaModel(BaseModel):
    """
    An LLM wrapper for Ollama models.
    """
    def __init__(self,model_name: Optional[str] = None, 
                    context_window: Optional[int] = None, 
                    temperature: Optional[float] = None, 
                    max_messages: Optional[int] = None,
                    message_window_size: Optional[int] = None,
                    verbose: bool = False,
                    stream_generate_to_console: bool = False) -> None:
       
        super().__init__(api_type = "ollama", 
                         verbose = verbose, stream_generate_to_console = stream_generate_to_console)
        
        self.model_name: str = model_name or self.CONFIG['model_name']
        self.context_window: int = context_window or self.CONFIG['generation_context_window']
        self.temperature: float = temperature or self.CONFIG['default_temperature']
        
        self.max_messages = max_messages or self.CONFIG['max_messages']
        self.message_window_size = message_window_size or self.CONFIG['message_window_size']

        self.default_system_prompt: str = self.CONFIG['default_system_prompt']

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for a given text using the model.

        Args:
            text (str): The input text to generate embeddings for.

        Returns:
            List[float]: A list of floating-point numbers representing the embeddings.
        """
        response = ollama.embeddings(model=self.model_name, prompt=text)
        if 'embedding' not in response:
            self.logger.error("No embedding found in the response.")
            raise ValueError("Failed to generate embedding.")
        return list(response['embedding'])


    def generate(self, user_prompt: str, 
                       system_prompt  : Optional[str] = None,
                       temperature    : Optional[float] = None, 
                       context_window : Optional[int] = None,
                       num_predict: Optional[int] = None) -> str:
                      
        """
        Generate a response from the LLM model.

        Args:
            user_prompt (str): The user prompt to send to the model.
            system_prompt (Optional[str]): The system prompt to use. Defaults to the default system prompt.
            temperature (Optional[float]): The sampling temperature. Defaults to the set temperature.
            context_window (Optional[int]): The token limit for the context window. Defaults to the set value.

        Returns:
            str: The model's response as a string.
        """

        # default overrides
        gen_temperature   :float  = temperature or self.temperature
        gen_context_window:int = context_window or self.context_window
        gen_system_prompt :str = system_prompt or self.default_system_prompt

        start_time = time.time() # generation timing
        
        # Prepare parameters for ollama.generate
        generate_params = {
            'model': self.model_name,
            'options': Options(num_ctx=gen_context_window, temperature=gen_temperature),
            'system': gen_system_prompt,
            'prompt': user_prompt
        }

        if self.stream_generate_to_console:
            generate_params['stream'] = True

        if num_predict:
            generate_params['options']['num_predict'] = num_predict

        response: Any = ollama.generate(**generate_params)

        if self.stream_generate_to_console:
            final_response = ""
            self.logger.info("Start to string response")
            for chunk in response:
                final_response += chunk['response']
                print(chunk['response'], end='', flush=True)
        else:
            final_response = response['response']

        elapsed_time = time.time() - start_time
        
        if self.verbose:
            self.logger.info(f"Response generated in {elapsed_time:.2f} seconds")
            self.logger.info(f"FinalResponse: {response}")

        return final_response
    
    def chat(self, user_prompt: str,
                system_prompt: Optional[str] = None, 
                temperature: Optional[float] = None, 
                context_window: Optional[int] = None,
                num_predict: Optional[int] = None) -> Iterator[str]:
        """
        Chat with the model, automatically compressing history if needed, and generating a response.

        Args:
            user_prompt (str): The user's message to send to the model.
            system_prompt (Optional[str]): An optional system message to prepend to the conversation.
            temperature (Optional[float]): The sampling temperature for response generation.
            context_window (Optional[int]): The maximum token context window.

        Returns:
            Iterator[str]: An iterator that streams the assistant's response.
        """
        if system_prompt is not None:
            self.append_to_history("system", system_prompt)
        
        self.append_to_history("user", user_prompt)
        self.logger.info(f"User Prompt: {user_prompt}")
        
        if self.verbose:    
            if system_prompt is not None:
                self.logger.info(f"System Prompt: {system_prompt}")
            else:
                self.logger.info(f"No system prompt provided.")
           

        self.manage_message_history() # this will compress the message history if needed

        ollama_messages = [OllamaMessage(role=msg['role'], content=msg['content']) for msg in self.message_history]

        # Prepare parameters for ollama.generate
        generate_params = {
            'model': self.model_name,
            'options': Options(num_ctx=context_window or self.context_window, 
                               temperature=temperature or self.temperature),
            'messages': ollama_messages,
            'stream': True
        }
        if num_predict:
            generate_params['options']['num_predict'] = num_predict

        response: Iterator[Mapping] = ollama.chat(
            **generate_params
        )

        def response_generator() -> Iterator[str]:
            assistant_response = ''
            for chunk in response:
                assistant_response += chunk['message']['content']
                yield chunk['message']['content']

            self.append_to_history("assistant", assistant_response)

        return response_generator()