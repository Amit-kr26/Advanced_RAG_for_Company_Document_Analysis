import os
import json
from dotenv import load_dotenv
from typing import Union, List, Dict, Type, Optional, Literal
from openai import OpenAI
import asyncio
from src.api_request_parallel_processor import process_api_requests_from_file
from openai.lib._parsing import type_to_response_format_param 
import tiktoken
import src.prompts as prompts
from json_repair import repair_json
from pydantic import BaseModel
import google.generativeai as genai
from copy import deepcopy
from tenacity import retry, stop_after_attempt, wait_fixed


class BaseOpenaiProcessor:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.default_model = 'gpt-4o-2024-08-06'

    def set_up_llm(self):
        """Sets up the OpenAI client with API key and configuration.

        Returns:
            OpenAI: Configured OpenAI client instance.
        """
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
            )
        return llm

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None
        ):
        """Sends a message to the OpenAI API and processes the response.

        Args:
            model (str, optional): The model to use for the request. Defaults to None.
            temperature (float): Controls randomness of the response. Defaults to 0.5.
            seed (int, optional): Seed for deterministic outputs. Defaults to None.
            system_content (str): The system prompt. Defaults to 'You are a helpful assistant.'.
            human_content (str): The user prompt. Defaults to 'Hello!'.
            is_structured (bool): Whether to expect a structured response. Defaults to False.
            response_format (type, optional): Schema for structured response parsing. Defaults to None.

        Returns:
            Union[str, dict]: The response content, either as a string or parsed dict.
        """
        if model is None:
            model = self.default_model
        params = {
            "model": model,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content}
            ]
        }
        
        if "o3-mini" not in model:
            params["temperature"] = temperature
            
        if not is_structured:
            completion = self.llm.chat.completions.create(**params)
            content = completion.choices[0].message.content
        elif is_structured:
            params["response_format"] = response_format
            completion = self.llm.beta.chat.completions.parse(**params)
            response = completion.choices[0].message.parsed
            content = response.dict()

        self.response_data = {"model": completion.model, "input_tokens": completion.usage.prompt_tokens, "output_tokens": completion.usage.completion_tokens}
        print(self.response_data)
        return content

    @staticmethod
    def count_tokens(string, encoding_name="o200k_base"):
        """Counts the number of tokens in a string using the specified encoding.

        Args:
            string (str): The string to tokenize.
            encoding_name (str): The name of the token encoding scheme. Defaults to 'o200k_base'.

        Returns:
            int: The number of tokens in the string.
        """
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(string)
        token_count = len(tokens)
        return token_count


class BaseGeminiProcessor:
    def __init__(self):
        self.llm = self._set_up_llm()
        self.default_model = 'gemini-2.0-flash-001'
        
    def _set_up_llm(self):
        """Sets up the Gemini client with API key configuration.

        Returns:
            module: Configured google.generativeai module.
        """
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        return genai

    def list_available_models(self):
        """Prints available Gemini models that support text generation."""
        print("Available models for text generation:")
        for model in self.llm.list_models():
            if "generateContent" in model.supported_generation_methods:
                print(f"- {model.name}")
                print(f"  Input token limit: {model.input_token_limit}")
                print(f"  Output token limit: {model.output_token_limit}")
                print()

    @retry(
        wait=wait_fixed(20),
        stop=stop_after_attempt(3),
        before_sleep=lambda retry_state: print(f"\nAPI Error encountered: {str(retry_state.outcome.exception())}\nWaiting 20 seconds before retry...\n")
    )
    def _generate_with_retry(self, model, human_content, generation_config):
        """Generates content with retry logic for robustness.

        Args:
            model: The Gemini model instance for generation.
            human_content (str): The user prompt to process.
            generation_config (dict): Configuration for generation (e.g., temperature).

        Returns:
            object: The generated response object.
        """
        try:
            return model.generate_content(
                human_content,
                generation_config=generation_config
            )
        except Exception as e:
            if getattr(e, '_attempt_number', 0) == 3:
                print(f"\nRetry failed. Error: {str(e)}\n")
            raise

    def _parse_structured_response(self, response_text, response_format):
        """Parses a structured response into a validated format.

        Args:
            response_text (str): The raw response text to parse.
            response_format (type): The Pydantic model for validation.

        Returns:
            Union[dict, str]: The parsed and validated response, or the reparsed result on error.
        """
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            validated_data = response_format.model_validate(parsed_dict)
            return validated_data.model_dump()
        except Exception as err:
            print(f"Error parsing structured response: {err}")
            print("Attempting to reparse the response...")
            reparsed = self._reparse_response(response_text, response_format)
            return reparsed

    def _reparse_response(self, response, response_format):
        """Reparses invalid JSON responses using the model itself.

        Args:
            response (str): The original response to reparse.
            response_format (type): The Pydantic model for validation.

        Returns:
            Union[dict, str]: The reparsed and validated response, or the raw response on error.
        """
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=prompts.AnswerSchemaFixPrompt.system_prompt,
            response=response
        )
        
        try:
            reparsed_response = self.send_message(
                model="gemini-2.0-flash-001",
                system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
                human_content=user_prompt,
                is_structured=False
            )
            try:
                repaired_json = repair_json(reparsed_response)
                reparsed_dict = json.loads(repaired_json)
                try:
                    validated_data = response_format.model_validate(reparsed_dict)
                    print("Reparsing successful!")
                    return validated_data.model_dump()
                except Exception:
                    return reparsed_dict
            except Exception as reparse_err:
                print(f"Reparse failed with error: {reparse_err}")
                print(f"Reparsed response: {reparsed_response}")
                return response
        except Exception as e:
            print(f"Reparse attempt failed: {e}")
            return response

    def send_message(
        self,
        model=None,
        temperature: float = 0.5,
        seed=12345,
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
    ):
        """Sends a message to the Gemini API and processes the response.

        Args:
            model (str, optional): The model to use. Defaults to None.
            temperature (float): Controls randomness. Defaults to 0.5.
            seed (int): For back compatibility. Defaults to 12345.
            system_content (str): System prompt. Defaults to 'You are a helpful assistant.'.
            human_content (str): User prompt. Defaults to 'Hello!'.
            is_structured (bool): Whether to expect a structured response. Defaults to False.
            response_format (type, optional): Schema for structured response parsing. Defaults to None.

        Returns:
            Union[str, dict, None]: The response content, either as a string, parsed dict, or None on error.
        """
        if model is None:
            model = self.default_model
        generation_config = {"temperature": temperature}
        prompt = f"{system_content}\n\n---\n\n{human_content}"
        model_instance = self.llm.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )
        try:
            response = self._generate_with_retry(model_instance, prompt, generation_config)
            self.response_data = {
                "model": response.model_version,
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }
            print(self.response_data)
            if is_structured and response_format is not None:
                return self._parse_structured_response(response.text, response_format)
            return response.text
        except Exception as e:
            raise Exception(f"API request failed after retries: {str(e)}")


class APIProcessor:
    def __init__(self, provider: Literal["openai", "gemini"] ="openai"):
        """Initializes the API processor for the specified provider.

        Args:
            provider (Literal["openai", "gemini"]): The API provider to use. Defaults to 'openai'.
        """
        self.provider = provider.lower()
        if self.provider == "openai":
            self.processor = BaseOpenaiProcessor()
        elif self.provider == "gemini":
            self.processor = BaseGeminiProcessor()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """Routes the send_message call to the appropriate processor.

        Args:
            model (str, optional): The model to use. Defaults to None.
            temperature (float): Controls randomness. Defaults to 0.5.
            seed (int, optional): Seed for deterministic outputs. Defaults to None.
            system_content (str): System prompt. Defaults to 'You are a helpful assistant.'.
            human_content (str): User prompt. Defaults to 'Hello!'.
            is_structured (bool): Whether to expect a structured response. Defaults to False.
            response_format (type, optional): Schema for structured response parsing. Defaults to None.
            **kwargs: Additional parameters for the API request.

        Returns:
            Union[str, dict, None]: The response content from the selected processor.
        """
        if model is None:
            model = self.processor.default_model
        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs
        )

    def get_answer_from_rag_context(self, question, rag_context, schema, model):
        """Gets an answer from RAG context using the specified schema and model.

        Args:
            question (str): The question to answer.
            rag_context (str): The RAG context for the question.
            schema (str): The schema type for the response (e.g., 'name', 'number').
            model (str): The model to use for the request.

        Returns:
            dict: The structured answer from the API.
        """
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)
        answer_dict = self.processor.send_message(
            model=model,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            is_structured=True,
            response_format=response_format
        )
        self.response_data = self.processor.response_data
        return answer_dict

    def _build_rag_context_prompts(self, schema):
        """Builds prompts and response format for RAG context queries.

        Args:
            schema (str): The schema type (e.g., 'name', 'number', 'boolean').

        Returns:
            tuple: (system_prompt, response_format, user_prompt) for the query.
        """
        use_schema_prompt = True if self.provider == "gemini" else False
        if schema == "name":
            system_prompt = (prompts.AnswerWithRAGContextNamePrompt.system_prompt_with_schema 
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamePrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamePrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamePrompt.user_prompt
        elif schema == "number":
            system_prompt = (prompts.AnswerWithRAGContextNumberPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNumberPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNumberPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNumberPrompt.user_prompt
        elif schema == "boolean":
            system_prompt = (prompts.AnswerWithRAGContextBooleanPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextBooleanPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextBooleanPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextBooleanPrompt.user_prompt
        elif schema == "names":
            system_prompt = (prompts.AnswerWithRAGContextNamesPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamesPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamesPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamesPrompt.user_prompt
        elif schema == "comparative":
            system_prompt = (prompts.ComparativeAnswerPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.ComparativeAnswerPrompt.system_prompt)
            response_format = prompts.ComparativeAnswerPrompt.AnswerSchema
            user_prompt = prompts.ComparativeAnswerPrompt.user_prompt
        else:
            raise ValueError(f"Unsupported schema: {schema}")
        return system_prompt, response_format, user_prompt

    def get_rephrased_questions(self, original_question: str, companies: List[str]):
        """Uses the LLM to break down a comparative question into individual questions.

        Args:
            original_question (str): The original comparative question.
            companies (List[str]): List of company names to rephrase for.

        Returns:
            Dict[str, str]: Dictionary mapping company names to rephrased questions.
        """
        answer_dict = self.processor.send_message(
            system_content=prompts.RephrasedQuestionsPrompt.system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                companies=", ".join([f'"{company}"' for company in companies])
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.RephrasedQuestions
        )
        questions_dict = {item["company_name"]: item["question"] for item in answer_dict["questions"]}
        return questions_dict


class AsyncOpenaiProcessor:
    
    def _get_unique_filepath(self, base_filepath):
        """Generates a unique filepath by appending a counter if the file exists.

        Args:
            base_filepath (str): The base filepath to check.

        Returns:
            str: A unique filepath.
        """
        if not os.path.exists(base_filepath):
            return base_filepath
        base, ext = os.path.splitext(base_filepath)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        return f"{base}_{counter}{ext}"

    async def process_structured_ouputs_requests(
        self,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        queries=None,
        response_format=None,
        requests_filepath='./temp_async_llm_requests.jsonl',
        save_filepath='./temp_async_llm_results.jsonl',
        preserve_requests=False,
        preserve_results=True,
        request_url="https://api.openai.com/v1/chat/completions",
        max_requests_per_minute=3_500,
        max_tokens_per_minute=3_500_000,
        token_encoding_name="o200k_base",
        max_attempts=5,
        logging_level=20,
        progress_callback=None
    ):
        """Processes multiple structured output requests asynchronously to the OpenAI API.

        Args:
            model (str): The model to use. Defaults to 'gpt-4o-mini-2024-07-18'.
            temperature (float): Controls randomness. Defaults to 0.5.
            seed (int, optional): Seed for deterministic outputs. Defaults to None.
            system_content (str): System prompt. Defaults to 'You are a helpful assistant.'.
            queries (list): List of user queries to process. Defaults to None.
            response_format (type, optional): Schema for structured response parsing. Defaults to None.
            requests_filepath (str): Path for temporary request JSONL file. Defaults to './temp_async_llm_requests.jsonl'.
            save_filepath (str): Path for saving results JSONL file. Defaults to './temp_async_llm_results.jsonl'.
            preserve_requests (bool): Whether to keep the request file. Defaults to False.
            preserve_results (bool): Whether to keep the results file. Defaults to True.
            request_url (str): API endpoint URL. Defaults to 'https://api.openai.com/v1/chat/completions'.
            max_requests_per_minute (int): Max requests per minute. Defaults to 3,500.
            max_tokens_per_minute (int): Max tokens per minute. Defaults to 3,500,000.
            token_encoding_name (str): Token encoding scheme. Defaults to 'o200k_base'.
            max_attempts (int): Max retry attempts. Defaults to 5.
            logging_level (int): Logging level. Defaults to 20 (INFO).
            progress_callback (callable, optional): Callback for progress updates. Defaults to None.

        Returns:
            list: List of dicts containing questions and validated answers.
        """
        jsonl_requests = []
        for idx, query in enumerate(queries):
            request = {
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query},
                ],
                'response_format': type_to_response_format_param(response_format),
                'metadata': {'original_index': idx}
            }
            jsonl_requests.append(request)
            
        requests_filepath = self._get_unique_filepath(requests_filepath)
        save_filepath = self._get_unique_filepath(save_filepath)

        with open(requests_filepath, "w") as f:
            for request in jsonl_requests:
                json_string = json.dumps(request)
                f.write(json_string + "\n")

        total_requests = len(jsonl_requests)

        async def monitor_progress():
            last_count = 0
            while True:
                try:
                    with open(save_filepath, 'r') as f:
                        current_count = sum(1 for _ in f)
                        if current_count > last_count:
                            if progress_callback:
                                for _ in range(current_count - last_count):
                                    progress_callback()
                            last_count = current_count
                        if current_count >= total_requests:
                            break
                except FileNotFoundError:
                    pass
                await asyncio.sleep(0.1)

        async def process_with_progress():
            await asyncio.gather(
                process_api_requests_from_file(
                    requests_filepath=requests_filepath,
                    save_filepath=save_filepath,
                    request_url=request_url,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    max_requests_per_minute=max_requests_per_minute,
                    max_tokens_per_minute=max_tokens_per_minute,
                    token_encoding_name=token_encoding_name,
                    max_attempts=max_attempts,
                    logging_level=logging_level
                ),
                monitor_progress()
            )

        await process_with_progress()

        with open(save_filepath, "r") as f:
            validated_data_list = []
            results = []
            for line_number, line in enumerate(f, start=1):
                raw_line = line.strip()
                try:
                    result = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Line {line_number}: Failed to load JSON from line: {raw_line}")
                    continue
                finish_reason = result[1]['choices'][0].get('finish_reason', '')
                if finish_reason != "stop":
                    print(f"[WARNING] Line {line_number}: finish_reason is '{finish_reason}' (expected 'stop').")
                try:
                    answer_content = result[1]['choices'][0]['message']['content']
                    answer_parsed = json.loads(answer_content)
                    answer = response_format(**answer_parsed).model_dump()
                except Exception as e:
                    print(f"[ERROR] Line {line_number}: Failed to parse answer JSON. Error: {e}.")
                    answer = ""
                results.append({
                    'index': result[2],
                    'question': result[0]['messages'],
                    'answer': answer
                })
            validated_data_list = [
                {'question': r['question'], 'answer': r['answer']} 
                for r in sorted(results, key=lambda x: x['index']['original_index'])
            ]

        if not preserve_requests:
            os.remove(requests_filepath)
        if not preserve_results:
            os.remove(save_filepath)
        else:
            with open(save_filepath, "r") as f:
                results = [json.loads(line) for line in f]
            sorted_results = sorted(results, key=lambda x: x[2]['original_index'])
            with open(save_filepath, "w") as f:
                for result in sorted_results:
                    json_string = json.dumps(result)
                    f.write(json_string + "\n")
        return validated_data_list