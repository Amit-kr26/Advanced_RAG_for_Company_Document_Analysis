import aiohttp
import argparse
import asyncio
import json
import logging
import os
import re
import tiktoken
import time
from dataclasses import dataclass, field


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
):
    """Processes API requests from a file in parallel, throttling to stay under rate limits.

    Args:
        requests_filepath (str): Path to the file containing API request JSON objects.
        save_filepath (str): Path where results and errors are saved in JSONL format.
        request_url (str): The URL of the API endpoint to send requests to.
        api_key (str): API key for authentication with the API.
        max_requests_per_minute (float): Maximum number of requests allowed per minute.
        max_tokens_per_minute (float): Maximum number of tokens allowed per minute.
        token_encoding_name (str): Name of the token encoding scheme (e.g., 'cl100k_base').
        max_attempts (int): Maximum number of attempts to retry a failed request.
        logging_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001

    logging.basicConfig(level=logging_level)

    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    if "/deployments" in request_url:
        request_header = {"api-key": f"{api_key}"}

    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()
    status_tracker = StatusTracker()
    next_request = None

    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    file_not_finished = True

    with open(requests_filepath) as file:
        requests = file.__iter__()
        async with aiohttp.ClientSession() as session:
            while True:
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                    elif file_not_finished:
                        try:
                            request_json = json.loads(next(requests))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                        except StopIteration:
                            file_not_finished = False

                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None

                if status_tracker.num_tasks_in_progress == 0:
                    break

                await asyncio.sleep(seconds_to_sleep_each_loop)

                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)


@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0


@dataclass
class APIRequest:
    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Makes an API call, handles responses, and manages retries and errors.

        Args:
            session (aiohttp.ClientSession): The HTTP session for making the API call.
            request_url (str): The URL of the API endpoint.
            request_header (dict): Headers for the API request, including authentication.
            retry_queue (asyncio.Queue): Queue to hold requests that need to be retried.
            save_filepath (str): Path to save the results or errors in JSONL format.
            status_tracker (StatusTracker): Tracks the progress and status of API requests.
        """
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                status_tracker.num_api_errors += 1
                error = response
                if "rate limit" in response["error"].get("message", "").lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1

        except Exception as e:
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1


def api_endpoint_from_url(request_url: str) -> str:
    """Extracts the API endpoint from the provided request URL.

    Args:
        request_url (str): The full URL of the API request.

    Returns:
        str: The extracted API endpoint path.
    """
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        match = re.search(
            r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url
        )
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Appends a JSON payload to the end of a JSONL file.

    Args:
        data: The data to be serialized and appended as a JSON line.
        filename (str): The path to the JSONL file where data will be appended.
    """
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
) -> int:
    """Counts the number of tokens consumed by a request, supporting completion and embedding requests.

    Args:
        request_json (dict): The JSON payload of the API request.
        api_endpoint (str): The API endpoint being called (e.g., 'completions', 'embeddings').
        token_encoding_name (str): The name of the token encoding scheme (e.g., 'cl100k_base').

    Returns:
        int: The total number of tokens consumed by the request.
    """
    encoding = tiktoken.get_encoding(token_encoding_name)
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens -= 1
            num_tokens += 2
            return num_tokens + completion_tokens
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )


def task_id_generator_function():
    """Generates a sequence of unique integer task IDs starting from 0.

    Yields:
        int: The next task ID in the sequence (0, 1, 2, ...).
    """
    task_id = 0
    while True:
        yield task_id
        task_id += 1